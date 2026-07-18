#!/usr/bin/env python3
"""
adaptive_place_right/main.py
============================
Standalone CLI driver for the **right-arm** wrist force-torque placement
(tactile-free build).

The arm descends until the wrist FT sensor detects table contact, presses
until the table bears the load (``gamma`` from the FT), then releases the
object via an injected gripper hook and lifts.  There is no fingertip
tactile feedback and no grip-decay loop.

The gripper is an injected :class:`gripper_hook.GripperHook`.  This standalone
build defaults to ``NoopGripper`` (logs only); wire your own Robotiq-only
wrapper in ``main()`` before running on hardware that should actually grip.

Usage:

    # From the ft_place_right/ folder (this folder must contain the
    # sibling dirs handover/ and admittance_control/; xarm_sdk is taken
    # from the system install):
    cd /home/nvidia/niu/ft_place_right

    # Make sure the xarm drivers, FT sensor, and MoveIt stack are up.
    python3 -m adaptive_place_right.main --g-obj-mode manual --object-weight-N 1.2

Common overrides:

    # Auto weight characterisation needs a real gripper hook (open/close
    # to sample F_empty/F_loaded).  With the NoopGripper stub, use manual:
    python3 -m adaptive_place_right.main --g-obj-mode manual --object-weight-N 1.2

    # Skip the pre-place jointspace move (descend from current pose).
    python3 -m adaptive_place_right.main --no-pre-place-move

    # Record a CSV trace of Fz, gamma, ... for post-run plots.
    python3 -m adaptive_place_right.main --log-csv /tmp/adp_r_run.csv

The script leaves the hardware in a safe state (right arm at the
lifted pose, gripper opened via the hook) even on Ctrl+C.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace as _dc_replace
from typing import List, Optional

# --- sibling-aware bootstrap -----------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
for _sib in (
    os.path.join(_WORKSPACE_ROOT, "admittance_control"),
    os.path.join(_WORKSPACE_ROOT, "xarm_sdk"),
    _WORKSPACE_ROOT,
):
    if _sib not in sys.path:
        sys.path.insert(0, _sib)

import rclpy  # noqa: E402

from xarm_sdk import XARM_manager, TopicPublisher, ActionCall  # noqa: E402
from xarm_sdk.tools import set_node_parameter  # noqa: E402

from AdmittanceController_v3 import TFHelper  # noqa: E402

from handover.admittance_arm import AdmittanceArm  # noqa: E402
from adaptive_place_right.spin_thread import _SpinThread  # noqa: E402
from adaptive_place_right.gripper_hook import NoopGripper  # noqa: E402

from adaptive_place_right.config import (  # noqa: E402
    RightAdaptivePlaceConfig, DEFAULT_RIGHT_ADAPTIVE_PLACE, AdmittanceGains,
)
from adaptive_place_right.adaptive_placer_right import (  # noqa: E402
    RightAdaptivePlacer,
)


def _parse_joints_csv(s: Optional[str]) -> Optional[List[float]]:
    if s is None:
        return None
    try:
        js = [float(x) for x in s.split(",")]
    except ValueError as e:
        raise SystemExit(f"--pre-place-joints: could not parse '{s}': {e}")
    if len(js) != 7:
        raise SystemExit(
            f"--pre-place-joints: expected 7 values, got {len(js)}"
        )
    return js


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Force-driven contact-decay placement on the RIGHT arm "
            "(Robotiq 2F + Tac3D).  Standalone test harness."
        ),
    )
    p.add_argument(
        "--g-obj-mode", choices=("auto", "manual"),
        default=DEFAULT_RIGHT_ADAPTIVE_PLACE.g_obj_estimation,
        help="How to estimate G_obj (object weight in waist-Z Newtons).",
    )
    p.add_argument(
        "--object-weight-N", type=float,
        default=DEFAULT_RIGHT_ADAPTIVE_PLACE.object_weight_N,
        help="Manual G_obj override (used when --g-obj-mode=manual).",
    )
    p.add_argument(
        "--k", type=float,
        default=DEFAULT_RIGHT_ADAPTIVE_PLACE.decay_exponent_k,
        help="Decay exponent for C_target = C_initial * (1 - gamma)^k.",
    )
    p.add_argument(
        "--gamma-release", type=float,
        default=DEFAULT_RIGHT_ADAPTIVE_PLACE.gamma_release_threshold,
        help="Release when gamma stays above this (default: 0.90).",
    )
    p.add_argument(
        "--descent-speed", type=float,
        default=DEFAULT_RIGHT_ADAPTIVE_PLACE.descent_speed_mps,
        help=(
            "SLOW-phase admittance equilibrium descent speed (m/s).  "
            f"HARD-CLIPPED at descent_speed_max_mps="
            f"{DEFAULT_RIGHT_ADAPTIVE_PLACE.descent_speed_max_mps:.3f} m/s "
            f"(default {DEFAULT_RIGHT_ADAPTIVE_PLACE.descent_speed_mps:.3f}). "
            "Going above ~20 mm/s makes the QP saturate and z_eq "
            "outruns the arm.  For overall descent time, raise "
            "--fast-descent-speed instead (the slow phase only covers "
            "the last few cm near contact)."
        ),
    )
    p.add_argument(
        "--fast-descent", dest="fast_descent_enabled",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RIGHT_ADAPTIVE_PLACE.fast_descent_enabled,
        help=(
            "Two-phase descent: fast direct-QP (admittance paused) for "
            "the bulk of travel, then slow admittance for the last "
            f"{DEFAULT_RIGHT_ADAPTIVE_PLACE.fast_slow_band_m * 100.0:.1f} cm "
            "near contact.  Default: on."
        ),
    )
    p.add_argument(
        "--fast-descent-speed", type=float,
        default=DEFAULT_RIGHT_ADAPTIVE_PLACE.fast_descent_speed_mps,
        help=(
            "Fast-phase descent speed (m/s), used when --fast-descent is on.  "
            f"Default: {DEFAULT_RIGHT_ADAPTIVE_PLACE.fast_descent_speed_mps:.3f} "
            "(60 mm/s).  Safe range is roughly 40-80 mm/s."
        ),
    )
    p.add_argument(
        "--fast-slow-band", type=float,
        default=DEFAULT_RIGHT_ADAPTIVE_PLACE.fast_slow_band_m,
        help=(
            "Metres above the safety floor where the slow admittance phase "
            f"takes over from the fast QP phase (default "
            f"{DEFAULT_RIGHT_ADAPTIVE_PLACE.fast_slow_band_m:.3f} = 4 cm)."
        ),
    )
    p.add_argument(
        "--force-threshold", type=float,
        default=DEFAULT_RIGHT_ADAPTIVE_PLACE.descent_force_threshold_N,
        help="Waist-Z delta above baseline that trips descent contact.",
    )
    p.add_argument(
        "--grasp-first", dest="grasp_first",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RIGHT_ADAPTIVE_PLACE.grasp_first,
        help=(
            "Run open + close_to_contact + adjust_grasp before descent "
            "(disable when the gripper is already holding the object, "
            "e.g. chaining after graspnet_grasp)."
        ),
    )
    p.add_argument(
        "--contact-timeout", type=float,
        default=DEFAULT_RIGHT_ADAPTIVE_PLACE.contact_timeout_sec,
        help=(
            "Seconds to wait for Tac3D contact in close_to_contact "
            "(default: 20).  If the gripper closes on nothing, you'll "
            "hit this timeout."
        ),
    )
    p.add_argument(
        "--target-cf", type=float, default=None,
        help=(
            "Override target_cf for both the hold phase "
            "(TactileFeedback.adjust_grasp) and the decay phase. "
            "Default: RightGripperConfig.target_cf = "
            f"{DEFAULT_RIGHT_ADAPTIVE_PLACE.right_gripper.target_cf}. "
            "Tighter cf (e.g. 0.45) helps on slippery / heavy objects; "
            "loose cf (e.g. 0.20) reduces over-squeeze on deformables."
        ),
    )
    p.add_argument(
        "--hold-mode",
        choices=("adaptive_pd", "motor_hold"),
        default=None,
        help=(
            "Hold-thread strategy between close_to_contact and the "
            "Phase C handoff.  'adaptive_pd' (default) runs the "
            "stock TactileFeedback.adjust_grasp PD loop in a thread "
            "and only fires goTo when |residual_cf|>tolerance -- "
            "same as the graspnet_grasp pipeline and recommended for "
            "deformable objects.  'motor_hold' periodically refreshes "
            "goTo(locked_pos, force=hold_force_pct) -- sustains motor "
            "torque and can cause visible creep on soft objects; "
            "only use for rigid targets."
        ),
    )
    p.add_argument(
        "--hold-force-pct", type=float, default=None,
        help=(
            "Motor-torque ceiling (0-100 percent) used by the hold "
            "thread.  Default "
            f"{DEFAULT_RIGHT_ADAPTIVE_PLACE.hold_force_pct:.0f} percent. "
            "Lower values (e.g. 25-30) produce a more compliant hold "
            "on delicate / deformable objects."
        ),
    )
    p.add_argument(
        "--pre-place-joints", type=str, default=None,
        help=(
            "Comma-separated 7 joint values for the RIGHT pre-place pose. "
            "Defaults to the config's pre_place_joints "
            "([0.0, -1.18, 0.0, -1.3, 0.30, -0.20, 0.0]).  Pass an empty "
            "string '' to skip the pre-place move."
        ),
    )
    p.add_argument(
        "--no-pre-place-move", action="store_true",
        help=(
            "Skip the RIGHT jointspace move to pre_place_joints "
            "(descend from wherever the RIGHT arm currently is)."
        ),
    )
    p.add_argument(
        "--home-move", action="store_true",
        help=(
            "First drive the RIGHT arm to the config's right_home_joints "
            "before anything else (off by default)."
        ),
    )
    p.add_argument(
        "--log-csv", type=str, default=None,
        help=(
            "Optional CSV path for (t, phase, Fz, F_support, gamma, "
            "target_cf, cf, mu, z_eq) trace."
        ),
    )
    # ---- Phase D (release + lift) speed knobs --------------------------
    p.add_argument(
        "--lift-speed", type=float, default=None,
        help=(
            "Speed (m/s) of the admittance-equilibrium lift after release. "
            f"Default {DEFAULT_RIGHT_ADAPTIVE_PLACE.lift_speed_mps:.3f} "
            "(20 mm/s).  Going to 0.05-0.08 cuts ~3 s off the tail."
        ),
    )
    p.add_argument(
        "--lift-height", type=float, default=None,
        help=(
            "Distance (m) lifted above release pose.  Default "
            f"{DEFAULT_RIGHT_ADAPTIVE_PLACE.lift_after_release_m:.3f} "
            "(8 cm).  Drop to 0.04 (4 cm) if clearance allows."
        ),
    )
    p.add_argument(
        "--post-release-dwell", type=float, default=None,
        help=(
            "Seconds to wait after the gripper opens before starting the "
            f"lift.  Default "
            f"{DEFAULT_RIGHT_ADAPTIVE_PLACE.post_release_dwell_sec:.2f} s."
        ),
    )
    # ---- Phase C (decay) speed knobs -----------------------------------
    # (Only active when G_obj >= min_Gobj_N; otherwise decay is skipped
    # and the release happens on contact dwell.)
    p.add_argument(
        "--gamma-debounce", type=int, default=None,
        help=(
            "Number of consecutive gamma>=threshold samples required to "
            f"trigger release.  Default "
            f"{DEFAULT_RIGHT_ADAPTIVE_PLACE.gamma_release_debounce}. "
            "Lower (e.g. 3) releases sooner."
        ),
    )
    p.add_argument(
        "--decay-release-step", type=float, default=None,
        help=(
            "Max per-step release size in Robotiq encoder ticks during "
            f"decay.  Default "
            f"{DEFAULT_RIGHT_ADAPTIVE_PLACE.decay_max_step_release_ticks:.1f}. "
            "Larger (e.g. 6.0) opens the jaws faster once decay kicks in."
        ),
    )
    p.add_argument(
        "--transfer-press-depth", type=float, default=None,
        help=(
            "Meters to drop the admittance target_Z below FAST/SLOW "
            "contact at the start of Phase C.  Combined with "
            "transfer_gains.K_z this produces a steady press force of "
            "K_z*depth Newtons that drives gamma up BEFORE the gripper "
            f"opens.  Default "
            f"{DEFAULT_RIGHT_ADAPTIVE_PLACE.transfer_press_depth_m * 1000.0:.1f}"
            " mm.  Raise to 0.020 (20 mm) for very light (<200 g) "
            "objects if gamma still stalls at 0.  Set to 0 to disable "
            "(restores the old hover-in-place behaviour)."
        ),
    )
    p.add_argument(
        "--transfer-kz", type=float, default=None,
        help=(
            "Phase-C admittance Z-stiffness (N/m).  Default "
            f"{float(list(DEFAULT_RIGHT_ADAPTIVE_PLACE.transfer_gains.K)[2]):.1f}. "
            "Raise (e.g. 60) for more press force per mm of "
            "press-depth; lower (e.g. 10) for softer placements on "
            "fragile surfaces.  K_z=1 (old default) disables the press "
            "effect -- the arm just drifts off the table."
        ),
    )
    p.add_argument(
        "--verbose", action=argparse.BooleanOptionalAction, default=True,
    )
    return p


def build_config_from_args(args) -> RightAdaptivePlaceConfig:
    cfg = DEFAULT_RIGHT_ADAPTIVE_PLACE

    # Optional pre-place-joints override.
    pp: Optional[List[float]] = cfg.pre_place_joints
    if args.no_pre_place_move:
        pp = None
    elif args.pre_place_joints is not None:
        if args.pre_place_joints.strip() == "":
            pp = None
        else:
            pp = _parse_joints_csv(args.pre_place_joints)

    # Hard-clip --descent-speed at the configured ceiling.  See
    # ``adaptive_place/main.py`` for the rationale -- same failure
    # mode applies to the right arm's QP.
    requested_ds = float(args.descent_speed)
    ds_cap = float(cfg.descent_speed_max_mps)
    if requested_ds > ds_cap:
        print(
            f"[ADP-R] WARNING: --descent-speed {requested_ds:.3f} m/s "
            f"exceeds the configured safety cap descent_speed_max_mps="
            f"{ds_cap:.3f} m/s.  Clipping -- the QP cannot reliably follow "
            "faster refs and z_eq will otherwise outrun the arm.  Raise "
            "descent_speed_max_mps in config.py only if you've also "
            "stiffened descent_gains to track the new speed."
        )
        requested_ds = ds_cap
    if requested_ds <= 0.0:
        print(
            f"[ADP-R] WARNING: --descent-speed {requested_ds} <= 0; using "
            f"default {DEFAULT_RIGHT_ADAPTIVE_PLACE.descent_speed_mps:.3f} "
            "m/s."
        )
        requested_ds = DEFAULT_RIGHT_ADAPTIVE_PLACE.descent_speed_mps

    FAST_SPEED_CAP = 0.10  # 100 mm/s -- upper edge of safe
    requested_fs = float(args.fast_descent_speed)
    if requested_fs > FAST_SPEED_CAP:
        print(
            f"[ADP-R] WARNING: --fast-descent-speed {requested_fs:.3f} m/s "
            f"exceeds the safety cap {FAST_SPEED_CAP:.3f} m/s.  Clipping."
        )
        requested_fs = FAST_SPEED_CAP
    if requested_fs <= 0.0:
        print(
            "[ADP-R] WARNING: --fast-descent-speed must be > 0; using "
            f"default {DEFAULT_RIGHT_ADAPTIVE_PLACE.fast_descent_speed_mps:.3f} "
            "m/s."
        )
        requested_fs = DEFAULT_RIGHT_ADAPTIVE_PLACE.fast_descent_speed_mps

    cfg = _dc_replace(
        cfg,
        g_obj_estimation=args.g_obj_mode,
        object_weight_N=float(args.object_weight_N),
        decay_exponent_k=float(args.k),
        gamma_release_threshold=float(args.gamma_release),
        descent_speed_mps=requested_ds,
        fast_descent_enabled=bool(args.fast_descent_enabled),
        fast_descent_speed_mps=requested_fs,
        fast_slow_band_m=float(args.fast_slow_band),
        descent_force_threshold_N=float(args.force_threshold),
        grasp_first=bool(args.grasp_first),
        contact_timeout_sec=float(args.contact_timeout),
        pre_place_joints=pp,
        log_csv_path=(args.log_csv if args.log_csv else None),
        verbose=bool(args.verbose),
    )
    # Optional target_cf override; propagate into the nested
    # RightGripperConfig so RightGripperTactile picks it up.
    if args.target_cf is not None:
        cfg = _dc_replace(
            cfg,
            right_gripper=_dc_replace(
                cfg.right_gripper, target_cf=float(args.target_cf)
            ),
        )

    # Optional hold-thread tuning.
    if getattr(args, "hold_mode", None) is not None:
        cfg = _dc_replace(cfg, hold_mode=str(args.hold_mode))
    if getattr(args, "hold_force_pct", None) is not None:
        hfp = max(0.0, min(100.0, float(args.hold_force_pct)))
        cfg = _dc_replace(cfg, hold_force_pct=hfp)

    # Phase-D release/lift speed overrides.
    if getattr(args, "lift_speed", None) is not None:
        ls = max(1e-3, float(args.lift_speed))
        if ls > 0.15:
            print(
                f"[ADP-R] WARNING: --lift-speed {ls:.3f} m/s is very high; "
                "clipping to 0.15 m/s to stay within safe admittance "
                "tracking."
            )
            ls = 0.15
        cfg = _dc_replace(cfg, lift_speed_mps=ls)
    if getattr(args, "lift_height", None) is not None:
        lh = max(0.0, float(args.lift_height))
        cfg = _dc_replace(cfg, lift_after_release_m=lh)
    if getattr(args, "post_release_dwell", None) is not None:
        prd = max(0.0, float(args.post_release_dwell))
        cfg = _dc_replace(cfg, post_release_dwell_sec=prd)

    # Phase-C decay speed overrides.
    if getattr(args, "gamma_debounce", None) is not None:
        gdb = max(1, int(args.gamma_debounce))
        cfg = _dc_replace(cfg, gamma_release_debounce=gdb)
    if getattr(args, "decay_release_step", None) is not None:
        drs = max(0.5, float(args.decay_release_step))
        cfg = _dc_replace(cfg, decay_max_step_release_ticks=drs)

    # Phase-C press tuning.  Clamp to safe range to avoid smashing
    # objects if the user fat-fingers a big number.
    if getattr(args, "transfer_press_depth", None) is not None:
        pd = max(0.0, min(0.05, float(args.transfer_press_depth)))
        cfg = _dc_replace(cfg, transfer_press_depth_m=pd)
    if getattr(args, "transfer_kz", None) is not None:
        kz_new = max(0.5, min(200.0, float(args.transfer_kz)))
        base = cfg.transfer_gains
        K_new = list(base.K)
        K_new[2] = kz_new
        # Keep damping roughly critical: B_z ~ 2*sqrt(K*M).  With
        # M_z = 0.15 this gives B ~ 0.775*sqrt(K).  Match the default
        # ratio (B_z=6.0 at K_z=30.0 -> 1.095*sqrt(K)) so we stay
        # slightly over-damped at all K_z values.
        B_new = list(base.B)
        B_new[2] = 1.1 * (kz_new ** 0.5)
        cfg = _dc_replace(
            cfg,
            transfer_gains=AdmittanceGains(
                M=list(base.M),
                B=B_new,
                K=K_new,
                M_rot=list(base.M_rot),
                B_rot=list(base.B_rot),
                K_rot=list(base.K_rot),
                max_vel=base.max_vel,
                max_omega=base.max_omega,
                rot_lead_time=base.rot_lead_time,
            ),
        )
    return cfg


def _right_giver_gains():
    """Boot gains for the right admittance arm during the first ~2 s of
    FT offset calibration.  We pick RIGHT_GIVER_GAINS from handover
    (K=[15,15,15]) for a known-good "stiff-but-not-rigid" baseline; the
    placer switches to ``cfg.hold_gains`` during characterisation and
    ``cfg.descent_gains`` at the start of descent.
    """
    from handover.config import RIGHT_GIVER_GAINS
    return RIGHT_GIVER_GAINS


def run_chained_placement(
    *,
    xarm: XARM_manager,
    right_gripper,
    argv: List[str],
) -> bool:
    """Run :class:`RightAdaptivePlacer` in-process with an existing gripper.

    Intended for ``bottle_cup_pour_place`` Phase 6: the parent already
    holds the cup with ``start_adaptive_hold()`` and must **not** call
    :meth:`~handover.right_gripper.RightGripperTactile.suspend_tac3d_for_subprocess`,
    which drops Tac3D UDP receivers and creates a long dead zone.

    The parent's tactile PF thread stays in-process until Phase C replaces it with
    decay (:meth:`~handover.right_gripper.RightGripperTactile.stop_adaptive_thread`);
    Tac3D UDP stays bound (no subprocess suspend).

    Unlike :func:`main`, chained placement **does not** call
    ``xarm_deactivate_all_controller``: that halts **every** controller (both
    arms) at once, would unsettle the left arm while holding the bottle, and can
    invalidate the rest of the scripted joint sequence (including the recorded
    cup-grasp pose replay). Only the RIGHT QP stack is brought up below.

    Does **not** destroy ``xarm`` or call ``rclpy.shutdown()`` — the caller
    owns the ROS context.

    Parameters
    ----------
    argv
        Tokens as for ``python -m adaptive_place_right.main`` (no script name).
    """
    args = build_parser().parse_args(list(argv))
    cfg = build_config_from_args(args)

    # FT-only build: no tactile preflight (Robotiq/Tac3D import checks removed).

    topic_pub = TopicPublisher(xarm)
    action = ActionCall(xarm)
    tf_helper = TFHelper(xarm)

    xarm.hardware_arm_enable(True)
    xarm.hardware_arm_mode(3)
    xarm.get_logger().info(
        "[ADP-R] Chained placement bring-up (no deactivate-all; left arm state "
        "unchanged until RIGHT QP switch)."
    )

    if args.home_move:
        xarm.get_logger().info(
            f"[ADP-R] Moving RIGHT arm to home joints {cfg.right_home_joints}"
        )
        action.jointspace_arm_R_controller(list(cfg.right_home_joints))
        xarm.get_logger().info("[ADP-R] RIGHT arm at home.")

    if cfg.pre_place_joints is not None:
        xarm.get_logger().info(
            f"[ADP-R] Moving RIGHT arm to pre-place joints "
            f"{cfg.pre_place_joints}"
        )
        action.jointspace_arm_R_controller(list(cfg.pre_place_joints))
        xarm.get_logger().info("[ADP-R] RIGHT arm at pre-place.")
    else:
        xarm.get_logger().info(
            "[ADP-R] Skipping pre-place move (descending from current pose)."
        )

    xarm.get_logger().info("[ADP-R] Using injected gripper hook (chained).")

    xarm.xarm_activate_controller([cfg.right_qp_controller])
    set_node_parameter(
        xarm, cfg.right_qp_controller, "otg_p_step", cfg.otg_p_step
    )
    set_node_parameter(
        xarm, cfg.right_qp_controller, "otg_r_step", cfg.otg_r_step
    )
    xarm.get_logger().info("[ADP-R] RIGHT QP controller active.")

    t_end = time.monotonic() + 0.4
    while time.monotonic() < t_end and rclpy.ok():
        rclpy.spin_once(xarm, timeout_sec=0.02)

    right_adm = AdmittanceArm(
        side="right",
        xarm_manager=xarm,
        topic_pub=topic_pub,
        tf_helper=tf_helper,
        tcp_frame=cfg.right_tcp_frame,
        waist_frame=cfg.waist_frame,
        force_topic=cfg.right_force_topic,
        qp_controller=cfg.right_qp_controller,
        calib_path=cfg.right_ft_calibration_path,
        initial_gains=_right_giver_gains(),
        force_proc=cfg.force_proc,
        loop_period=0.004,
        name="[ADP-R/RIGHT-ADM]",
    )
    if not right_adm.capture_current_pose_as_target(timeout_sec=0.5):
        raise RuntimeError(
            "[ADP-R] Could not capture RIGHT TCP pose as admittance target."
        )

    spin = _SpinThread(xarm, spin_timeout_sec=0.01)
    spin.start()
    right_adm.start()
    xarm.get_logger().info(
        f"[ADP-R] RIGHT admittance running; waiting "
        f"{cfg.admittance_calibration_sec:.1f} s for FT-offset calibration..."
    )
    time.sleep(float(cfg.admittance_calibration_sec))

    placer = RightAdaptivePlacer(
        cfg=cfg,
        xarm_manager=xarm,
        topic_pub=topic_pub,
        tf_helper=tf_helper,
        right_gripper=right_gripper,
        right_adm=right_adm,
    )
    ok = False
    try:
        ok = placer.run()
        if ok:
            xarm.get_logger().info(
                "[ADP-R] Adaptive place finished: RELEASED (chained)."
            )
        else:
            xarm.get_logger().warn(
                "[ADP-R] Adaptive place finished WITHOUT clean release "
                "(chained; see logs above)."
            )
    except KeyboardInterrupt:
        print("\n[ADP-R] KeyboardInterrupt -- shutting down safely (chained).")
    except Exception as e:
        print(f"[ADP-R] ERROR (chained): {e!r}")
        import traceback

        traceback.print_exc()
    finally:
        try:
            right_adm.stop()
        except Exception as e:
            print(f"[ADP-R] right_adm.stop warning: {e}")
        try:
            spin.stop()
        except Exception as e:
            print(f"[ADP-R] spin.stop warning: {e}")
        if not ok:
            print(
                "[ADP-R] Pipeline did NOT finish cleanly; the right "
                "gripper was left CLOSED on purpose (payload safety)."
            )
        # Skip right_gripper.shutdown() here — the placer already
        # released the gripper during release_and_lift when ok=True,
        # and the caller's finally block handles final cleanup.
        try:
            xarm.xarm_deactivate_all_controller()
        except Exception:
            pass
        print(f"[ADP-R] Exit chained (ok={ok}).")
        sys.stdout.flush()
        sys.stderr.flush()
    return ok


def main() -> None:
    args = build_parser().parse_args()
    cfg = build_config_from_args(args)

    # FT-only build: no tactile preflight (Robotiq/Tac3D import checks removed).

    # ---------------- ROS / hardware bring-up ------------------------
    if not rclpy.ok():
        rclpy.init()
    xarm = XARM_manager()
    topic_pub = TopicPublisher(xarm)
    action = ActionCall(xarm)
    tf_helper = TFHelper(xarm)

    xarm.xarm_deactivate_all_controller()
    xarm.hardware_arm_enable(True)
    xarm.hardware_arm_mode(3)
    xarm.get_logger().info("[ADP-R] RIGHT arm enabled, mode 3.")

    # Optional home move.
    if args.home_move:
        xarm.get_logger().info(
            f"[ADP-R] Moving RIGHT arm to home joints {cfg.right_home_joints}"
        )
        action.jointspace_arm_R_controller(list(cfg.right_home_joints))
        xarm.get_logger().info("[ADP-R] RIGHT arm at home.")

    # Pre-place jointspace move (this deactivates QP internally).
    if cfg.pre_place_joints is not None:
        xarm.get_logger().info(
            f"[ADP-R] Moving RIGHT arm to pre-place joints "
            f"{cfg.pre_place_joints}"
        )
        action.jointspace_arm_R_controller(list(cfg.pre_place_joints))
        xarm.get_logger().info("[ADP-R] RIGHT arm at pre-place.")
    else:
        xarm.get_logger().info(
            "[ADP-R] Skipping pre-place move (descending from current pose)."
        )

    # ---------------- Right gripper (injected hook) ------------------
    # FT-only build: no tactile gripper.  Default is a no-op stub that
    # only logs; replace ``NoopGripper`` with your own
    # ``gripper_hook.GripperHook`` implementation (e.g. a Robotiq-only
    # wrapper) on the target machine.
    right_gripper = NoopGripper()
    xarm.get_logger().info("[ADP-R] Gripper hook ready (NoopGripper stub).")

    # ---------------- RIGHT QP + admittance --------------------------
    xarm.xarm_activate_controller([cfg.right_qp_controller])
    set_node_parameter(
        xarm, cfg.right_qp_controller, "otg_p_step", cfg.otg_p_step
    )
    set_node_parameter(
        xarm, cfg.right_qp_controller, "otg_r_step", cfg.otg_r_step
    )
    xarm.get_logger().info("[ADP-R] RIGHT QP controller active.")

    # Flush TF so capture_current_pose_as_target succeeds.
    t_end = time.monotonic() + 0.4
    while time.monotonic() < t_end and rclpy.ok():
        rclpy.spin_once(xarm, timeout_sec=0.02)

    right_adm = AdmittanceArm(
        side="right",
        xarm_manager=xarm,
        topic_pub=topic_pub,
        tf_helper=tf_helper,
        tcp_frame=cfg.right_tcp_frame,
        waist_frame=cfg.waist_frame,
        force_topic=cfg.right_force_topic,
        qp_controller=cfg.right_qp_controller,
        calib_path=cfg.right_ft_calibration_path,
        # Boot with the familiar RIGHT_GIVER_GAINS (stiff-but-not-rigid)
        # so the arm doesn't surprise anyone; the placer switches to
        # ``hold_gains`` for characterisation.
        initial_gains=_right_giver_gains(),
        force_proc=cfg.force_proc,
        loop_period=0.004,
        name="[ADP-R/RIGHT-ADM]",
    )
    if not right_adm.capture_current_pose_as_target(timeout_sec=0.5):
        raise RuntimeError(
            "[ADP-R] Could not capture RIGHT TCP pose as admittance target."
        )

    # Start spinning + admittance.
    spin = _SpinThread(xarm, spin_timeout_sec=0.01)
    spin.start()
    right_adm.start()
    xarm.get_logger().info(
        f"[ADP-R] RIGHT admittance running; waiting "
        f"{cfg.admittance_calibration_sec:.1f} s for FT-offset calibration..."
    )
    time.sleep(float(cfg.admittance_calibration_sec))

    # ---------------- Run the placer ---------------------------------
    placer = RightAdaptivePlacer(
        cfg=cfg,
        xarm_manager=xarm,
        topic_pub=topic_pub,
        tf_helper=tf_helper,
        right_gripper=right_gripper,
        right_adm=right_adm,
    )
    ok = False
    try:
        ok = placer.run()
        if ok:
            xarm.get_logger().info("[ADP-R] Adaptive place finished: RELEASED.")
        else:
            xarm.get_logger().warn(
                "[ADP-R] Adaptive place finished WITHOUT clean release "
                "(see logs above)."
            )
    except KeyboardInterrupt:
        print("\n[ADP-R] KeyboardInterrupt -- shutting down safely.")
    except Exception as e:
        print(f"[ADP-R] ERROR: {e!r}")
        import traceback
        traceback.print_exc()
    finally:
        # ---------------- Teardown -----------------------------------
        try:
            right_adm.stop()
        except Exception as e:
            print(f"[ADP-R] right_adm.stop warning: {e}")
        try:
            spin.stop()
        except Exception as e:
            print(f"[ADP-R] spin.stop warning: {e}")
        try:
            # Only open the gripper at teardown if the pipeline
            # succeeded.  On failure we leave whatever grasp the
            # gripper is in so a real payload does not drop --
            # operator can then inspect and decide manually.
            if ok:
                right_gripper.shutdown()
            else:
                print(
                    "[ADP-R] Pipeline did NOT finish cleanly; the right "
                    "gripper was left CLOSED on purpose (payload safety). "
                    "To open manually rerun with a fresh grasp."
                )
        except Exception as e:
            print(f"[ADP-R] right_gripper.shutdown warning: {e}")
        try:
            xarm.xarm_deactivate_all_controller()
        except Exception:
            pass
        try:
            xarm.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()
        print(f"[ADP-R] Exit (ok={ok}).")
        sys.stdout.flush()
        sys.stderr.flush()
        # Same as adaptive_place: avoid PyGILState_Release on process exit when
        # rclpy + tf2 + tac3d layers tear down (see adaptive_place/main.py).
        if os.environ.get("ADAPTIVE_PLACE_NORMAL_EXIT", "") != "1":
            os._exit(0 if ok else 1)


if __name__ == "__main__":
    main()
