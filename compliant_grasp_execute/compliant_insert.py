#!/usr/bin/env python3
"""
compliant_grasp_execute/compliant_insert.py
============================================
Force-compliant final grasp insert (v1).

Replaces the position-controlled "descend to the grasp pose" of the original
grasp pipeline with an *admittance-controlled* descent that:

  * slews the equilibrium target from the approach pose toward the planned
    grasp pose along the straight insertion axis, at a slow controlled speed;
  * is SOFT along the insertion axis (low stiffness) and STIFF laterally, with
    the wrist orientation held RIGID -- so a hard contact with the table or the
    object does not slam / fault a joint, it just stops the arm;
  * monitors the F/T sensor and STOPS the descent the instant the resisting
    force along the insertion axis exceeds a threshold (table / object contact).

After the gripper closes (done by the caller while this controller keeps the
arm compliant at the contact pose), the caller stops the controller and rotates
the wrist away from the table (``build_table_clear_pose7``) before the usual
lift / retract / home.

This module is intentionally self-contained: it uses the proven AdmittanceArm
runner copied into ``compliant_grasp_execute.admittance`` and the F/T
calibration produced by ``ft_calibration/calibrate_ft.py``. It does NOT modify
the original grasp project or ft_place_right.

v1 scope (per design): NO object-weight re-zero and NO compliant lift -- the
compliant phase is descend-to-contact + hold-while-closing only.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from compliant_grasp_execute.admittance import (
    AdmittanceArm,
    AdmittanceGains,
    ForceProcessing,
    TFHelper,
    _SpinThread,
    DEFAULT_LEFT_FT_CALIB,
    DEFAULT_RIGHT_FT_CALIB,
)

_TAG = "[COMPLIANT-INSERT]"


def _log(msg: str) -> None:
    print(f"{_TAG} {msg}", flush=True)


def default_ft_topic(arm: str) -> str:
    return "/arm_6dof_left" if arm == "left" else "/arm_6dof_right"


def default_ft_calib_path(arm: str) -> str:
    return DEFAULT_LEFT_FT_CALIB if arm == "left" else DEFAULT_RIGHT_FT_CALIB


def _qp_controller_name(arm: str) -> str:
    return (
        "endpose_single_arm_qp_L_controller"
        if arm == "left"
        else "endpose_single_arm_qp_R_controller"
    )


@dataclass
class CompliantInsertParams:
    """Tunables for the compliant descend-to-contact insert.

    Defaults follow the proven ft_place_right descent profile (soft, low-damping
    admittance + a lag-aware throttle) rather than a stiff/high-damping profile,
    because the latter cannot follow a slewed equilibrium (the arm lags and the
    commanded target races ahead -> the gripper stops short and grasps air).
    """

    # --- F/T sensor + calibration -------------------------------------
    ft_topic: Optional[str] = None          # default: per-arm /arm_6dof_*
    calib_path: Optional[str] = None        # default: per-arm ft_calibration_*.json

    # --- F/T signal conditioning (SAME for both arms) -----------------
    # The gains/calibration are symmetric, so a side that "shakes" while the
    # other is smooth is almost always a NOISIER wrench on that arm's sensor
    # leaking through into the admittance command (and getting amplified by the
    # velocity-based carrot). Filter harder + widen the deadzone here. This is
    # applied identically to left and right.
    filter_alpha: float = 0.35              # EMA: prev = a*new + (1-a)*prev.
                                            #   LOWER = more smoothing (less
                                            #   noise -> less shake). 0.8 (raw
                                            #   default) barely filters.
    force_deadzone: float = 0.8             # N below which the wrench is ignored
                                            #   (must stay < contact_force_n).
    torque_deadzone: float = 0.08           # Nm deadzone (wrist held rigid, so
                                            #   torque only needs noise rejection)

    # --- contact detection --------------------------------------------
    contact_force_n: float = 1.5            # resist along insertion axis -> contact
    contact_debounce: int = 3               # consecutive samples over threshold
    min_insert_m: float = 0.008             # min ACTUAL TCP drop before a trip counts
                                            #   (rejects wrist reaction, not contact)
    overshoot_m: float = 0.02               # how far PAST the planned grasp depth we
                                            #   may keep descending to find contact

    # --- stall (blocked-arm) contact detection -----------------------
    # A compliant arm yields on contact, so against a light object / hard table the
    # force may never reach contact_force_n -- but the TCP STOPS making progress
    # (throttle pins, tcp_drop plateaus). Treat "commanded deeper but not moving"
    # as contact; this is the primary, reliable stop for compliant grasping.
    stall_window_s: float = 0.7             # window over which to measure progress.
                                            #   Must be long enough that a healthy
                                            #   (slow, compliant) descent clears
                                            #   stall_eps: at ~1.3cm/s a 0.7s window
                                            #   travels ~0.9cm >> stall_eps, while a
                                            #   truly blocked arm (0 progress) still
                                            #   trips within 0.7s.
    stall_eps_m: float = 0.004              # < this much progress in the window => stalled
    max_insert_m: float = 0.20              # safety ceiling on total travel (must be
                                            #   >= planned insertion depth + overshoot)

    # --- descent motion -----------------------------------------------
    insert_speed_mps: float = 0.020         # equilibrium slew speed (20 mm/s)
    max_lag_m: float = 0.025                # lag throttle: HOLD the slew when the TCP
                                            #   lags the commanded equilibrium by more
                                            #   than this along the insertion axis
    control_rate_hz: float = 100.0          # descent supervisory loop rate
    max_vel: float = 0.20                   # HARD cap (m/s) on admittance output
                                            #   velocity (a safety backstop on the
                                            #   force-driven runaway, NOT the primary
                                            #   guard -- see runaway_lag below). MUST
                                            #   stay well above insert_speed_mps: the
                                            #   published carrot is v_cmd*lead_time
                                            #   and the QP only tracks a fraction of
                                            #   it, so a healthy descent needs
                                            #   v_cmd ~5x the slew speed. Capping too
                                            #   low starves the carrot and the arm
                                            #   creeps -> false stall (grasp air).
    max_omega: float = 0.5                  # HARD cap (rad/s) on admittance angular
                                            #   velocity (wrist is held, so small).
    loop_period: float = 0.004              # admittance integration period
    trans_lead_time: float = 0.08           # publish carrot = v_cmd * this (NOT
                                            #   v_cmd*loop_period, which is sub-mm
                                            #   for slow motion and stalls the QP).
                                            #   Larger amplifies velocity jitter into
                                            #   target jitter (shaking) -- keep modest.
    otg_p_step: float = 0.008               # QP position OTG step for the insert
    otg_r_step: float = 0.005               # QP rotation OTG step for the insert

    # --- admittance gains (waist frame, per axis x,y,z) ---------------
    # SOFT along the insertion direction (compliant on contact), stiffer on the
    # axes orthogonal to it, wrist orientation rigid (hold_orientation).
    # IMPORTANT: a tilted grasp insertion spans several waist axes, so EVERY axis
    # with a meaningful insertion component must be soft -- otherwise a stiff axis
    # fights the diagonal descent and its spring force masquerades as contact
    # (false trip -> grasp air). Damping is set per-axis from `damping_ratio` so
    # each axis is near-critically damped regardless of its K (a single scalar B
    # leaves the stiff axes underdamped -> shaking in free space, bounce on the
    # table).
    lateral_stiffness: float = 40.0         # K on axes orthogonal to insertion (N/m)
    insertion_stiffness: float = 20.0       # K on axes along insertion (N/m)
    soften_threshold: float = 0.30          # |dir component| above which an axis is
                                            #   treated as "along insertion" -> soft
    damping_ratio: float = 1.4              # target damping ratio (>=1 => no bounce)
    damping: float = 3.0                    # absolute floor on B (N s/m)
    mass: float = 0.1                       # M on all translation axes (kg)
    hold_stiffness: float = 150.0           # K (isotropic) for the post-contact HOLD
                                            #   while the gripper closes. Stiffer than
                                            #   the descent so the arm stays planted on
                                            #   the residual contact force instead of
                                            #   springing back (bounce).

    # --- timing -------------------------------------------------------
    offset_calib_timeout_s: float = 4.0     # wait for runtime force-offset calib
    settle_after_contact_s: float = 0.2     # brief hold once contact detected
    settle_before_calib_s: float = 0.3      # HOLD still BEFORE the runtime force
                                            #   offset is captured, so the baseline
                                            #   is not contaminated by residual
                                            #   sway/motion left over from the
                                            #   approach (a contaminated baseline
                                            #   drives a phantom force -> runaway).

    # --- bad-baseline / backward-runaway safety -----------------------
    # In PLACEMENT the gripper holds an object, so an uncompensated held-object
    # weight or a still-swinging arm at calibration time can leave a large phantom
    # force on the insertion axis. The compliant descent would then chase that
    # phantom and drive the arm AWAY from the target (the "lift up and hold"
    # failure). These two guards catch that and abort cleanly instead.
    baseline_force_max_n: float = 5.0       # right after calibration + settle, if
                                            #   |resist| (or a large net force)
                                            #   exceeds this the baseline is bad ->
                                            #   abort (do NOT drive the arm).
    baseline_check_s: float = 0.25          # hold-and-sample window used to measure
                                            #   the post-calibration baseline force.
    backward_abort_m: float = 0.015         # if the tool moves this far BACKWARD
                                            #   along the insertion axis before any
                                            #   real forward progress, a phantom
                                            #   force is driving it away -> abort
                                            #   immediately (stops the lift-up).


class CompliantInsertHandle:
    """Live handle to a compliant insert that is holding at the contact pose.

    The descent has already completed by the time this is returned; the
    AdmittanceArm thread keeps the arm compliant at the contact pose so the
    caller can close the gripper without the arm fighting. Call ``stop()`` once
    the gripper is closed (before the position-controlled lift).
    """

    def __init__(self, admit: AdmittanceArm, spin: _SpinThread, result: Dict[str, Any]):
        self._admit = admit
        self._spin = spin
        self.result = result

    @property
    def ok(self) -> bool:
        return bool(self.result.get("ok"))

    @property
    def contact(self) -> bool:
        return bool(self.result.get("contact"))

    def stop(self) -> None:
        try:
            self._admit.stop()
        finally:
            self._spin.stop()


def run_compliant_insert(
    *,
    xarm: Any,
    topic_pub: Any,
    arm: str,
    waist_frame: str,
    tcp_frame: str,
    approach_pose7: List[float],
    grasp_pose7: List[float],
    params: Optional[CompliantInsertParams] = None,
    guard: Any = None,
    otg_p_step: float = 0.005,
    otg_r_step: float = 0.005,
) -> CompliantInsertHandle:
    """Run the compliant descend-to-contact insert and return a live handle.

    The returned handle is holding the arm compliant at the contact (or
    planned-depth) pose. Close the gripper, then call ``handle.stop()``.
    """
    from xarm_sdk.tools import set_node_parameter

    params = params or CompliantInsertParams()
    ft_topic = params.ft_topic or default_ft_topic(arm)
    calib_path = params.calib_path or default_ft_calib_path(arm)
    ctrl_name = _qp_controller_name(arm)

    if not os.path.exists(calib_path):
        _log(
            f"WARNING: F/T calibration not found at {calib_path}. "
            "Run ft_calibration/calibrate_ft.py --arm "
            f"{arm} FIRST. Admittance will use an identity fallback and WILL "
            "drift/push -- compliant insert is unsafe without calibration."
        )

    # The endpose QP controller must be active to receive the admittance target
    # poses (same controller the position QP-stream uses). Fine OTG steps so the
    # compliant descent stays smooth.
    xarm.xarm_activate_controller([ctrl_name])
    set_node_parameter(xarm, ctrl_name, "otg_p_step", float(otg_p_step))
    set_node_parameter(xarm, ctrl_name, "otg_r_step", float(otg_r_step))

    # Background spin so the AdmittanceArm's TF + F/T subscriptions stay live.
    spin = _SpinThread(xarm)
    spin.start()

    tf_helper = TFHelper(xarm)

    descent_gains = AdmittanceGains(
        M=[params.mass, params.mass, params.mass],
        B=[params.damping, params.damping, params.damping],
        # K filled in per-axis below once the insertion axis is known.
        K=[params.lateral_stiffness, params.lateral_stiffness, params.lateral_stiffness],
        # HARD velocity cap: a residual downward force on a soft axis would
        # otherwise drive a runaway (TCP races ahead of the slew at ~0.9 m/s and
        # slams the table). Capped near the slew speed this can't happen.
        max_vel=float(params.max_vel),
        max_omega=float(params.max_omega),
        rot_lead_time=0.15,
    )

    # Force conditioning -- IDENTICAL for both arms. Strong filtering + a wider
    # deadzone keeps a noisier sensor (typically the side that "shakes") from
    # driving the admittance in free space.
    force_proc = ForceProcessing(
        force_deadzone=params.force_deadzone,
        torque_deadzone=params.torque_deadzone,
        filter_alpha=params.filter_alpha,
    )

    admit = AdmittanceArm(
        side=arm,
        xarm_manager=xarm,
        topic_pub=topic_pub,
        tf_helper=tf_helper,
        tcp_frame=tcp_frame,
        waist_frame=waist_frame,
        force_topic=ft_topic,
        qp_controller=ctrl_name,
        calib_path=calib_path,
        initial_gains=descent_gains,
        force_proc=force_proc,
        loop_period=params.loop_period,
        trans_lead_time=params.trans_lead_time,
        name=f"[COMPLIANT-{arm.upper()}]",
        hold_orientation=True,   # rigid wrist during the insert
    )

    result: Dict[str, Any] = {
        "ok": False,
        "contact": False,
        "reason": None,
        "resist_n": 0.0,
        "traveled_m": 0.0,
        "planned_depth_m": 0.0,
        "ft_topic": ft_topic,
        "calib_path": calib_path,
        "calib_present": os.path.exists(calib_path),
    }

    # Capture the current TCP pose (the post-approach standoff pose) as the
    # admittance equilibrium and seed the QP target, then start the loop.
    if not admit.capture_current_pose_as_target(timeout_sec=1.0):
        _log("ERROR: could not capture current TCP pose; aborting compliant insert")
        result["reason"] = "no_tcp_pose"
        spin.stop()
        return CompliantInsertHandle(admit, spin, result)

    # _target_p_ref / _target_q_ref were just set by
    # capture_current_pose_as_target(); read them as the insert start pose.
    start_xyz = np.asarray(admit._target_p_ref, dtype=float).copy()
    q_ref = np.asarray(admit._target_q_ref, dtype=float).copy()

    # Let the arm + held object settle BEFORE the runtime force-offset is
    # captured. The compliant descent starts right after the approach/reorient, so
    # without this dwell the offset can be averaged over a still-decelerating,
    # swinging wrench -> a contaminated baseline that later reappears as a phantom
    # force and drives the arm away from the target (the "lift up" failure).
    if float(params.settle_before_calib_s) > 0.0:
        _log(
            f"settling {params.settle_before_calib_s:.2f}s before force-offset "
            "calibration (clean baseline)"
        )
        time.sleep(float(params.settle_before_calib_s))

    admit.start()

    # Wait for the runtime force-offset calibration (keep the arm still).
    t0 = time.monotonic()
    while not admit.runtime_offset.calibrated:
        if time.monotonic() - t0 > params.offset_calib_timeout_s:
            _log(
                "WARNING: runtime force-offset calibration timed out; "
                "proceeding (contact detection may be noisy)"
            )
            break
        time.sleep(0.02)
    if admit.runtime_offset.calibrated:
        _log(f"runtime force-offset calibrated in {time.monotonic() - t0:.2f}s")

    # Insertion geometry (waist frame).
    grasp_xyz = np.asarray([float(v) for v in grasp_pose7[:3]], dtype=float)
    insert_vec = grasp_xyz - start_xyz
    planned_depth = float(np.linalg.norm(insert_vec))
    result["planned_depth_m"] = planned_depth
    if planned_depth < 1e-6:
        direction = np.array([0.0, 0.0, -1.0])  # degenerate: descend straight down
        _log("WARNING: approach==grasp (no standoff); defaulting insertion to -Z")
    else:
        direction = insert_vec / planned_depth

    # Soften EVERY waist axis that carries a meaningful share of the (possibly
    # tilted) insertion direction; keep the rest stiffer to hold the planned line.
    # A single-soft-axis scheme leaves a major insertion component (e.g. the Z of a
    # 45-deg grasp) stiff, so its spring force fights the descent and reads as a
    # false contact -> grasp air. Damping is then set PER AXIS for ~critical
    # damping (B = damping_ratio * 2*sqrt(K*M)), with an absolute floor, so the
    # stiffer axes are not underdamped (which causes free-space shaking and table
    # bounce).
    soft_mask = np.abs(direction) >= float(params.soften_threshold)
    K = np.where(
        soft_mask, float(params.insertion_stiffness), float(params.lateral_stiffness)
    ).astype(float)
    B = np.maximum(
        float(params.damping_ratio) * 2.0 * np.sqrt(K * float(params.mass)),
        float(params.damping),
    )
    descent_gains.K = [float(v) for v in K]
    descent_gains.B = [float(v) for v in B]
    admit.set_gains(descent_gains)
    soft_axes = "".join(c for c, m in zip("xyz", soft_mask) if m) or "(none)"
    _log(
        f"insertion dir=[{direction[0]:+.2f},{direction[1]:+.2f},{direction[2]:+.2f}] "
        f"planned_depth={planned_depth*100:.1f}cm soft_axes={soft_axes} "
        f"K={[round(v,1) for v in descent_gains.K]} "
        f"B={[round(v,2) for v in descent_gains.B]} "
        f"contact_thresh={params.contact_force_n:.1f}N"
    )

    # Baseline sanity check: with the insertion axis now known, hold still briefly
    # and measure the resting wrench. Right after the runtime offset calibration
    # nothing is touching, so the force along the insertion axis should be ~0. A
    # large, persistent reading means the baseline is bad (uncompensated held
    # object, a still-swinging arm, or a faulty F/T) -- driving the compliant
    # descent now would chase that phantom force and run the arm AWAY from the
    # target (the "lift up and hold" failure). Abort with a clear message instead.
    if float(params.baseline_check_s) > 0.0 and float(params.baseline_force_max_n) > 0.0:
        t_b = time.monotonic()
        resist_samples: List[float] = []
        fmag_samples: List[float] = []
        while time.monotonic() - t_b < float(params.baseline_check_s):
            Fb = np.asarray(admit.last_F_ctrl[:3], dtype=float)
            resist_samples.append(float(-np.dot(Fb, direction)))
            fmag_samples.append(float(np.linalg.norm(Fb)))
            time.sleep(0.01)
        resist0 = float(np.median(resist_samples)) if resist_samples else 0.0
        fmag0 = float(np.median(fmag_samples)) if fmag_samples else 0.0
        thresh = float(params.baseline_force_max_n)
        if abs(resist0) > thresh or fmag0 > 2.0 * thresh:
            result["ok"] = False
            result["contact"] = False
            result["reason"] = "bad_ft_baseline"
            result["resist_n"] = resist0
            result["baseline_force_n"] = fmag0
            _log(
                f"ABORT bad F/T baseline: resting resist={resist0:+.1f}N "
                f"|F|={fmag0:.1f}N exceeds {thresh:.1f}N BEFORE any contact. The "
                "force offset is contaminated -- most likely the held object's "
                "weight is not compensated, the arm was still moving/swinging at "
                "calibration, or the F/T needs recalibration. NOT descending "
                "(would drive the arm away from the target -> lift up). Re-run "
                "after the arm settles, or recalibrate this arm's F/T."
            )
            return CompliantInsertHandle(admit, spin, result)
        _log(
            f"baseline OK: resting resist={resist0:+.1f}N |F|={fmag0:.1f}N "
            f"(< {thresh:.1f}N)"
        )

    # Fresh trend history so the recovery-aware guard grants its grace window to
    # any joint parked inside the margin at phase start (parity with QP stream).
    if guard is not None and hasattr(guard, "reset_recovery_state"):
        try:
            guard.reset_recovery_state(arm)
        except Exception:  # noqa: BLE001
            pass

    # Total commanded travel: planned depth + overshoot. The planned depth (the
    # approach standoff distance) MUST be covered, so max_insert_m is only a
    # safety ceiling above it -- never a cap below it (capping below planned_depth
    # is what leaves the gripper short of the object and grasping air).
    s_max = planned_depth + float(params.overshoot_m)
    if s_max > float(params.max_insert_m):
        _log(
            f"WARNING: planned insertion {planned_depth*100:.1f}cm + overshoot "
            f"{params.overshoot_m*100:.1f}cm = {s_max*100:.1f}cm exceeds safety "
            f"ceiling max_insert_m={params.max_insert_m*100:.1f}cm; clamping. The "
            "gripper may stop short -- raise --compliant-max-insert-m."
        )
        s_max = float(params.max_insert_m)

    dt = 1.0 / max(5.0, float(params.control_rate_hz))
    max_lag = float(params.max_lag_m)
    # "Arrived" band once the commanded slew (s) has saturated at s_max. This MUST
    # stay small and independent of max_lag: it is the distance short of s_max at
    # which we declare the descent done. max_lag is the (now larger, for speed)
    # carrot lead; tying reach_tol to it made the descent quit ~max_lag short of
    # s_max -- i.e. ~0.25cm short of the PLANNED depth, never consuming the
    # overshoot that pushes the tool down until the table/object trips contact.
    # That left the gripper ~1cm high -> grasp air (worse on the arm whose planned
    # pose already lands a touch high). Keep it a small fixed band so the tool
    # actually descends into the overshoot and reaches contact.
    reach_tol = max(0.003, min(0.008, 0.5 * max_lag))  # "arrived" band at s_max
    s = 0.0
    over = 0
    contact = False
    contact_kind = "force"
    aborted = False
    last_resist = 0.0
    actual_drop = 0.0
    throttle_events = 0
    t_start = time.monotonic()
    # Worst case the throttle stalls progress, so allow generous extra time.
    deadline = t_start + (s_max / max(1e-4, params.insert_speed_mps)) + 8.0
    last_log = t_start
    stall_window_s = float(params.stall_window_s)
    stall_eps_m = float(params.stall_eps_m)
    stall_t_ref = t_start
    stall_drop_ref = 0.0

    while time.monotonic() < deadline:
        # Joint-limit watchdog (the arm is compliant, but a slewed target could
        # still drive a wrist joint toward a hard stop).
        if guard is not None and getattr(guard, "enabled", False):
            ev = guard.check_live(arm)
            if ev is not None and ev.get("should_abort"):
                b = ev["breached"][0]
                eff_margin = guard.margin_overrides.get(b["joint"], guard.margin)
                _log(
                    f"ABORT joint-limit guard - '{b['joint']}'={b['value']} within "
                    f"{eff_margin:.3f}rad of limit; holding compliant pose"
                )
                guard.last_event = {**ev, "phase": "compliant_grasp"}
                aborted = True
                break

        # Live TCP (waist frame) from the admittance loop. The signed progress of
        # the ACTUAL tool along the insertion axis -- not the commanded target --
        # is what gates both contact and the lag throttle.
        x_cur = np.asarray(admit.last_x_current[:3], dtype=float)
        actual_drop = float(np.dot(x_cur - start_xyz, direction))
        lag = s - actual_drop  # how far the commanded equilibrium is ahead of TCP

        # Early backward-runaway guard: before ANY real forward progress the tool
        # must never move BACKWARD along the insertion axis. If it does, a phantom /
        # uncompensated force is driving the arm AWAY from the target -- abort at
        # once instead of letting it rise until the stall timeout (the "lift up and
        # hold" failure, seen as actual_drop -> -9cm). The forward runaway/overrun
        # guards below only trigger AFTER min_insert, so they never catch this.
        if actual_drop < -float(params.backward_abort_m):
            aborted = True
            contact_kind = "backward_runaway"
            break

        # Overrun guard: a residual / mis-compensated force on a soft axis can push
        # the ACTUAL tool PAST the deepest commanded depth (lag goes negative) even
        # while the slew is throttled/held. s_max (planned depth + overshoot) is the
        # deepest we ever intend to go, so if the tool reaches it, STOP -- otherwise
        # the soft spring keeps driving the arm into the table (the right-arm
        # runaway: TCP raced to 16.9cm of an 11.4cm plan and hit 10.5N). Re-anchor
        # at the reached pose and hold for the gripper close.
        if actual_drop >= s_max:
            contact = True
            contact_kind = "overrun"
            break

        # Runaway guard (the PRIMARY protection against a force-driven dive). In a
        # healthy descent the TCP trails the commanded slew (lag > 0). If lag goes
        # strongly NEGATIVE the tool is racing AHEAD of the command -- only possible
        # if a residual / mis-compensated force is dragging it down the insertion
        # axis (seen on an arm with a biased wrench: lag -> -4.5cm, v -> 0.9 m/s,
        # then slams the table). Stop immediately and re-anchor; the soft spring can
        # otherwise keep accelerating into the surface. Gated on min travel so the
        # initial wrist settle is not mistaken for it. Symmetric with the throttle:
        # |lag| is bounded by max_lag in BOTH directions.
        if actual_drop >= params.min_insert_m and lag < -max_lag:
            contact = True
            contact_kind = "runaway"
            break

        # Contact = resisting force along the insertion axis, gated on a minimum
        # ACTUAL travel so wrist reaction at the very start is not mistaken for it.
        F = np.asarray(admit.last_F_ctrl[:3], dtype=float)
        resist = float(-np.dot(F, direction))  # opposing the descent
        last_resist = resist
        if resist > params.contact_force_n and actual_drop >= params.min_insert_m:
            over += 1
            if over >= int(params.contact_debounce):
                contact = True
                contact_kind = "force"
                break
        else:
            over = 0

        # Stall = the TCP has stopped progressing along the insertion axis while we
        # are still trying to descend (arm blocked by object/table). A compliant arm
        # yields, so the force may never reach the threshold -- but progress stops.
        # This is the primary, reliable stop and prevents the long grind to full
        # overshoot depth when the surface is reached.
        # Startup grace: the admittance makes a brief transient as it establishes
        # the carrot at the start, which front-loads progress into the first window
        # and can make the settle that follows look like a stall. Don't evaluate
        # stall until the descent has been running long enough to be in steady
        # state.
        now = time.monotonic()
        in_stall_grace = (now - t_start) < (2.0 * stall_window_s)
        if not in_stall_grace and now - stall_t_ref >= stall_window_s:
            progress = actual_drop - stall_drop_ref
            if progress < stall_eps_m:
                if actual_drop >= params.min_insert_m:
                    # Made meaningful forward progress earlier, now stopped: the
                    # compliant arm is resting on the table/object -> contact.
                    contact = True
                    contact_kind = "stall"
                    break
                # Never reached min_insert_m AND no progress in the window: the
                # arm is blocked BEFORE the descent even starts -- a wrist joint
                # saturated against its hard stop (the right-arm wrist_roll case:
                # the seed-anchored orientation demands wrist_roll past -1.3 at
                # this pose), or a residual force is pushing the tool backward
                # (actual_drop goes negative). Do NOT grind against the stop /
                # drift backward until the deadline -- abort so the caller can
                # recover or re-teach the seed, instead of the arm straining.
                aborted = True
                contact_kind = "blocked_no_progress"
                break
            stall_t_ref = now
            stall_drop_ref = actual_drop
        elif in_stall_grace:
            # keep the baseline current during grace so the first real window
            # measures progress from the end of the transient, not from zero.
            stall_t_ref = now
            stall_drop_ref = actual_drop

        # Advance the equilibrium with a SMOOTH "leash" rather than a bang-bang
        # throttle. Keep the commanded target a fixed small lead ahead of the
        # ACTUAL tool, rate-limited by insert_speed. This self-paces the descent to
        # whatever rate the arm can actually achieve (set by the soft admittance +
        # carrot), so the target never races to the max_lag ceiling and chatter
        # (advance/hold/advance/hold). That chatter is what shook the arm and -- via
        # inertial reaction on the ~1.4kg payload -- created a spurious force that
        # both desensitised contact detection AND pushed the tool sideways on the
        # soft lateral axis (grasp air). With the leash, lag stays ~constant and the
        # descent is smooth, like a well-behaved arm.
        # Hold the commanded target a CONSTANT lead ahead of the ACTUAL tool. This
        # lead is the carrot that pulls the arm along the insertion axis. The proven
        # bang-bang slew let lag sit at ~max_lag (a strong carrot, ~1.5cm/s descent)
        # but chattered at the ceiling. We reproduce that strong, steady carrot
        # smoothly here. CRITICAL: do NOT rate-limit s to insert_speed -- if the arm
        # keeps pace, s and actual advance together and the lead never builds, so the
        # carrot collapses to near zero and the descent creeps until the stall
        # detector false-trips in the first second (grasp air). Tying s directly to
        # actual_drop keeps the carrot at exactly `lead` regardless of arm speed;
        # max_vel + the runaway/overrun guards bound any transient. insert_speed is
        # retained only as a gentle ceiling on how fast the lead may be *grown*.
        if resist <= params.contact_force_n:
            lead_target = 0.9 * max_lag                 # constant carrot (~max_lag)
            s_desired = actual_drop + lead_target
            # Grow s toward the lead quickly (5x slew) so the carrot is established
            # in a fraction of a second, then it self-maintains as the tool moves.
            s_step = 5.0 * params.insert_speed_mps * dt
            s = min(s_max, max(s, min(s + s_step, s_desired)))
            if lag > max_lag:
                throttle_events += 1                    # telemetry only (now rare)
        target_p = start_xyz + s * direction
        admit.update_target(target_p, q_ref)

        # Done once we have commanded the full depth AND the arm has caught up.
        if s >= s_max - 1e-9 and lag <= reach_tol:
            break

        now = time.monotonic()
        if now - last_log >= 1.0:
            _log(
                f"insert s={s*100:.1f}cm tcp_drop={actual_drop*100:.1f}cm "
                f"lag={lag*100:+.1f}cm resist={resist:+.2f}N "
                f"(throttle x{throttle_events})"
            )
            last_log = now

        time.sleep(dt)

    result["traveled_m"] = float(actual_drop)
    result["commanded_m"] = float(s)
    result["resist_n"] = float(last_resist)
    result["contact"] = bool(contact)
    result["throttle_events"] = int(throttle_events)

    # Anchor the compliant hold at the ACTUAL pose reached, with zero velocity,
    # and stiffen the spring for the hold. Critical for avoiding the post-contact
    # bounce: the descent commanded equilibrium lags ~max_lag BELOW the surface,
    # so if we leave it there the soft spring keeps driving the arm into the table
    # and rebounds during the gripper close. Re-anchoring at the reached pose (the
    # surface) removes the downward drive; a stiffer hold K then keeps the arm
    # planted while the gripper closes instead of springing on the residual force.
    if not aborted:
        try:
            x_hold = np.asarray(admit.last_x_current[:3], dtype=float).copy()
            hold_gains = AdmittanceGains(
                M=[params.mass, params.mass, params.mass],
                B=[float(v) for v in (
                    float(params.damping_ratio) * 2.0
                    * np.sqrt(np.full(3, params.hold_stiffness) * float(params.mass))
                )],
                K=[float(params.hold_stiffness)] * 3,
                max_vel=descent_gains.max_vel,
                max_omega=descent_gains.max_omega,
                rot_lead_time=descent_gains.rot_lead_time,
            )
            admit.set_gains(hold_gains)
            admit.set_target(x_hold, q_ref)  # set_target zeroes the integrator
        except Exception:  # noqa: BLE001
            pass

    if aborted:
        result["ok"] = False
        if contact_kind == "backward_runaway":
            result["reason"] = "backward_runaway"
            result["contact_kind"] = contact_kind
            _log(
                f"ABORT backward runaway: tool moved {actual_drop*100:.1f}cm BACKWARD "
                f"along the insertion axis (resist {last_resist:+.1f}N) before any "
                "forward progress. A phantom / uncompensated F/T force is driving the "
                "arm AWAY from the target (the 'lift up' failure). Stopped early. Most "
                "likely the held object's weight is not compensated or the F/T baseline "
                "was captured while the arm was still moving -- let the arm settle "
                "longer before the set-down, or recalibrate this arm's F/T."
            )
        elif contact_kind == "blocked_no_progress":
            result["reason"] = "blocked_no_progress"
            result["contact_kind"] = contact_kind
            _log(
                f"ABORT no forward progress: tcp_drop {actual_drop*100:.1f}cm < min "
                f"{params.min_insert_m*100:.1f}cm (resist {last_resist:+.1f}N). The arm "
                "is blocked before the descent started -- a wrist joint is saturated at "
                "its hard stop, or a residual force is pushing the tool backward. "
                "Re-teach this arm's elbow-high seed (so the grasp orientation keeps the "
                "wrist in range at the grasp pose) or lower the grasp tilt; do NOT just "
                "retry -- the arm will grind against the stop again."
            )
        else:
            result["reason"] = "joint_limit_abort"
            _log(f"insert aborted by joint-limit guard after tcp_drop {actual_drop*100:.1f}cm")
    elif contact:
        result["ok"] = True
        result["reason"] = {
            "stall": "stall_contact",
            "overrun": "max_depth_overrun",
            "runaway": "force_runaway_stop",
        }.get(contact_kind, "contact")
        result["contact_kind"] = contact_kind
        if contact_kind in ("overrun", "runaway"):
            _log(
                f"{'MAX-DEPTH' if contact_kind == 'overrun' else 'RUNAWAY'} stop at "
                f"tcp_drop {actual_drop*100:.1f}cm (commanded {s*100:.1f}cm, "
                f"lag {(s-actual_drop)*100:+.1f}cm, resist {last_resist:+.1f}N): a "
                "residual force was driving the tool down the insertion axis faster "
                "than commanded. Re-anchoring + holding compliant for gripper close. "
                "If this recurs, RE-CALIBRATE this arm's F/T (resist should stay ~0 "
                "until real contact)."
            )
        else:
            _log(
                f"CONTACT ({contact_kind}) at tcp_drop {actual_drop*100:.1f}cm "
                f"(resist {last_resist:.1f}N, thresh {params.contact_force_n:.1f}N); "
                "holding compliant for gripper close"
            )
    else:
        # Reached planned depth + overshoot without a contact trip. Still a valid
        # grasp pose (object may be soft / detection slightly high); proceed.
        result["ok"] = True
        result["reason"] = "reached_depth_no_contact"
        _log(
            f"reached commanded depth (tcp_drop {actual_drop*100:.1f}cm of planned "
            f"{planned_depth*100:.1f}cm) WITHOUT contact (resist {last_resist:.1f}N "
            f"< {params.contact_force_n:.1f}N); proceeding to close anyway"
        )

    # Brief compliant settle so the arm is steady before the gripper closes.
    if result["ok"] and params.settle_after_contact_s > 0:
        time.sleep(float(params.settle_after_contact_s))

    return CompliantInsertHandle(admit, spin, result)


def build_table_clear_pose7(current_pose7: List[float], rotate_deg: float) -> List[float]:
    """Rotate the wrist about the waist Y axis to tilt the gripper away from the
    table after the grasp, keeping the TCP position fixed.

    Same convention as the pipeline's lift tilt:
        q_new = Ry(rotate_deg) * q_current
    A negative angle tilts the nose up/back (typical "clear the table" sense),
    matching ``--lift-tilt-y-deg`` in the base pipeline.
    """
    from scipy.spatial.transform import Rotation as R

    q = [float(v) for v in current_pose7[3:7]]
    q_new = (R.from_euler("y", float(rotate_deg), degrees=True) * R.from_quat(q)).as_quat()
    return [
        float(current_pose7[0]),
        float(current_pose7[1]),
        float(current_pose7[2]),
        float(q_new[0]),
        float(q_new[1]),
        float(q_new[2]),
        float(q_new[3]),
    ]
