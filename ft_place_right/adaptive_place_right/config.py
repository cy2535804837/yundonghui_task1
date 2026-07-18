"""
adaptive_place_right/config.py
==============================
Config for force-driven contact-decay placement, **right arm** variant.

Right-side specifics:

* TCP frame             ``right_tcp_link``
* QP controller         ``endpose_single_arm_qp_R_controller``
* Jointspace controller ``jointspace_arm_R_controller``
* F/T topic             ``/arm_6dof_right``
* F/T calibration JSON  ``admittance_control/ft_calibration_right.json``
  (loaded via ``handover.config.DEFAULT_RIGHT_FT_CALIB``)
* Gripper               Robotiq 2F-85 (1 DoF, 0-255 encoder ticks)
                        + two Tac3D tactile sensors (ports 9988/9989)
* Contact coefficient   computed by
                        ``TactileFeedback.compute_contact_coefficient``
                        inside ``sync_tactile_grasp/tianyi_tactile_grasp.py``
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

# Sibling packages on sys.path when invoked directly.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from dataclasses import replace as _dc_replace  # noqa: E402

from handover.config import (  # noqa: E402
    AdmittanceGains,
    DEFAULT_FORCE_PROC,
    DEFAULT_RIGHT_FT_CALIB,
    DEFAULT_RIGHT_GRIPPER,
    RightGripperConfig,
)


# ----------------------------------------------------------------------
# Optional firmer / narrower dead-band variant (NOT the tianyi default).
#
# ``sync_tactile_grasp/tianyi_tactile_grasp.GraspingManager`` uses
# ``target_cf=0.25, tolerance=0.03, predicted_grasp_force=-0.001`` --
# identical to ``handover.config.DEFAULT_RIGHT_GRIPPER``.
#
# For placement, a *tighter* ``cf_tolerance`` (e.g. 0.02) can reduce
# micro-moves during the F_loaded baseline window at the cost of more
# PD activity.  Use ``replace(DEFAULT_RIGHT_GRIPPER_FIRM, ...)`` or
# build a custom ``RightGripperConfig`` when you need that behaviour;
# the adaptive-place default is stock tianyi parity.
DEFAULT_RIGHT_GRIPPER_FIRM: RightGripperConfig = _dc_replace(
    DEFAULT_RIGHT_GRIPPER,
    cf_tolerance=0.02,
)


@dataclass
class RightAdaptivePlaceConfig:
    """Top-level configuration for the right-arm adaptive-place POC.

    All thresholds use the RIGHT wrist FT sensor's waist-frame Z
    component via the same calibration file used by the rest of the
    stack.

    The core tunable is ``decay_exponent_k`` -- see
    ``RightAdaptivePlacer`` for the physical meaning.
    """

    # ------------------------------------------------------------------
    # Right gripper (2F + Tac3D)
    # ------------------------------------------------------------------
    # Same defaults as ``GraspingManager`` in
    # ``sync_tactile_grasp/tianyi_tactile_grasp.py`` (via
    # ``DEFAULT_RIGHT_GRIPPER``).  For a narrower PD dead-band during
    # long F_loaded sampling, switch to ``DEFAULT_RIGHT_GRIPPER_FIRM``
    # or override ``cf_tolerance`` / ``target_cf`` / ``predicted_grasp_force``.
    right_gripper: RightGripperConfig = field(
        default_factory=lambda: DEFAULT_RIGHT_GRIPPER
    )
    # Timeout (seconds) for ``close_to_contact`` -- the Tac3D
    # ``grasp_to_stop`` loop polls until either jaw registers force
    # below ``predicted_grasp_force``.  Typical success is < 3 s on
    # normal objects; allow a generous ceiling in case the gripper
    # starts far from the object.
    contact_timeout_sec: float = 10.0

    # ------------------------------------------------------------------
    # ROS topics / frames
    # ------------------------------------------------------------------
    right_tcp_frame: str = "right_tcp_link"
    waist_frame: str = "waist_yaw_link"
    right_force_topic: str = "/arm_6dof_right"
    right_ft_calibration_path: str = DEFAULT_RIGHT_FT_CALIB
    right_qp_controller: str = "endpose_single_arm_qp_R_controller"
    right_jointspace_controller: str = "jointspace_arm_R_controller"
    otg_p_step: float = 0.005
    otg_r_step: float = 0.005

    # Home pose for the RIGHT arm (tucked-in HOME shared with the rest
    # of the stack; only used when ``--home-move`` is passed).
    right_home_joints: List[float] = field(
        default_factory=lambda: [1.0, -0.3, -0.4, -2.3, -0.45, -0.20, 0.18]
    )
    # ---- Pre-place staging pose (jointspace) -------------------------
    # User-provided descent-start pose.  At these joint values the
    # right gripper is parallel to the ground with the jaws vertical,
    # approximately 30-40 cm above the table in the standard test rig.
    # The adaptive place pipeline descends straight down from here.
    #
    # Source: user spec 2026-04-17 -- "the grasping pose and the
    # descent start pose should be [0.0, -1.18, 0.0, -1.3, 0.30,
    # -0.20, 0.00] which I make the gripper to be parallel with the
    # ground".
    pre_place_joints: Optional[List[float]] = field(
        default_factory=lambda: [0.0, -1.18, 0.0, -1.3, 0.30, -0.20, 0.00]
    )

    # ------------------------------------------------------------------
    # Object-weight characterisation
    # ------------------------------------------------------------------
    # How G_obj (object weight expressed as waist-Z force in N) is
    # found:
    #   "auto"    -- sample Fz with gripper OPEN BEFORE grasp
    #                (F_empty) and again with gripper CLOSED AFTER
    #                grasp settles (F_loaded); G_obj = F_empty - F_loaded.
    #   "manual"  -- skip step 2 and use ``object_weight_N`` directly
    #                (use when the gripper is already holding the
    #                object, e.g. chaining after graspnet_grasp).
    g_obj_estimation: str = "auto"
    object_weight_N: float = 1.5
    # Floor on G_obj to avoid divide-by-zero + wildly amplified gamma
    # when we hold a near-massless object (sensor noise dominates).
    # Below this the decay loop falls back to "release on threshold".
    min_object_weight_N: float = 0.3
    # Sample count + sleep for each of the empty / loaded baselines.
    weight_samples: int = 80
    weight_sleep_sec: float = 0.008

    # ------------------------------------------------------------------
    # Phase A -- grasp the object first (proof-of-concept convenience)
    # ------------------------------------------------------------------
    # If True, open -> close_to_contact -> adaptive settle before
    # descending.  Disable when the gripper is already holding the
    # object (e.g. chained after ``graspnet_grasp``).
    grasp_first: bool = True
    # Seconds to run the Tac3D PD loop (``adjust_grasp``) before
    # sampling F_loaded.  2 s was borderline on slippery objects (PD
    # loop hadn't stopped hunting when we froze the baseline);
    # 3.5 s leaves time for cf to settle inside the new 0.05 band.
    grasp_settle_sec: float = 3.5

    # Force used for the explicit ``goTo`` we issue immediately after
    # ``grasp_to_stop`` to re-assert rGTO=1 (the Robotiq ``stop()``
    # call inside ``grasp_to_stop`` clears rGTO).  Also the force
    # ceiling applied by the background adaptive-hold + decay threads
    # via ``goTo(..., force=<ticks>)``.  The Robotiq force register
    # is 0-255; we express it as 0-100% here for readability.  50%
    # (128/255) is the stock ``adjust_grasp`` value, which is plenty
    # to resist gravity on ~300 g objects while still being well
    # below the "crush" range for deformables.
    hold_force_pct: float = 50.0

    # --- Phase A hold (between close_to_contact and Phase C) ---------
    #
    # ``adaptive_pd``  (DEFAULT):
    #     Runs ``TactileFeedback.adjust_grasp`` in a background
    #     thread -- the SAME behaviour the proven ``graspnet_grasp``
    #     pipeline uses, and which the user confirmed does a good
    #     job.  Key property: ``adjust_grasp`` only issues a
    #     ``goTo(..., force=128)`` when ``|residual_cf| > tolerance``
    #     (there is a dead-band).  Between corrections the motor is
    #     passive -- NO continuous high-force push against the
    #     object.  This is why the stock flow doesn't over-squeeze:
    #     corrections are discrete, small, and conditional.
    #
    # ``motor_hold``:
    #     Periodically re-issues ``goTo(locked_position,
    #     force=hold_force)``.  Sounds gentle, but on a deformable
    #     object the Robotiq re-applies up to ``hold_force`` motor
    #     torque every refresh -- even a tiny creep in the jaws
    #     toward the target position is driven by active motor
    #     current, and the effect COMPOUNDS over many refreshes.
    #     Offered only as an opt-in for cases where the object is
    #     rigid and/or the user really wants a continuous active
    #     hold.  AVOID for soft / deformable targets.
    hold_mode: str = "adaptive_pd"

    # Refresh rate for the ``motor_hold`` loop.  10 Hz (0.1 s) is
    # plenty -- each goTo just re-asserts rGTO=1 with the same
    # position + force, so higher rates don't improve holding but do
    # load the Modbus serial bus that the Tac3D decay loop also uses.
    hold_refresh_period_sec: float = 0.1

    # ------------------------------------------------------------------
    # Phase A -- "hold" gains while we characterise the object
    # ------------------------------------------------------------------
    # During close_to_contact / adjust_grasp / F_loaded sampling the
    # arm must stay put.  With descent_gains (soft K_z) the admittance
    # yields ``G_obj / K_z`` (~1 cm per 0.06 N!) in -Z the moment the
    # gripper closes, so the arm would already have sagged before
    # Phase B starts.  Use stiff K here to hold pose; we relax to
    # ``descent_gains`` at the start of Phase B.
    #
    # NOTE: keep K_z moderate.  The FT-offset calibration still has to
    # absorb sensor noise, and very high K_rot amplifies wrist tremor
    # from the grasp dynamics (the 2F jaws can jitter audibly when the
    # Robotiq ``stop()`` command fires).
    hold_gains: AdmittanceGains = field(
        default_factory=lambda: AdmittanceGains(
            M=[0.1, 0.1, 0.1],
            B=[3.0, 3.0, 4.0],
            K=[40.0, 40.0, 40.0],
            M_rot=[0.01, 0.01, 0.01],
            B_rot=[1.0, 1.0, 1.0],
            K_rot=[5.0, 5.0, 5.0],
        )
    )

    # ------------------------------------------------------------------
    # Phase B -- descent until contact
    # ------------------------------------------------------------------
    # RIGHT admittance gains during descent.  Same reasoning as
    # ``adaptive_place/config.py`` (left):
    #   K_z=20, B_z=4, M_z=0.1  ->  slightly overdamped,
    #                              SS tracking lag at 10 mm/s = 2 mm,
    #                              compliance under G_obj=1.2 N ~ 60 mm
    # which yields gently on contact without slamming the object.
    descent_gains: AdmittanceGains = field(
        default_factory=lambda: AdmittanceGains(
            M=[0.1, 0.1, 0.1],
            B=[2.0, 2.0, 4.0],
            K=[20.0, 20.0, 20.0],
            M_rot=[0.01, 0.01, 0.01],
            B_rot=[1.0, 1.0, 1.0],
            K_rot=[5.0, 5.0, 5.0],
        )
    )
    # Just before the gripper opens: arm was on ``descent_gains`` and/or
    # ``transfer_gains`` with non-trivial K_z, so the spring continues to
    # push the object into the table and Fz / felt support look large even
    # though the gripper has not released yet.  This profile uses very
    # soft K_z, then we ``capture_current_pose_as_target`` in code so the
    # equilibrium matches the true TCP and the admittance does not "fight"
    # the support reaction.
    pre_release_gains: AdmittanceGains = field(
        default_factory=lambda: AdmittanceGains(
            M=[0.1, 0.1, 0.1],
            B=[2.0, 2.0, 3.0],
            K=[0.5, 0.5, 0.5],
            M_rot=[0.01, 0.01, 0.01],
            B_rot=[1.0, 1.0, 1.0],
            K_rot=[1.0, 1.0, 1.0],
        )
    )
    pre_release_settle_sec: float = 0.08
    # ------------------------------------------------------------------
    # TWO-PHASE descent: fast (direct QP) + slow (admittance)
    # ------------------------------------------------------------------
    # WHY two phases?  Admittance has a speed ceiling because tracking
    # lag = B*v/K and compliance under wrist load = F/K grow linearly
    # with descent speed (and K is intentionally soft so the arm
    # yields gently on contact).  The QP *without* admittance can track
    # ~50-60 mm/s reliably (same setting used by
    # ``tactile_place/tactile_place.py`` + ``qp_motion.stream_tcp_to_pose``).
    # So do the "get close to the table" part with direct QP streaming
    # (admittance paused), then hand off to admittance for the last
    # few cm where compliance actually matters.
    fast_descent_enabled: bool = True
    fast_descent_speed_mps: float = 0.06          # 60 mm/s
    fast_slow_band_m: float = 0.04                # last 4 cm are slow+compliant
    fast_descent_max_lag_m: float = 0.030
    fast_to_slow_dwell_sec: float = 0.25
    fast_descent_max_time_sec: float = 15.0

    descent_speed_mps: float = 0.015              # 15 mm/s
    descent_speed_max_mps: float = 0.025          # hard ceiling, see main.py
    descent_max_lag_m: float = 0.020              # runtime lag-aware throttle
    descent_min_z_m: float = 0.02                 # safety floor (waist-Z)
    descent_loop_period_sec: float = 0.01         # 100 Hz
    # Contact trip uses the same sign as ``F_support = max(Fz-baseline,0)``
    # unless ``descent_contact_signed_support`` is False (legacy |ΔFz|).
    # 0.45 N: sensitive enough for light / soft first contact; raise to ~0.6
    # if you get false trips from vibration only (CLI: --force-threshold).
    descent_force_threshold_N: float = 0.45
    descent_force_debounce: int = 4               # consecutive samples > threshold
    # Minimum true TCP drop (m) from the start_z of the current descent phase
    # before an Fz trip is accepted — rejects noise/jerk when still above table.
    descent_min_tcp_drop_m: float = 0.008
    # Idle after ``capture_current_pose_as_target`` before sampling descent
    # baseline Fz (arm + FT settle after gain / pause / resume transitions).
    descent_baseline_settle_sec: float = 0.35
    # If True (default), only (Fz - baseline) > threshold counts as contact
    # (table pushing up).  If False, abs(Fz-baseline) > threshold (old behaviour
    # — sensitive to negative spikes from vibration / gripper transients).
    descent_contact_signed_support: bool = True
    admittance_calibration_sec: float = 2.5

    # ------------------------------------------------------------------
    # Phase C -- load transfer / adaptive release (the novel part)
    # ------------------------------------------------------------------
    # RIGHT admittance gains during transfer.  K_z is the BIG knob
    # here: with a stiff K_z the arm actively resists the table's
    # upward push, so an equilibrium target set ``transfer_press_depth_m``
    # below the contact surface produces a steady
    # ``F_press = K_z * press_depth`` of downward force on the object.
    # That force appears on the F/T sensor as positive ``F_support``
    # and is what drives ``gamma -> 1`` for light objects (~<1 N)
    # *before* the gripper has opened -- without it the arm just
    # drifts up off the table as the gripper loosens and gamma
    # stays at 0 until the object falls (which is the 8 s stall you
    # see on sub-200 g objects).
    #
    # K_z=30 + press_depth=0.010 m gives ~0.3 N of steady press.
    # For a 0.5-1 N object that's enough to jump-start the feedback
    # loop (gamma starts at ~0.3-0.6 right at the beginning of
    # Phase C).  For heavier objects the object's own weight
    # dominates and the press is just insurance.  Raise K_z to ~60
    # N/m (or press_depth to 0.020 m) if you still see gamma
    # stalling at 0 on very light targets.
    transfer_gains: AdmittanceGains = field(
        default_factory=lambda: AdmittanceGains(
            M=[0.1, 0.1, 0.15],
            B=[2.0, 2.0, 3.0],
            K=[2.0, 2.0, 1.0],
            M_rot=[0.01, 0.01, 0.01],
            B_rot=[1.0, 1.0, 1.0],
            K_rot=[5.0, 5.0, 5.0],
        )
    )
    # Target Z is dropped by this much below the FAST/SLOW contact
    # pose the moment Phase C starts.  With ``transfer_gains.K_z``
    # the arm cannot actually go that low (the table stops it), so
    # the remaining error manifests as a steady press force.  Set
    # to 0.0 to disable (restores the old "compliant hover" behaviour).
    transfer_press_depth_m: float = 0.010
    # C_target(t) = C_initial * (1 - gamma(t)) ** decay_exponent_k
    #   k = 1.0  -- linear decay.
    #   k < 1.0  -- release quickly once the table takes any support
    #               (best for rigid objects on hard surface).
    #   k > 1.0  -- release slowly at first, quickly at the end
    #               (best for fragile / compliant objects).
    # k≈1.2: relax C_target quickly as gamma rises once the table shares
    # load (k=2.0 keeps a firm grasp longer — feels "unintelligent").
    decay_exponent_k: float = 1.0
    fz_filter_alpha: float = 0.8        # matches tactile_place filter_alpha
    transfer_loop_hz: float = 30.0
    transfer_timeout_sec: float = 15.0
    # Release handshake: lower threshold + shorter dwell + fewer
    # debounced samples so the hand opens soon after real support.
    gamma_release_threshold: float = 0.40
    gamma_release_debounce: int = 2
    transfer_min_dwell_sec: float = 0.22
    # After forcing C_target=0, brief wait so the decay thread advances
    # the jaw before join (then ``release()`` finishes the open).
    transfer_decay_finish_sec: float = 0.10

    # Underlying adaptive-grasp loop period for the decay thread (the
    # Tac3D PD loop).  20-100 Hz is fine; 10 ms matches the stock
    # ``TactileFeedback.adjust_grasp`` cadence.
    decay_loop_period_sec: float = 0.01
    # Max tick-size-per-step (Robotiq 0-255 encoder ticks).  The stock
    # ``adjust_grasp`` uses ``k_p_residual = 10.0`` which at
    # residual=0.35 would command 3.5 ticks -- fine.  We expose this so
    # decay runs can be made more/less aggressive without patching
    # ``sync_tactile_grasp``.
    decay_k_p_residual: float = 10.0
    # Asymmetric step caps during the decay phase.  We want slow
    # release (avoid suddenly dropping the object) and ~nothing on the
    # tighten side (if cf somehow climbs, re-tightening during release
    # is the wrong move -- the point is to open).
    decay_max_step_tighten_ticks: float = 1.0
    decay_max_step_release_ticks: float = 6.0

    # ------------------------------------------------------------------
    # Phase D -- full release + lift
    # ------------------------------------------------------------------
    post_release_dwell_sec: float = 0.22
    lift_after_release_m: float = 0.08
    lift_speed_mps: float = 0.02

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    log_csv_path: Optional[str] = None
    verbose: bool = True

    # Force processing (re-exported for convenience).
    force_proc = DEFAULT_FORCE_PROC


DEFAULT_RIGHT_ADAPTIVE_PLACE = RightAdaptivePlaceConfig()


__all__ = [
    "RightAdaptivePlaceConfig",
    "DEFAULT_RIGHT_ADAPTIVE_PLACE",
    "DEFAULT_RIGHT_GRIPPER_FIRM",
    "AdmittanceGains",
]
