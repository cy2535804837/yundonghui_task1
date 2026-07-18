"""
handover/config.py
==================
Central configuration for the dual-arm handover pipeline.

All tunables live here so the pipeline file stays short and the operator has
one place to adjust stiffness, damping, force thresholds, hand ports, etc.

The two arms use DIFFERENT stiffness profiles during each phase:

* RIGHT (giver, Robotiq gripper holding the object):
    - HIGHER damping and HIGHER stiffness  →  more "solid"
    - Goal: keep the object steady while the left hand pinches it
    - Still compliant enough that a hard contact from the left hand will
      not spike forces on the Robotiq tactile sensors or the wrist F/T.

* LEFT (receiver, Revo2 dextrous hand closing in on the object):
    - LOWER damping and ZERO stiffness
    - Goal: maximum compliance so the hand can drift onto the object
      without pushing it out of the right gripper.
    - After the left hand confirms contact and target contact-coefficient,
      we transition to a "hold" profile with moderate damping to resist
      disturbances while the right hand opens.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


HANDOVER_DIR = os.path.dirname(os.path.abspath(__file__))
ADMITTANCE_DIR = os.path.abspath(os.path.join(HANDOVER_DIR, "..", "admittance_control"))
REVO2_DIR = os.path.abspath(os.path.join(HANDOVER_DIR, "..", "revo2_tactile_grasp"))
SYNC_TACTILE_DIR = os.path.abspath(os.path.join(HANDOVER_DIR, "..", "sync_tactile_grasp"))

DEFAULT_POSES_PATH = os.path.join(HANDOVER_DIR, "handover_recorded_poses.json")
DEFAULT_LEFT_FT_CALIB = os.path.join(ADMITTANCE_DIR, "ft_calibration.json")
DEFAULT_RIGHT_FT_CALIB = os.path.join(ADMITTANCE_DIR, "ft_calibration_right.json")


@dataclass
class AdmittanceGains:
    """Second-order admittance parameters for one arm (translation + rotation)."""

    M: List[float] = field(default_factory=lambda: [0.1, 0.1, 0.1])
    B: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    K: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    M_rot: List[float] = field(default_factory=lambda: [0.01, 0.01, 0.01])
    B_rot: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    K_rot: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    max_vel: float = 20.0
    max_omega: float = 20.0
    rot_lead_time: float = 0.15


# =====================================================================
# Admittance stiffness profiles per phase
# =====================================================================
# Right arm (giver)  — higher stiffness & damping, "solid" presentation.
RIGHT_GIVER_GAINS = AdmittanceGains(
    M=[0.1, 0.1, 0.1],
    B=[2.0, 2.0, 2.0],           # ~4x the left baseline
    K=[15.0, 15.0, 15.0],        # mild spring back to handover pose
    M_rot=[0.01, 0.01, 0.01],
    B_rot=[1.2, 1.2, 1.2],
    K_rot=[8.0, 8.0, 8.0],
    max_vel=20.0,
    max_omega=20.0,
    rot_lead_time=0.15,
)

# Left arm (receiver) — maximum compliance during approach.
LEFT_RECEIVER_GAINS = AdmittanceGains(
    M=[0.1, 0.1, 0.1],
    B=[0.4, 0.4, 0.4],           # very low damping = floats easily
    K=[0.0, 0.0, 0.0],           # zero stiffness = no pull-back
    M_rot=[0.01, 0.01, 0.01],
    B_rot=[0.1, 0.1, 0.1],
    K_rot=[0.0, 0.0, 0.0],
    max_vel=20.0,
    max_omega=20.0,
    rot_lead_time=0.15,
)

# Left arm after the hand has grasped — add some damping so the arm holds
# the object instead of drifting once the right gripper opens.
LEFT_HOLDING_GAINS = AdmittanceGains(
    M=[0.1, 0.1, 0.1],
    B=[1.5, 1.5, 1.5],
    K=[5.0, 5.0, 5.0],
    M_rot=[0.01, 0.01, 0.01],
    B_rot=[1.0, 1.0, 1.0],
    K_rot=[3.0, 3.0, 3.0],
    max_vel=20.0,
    max_omega=20.0,
    rot_lead_time=0.15,
)

# Used by adjust_handover_pose.py: BOTH arms with zero stiffness and low
# damping so the operator can hand-guide the end-effectors into the
# desired handover pose.  Slightly more damping than LEFT_RECEIVER_GAINS
# so the arm doesn't drift under sensor noise when the operator lets go.
ADJUSTMENT_GAINS = AdmittanceGains(
    M=[0.1, 0.1, 0.1],
    B=[0.05, 0.05, 0.05],           # a bit of viscous drag = no drift at rest
    K=[0.0, 0.0, 0.0],           # zero stiffness = operator drags freely
    M_rot=[0.1, 0.1, 0.1],
    B_rot=[0.1, 0.1, 0.1],
    K_rot=[0.0, 0.0, 0.0],
    max_vel=20.0,
    max_omega=20.0,
    rot_lead_time=0.15,
)

# Drag-to-teach profile for single-point positioning (e.g. touching an
# object center to record a ground-truth pose).  Unlike ADJUSTMENT_GAINS
# (very low damping = the arm darts away on a light touch, which feels
# like the robot "fighting" you), this profile is tuned for CONTROLLED
# hand-guiding:
#   - Translation: K = 0 (stays where you leave it) but MUCH higher
#     damping B so velocity under a firm push stays slow/controllable.
#   - Rotation: a stiff, overdamped spring back to the captured wrist
#     orientation (K_rot > 0, B_rot >> B_crit) so the wrist does NOT
#     wander under torque noise.  When the recorder is run with
#     ``hold_orientation=True`` the rotation admittance is bypassed
#     entirely and the wrist orientation is held rigidly.
# Damping scale note: the QP clips motion to otg_p_step (~0.005 m/cycle),
# and the admittance reaches that step when dx = (F_net/B)*dt is large
# enough.  So effective "draggability" is set by B relative to the push
# force AFTER the 0.5 N deadzone:
#   B=0.05 -> full speed at ~0.06 N  (darts on the faintest touch)
#   B=0.4  -> full speed at ~0.5 N   (light touches creep, firm pushes glide)
#   B=5.0  -> needs ~6 N to move     (feels immovable)
# 0.2-0.5 is the comfortable hand-guide range; tune with --trans-damping.
DRAG_TEACH_GAINS = AdmittanceGains(
    M=[0.1, 0.1, 0.1],
    B=[0.4, 0.4, 0.4],            # controlled drag (0.05=darty, 5.0=immovable)
    K=[0.0, 0.0, 0.0],            # no pull-back: arm stays where released
    M_rot=[0.02, 0.02, 0.02],
    B_rot=[2.0, 2.0, 2.0],        # overdamped (B_crit=2*sqrt(0.02*6)=0.69)
    # K_rot=[6.0, 6.0, 6.0],        # hold captured wrist orientation
    K_rot=[0.0, 0.0, 0.0],
    max_vel=20.0,
    max_omega=20.0,
    rot_lead_time=0.15,
)


# Used after the LEFT arm returns to HOME carrying the handed-over
# object.  Same idea as ADJUSTMENT_GAINS but tuned for carrying a small
# payload at HOME: zero stiffness (operator can push the arm anywhere),
# moderate damping (no spontaneous drift under FT sensor noise).
LEFT_HOME_COMPLIANT_GAINS = AdmittanceGains(
    M=[0.1, 0.1, 0.1],
    B=[0.6, 0.6, 0.6],
    K=[0.0, 0.0, 0.0],
    M_rot=[0.01, 0.01, 0.01],
    B_rot=[0.4, 0.4, 0.4],
    K_rot=[0.0, 0.0, 0.0],
    max_vel=20.0,
    max_omega=20.0,
    rot_lead_time=0.15,
)

# Right arm "release" admittance: the cup has been delivered at the
# handover pose and the operator drags the arm to wherever they want
# to take the cup before the gripper opens.  Critically different from
# RIGHT_GIVER_GAINS (which spring-pulls the arm back to the captured
# handover pose):
#   - K = 0      → no pull-back, arm stays where the operator leaves it
#   - K_rot = 0  → operator can rotate the cup freely
#   - B = 0.6    → moderate translational damping (prevents slow drift
#                  under FT sensor noise without feeling sluggish)
#   - B_rot = 0.4 → moderate rotational damping
# Same shape as LEFT_HOME_COMPLIANT_GAINS, named for the right-arm
# release semantics so callers know which profile to import.
RIGHT_RELEASE_GAINS = AdmittanceGains(
    M=[0.1, 0.1, 0.1],
    B=[0.6, 0.6, 0.6],
    K=[0.0, 0.0, 0.0],
    M_rot=[0.01, 0.01, 0.01],
    B_rot=[0.4, 0.4, 0.4],
    K_rot=[0.0, 0.0, 0.0],
    max_vel=20.0,
    max_omega=20.0,
    rot_lead_time=0.15,
)


# =====================================================================
# Force processing parameters (shared shape for both arms)
# =====================================================================
@dataclass
class ForceProcessing:
    force_deadzone: float = 0.5
    torque_deadzone: float = 0.05
    force_threshold: float = 0.5
    torque_threshold: float = 0.15
    filter_alpha: float = 0.8
    calib_samples: int = 200


DEFAULT_FORCE_PROC = ForceProcessing()


# =====================================================================
# Right hand (Robotiq 2F + Tac3D)
# =====================================================================
@dataclass
class RightGripperConfig:
    port: str = "/dev/serial/by-id/usb-FTDI_FT231X_USB_UART_D30JMIPY-if00-port0"
    slave_id: int = 9
    tactile_ports: tuple = (9988, 9989)
    predicted_grasp_force: float = -0.001     # N, per-sensor z trigger
    target_cf: float = 0.25                   # contact coefficient target
    cf_tolerance: float = 0.03
    release_speed_pct: int = 100              # when opening
    release_force_pct: int = 50
    # PF / adjust_grasp loop ``print`` spam (e.g. Estimating μ / Current cf)
    tactile_grasp_verbose: bool = False


DEFAULT_RIGHT_GRIPPER = RightGripperConfig()


# =====================================================================
# Left hand (Revo2 dextrous, pinch_2 = thumb + index)
# =====================================================================
@dataclass
class LeftHandConfig:
    port: str = "/dev/ttyUSB1"
    slave_id: int = 0x7E
    grasp_type: str = "pinch_2"               # thumb + index (phase 1)
    loop_rate: float = 20.0
    # Newton-calibrated (raw * 0.01)
    contact_threshold_N: float = 0.08
    contact_force_target_N: float = 0.35
    target_cf: float = 0.28
    tolerance: float = 0.03
    k_p: float = 6.0
    k_d: float = 0.4
    max_step_tighten: float = 3.0
    max_step_release: float = 5.0
    soft_limit_start_N: float = 2.0
    soft_limit_end_N: float = 4.0
    close_step_size: int = 5
    approach_speed: int = 80
    adapt_speed: int = 200
    max_adapt_range: int = 80
    max_release_range: int = 50
    # Keep thumb base centred, no adaptive lateral motion
    thumb_base_init_pos: int = 0
    thumb_base_scale: float = 0.3
    thumb_flex_position: int = 900
    thumb_flex_init_pos: Optional[int] = None  # if set, overrides thumb_flex_position/10 at init
    # Curled-finger positions for the three unused fingers (middle/ring/pinky)
    # Joint order: [thumb_base, thumb_flex, index, middle, ring, pinky]
    curl_positions: List[int] = field(default_factory=lambda: [0, 900, 0, 1000, 1000, 1000])
    curl_speeds: List[int] = field(default_factory=lambda: [400, 400, 400, 400, 400, 400])
    open_all_speed: int = 300
    # Background hold after grasp – how long to stay engaged before release
    hold_sec: float = 2.0
    ema_alpha: float = 0.4
    asymmetry_threshold_N: float = 0.5
    mu_ema_alpha: float = 0.3
    # Revo2AdaptiveGraspV6 ``[v6]`` per-loop prints
    adaptive_grasp_verbose: bool = True
    # Phase A (grasp_to_contact) safety: if one finger never crosses
    # ``contact_threshold_N`` (e.g. cylinder geometry), complete after
    # ``timeout_sec`` when at least ``min_fingers`` have real contact.
    # Default None/None = require all fingers (original behavior).
    phase_a_timeout_sec: Optional[float] = None
    phase_a_min_fingers: Optional[int] = None
    # With ``min_fingers`` set: complete partial grasp if this many
    # seconds elapse with no *new* finger contact (bottle [4/5] stall).
    phase_a_partial_stall_sec: Optional[float] = None


DEFAULT_LEFT_HAND = LeftHandConfig()


# =====================================================================
# Pipeline-level timing & safety
# =====================================================================
@dataclass
class PipelineConfig:
    poses_path: str = DEFAULT_POSES_PATH
    waist_frame: str = "waist_yaw_link"
    left_tcp_frame: str = "left_tcp_link"
    right_tcp_frame: str = "right_tcp_link"
    left_qp_controller: str = "endpose_single_arm_qp_L_controller"
    right_qp_controller: str = "endpose_single_arm_qp_R_controller"
    left_force_topic: str = "/arm_6dof_left"
    right_force_topic: str = "/arm_6dof_right"

    # ---- HOME joint configuration -----------------------------------
    # The "safe rest" pose.  Both arms are driven here at pipeline
    # start AND at pipeline end.  Deliberately tucked in so neither
    # arm is presenting to the world in the idle state.
    #
    # Source of truth: admittance_control/home_move_topic.py lines 92-93
    # (updated 2026-04-17).
    #
    # NOTE: the *previous* HOME values ([0, ±1.18, 0, -1.3, ±1.4, -0.13,
    # 0.18]) have been repurposed as "grasp staging" (right arm) and
    # "place staging" (left arm) poses in the detect_handover_place
    # pipeline — see ``detect_handover_place/config.py``
    # (DetectConfig.right_grasp_staging_joints and
    # PlaceConfig.pre_place_joints).
    
    # left_home_joints: List[float] = field(
    #     default_factory=lambda: [1.0, 0.3, 0.4, -2.3, 0.45, -0.20, 0.18]
    # )
    # right_home_joints: List[float] = field(
    #     default_factory=lambda: [1.0, -0.3, -0.4, -2.3, -0.45, -0.20, 0.18]
    # )

    left_home_joints: List[float] = field(
        default_factory=lambda: [0.0, 1.18, 0.0, -1.3, 1.4, -0.13, 0.18]
    )
    right_home_joints: List[float] = field(
        default_factory=lambda: [0.0, -1.18, 0.0, -1.3, -1.4, -0.13, 0.18]
    )

    otg_p_step: float = 0.005
    otg_r_step: float = 0.005

    # Motion to handover pose via MoveIt
    moveit_vel_scale: float = 0.15
    moveit_acc_scale: float = 0.15
    moveit_spin_timeout: float = 25.0
    plan_retries: int = 2

    # Admittance loop target period (s) — Orin can sustain well under this.
    loop_period: float = 0.004

    # Seconds to stay in "both-hands-holding" overlap before the right opens.
    overlap_sec: float = 0.8
    # Seconds to pause after right opens, before admittance transitions left.
    post_release_sec: float = 0.8
    # How far the right arm retracts (in -X of waist) after releasing.
    right_retract_dx: float = 0.15
    # After retreat, drive RIGHT arm back to its home pose (joint-space).
    retreat_to_home: bool = True
    # After RIGHT arm retreats, also drive the LEFT arm (still holding
    # the object!) back to its HOME joint pose.
    left_return_to_home: bool = True
    # Block on input() before releasing the left hand.  When True the
    # pipeline ends with both arms at HOME, the left hand still gripping
    # the object, and waits for the operator to press Enter.
    wait_for_enter_before_release: bool = True
    # After the LEFT arm reaches HOME while still gripping, re-engage
    # admittance on the LEFT arm with zero-stiffness gains so the
    # operator can hand-guide the arm (and the held object) freely
    # during the wait-for-Enter phase.  The RIGHT arm stays parked at
    # HOME under its jointspace controller.
    compliant_wait_at_home: bool = True


DEFAULT_PIPELINE = PipelineConfig()
