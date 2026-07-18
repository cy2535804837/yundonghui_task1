#!/usr/bin/env python3
"""
compliant_grasp_execute
========================

Use detected grasp pose from grasp_pose_generation and move one arm to grasp.

Configuration
-------------
All tunable parameters live in ``config.yaml`` (next to this file) and are
auto-loaded on every run, so the common case is simply::

    python3 -m compliant_grasp_execute.main

Edit ``config.yaml`` to change behaviour (object prompt, tilt, offsets, motion
strategy, ...). Any command-line flag still overrides the file for a one-off,
e.g. grasp a different object once::

    python3 -m compliant_grasp_execute.main --prompt sponge

Precedence: CLI flag > config.yaml > built-in default. Regenerate/refresh the
file (e.g. after adding a new flag) with::

    python3 -m compliant_grasp_execute.main --write-config

Pass ``--config ''`` to ignore the file and use built-in defaults, or
``--config my.yaml`` to use a different file.

Note: ``--prompt`` is an append flag, so passing it on the CLI ADDS to the
prompt(s) in the config. To change the grasped object permanently, edit the
``prompt`` list in ``config.yaml``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Pose

from compliant_grasp_execute.config_io import (
    add_config_args,
    apply_config_defaults,
    default_config_path,
    dump_config,
    maybe_write_config,
)
from compliant_grasp_execute.joint_limits import (
    ARM_JOINT_NAMES,
    TIANYI2_ARM_LIMITS,
    JointLimitGuard,
    fetch_arm_limits,
)
from grasp_pose_generation.internal.fast_seg_client import FastSegClient
from grasp_pose_generation.internal.object_pose_pipeline import get_object_pose_in_waist_yaw_link
from grasp_pose_generation.internal.perception_tools import PerceptionTool
from grasp_pose_generation.internal.pose_estimator import PoseEstimator, extract_masks_from_results
from xarm_sdk import ActionCall, MoveitCall, TopicPublisher, XARM_manager
try:
    from xarm_sdk.tools import set_node_parameter
except Exception:
    from xarm_sdk import set_node_parameter  # type: ignore

_TAG = "[GRASP-EXEC]"

# Calibrated defaults from bottle_cup_pour_place/detect_pour_place.py
_DEFAULT_RIGHT_QUAT = [0.4708, 0.5384, -0.5371, -0.4473]
# Left grasp orientation = sagittal-plane mirror of the calibrated right quat
# (reflect across the waist XZ-plane, +Y normal): (x,y,z,w) -> (-x, y, -z, w).
# The previous hand value [0.1378,-0.7086,0.0394,0.6909] was uncalibrated/wrong.
_DEFAULT_LEFT_QUAT = [
    -_DEFAULT_RIGHT_QUAT[0],
    _DEFAULT_RIGHT_QUAT[1],
    -_DEFAULT_RIGHT_QUAT[2],
    _DEFAULT_RIGHT_QUAT[3],
]
_DEFAULT_RIGHT_XYZ_OFFSET = (-0.25, 0.01, 0.01)
# Mirror of the right pour-place offset across the sagittal plane (negate Y).
_DEFAULT_LEFT_XYZ_OFFSET = (
    _DEFAULT_RIGHT_XYZ_OFFSET[0],
    -_DEFAULT_RIGHT_XYZ_OFFSET[1],
    _DEFAULT_RIGHT_XYZ_OFFSET[2],
)
# Home joint postures (rad, 7 joints) shared with home_move_topic.py. Both arms
# are sent here before the detect->grasp cycle starts, and returned here after.
_LEFT_ARM_HOME_JOINTS = [0.844, 0.026, -0.006, -2.216, -1.529, 0.039, 0.683]
_RIGHT_ARM_HOME_JOINTS = [0.844, -0.026, 0.006, -2.216, 1.529, 0.039, -0.683]
# Elbow-HIGH staging posture (rad, 7 joints): the arm is sent HERE from home
# BEFORE the elbow-high reconfigure (transition -> ready seed), and passes back
# through it on the way home. It is the previous "arm-up, elbow-bent" home, which
# is a clean base for the transition sweep into the elbow-high basin. The
# elbow-LOW path does NOT use it (it approaches straight from home).
_ELBOW_HIGH_STAGE_LEFT_JOINTS = [0.0, 1.18, 0.0, -1.3, 1.4, -0.13, 0.18]
_ELBOW_HIGH_STAGE_RIGHT_JOINTS = [0.0, -1.18, 0.0, -1.3, -1.4, -0.13, 0.18]
# Elbow-HIGH "top reconfiguration" seed posture (rad, 7 joints), hand-taught.
# Used to switch the redundant arm out of the elbow-low QP basin for awkward
# object poses. The right-arm default is the sagittal mirror (negate
# shoulder_roll, shoulder_yaw, elbow_yaw, wrist_roll). Near-limit joints are
# auto-clamped inward at runtime (see --elbow-high-clamp-margin-rad).
_ELBOW_HIGH_READY_LEFT_JOINTS = [-2.610, 0.178, -2.964, -1.817, -0.204, -0.644, -0.066]
_ELBOW_HIGH_READY_RIGHT_JOINTS = [-2.610, -0.178, 2.964, -1.817, 0.204, -0.644, 0.066]

_POUR_PLACE_QPIK_OTG_P_STEP = 0.0008
_POUR_PLACE_QPIK_OTG_R_STEP = 0.001


def _log(msg: str) -> None:
    print(f"{_TAG} {msg}", flush=True)


def _ensure_robotiq_usb_import_path() -> None:
    """Add local RobotiqUSB folder to sys.path for pure gripper control."""
    cand = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "RobotiqUSB")
    )
    if cand not in sys.path:
        sys.path.insert(0, cand)


class _AcceleratedSegToPoseAdapter:
    def __init__(self, *, base_url: str, camera_yaml: str) -> None:
        self._seg = FastSegClient(base_url=base_url)
        self._estimator = PoseEstimator.from_yaml(camera_yaml)

    def perception_pipeline(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        prompts: List[str],
        confidence: float = 0.8,
    ):
        results_2d = self._seg.perception_pipeline(
            rgb_image, prompts, confidence=float(confidence)
        )
        if not isinstance(results_2d, dict) or not results_2d.get("results"):
            return None, results_2d
        masks = extract_masks_from_results(results_2d, depth_image)
        detections = list(results_2d.get("results") or [])
        results_output = []
        for i, det in enumerate(detections):
            mask = masks[i] if i < len(masks) else None
            if mask is None:
                continue
            pose, info = self._estimator.estimate_pose(mask, depth_image)
            if "points_3d" not in info:
                continue
            bbox_3d = self._estimator.compute_bounding_box(info["points_3d"])
            results_output.append(
                {
                    "class_name": det.get("label", "unknown"),
                    "pose": pose,
                    "bbox_2d": det.get("bbox"),
                    "bbox_3d": bbox_3d,
                    "mask_id": i,
                    "grasp_axes": info.get("grasp_axes"),
                }
            )
        return (results_output if results_output else None), results_2d


def _resolve_camera_yaml(path: str) -> str:
    if os.path.isabs(path):
        return path
    local_candidate = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "grasp_pose_generation",
        "assets",
        path,
    )
    if os.path.exists(local_candidate):
        return local_candidate
    return os.path.abspath(path)


def _moveit_to_pose7(
    moveit: MoveitCall,
    xarm: XARM_manager,
    arm: str,
    pose7: List[float],
    waist_frame: str,
    vel_scale: float,
    acc_scale: float,
    label: str,
    *,
    use_cartesian_path: bool = False,
    moveit_timeout_sec: float = 120.0,
) -> bool:
    wp = [float(v) for v in pose7]
    if arm == "left":
        jd = moveit.build_left_arm_waypoints_json(
            left_waypoints=[wp],
            vel_scale=vel_scale,
            acc_scale=acc_scale,
            frame=waist_frame,
            mode="plan_and_execute",
            use_cartesian_path=use_cartesian_path,
        )
    else:
        jd = moveit.build_right_arm_waypoints_json(
            right_waypoints=[wp],
            vel_scale=vel_scale,
            acc_scale=acc_scale,
            frame=waist_frame,
            mode="plan_and_execute",
            use_cartesian_path=use_cartesian_path,
        )
    _log(
        f"MoveIt {arm} -> {label}"
        + (" (cartesian)" if use_cartesian_path else "")
        + f": pose7={[f'{v:.4f}' for v in wp]}"
    )
    fut = moveit.arm_waypoints_async(jd)
    if float(moveit_timeout_sec) > 0:
        rclpy.spin_until_future_complete(xarm, fut, timeout_sec=float(moveit_timeout_sec))
    else:
        rclpy.spin_until_future_complete(xarm, fut)
    if not fut.done():
        _log(f"MoveIt {label} timed out after {moveit_timeout_sec:.0f}s")
        return False
    res = fut.result()
    ok = bool(res and res[0])
    _log(f"MoveIt {label} result: {res} ok={ok}")
    return ok


def _moveit_to_pose7_with_fallback(
    moveit: MoveitCall,
    xarm: XARM_manager,
    arm: str,
    pose7: List[float],
    waist_frame: str,
    vel_scale: float,
    acc_scale: float,
    label: str,
    *,
    use_cartesian_path: bool = False,
    moveit_timeout_sec: float = 120.0,
) -> bool:
    """MoveIt once; on failure retry same XYZ with live TCP orientation (pour_place style)."""
    ok = _moveit_to_pose7(
        moveit,
        xarm,
        arm,
        pose7,
        waist_frame,
        vel_scale,
        acc_scale,
        label,
        use_cartesian_path=use_cartesian_path,
        moveit_timeout_sec=moveit_timeout_sec,
    )
    if ok:
        return True
    q = _read_tcp_quat(xarm, arm, waist_frame, pose7)
    retry_pose7 = [float(pose7[0]), float(pose7[1]), float(pose7[2]), *q]
    _log(
        f"MoveIt {label} failed; retry with current TCP orientation "
        f"q={[f'{v:.4f}' for v in q]}"
    )
    return _moveit_to_pose7(
        moveit,
        xarm,
        arm,
        retry_pose7,
        waist_frame,
        vel_scale,
        acc_scale,
        f"{label}_retry_current_quat",
        use_cartesian_path=use_cartesian_path,
        moveit_timeout_sec=moveit_timeout_sec,
    )


def _default_grasp_quat(arm: str) -> List[float]:
    return list(_DEFAULT_LEFT_QUAT if arm == "left" else _DEFAULT_RIGHT_QUAT)


def _default_grasp_xyz_offset(arm: str) -> tuple[float, float, float]:
    if arm == "left":
        return _DEFAULT_LEFT_XYZ_OFFSET
    return _DEFAULT_RIGHT_XYZ_OFFSET


def _resolve_grasp_quat(args: argparse.Namespace, arm: str) -> List[float]:
    if getattr(args, "grasp_quat", None):
        return [float(v) for v in args.grasp_quat]
    return _default_grasp_quat(arm)


def _resolve_grasp_xyz_offset(args: argparse.Namespace, arm: str) -> tuple[float, float, float]:
    # Any explicitly-set axis applies independently; unset axes default to 0.
    # (Previously a z-only offset was silently ignored unless --grasp-x-offset
    # was also given.)
    if any(
        v is not None
        for v in (args.grasp_x_offset, args.grasp_y_offset, args.grasp_z_offset)
    ):
        return (
            float(args.grasp_x_offset or 0.0),
            float(args.grasp_y_offset or 0.0),
            float(args.grasp_z_offset or 0.0),
        )
    if args.use_pour_place_offsets:
        return _default_grasp_xyz_offset(arm)
    return (0.0, 0.0, 0.0)


def _diagonal_schedule(
    args: argparse.Namespace, angle_from_x: Optional[float]
) -> tuple[float, float]:
    """Continuous tilt/yaw schedule for diagonal objects, keyed off the detected
    object long-axis angle from waist +X (deg).

    Returns ``(tilt_scale, yaw_clamp_deg)`` for the STANDARD grasp path (not the
    elbow-high path). The problem this solves: a fixed 45deg waist-Y tilt plus a
    jaw yaw clamped to +/-15deg works for perpendicular (~0deg) objects but fails
    for DIAGONAL objects (~20-60deg): the +/-15deg clamp can't align the jaws to
    the long axis (a 38deg object gets only 15deg of jaw yaw -> ~23deg misaligned)
    AND even that capped 15deg yaw drives ``wrist_pitch`` past its +1.05 upper stop
    at far-lateral reaches. So the grasp either twists the wrist into its limit or
    closes misaligned on the object.

    Between ``angle_start`` and ``angle_end`` the schedule scales the tilt DOWN
    (freeing ``wrist_pitch``) and the jaw-yaw clamp UP (actually aligning the jaws),
    linearly with the angle. Below ``angle_start`` it returns ``(1.0, base_clamp)``
    so the proven perpendicular behaviour is unchanged. At/above ``angle_end`` the
    elbow-high proactive path takes over anyway (its threshold defaults to 60deg),
    so the schedule end defaults to 60deg for a smooth handoff.
    """
    base_clamp = float(getattr(args, "continuous_grasp_max_yaw_deg", 15.0))
    try:
        ang = float(angle_from_x)
    except (TypeError, ValueError):
        return 1.0, base_clamp
    if not bool(getattr(args, "diagonal_schedule", True)):
        return 1.0, base_clamp
    start = float(getattr(args, "diagonal_schedule_angle_start_deg", 15.0))
    end = float(getattr(args, "diagonal_schedule_angle_end_deg", 60.0))
    if end <= start or ang <= start:
        return 1.0, base_clamp
    end_clamp = float(getattr(args, "diagonal_schedule_yaw_clamp_end_deg", 45.0))
    tilt_end = float(getattr(args, "diagonal_schedule_tilt_scale_end", 0.5))
    frac = min(1.0, max(0.0, (ang - start) / (end - start)))
    tilt_scale = 1.0 + (tilt_end - 1.0) * frac
    yaw_clamp = base_clamp + (end_clamp - base_clamp) * frac
    return tilt_scale, yaw_clamp


def _reach_tilt_scale(args: argparse.Namespace, det: Dict[str, Any]) -> float:
    """Scale (<=1.0) for the STANDARD (elbow-low) side-tilt, reduced for
    far-outboard objects so the wrist lands less outboard and stays reachable.

    A far-LATERAL object (large |Y| in the waist frame) drives ``elbow_pitch``
    into its -2.61 fully-folded stop: the 45deg nose-down side-tilt lever-arms
    the wrist further out than the object, and the elbow has to fold hard to hold
    that orientation at reach. Backing the tilt off (toward level) lets the elbow
    extend, pulling ``elbow_pitch`` off its stop -- the elbow-low way to reach a
    far object (the elbow-HIGH path is reserved for the parallel-axis bucket).
    Keyed off the object tip |Y| (a stable, tilt-independent proxy for reach).
    Below ``start`` returns 1.0 so centered objects keep the proven 45deg tilt.

    NB: this is keyed on LATERAL reach only. A far-FORWARD term was tried and
    reverted: it stacked multiplicatively with the diagonal schedule and
    over-flattened the tool to ~10deg (near-horizontal), raking the table, while
    not actually clearing the wrist_pitch upper stop it targeted -- far-forward
    objects are simply at the arm's downward-grasp reach limit.
    """
    if not bool(getattr(args, "reach_tilt_reduce", True)):
        return 1.0
    try:
        tip = det.get("pose_tip_waist_yaw_link_rpy") or {}
        ay = abs(float(tip.get("y")))
    except (TypeError, ValueError):
        return 1.0
    start = float(getattr(args, "reach_tilt_start_abs_y_m", 0.18))
    end = float(getattr(args, "reach_tilt_end_abs_y_m", 0.28))
    if end <= start or ay <= start:
        return 1.0
    scale_end = float(getattr(args, "reach_tilt_scale_end", 0.4))
    frac = min(1.0, max(0.0, (ay - start) / (end - start)))
    return 1.0 + (scale_end - 1.0) * frac


def _symmetric_jaw_yaw(
    base_jaw_yaw_deg: float,
    long_yaw_deg: float,
    roll0_rad: float,
    roll_lo_rad: float,
    roll_hi_rad: float,
    margin_rad: float,
    max_yaw_deg: float,
    flip: bool = False,
    target_heading_deg: Optional[float] = None,
) -> float:
    """Pick the waist-Z yaw (deg) that aligns the jaws to grasp the object's
    SHORT axis, keeping ``wrist_roll`` in range.

    This is an ABSOLUTE alignment, not a relative nudge. The gripper's
    jaw-closing axis is tool_x; to grasp the short axis it must point along the
    short-axis direction ``long_yaw + 90deg`` (a line, so mod 180). A waist-Z
    rotation by ``theta`` turns the base jaw-axis heading ``base_jaw_yaw_deg`` to
    ``base_jaw_yaw_deg + theta``, so the required rotation is
    ``theta = (long_yaw + 90) - base_jaw_yaw`` folded into ``(-90, 90]`` (jaw
    line symmetry). We must compute this from the base's MEASURED jaw heading
    because the elbow-high seed is hand-taught: its baseline jaw axis sits at an
    arbitrary heading, so applying the object yaw *relatively* (the old code)
    left the jaws pointing near the LONG axis instead (grasping end-to-end).

    In the ~top-down elbow-high posture a waist-Z yaw maps ~1:1 onto
    ``wrist_roll`` (predicted roll ~= roll0 + theta). The parallel-jaw 180deg
    symmetry gives equivalents ``theta + k*180``; we pick the one that (a) stays
    within ``+/-max_yaw_deg`` and (b) keeps predicted ``wrist_roll`` inside its
    (asymmetric) limits with ``margin_rad`` to spare, ties broken toward the
    smallest wrist travel. This is what lets the left arm roll ~-90deg and the
    right ~+90deg for a parallel object, matching where each arm's roll has room.
    """
    import math

    if target_heading_deg is not None:
        # Fixed jaw-line heading (e.g. waist +X = 0deg) instead of the object's
        # short axis. Used as a wrist_roll-saturation fallback: the diagonal
        # short axis can need a large yaw that pins wrist_roll on its cramped
        # stop, whereas a fixed heading near the fore-aft seed needs only a
        # small roll that stays in range (at the cost of short-axis alignment).
        target = float(target_heading_deg)
    else:
        target = float(long_yaw_deg) + 90.0  # short-axis line heading
    raw = target - float(base_jaw_yaw_deg)
    raw = ((raw + 90.0) % 180.0) - 90.0  # minimal jaw-line-equivalent delta
    if flip:
        # Same short-axis grip, opposite 180deg jaw-line equivalent. The minimal
        # rotation ``raw`` can drive wrist_roll into the arm's CRAMPED stop; the
        # complement (raw +/- 180) reaches the identical jaw LINE from the
        # mirrored side, rolling the wrist onto its ROOMY side. Bypass the
        # max_yaw clamp and the roll-feasibility search -- the whole point is to
        # go past +/-90deg to the other branch.
        return float(raw + 180.0 if raw <= 0.0 else raw - 180.0)
    best: Optional[tuple] = None
    for k in (-1, 0, 1, 2, -2):
        c = raw + 180.0 * k
        if abs(c) > float(max_yaw_deg) + 1e-6:
            continue
        roll = float(roll0_rad) + math.radians(c)
        viol = max(0.0, (roll_lo_rad + margin_rad) - roll) + max(
            0.0, roll - (roll_hi_rad - margin_rad)
        )
        score = (viol, abs(c))
        if best is None or score < best[0]:
            best = (score, c)
    if best is None:
        # Nothing within max_yaw: return the minimal delta, clamped.
        return float(max(-max_yaw_deg, min(max_yaw_deg, raw)))
    return float(best[1])


def _build_grasp_pose7(
    det: Dict[str, Any],
    args: argparse.Namespace,
    arm: str,
    extra_yaw_scale: float = 1.0,
    extra_tilt_scale: float = 1.0,
    tilt_override_deg: Optional[float] = None,
    max_yaw_override_deg: Optional[float] = None,
    base_quat_override: Optional[List[float]] = None,
    jaw_yaw_symmetry: bool = False,
    jaw_yaw_roll0: float = 0.0,
    jaw_yaw_roll_limits: Optional[Tuple[float, float]] = None,
    jaw_yaw_margin: float = 0.05,
    jaw_yaw_flip: bool = False,
    jaw_target_heading_deg: Optional[float] = None,
) -> List[float]:
    """Build a self-consistent grasp TCP pose7 in the waist frame.

    ``extra_yaw_scale`` scales the adaptive/continuous yaw applied about waist Z
    (1.0 = full alignment). The approach retry uses a value < 1.0 to back off the
    alignment when full alignment drives a wrist joint past its limit.

    Bug-A fix: the detected ``pose_tcp_waist_yaw_link_pose7`` applies the
    tcp<-tip offset along the *detection-time* orientation. If we then override
    the orientation with a fixed grasp quat, the 0.2 m lever arm makes the
    fingertip miss the object by ``(R_detect - R_final) @ offset``.

    So we recompute the TCP position from the object centroid
    (``pose_tip_waist_yaw_link_rpy``) using the *final* orientation we will
    actually command, then apply optional waist-frame XYZ offsets.
    """
    from scipy.spatial.transform import Rotation as R

    detected_tcp = [float(v) for v in det["pose_tcp_waist_yaw_link_pose7"]]

    if base_quat_override is not None:
        # Anchor on a supplied orientation (e.g. the measured elbow-high seed
        # orientation) instead of the configured/ detected grasp quat. tilt/yaw
        # overrides still apply on top.
        quat = [float(v) for v in base_quat_override]
    elif args.use_fixed_grasp_quat:
        quat = [float(v) for v in _resolve_grasp_quat(args, arm)]
    else:
        quat = [float(v) for v in detected_tcp[3:7]]

    # Human-like tilt: rotate the grasp orientation about the waist-frame Y axis
    # so the gripper noses down instead of staying parallel to the ground.
    _sched_tilt_scale = 1.0
    _reach_ts = 1.0
    _sched_yaw_clamp: Optional[float] = None
    if tilt_override_deg is not None:
        # Absolute tilt (e.g. 90deg = pure top-down for the elbow-high path);
        # ignores grasp_tilt_y_deg and extra_tilt_scale. The diagonal schedule
        # does NOT apply on the elbow-high path.
        tilt_deg = float(tilt_override_deg)
    else:
        _sched_tilt_scale, _sched_yaw_clamp = _diagonal_schedule(
            args, det.get("object_angle_from_waist_x_deg")
        )
        _reach_ts = _reach_tilt_scale(args, det)
        tilt_deg = (
            float(getattr(args, "grasp_tilt_y_deg", 0.0))
            * float(extra_tilt_scale)
            * float(_sched_tilt_scale)
            * float(_reach_ts)
        )
    if abs(tilt_deg) > 1e-6:
        quat_tilted = (
            R.from_euler("y", tilt_deg, degrees=True) * R.from_quat(quat)
        ).as_quat()
        quat = [float(v) for v in quat_tilted]
        _tilt_mods = (
            abs(float(extra_tilt_scale) - 1.0) > 1e-6
            or abs(float(_sched_tilt_scale) - 1.0) > 1e-6
            or abs(float(_reach_ts) - 1.0) > 1e-6
        )
        tilt_scale_str = (
            f" (tilt scale {float(extra_tilt_scale):.2f}"
            + (
                f" x diag {float(_sched_tilt_scale):.2f}"
                if abs(float(_sched_tilt_scale) - 1.0) > 1e-6
                else ""
            )
            + (
                f" x reach {float(_reach_ts):.2f}"
                if abs(float(_reach_ts) - 1.0) > 1e-6
                else ""
            )
            + ")"
            if _tilt_mods
            else ""
        )
        if abs(float(_sched_tilt_scale) - 1.0) > 1e-6:
            _log(
                f"diagonal schedule: angle-from-X="
                f"{det.get('object_angle_from_waist_x_deg')}deg -> tilt_scale="
                f"{float(_sched_tilt_scale):.2f}, yaw_clamp="
                f"{float(_sched_yaw_clamp):.1f}deg (freed wrist_pitch, raised jaw "
                "alignment)"
            )
        if abs(float(_reach_ts) - 1.0) > 1e-6:
            _tip_y = (det.get("pose_tip_waist_yaw_link_rpy") or {}).get("y")
            _log(
                f"reach tilt reduction: object |Y|="
                f"{abs(float(_tip_y)):.3f}m -> tilt_scale={float(_reach_ts):.2f} "
                "(backed the side-tilt off so the elbow can extend to a "
                "far-outboard object instead of folding into its stop)"
            )
        _log(f"applied grasp tilt {tilt_deg:+.1f}deg about waist Y{tilt_scale_str} -> quat={[f'{v:.4f}' for v in quat]}")

    # Adaptive grasp orientation for elongated objects (e.g. banana). The
    # detection exposes the object's long-axis heading in the waist frame:
    #   angle_from_x ~ 0deg  -> long axis forward (waist +X): "vertical to body"
    #   angle_from_x ~ 90deg -> long axis left/right (waist Y): "parallel to body"
    # Two ways to adapt:
    #   * continuous (--continuous-grasp-orientation): rotate the fixed quat by
    #     the measured long-axis yaw so the grasp tracks the object's heading
    #     instead of snapping to a discrete bucket. Takes precedence and works
    #     independently of the binary parallel/vertical classifier.
    #   * binary (--adaptive-grasp-orientation): when the object lies parallel to
    #     the body the fixed orientation cannot grasp across it, so add a fixed
    #     extra yaw about waist Z (+ for right arm, - for left arm).
    angle_from_x = det.get("object_angle_from_waist_x_deg")
    long_yaw = det.get("object_long_axis_yaw_waist_deg")
    continuous = bool(getattr(args, "continuous_grasp_orientation", False))
    adaptive = bool(getattr(args, "adaptive_grasp_orientation", False))
    if continuous:
        if long_yaw is not None:
            gain = float(getattr(args, "continuous_grasp_yaw_gain", 1.0))
            if max_yaw_override_deg is not None:
                max_yaw = float(max_yaw_override_deg)
            elif _sched_yaw_clamp is not None:
                # Diagonal schedule raised the jaw-yaw clamp so the jaws can
                # actually track the object's long axis (the base +/-15deg clamp
                # leaves diagonal objects ~20deg misaligned).
                max_yaw = float(_sched_yaw_clamp)
            else:
                max_yaw = float(getattr(args, "continuous_grasp_max_yaw_deg", 90.0))
            if jaw_yaw_symmetry and jaw_yaw_roll_limits is not None:
                # Top-down (elbow-high): ABSOLUTELY align the jaws to the object's
                # short axis. The gripper jaw-closing axis is tool_x; measure its
                # current horizontal heading off the (hand-taught) base seed quat,
                # then rotate about waist Z so it lands on the short-axis direction
                # -- picking the 180deg-equivalent that keeps wrist_roll in range.
                # (A relative nudge is wrong here: the seed's baseline jaw heading
                # is arbitrary, so +long_yaw left the jaws near the LONG axis.)
                _tx = R.from_quat(quat).as_matrix()[:, 0]
                _base_jaw_yaw = float(np.degrees(np.arctan2(_tx[1], _tx[0])))
                extra_yaw = _symmetric_jaw_yaw(
                    _base_jaw_yaw,
                    float(long_yaw),
                    float(jaw_yaw_roll0),
                    float(jaw_yaw_roll_limits[0]),
                    float(jaw_yaw_roll_limits[1]),
                    float(jaw_yaw_margin),
                    float(max_yaw),
                    flip=bool(jaw_yaw_flip),
                    target_heading_deg=jaw_target_heading_deg,
                )
                extra_yaw *= float(extra_yaw_scale)
                if jaw_target_heading_deg is not None:
                    _tgt_str = f"waist-axis {float(jaw_target_heading_deg):+.1f}deg"
                else:
                    _tgt_str = f"short-axis {float(long_yaw) + 90.0:+.1f}deg"
                _yaw_mode_str = (
                    f" [top-down jaw-align: base jaw heading {_base_jaw_yaw:+.1f}deg"
                    f" -> {_tgt_str}"
                    + (" FLIP(roomy-roll side)" if jaw_yaw_flip else "")
                    + "]"
                )
            else:
                extra_yaw = float(np.clip(gain * float(long_yaw), -max_yaw, max_yaw))
                extra_yaw *= float(extra_yaw_scale)
                _yaw_mode_str = ""
            quat_yawed = (
                R.from_euler("z", extra_yaw, degrees=True) * R.from_quat(quat)
            ).as_quat()
            quat = [float(v) for v in quat_yawed]
            ang_str = (
                f"{float(angle_from_x):.1f}" if angle_from_x is not None else "n/a"
            )
            scale_str = (
                f" x scale {float(extra_yaw_scale):.2f}"
                if abs(float(extra_yaw_scale) - 1.0) > 1e-6
                else ""
            )
            _log(
                f"continuous orientation: object long-axis yaw {float(long_yaw):+.1f}deg "
                f"(angle-from-X {ang_str}deg); applied continuous yaw {extra_yaw:+.1f}deg "
                f"about waist Z (gain={gain:.2f}, clamp +/-{max_yaw:.0f}deg{scale_str})"
                f"{_yaw_mode_str} -> quat="
                f"{[f'{v:.4f}' for v in quat]}"
            )
        else:
            _log(
                "continuous orientation: no object long-axis yaw from detection; "
                "keeping fixed orientation"
            )
    elif adaptive and angle_from_x is not None:
        threshold = float(getattr(args, "parallel_detect_threshold_deg", 45.0))
        extra_mag = float(getattr(args, "parallel_extra_yaw_deg", 15.0))
        is_parallel = float(angle_from_x) >= threshold
        # Geometric "align" suggestion: yaw about Z that would rotate a
        # forward-pointing grasp onto the object's long axis (sign per waist
        # frame; logged for comparison with the fixed +/- value actually used).
        cont_suggest = float(long_yaw) if long_yaw is not None else float("nan")
        klass = "parallel" if is_parallel else "vertical"
        if is_parallel:
            extra_yaw = (extra_mag if arm == "right" else -extra_mag) * float(extra_yaw_scale)
            quat_yawed = (
                R.from_euler("z", extra_yaw, degrees=True) * R.from_quat(quat)
            ).as_quat()
            quat = [float(v) for v in quat_yawed]
            _log(
                f"adaptive orientation: object long-axis {float(angle_from_x):.1f}deg from "
                f"waist X -> '{klass}' (>= {threshold:.0f}deg); applied extra yaw "
                f"{extra_yaw:+.1f}deg about waist Z ({arm}) -> quat="
                f"{[f'{v:.4f}' for v in quat]} | continuous-align suggestion="
                f"{cont_suggest:+.1f}deg"
            )
        else:
            _log(
                f"adaptive orientation: object long-axis {float(angle_from_x):.1f}deg from "
                f"waist X -> '{klass}' (< {threshold:.0f}deg); keeping fixed orientation "
                f"| continuous-align suggestion={cont_suggest:+.1f}deg"
            )
    elif adaptive:
        _log("adaptive orientation: no object long-axis from detection; keeping fixed orientation")

    tip = det.get("pose_tip_waist_yaw_link_rpy")
    offset = np.array(
        [float(args.tcp_to_tip_x), float(args.tcp_to_tip_y), float(args.tcp_to_tip_z)],
        dtype=float,
    )
    if isinstance(tip, dict) and all(k in tip for k in ("x", "y", "z")):
        tip_xyz = np.array([float(tip["x"]), float(tip["y"]), float(tip["z"])], dtype=float)
        # Reach-aware reorientation (standard path only). The fixed grasp
        # orientation points the gripper sideways, so tool_z carries a large Y
        # component and the wrist TCP lands further outboard than the object tip
        # (TCP_Y = tip_Y + lever * tool_z_Y, lever ~0.266m). For a far-side
        # object that puts the TCP past the elbow-low basin's reach (elbow_pitch
        # -> -2.61 stop at |TCP_Y| ~0.25), forcing a fallback to elbow-high even
        # for a perpendicular object. The Y-tilt can't fix this (a waist-Y
        # rotation leaves tool_z_Y unchanged -- verified), so instead rotate the
        # grasp orientation about waist X to flatten tool_z_Y: the wrist then
        # stays over the object and the standard basin gets a chance to reach it
        # with a steeper (more top-down) approach. Capped to limit jaw
        # misalignment; if the cap leaves |TCP_Y| still past the edge, the
        # proactive elbow-high trigger catches it downstream.
        if (
            tilt_override_deg is None
            and base_quat_override is None
            and bool(getattr(args, "elbow_low_reorient", True))
        ):
            _lever = -float(offset[2])  # TCP = tip + lever * tool_z (lever ~+0.266)
            _tz_pre = R.from_quat(quat).as_matrix()[:, 2]
            _would_y = float(tip_xyz[1]) + _lever * float(_tz_pre[1])
            _reach_thr = float(
                getattr(args, "elbow_low_reorient_trigger_abs_y_m", 0.24)
            )
            if (
                abs(_would_y) >= _reach_thr
                and abs(float(_tz_pre[1])) > 1e-3
                and abs(float(_tz_pre[2])) > 1e-3
            ):
                _phi = float(
                    np.degrees(
                        np.arctan2(
                            abs(float(_tz_pre[1])), abs(float(_tz_pre[2]))
                        )
                    )
                )
                _phi_cap = float(getattr(args, "elbow_low_reorient_max_deg", 20.0))
                _phi = min(_phi, _phi_cap)
                _sign = 1.0 if float(_tz_pre[1]) > 0 else -1.0
                quat_re = (
                    R.from_euler("x", _sign * _phi, degrees=True)
                    * R.from_quat(quat)
                ).as_quat()
                _tz_post = R.from_quat(quat_re).as_matrix()[:, 2]
                _post_y = float(tip_xyz[1]) + _lever * float(_tz_post[1])
                _log(
                    f"reach-aware reorient: standard TCP |Y| would be "
                    f"{abs(_would_y):.3f}m (>= {_reach_thr}m, past the elbow-low "
                    f"edge); rotated grasp {_sign * _phi:+.1f}deg about waist X to "
                    f"flatten tool_z_Y {float(_tz_pre[1]):.3f}->"
                    f"{float(_tz_post[1]):.3f} -> TCP |Y| {abs(_post_y):.3f}m "
                    "(wrist back over the object so the standard basin can reach "
                    "it with a steeper approach)."
                )
                quat = [float(v) for v in quat_re]
        rot = R.from_quat(quat).as_matrix()
        # tcp = tip + R_final @ (-offset)  (mirrors tip_pose_to_tcp_pose)
        tcp_xyz = tip_xyz + rot @ (-offset)
        _log(
            f"recomputed TCP along final quat: detected_tcp_xyz="
            f"{[f'{v:.4f}' for v in detected_tcp[:3]]} -> "
            f"consistent_tcp_xyz={[f'{v:.4f}' for v in tcp_xyz.tolist()]}"
        )
    else:
        _log("WARNING: detection missing pose_tip_waist_yaw_link_rpy; using detected TCP xyz as-is")
        tcp_xyz = np.array(detected_tcp[:3], dtype=float)

    out = [
        float(tcp_xyz[0]),
        float(tcp_xyz[1]),
        float(tcp_xyz[2]),
        float(quat[0]),
        float(quat[1]),
        float(quat[2]),
        float(quat[3]),
    ]
    dx, dy, dz = _resolve_grasp_xyz_offset(args, arm)
    _offset_frame = str(getattr(args, "grasp_offset_frame", "object")).lower()
    if (
        _offset_frame == "object"
        and long_yaw is not None
        and (abs(dx) > 1e-9 or abs(dy) > 1e-9)
    ):
        # Apply the horizontal grasp offset in the OBJECT's axis frame (not the
        # waist frame) so a FIXED offset grasps the same physical spot no matter
        # how the object is rotated on the table. dx runs ALONG the object's long
        # axis, dy ACROSS it (the jaw-gap / short-axis direction). Applied in the
        # waist frame, the same dx lands at a different point on the object as it
        # turns (a perpendicular object shifts along its length, a parallel one
        # across its width), which is why the offset had to be retuned per pose.
        _yaw = math.radians(float(long_yaw))
        # long_yaw is the axis heading in [-90,90], so the long-axis unit vector
        # points forward-ish (+X); this fixes the sign so +dx is consistent.
        _cos, _sin = math.cos(_yaw), math.sin(_yaw)
        long_hat = (_cos, _sin)          # along the object long axis (horizontal)
        short_hat = (-_sin, _cos)        # across it (perpendicular, horizontal)
        out[0] += dx * long_hat[0] + dy * short_hat[0]
        out[1] += dx * long_hat[1] + dy * short_hat[1]
        out[2] += dz
        _log(
            f"grasp offset (object frame): along-axis dx={dx:+.3f}m, "
            f"across-axis dy={dy:+.3f}m, dz={dz:+.3f}m applied along object "
            f"long-axis heading {float(long_yaw):+.1f}deg -> waist "
            f"dXY=[{dx * long_hat[0] + dy * short_hat[0]:+.3f},"
            f"{dx * long_hat[1] + dy * short_hat[1]:+.3f}]m (pose-invariant: the "
            "same offset grasps the same spot at any object heading)"
        )
    else:
        out[0] += dx
        out[1] += dy
        out[2] += dz
    return out


def _build_topdown_grasp_pose7(
    det: Dict[str, Any],
    args: argparse.Namespace,
    arm: str,
    extra_yaw_scale: float = 1.0,
    tilt_deg: Optional[float] = None,
) -> List[float]:
    """Top-down grasp pose for the elbow-high path.

    Forces the approach axis fully vertical (tool noses straight down,
    ``elbow_high_topdown_tilt_deg`` ~ 90deg) so the wrist stays centred instead
    of saturating the way a tilted side-grasp does in BOTH elbow branches. The
    yaw still tracks the object long axis, but with a wider clamp
    (``elbow_high_topdown_max_yaw_deg``) since rotating about the (now vertical)
    tool axis is cheap and aligns the jaws across an object lying parallel to the
    body. The approach standoff derived from this pose is directly ABOVE the
    object, so the compliant insert descends straight down onto it.
    """
    tilt = (
        float(tilt_deg)
        if tilt_deg is not None
        else float(getattr(args, "elbow_high_topdown_tilt_deg", 90.0))
    )
    return _build_grasp_pose7(
        det,
        args,
        arm,
        extra_yaw_scale=extra_yaw_scale,
        tilt_override_deg=tilt,
        max_yaw_override_deg=float(getattr(args, "elbow_high_topdown_max_yaw_deg", 90.0)),
    )


def _topdown_retry_tilt_deg(args: argparse.Namespace, guard: Optional[JointLimitGuard]) -> float:
    """Tilt (deg) for a top-down retry.

    A wrist-pitch saturation on the elbow-high top-down approach means the tool is
    too vertical to reach at this object distance: a less-vertical tilt moves
    wrist_pitch back off its (asymmetric, only -0.785rad) lower stop AND pulls the
    TCP target closer, so the retry uses the shallower
    ``elbow_high_topdown_retry_tilt_deg`` when the guard tripped; otherwise it keeps
    the full top-down tilt.
    """
    full = float(getattr(args, "elbow_high_topdown_tilt_deg", 90.0))
    if (
        bool(getattr(args, "joint_limit_reduce_tilt_on_retry", True))
        and guard is not None
        and guard.last_event is not None
    ):
        return float(getattr(args, "elbow_high_topdown_retry_tilt_deg", 55.0))
    return full


def _reduce_approach_tilt(quat_xyzw: List[float], reduce_deg: float) -> List[float]:
    """Rotate an orientation so its approach axis (tool Z) tilts ``reduce_deg``
    closer to LEVEL (less nose-down), keeping its heading (azimuth) the same.

    A nose-down grasp needs the wrist to flex further (wrist_pitch toward its
    lower stop) the deeper it descends; making the tool less vertical relieves
    that. We rotate about the horizontal axis perpendicular to the tool's
    heading and pick the sign that REDUCES the elevation magnitude.
    """
    if abs(float(reduce_deg)) < 1e-6:
        return list(quat_xyzw)
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    rot = R.from_quat(list(quat_xyzw))
    tz = rot.as_matrix()[:, 2]
    horiz = np.array([tz[0], tz[1], 0.0])
    if float(np.linalg.norm(horiz)) < 1e-6:
        return list(quat_xyzw)  # tool already vertical; no defined heading
    horiz /= np.linalg.norm(horiz)
    axis = np.cross(np.array([0.0, 0.0, 1.0]), horiz)
    if float(np.linalg.norm(axis)) < 1e-6:
        return list(quat_xyzw)
    axis /= np.linalg.norm(axis)

    def _elev(v) -> float:
        n = float(np.linalg.norm(v))
        return math.degrees(math.asin(max(-1.0, min(1.0, float(v[2]) / n))))

    el0 = abs(_elev(tz))
    best_quat = list(quat_xyzw)
    best_el = el0
    for sign in (1.0, -1.0):
        dR = R.from_rotvec(axis * math.radians(float(reduce_deg)) * sign)
        nz = (dR * rot).as_matrix()[:, 2]
        el = abs(_elev(nz))
        if el < best_el:
            best_el = el
            best_quat = (dR * rot).as_quat().tolist()
    return [float(v) for v in best_quat]


def _seed_tilt_up_deg(args: argparse.Namespace, guard: Optional[JointLimitGuard]) -> float:
    """Tilt-up (deg) for the seed-anchored elbow-high grasp.

    Uses ``elbow_high_seed_tilt_up_deg`` normally and the larger
    ``elbow_high_seed_retry_tilt_up_deg`` once the guard has tripped (wrist
    saturated) -- a shallower tool needs less wrist_pitch flexion to descend.
    """
    base = float(getattr(args, "elbow_high_seed_tilt_up_deg", 20.0))
    if (
        bool(getattr(args, "joint_limit_reduce_tilt_on_retry", True))
        and guard is not None
        and guard.last_event is not None
    ):
        return float(getattr(args, "elbow_high_seed_retry_tilt_up_deg", 35.0))
    return base


def _qp_ctrl_name(arm: str) -> str:
    return (
        "endpose_single_arm_qp_L_controller"
        if str(arm) == "left"
        else "endpose_single_arm_qp_R_controller"
    )


def _read_arm_joints(xarm: XARM_manager, arm: str) -> Optional[List[float]]:
    """Current 7 arm joint angles (controller order), or None."""
    try:
        joints = (
            xarm.xarm_left_arm_joint_angles()
            if str(arm) == "left"
            else xarm.xarm_right_arm_joint_angles()
        )
    except Exception:  # noqa: BLE001
        return None
    if not joints or len(joints) < 7 or any(j is None for j in joints[:7]):
        return None
    return [float(j) for j in joints[:7]]


def _set_qp_joint_window(
    xarm: XARM_manager,
    arm: str,
    seed_joints: List[float],
    margin_rad: float,
    lock_names: List[str],
    hard_limits: List[Tuple[float, float]],
    drop_margin_rad: Optional[float] = None,
) -> Optional[Tuple[List[float], List[float]]]:
    """Tighten the QP controller's joint limits into a window around the
    elbow-high seed for the named joints, so the QP solver keeps the arm in the
    elbow-high basin (the elbow can't sag) while it streams to the grasp.

    The window is ASYMMETRIC. ``margin_rad`` bounds the "flex" side (the joint
    may move away from the seed toward more flexion / further into the
    elbow-high posture). ``drop_margin_rad`` bounds the "drop" side -- the
    direction that LOSES the elbow-high posture (the elbow extending back down,
    the shoulder unwinding). It defaults to ``margin_rad`` (symmetric) for
    backward compatibility, but should be set much smaller so the elbow is
    actually held up. The drop direction per joint is inferred from the seed
    sign: a negative seed drops by increasing toward 0, a positive seed drops
    by decreasing toward 0.

    Leaves un-named joints (e.g. the wrists) at their true hard limits so the
    wrist can still rotate to track the grasp orientation. Returns the ORIGINAL
    (lower, upper) lists to restore afterwards, or None on failure.
    """
    orig_lo = [float(hard_limits[i][0]) for i in range(7)]
    orig_hi = [float(hard_limits[i][1]) for i in range(7)]
    lo = list(orig_lo)
    hi = list(orig_hi)
    if drop_margin_rad is None or float(drop_margin_rad) <= 0.0:
        drop_margin_rad = margin_rad
    flex_m = float(margin_rad)
    drop_m = float(drop_margin_rad)
    locked: List[str] = []
    for i, name in enumerate(ARM_JOINT_NAMES):
        if name in lock_names:
            s = float(seed_joints[i])
            if s < 0.0:
                # negative seed: drops by increasing toward 0 -> tight UPPER
                lo[i] = max(orig_lo[i], s - flex_m)
                hi[i] = min(orig_hi[i], s + drop_m)
            else:
                # positive seed: drops by decreasing toward 0 -> tight LOWER
                lo[i] = max(orig_lo[i], s - drop_m)
                hi[i] = min(orig_hi[i], s + flex_m)
            locked.append(f"{name}[{lo[i]:.2f},{hi[i]:.2f}]")
    ctrl = _qp_ctrl_name(arm)
    try:
        set_node_parameter(xarm, ctrl, "joint_lower_limits", lo)
        set_node_parameter(xarm, ctrl, "joint_upper_limits", hi)
    except Exception as e:  # noqa: BLE001
        _log(f"ELBOW-HIGH QP joint window: set failed ({e!r}); leaving limits as-is")
        return None
    _log(
        f"ELBOW-HIGH QP joint window set on {ctrl} (flex {flex_m:.2f}rad, drop "
        f"{drop_m:.2f}rad): " + ", ".join(locked)
        + ". Asymmetric: tight on the drop side so the elbow is actually held "
        "high; loose on the flex side so the arm can still reach."
    )
    return orig_lo, orig_hi


def _restore_qp_joint_limits(
    xarm: XARM_manager, arm: str, lower: List[float], upper: List[float]
) -> None:
    ctrl = _qp_ctrl_name(arm)
    try:
        set_node_parameter(xarm, ctrl, "joint_lower_limits", list(lower))
        set_node_parameter(xarm, ctrl, "joint_upper_limits", list(upper))
        _log(f"ELBOW-HIGH QP joint limits restored on {ctrl} (back to hard limits).")
    except Exception as e:  # noqa: BLE001
        _log(f"WARNING: failed to restore QP joint limits on {ctrl}: {e!r}")


def _derive_motion_poses(
    grasp_pose7: List[float], args: argparse.Namespace
) -> Tuple[List[float], List[float], List[float]]:
    """Derive (approach, lift, pre_home) pose7s from a grasp pose7.

    Factored out so the approach can be retried with a re-built grasp pose
    (e.g. the 180deg gripper-symmetry equivalent) without duplicating the
    waypoint geometry.
    """
    from scipy.spatial.transform import Rotation as R

    approach_pose7 = list(grasp_pose7)
    if bool(args.approach_along_axis):
        # Back off the approach waypoint ALONG the grasp approach axis (tool +Z).
        tool_z = R.from_quat([float(v) for v in grasp_pose7[3:7]]).as_matrix()[:, 2]
        approach_xyz = np.asarray(grasp_pose7[:3], dtype=float) + float(
            args.approach_dist
        ) * np.asarray(tool_z, dtype=float)
        approach_pose7[0] = float(approach_xyz[0])
        approach_pose7[1] = float(approach_xyz[1])
        approach_pose7[2] = float(approach_xyz[2])
        # Lateral clamp: tool_z almost always has a small outboard Y component
        # (an artifact of the arm's fixed grasp orientation, not a deliberate
        # approach direction). Backing out along tool_z therefore pushes the
        # approach standoff FURTHER from the body center (|Y|) than the grasp
        # itself. For far-outboard reaches that extra lateral push is what folds
        # the elbow into its stop (left arm: elbow_pitch -> -2.56 at approach
        # Y=+0.276 vs grasp Y=+0.254) -- and it's the APPROACH start that's
        # unreachable, not the grasp. The standoff only needs to be safely
        # above/behind, so clamp the approach |Y| to the grasp |Y|: the approach
        # backs out in X-Z at the same lateral position as the grasp. No-op when
        # the approach is already inboard of the grasp.
        if bool(getattr(args, "approach_clamp_lateral", True)):
            gy = float(grasp_pose7[1])
            ay = float(approach_pose7[1])
            if abs(ay) > abs(gy) + 1e-6:
                approach_pose7[1] = gy
                _log(
                    f"approach lateral clamp: approach Y {ay:+.4f} -> {gy:+.4f} "
                    f"(= grasp Y); tool_z outboard component would have pushed the "
                    "standoff further from center than the grasp, which folds the "
                    "elbow into its stop at far-outboard reaches."
                )
        _log(
            f"approach along grasp axis: dist={args.approach_dist:.3f}m "
            f"tool_z=[{tool_z[0]:.3f},{tool_z[1]:.3f},{tool_z[2]:.3f}] "
            f"approach_xyz={[f'{v:.4f}' for v in approach_pose7[:3]]}"
        )
    else:
        approach_pose7[0] += float(args.approach_dx)
        approach_pose7[1] += float(args.approach_dy)
        approach_pose7[2] += float(args.approach_dz)

    lift_pose7 = list(grasp_pose7)
    lift_pose7[2] += float(args.lift_z)
    # Optionally rotate the wrist about waist Y during the lift (delta from the
    # grasp orientation) so the following joint-space move to the pre-home
    # waypoint starts from a friendlier wrist orientation (no conflict).
    lift_tilt_deg = float(args.lift_tilt_y_deg)
    if abs(lift_tilt_deg) > 1e-6:
        lq = (
            R.from_euler("y", lift_tilt_deg, degrees=True)
            * R.from_quat(grasp_pose7[3:7])
        ).as_quat()
        lift_pose7[3:7] = [float(v) for v in lq]
        _log(
            f"applied lift tilt {lift_tilt_deg:+.1f}deg about waist Y -> "
            f"lift quat={[f'{v:.4f}' for v in lift_pose7[3:7]]}"
        )

    # High, toward-body waypoint used after the lift so the joint-space home
    # move stays high/tucked instead of sweeping the gripper through the table.
    pre_home_pose7 = list(lift_pose7)
    pre_home_pose7[0] -= float(args.retract_toward_body_m)  # waist -X = toward body
    pre_home_pose7[1] -= float(args.retract_toward_body_y_m)  # waist -Y to unload a joint
    pre_home_pose7[2] += float(args.retract_extra_z)
    return approach_pose7, lift_pose7, pre_home_pose7


def _wait_arm_near_pose7(
    xarm: XARM_manager,
    arm: str,
    waist_frame: str,
    target_pose7: List[float],
    *,
    pos_tol_m: float,
    timeout_sec: float,
) -> bool:
    """Wait until TCP is near target XYZ (QPIK may return before motion finishes)."""
    import numpy as np

    target = np.asarray(target_pose7[:3], dtype=float)
    deadline = time.monotonic() + max(0.1, float(timeout_sec))
    while time.monotonic() < deadline and rclpy.ok():
        cur = xarm.get_tcp_pose(arm=arm, base_frame=waist_frame, timeout=1.0)
        if cur is not None:
            pos = np.asarray(cur["translation"], dtype=float)
            dist = float(np.linalg.norm(pos - target))
            if dist <= float(pos_tol_m):
                _log(f"arm reached grasp neighborhood: dist={dist*100:.1f}cm")
                return True
        rclpy.spin_once(xarm, timeout_sec=0.05)
    _log(f"WARNING: arm not within {pos_tol_m*100:.1f}cm of grasp pose after {timeout_sec:.1f}s")
    return False


def _tcp_tracking_error(
    xarm: XARM_manager,
    arm: str,
    waist_frame: str,
    commanded_pose7: List[float],
    label: str,
) -> Optional[Dict[str, Any]]:
    """Read achieved TCP and report XYZ error vs the commanded target.

    Separates 'wrong target math' (detection/Bug-A) from 'controller did not
    track' (motion). A large |err| here means the arm did not reach the
    commanded pose; a small |err| here with a bad grasp means the commanded
    target itself was off.
    """
    cur = xarm.get_tcp_pose(arm=arm, base_frame=waist_frame, timeout=2.0)
    if cur is None:
        _log(f"{label} TCP tracking: cannot read current TCP pose")
        return None
    cmd = np.asarray(commanded_pose7[:3], dtype=float)
    ach = np.asarray(cur["translation"], dtype=float)
    err = ach - cmd
    err_mm = (err * 1000.0).tolist()
    norm_mm = float(np.linalg.norm(err)) * 1000.0

    # Orientation diagnostics: how far the gripper approach axis (tool +Z) tilts
    # above/below the horizontal plane. ~0 deg = parallel to the ground.
    out: Dict[str, Any] = {
        "commanded_xyz": [float(v) for v in cmd.tolist()],
        "achieved_xyz": [float(v) for v in ach.tolist()],
        "error_mm": [float(v) for v in err_mm],
        "error_norm_mm": norm_mm,
    }
    tilt_str = ""
    try:
        from scipy.spatial.transform import Rotation as R

        ach_quat = [float(v) for v in cur["rotation"]]
        cmd_quat = [float(v) for v in commanded_pose7[3:7]]
        ach_zaxis = R.from_quat(ach_quat).as_matrix()[:, 2]
        cmd_zaxis = R.from_quat(cmd_quat).as_matrix()[:, 2]
        ach_tilt = math.degrees(math.asin(max(-1.0, min(1.0, float(ach_zaxis[2])))))
        cmd_tilt = math.degrees(math.asin(max(-1.0, min(1.0, float(cmd_zaxis[2])))))
        out["achieved_quat"] = ach_quat
        out["approach_axis_tilt_deg"] = ach_tilt
        out["commanded_approach_axis_tilt_deg"] = cmd_tilt
        # Full orientation error commanded->achieved. This matters because the
        # fingertip is offset ~tcp_to_tip metres along the tool axis, so an
        # orientation miss throws the fingertip off by ~|offset|*sin(err) even
        # when the TCP *position* tracks perfectly (the cause of "reached the
        # pose but grasped air"). Report the implied fingertip miss too.
        ori_err = float(
            (R.from_quat(cmd_quat).inv() * R.from_quat(ach_quat)).magnitude()
        )
        ori_err_deg = math.degrees(ori_err)
        out["orientation_error_deg"] = ori_err_deg
        tilt_str = (
            f" approach-axis tilt: achieved={ach_tilt:+.1f}deg "
            f"commanded={cmd_tilt:+.1f}deg (0=level) ori-err={ori_err_deg:.1f}deg"
        )
    except Exception:  # noqa: BLE001
        pass

    _log(
        f"{label} TCP tracking: "
        f"commanded=[{cmd[0]:.4f},{cmd[1]:.4f},{cmd[2]:.4f}] "
        f"achieved=[{ach[0]:.4f},{ach[1]:.4f},{ach[2]:.4f}] "
        f"err=[{err_mm[0]:+.1f},{err_mm[1]:+.1f},{err_mm[2]:+.1f}]mm "
        f"|err|={norm_mm:.1f}mm" + tilt_str
    )
    return out


def _record_joint_limits(
    guard: Optional[JointLimitGuard],
    arm: str,
    label: str,
    result: Dict[str, Any],
) -> None:
    """Snapshot how close the arm joints are to their limits after a phase."""
    if guard is None:
        return
    ev = guard.report(arm)
    if ev is not None:
        result["motion"][f"{label}_joint_limits"] = {
            "closest": ev["closest"],
            "breached": ev["breached"],
            "margin_rad": ev["margin_rad"],
        }
        # Full per-joint snapshot so the elbow-high QP window can be verified:
        # shows every joint's value + distance-to-limit, not just the closest.
        per_joint = ev.get("per_joint", [])
        if per_joint:
            parts = [
                f"{pj['joint']}={pj['value']:+.3f}[{pj['lower']:+.2f},{pj['upper']:+.2f}]"
                f"({pj['dist_to_limit']:.3f})"
                for pj in per_joint
            ]
            print(
                f"[JOINT-LIMIT] {arm} {label} per-joint: " + " ".join(parts),
                flush=True,
            )


def _seed_arm_controller(xarm: XARM_manager, seconds: float = 0.2) -> None:
    t_end = time.monotonic() + float(seconds)
    while time.monotonic() < t_end and rclpy.ok():
        rclpy.spin_once(xarm, timeout_sec=0.02)


def _arm_to_frame(arm: str) -> str:
    return "left_tcp_link" if arm == "left" else "right_tcp_link"


def _pose7_to_pose_msg(pose7: List[float]) -> Pose:
    p = Pose()
    p.position.x = float(pose7[0])
    p.position.y = float(pose7[1])
    p.position.z = float(pose7[2])
    p.orientation.x = float(pose7[3])
    p.orientation.y = float(pose7[4])
    p.orientation.z = float(pose7[5])
    p.orientation.w = float(pose7[6])
    return p


def _read_tcp_quat(xarm: XARM_manager, arm: str, waist_frame: str, fallback_pose7: List[float]) -> List[float]:
    cur = xarm.get_tcp_pose(arm=arm, base_frame=waist_frame, timeout=2.0)
    if cur is None:
        _log(f"WARNING: cannot read current {arm} TCP pose; using target quaternion fallback")
        return [float(v) for v in fallback_pose7[3:]]
    return [float(v) for v in cur["rotation"]]


def _qpik_to_pose7(
    action: ActionCall,
    xarm: XARM_manager,
    arm: str,
    pose7: List[float],
    waist_frame: str,
    label: str,
    *,
    otg_p_step: float,
    otg_r_step: float,
    keep_current_orientation: bool,
) -> bool:
    pose = list(float(v) for v in pose7)
    ctrl_name = (
        "endpose_single_arm_qpik_L_controller"
        if arm == "left"
        else "endpose_single_arm_qpik_R_controller"
    )
    set_node_parameter(xarm, ctrl_name, "otg_p_step", float(otg_p_step))
    set_node_parameter(xarm, ctrl_name, "otg_r_step", float(otg_r_step))
    _seed_arm_controller(xarm, seconds=0.2)

    target = _pose7_to_pose_msg(pose)
    if keep_current_orientation:
        q = _read_tcp_quat(xarm, arm, waist_frame, pose)
        target.orientation.x = float(q[0])
        target.orientation.y = float(q[1])
        target.orientation.z = float(q[2])
        target.orientation.w = float(q[3])

    _log(
        f"QPIK {arm} -> {label}: pose7={[f'{v:.4f}' for v in pose]} "
        f"(keep_current_orientation={keep_current_orientation})"
    )
    t0 = time.monotonic()
    if arm == "left":
        res = action.endpose_single_arm_qpik_L_controller(
            target,
            from_frame=waist_frame,
            to_frame=_arm_to_frame(arm),
            offset=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
    else:
        res = action.endpose_single_arm_qpik_R_controller(
            target,
            from_frame=waist_frame,
            to_frame=_arm_to_frame(arm),
            offset=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
    ok = res is not None
    _log(f"QPIK {label} result: {res} elapsed={time.monotonic() - t0:.2f}s ok={ok}")
    return ok


def _qp_stream_to_pose7(
    topic_pub: TopicPublisher,
    xarm: XARM_manager,
    arm: str,
    pose7: List[float],
    waist_frame: str,
    label: str,
    *,
    otg_p_step: float,
    otg_r_step: float,
    stream_duration_sec: float,
    stream_rate_hz: float,
    keep_current_orientation: bool,
    closed_loop: bool = False,  # deprecated; kept for API compatibility
    step_m: float = 0.0,  # optional hard cap on per-cycle position advance (0=off)
    waypoint_tol_m: float = 0.004,  # deprecated; kept for API compatibility
    waypoint_timeout_s: float = 0.6,  # deprecated; kept for API compatibility
    speed_scale: float = 0.7,
    hold_sec: float = 0.4,
    lag_guard_m: float = 0.0,  # deprecated; kept for API compatibility
    stall_timeout_s: float = 2.0,
    lookahead_m: float = 0.08,
    pos_tol_m: float = 0.01,
    ori_lead_rad: float = 0.6,
    guard: Optional[JointLimitGuard] = None,
) -> bool:
    """Drive the single-arm QP controller along a straight Cartesian line using
    closed-loop pure pursuit.

    If ``guard`` is provided, each streaming cycle checks the live arm joints
    against their limits; if any joint comes within the guard margin of a hard
    limit, the stream HOLDS the current pose and aborts (returns False) instead
    of driving the joint into its hard stop (which faults the motor). The caller
    can inspect ``guard.last_event`` to see which joint/phase tripped.

    The QP controller (single_arm_qp_controller) has its OWN task-space OTG with
    a per-cycle cap (otg_p_step) and a tracking-error bound (dis_err_bound, 0.2 m
    by default). Streaming setpoints faster than the arm can move makes the
    commanded point race ahead, trips ``目标超出跟踪限`` and keeps resetting the
    OTG (the arm stutters and falls far behind). Publishing the far final target
    in one shot saturates the same error bound.

    Instead we keep the commanded setpoint a fixed ``lookahead_m`` AHEAD of the
    ACTUAL TCP along the line (a "carrot"), re-read each cycle:
      - the tracking error stays ~lookahead (< dis_err_bound) -> no warnings, the
        OTG builds velocity smoothly (no stutter),
      - the carrot stays ON the line -> straight Cartesian path,
      - progress is paced by the ARM, so it self-adapts to whatever speed the
        controller can actually achieve (works for a 1 cm insert or a 50 cm
        transit), and a stall (no progress) aborts instead of diverging.
    Orientation is slerped by the carrot's progress fraction so it arrives with
    the position.
    """
    from scipy.spatial.transform import Rotation as R, Slerp

    ctrl_name = (
        "endpose_single_arm_qp_L_controller"
        if arm == "left"
        else "endpose_single_arm_qp_R_controller"
    )
    xarm.xarm_activate_controller([ctrl_name])
    set_node_parameter(xarm, ctrl_name, "otg_p_step", float(otg_p_step))
    set_node_parameter(xarm, ctrl_name, "otg_r_step", float(otg_r_step))

    # Fresh trend history for this motion so the recovery-aware guard grants its
    # grace window to any joint already parked inside the margin at phase start
    # (e.g. after a previous abort) and lets this command pull it back out.
    if guard is not None:
        guard.reset_recovery_state(arm)

    cur = xarm.get_tcp_pose(arm=arm, base_frame=waist_frame, timeout=2.0)
    if cur is None:
        _log(f"QP-Stream {label}: cannot read current TCP pose")
        return False
    start_xyz = np.asarray([float(v) for v in cur["translation"]], dtype=float)
    start_quat = np.asarray([float(v) for v in cur["rotation"]], dtype=float)
    end_xyz = np.asarray([float(pose7[0]), float(pose7[1]), float(pose7[2])], dtype=float)
    end_quat = (
        start_quat.copy()
        if keep_current_orientation
        else np.asarray([float(v) for v in pose7[3:7]], dtype=float)
    )

    line_vec = end_xyz - start_xyz
    line_len = float(np.linalg.norm(line_vec))
    direction = line_vec / line_len if line_len > 1e-9 else np.zeros(3)

    # Shortest-path geodesic angle between the two orientations (rad).
    sq = start_quat / (np.linalg.norm(start_quat) + 1e-12)
    eq = end_quat / (np.linalg.norm(end_quat) + 1e-12)
    dot = max(-1.0, min(1.0, float(abs(np.dot(sq, eq)))))
    ori_angle = 2.0 * math.acos(dot)
    slerp = None
    if ori_angle > 1e-4:
        slerp = Slerp([0.0, 1.0], R.from_quat(np.vstack([start_quat, end_quat])))

    # Cap the read/publish rate so each cycle has time for a TF lookup. Keep the
    # carrot well inside the controller's dis_err_bound (0.2 m).
    rate = min(max(1.0, float(stream_rate_hz)), 50.0)
    dt = 1.0 / rate
    lookahead = max(0.01, min(float(lookahead_m), 0.15))

    pose = Pose()
    publish_fn = (
        topic_pub.publish_endposetarget_L
        if arm == "left"
        else topic_pub.publish_endposetarget_R
    )

    def _publish(xyz: np.ndarray, frac: float) -> None:
        pose.position.x = float(xyz[0])
        pose.position.y = float(xyz[1])
        pose.position.z = float(xyz[2])
        q = slerp([max(0.0, min(1.0, frac))])[0].as_quat() if slerp is not None else end_quat
        pose.orientation.x = float(q[0])
        pose.orientation.y = float(q[1])
        pose.orientation.z = float(q[2])
        pose.orientation.w = float(q[3])
        publish_fn(pose, from_frame=waist_frame)

    # The QP controller has a non-zero steady-state tracking error (~1-2 cm), so
    # an exact pos_tol is often unreachable. We therefore exit on the FIRST of:
    #   - reached:   within pos_tol_m (ideal), or
    #   - settled:   within reach_accept_m AND no further improvement for settle_s
    #                (the controller has converged as far as it will -> exit now
    #                instead of hovering until the deadline), or
    #   - deadline:  hard backstop.
    # Progress is tracked monotonically (best-so-far) so a transient stale/failed
    # TF read never makes the carrot retreat or trips a false stall.
    reach_accept = max(float(pos_tol_m), 0.025)
    settle_s = 0.6
    _log(
        f"QP-Stream {arm} -> {label} (pure-pursuit): pos={line_len*100:.1f}cm "
        f"rot={math.degrees(ori_angle):.1f}deg, lookahead={lookahead*100:.0f}cm "
        f"@ {rate:.0f}Hz, pos_tol={pos_tol_m*100:.1f}cm accept={reach_accept*100:.1f}cm "
        f"keep_ori={keep_current_orientation}"
    )

    t_start = time.monotonic()
    deadline = t_start + max(8.0, line_len / 0.03 + 6.0)
    cur_xyz = start_xyz.copy()
    s_best = 0.0
    end_dist = line_len
    best_end_dist = line_len
    last_improve_t = t_start
    got_read = False
    done = False
    settled = False
    # Measured fraction of the orientation geodesic already achieved (0..1). Used
    # to LEASH the orientation carrot: the published orientation is held within
    # ori_lead_rad of the ACTUAL orientation so a large reorientation does not
    # over-command the QP and blow its orientation tracking bound (`目标超出跟踪限`).
    ori_frac_meas = 0.0
    while rclpy.ok() and time.monotonic() < deadline:
        # Joint-limit watchdog: stop BEFORE a joint reaches its hard stop.
        if guard is not None and guard.enabled:
            ev = guard.check_live(arm)
            if ev is not None and ev.get("should_abort"):
                b = ev["breached"][0]
                eff_margin = guard.margin_overrides.get(b["joint"], guard.margin)
                _log(
                    f"QP-Stream {label}: ABORT joint-limit guard - '{b['joint']}'"
                    f"={b['value']} within {eff_margin:.3f}rad of limit "
                    f"[{b['lower']},{b['upper']}] (side={b['nearest_side']}); "
                    "holding pose to avoid the hard stop"
                )
                hold = xarm.get_tcp_pose(arm=arm, base_frame=waist_frame, timeout=0.2)
                if hold is not None:
                    hp = np.asarray([float(v) for v in hold["translation"]], dtype=float)
                    hq = np.asarray([float(v) for v in hold["rotation"]], dtype=float)
                else:
                    hp, hq = cur_xyz, end_quat
                for _ in range(max(1, int(round(rate * 0.2)))):
                    if not rclpy.ok():
                        break
                    pose.position.x = float(hp[0])
                    pose.position.y = float(hp[1])
                    pose.position.z = float(hp[2])
                    pose.orientation.x = float(hq[0])
                    pose.orientation.y = float(hq[1])
                    pose.orientation.z = float(hq[2])
                    pose.orientation.w = float(hq[3])
                    publish_fn(pose, from_frame=waist_frame)
                    rclpy.spin_once(xarm, timeout_sec=dt)
                guard.last_event = {**ev, "phase": label}
                return False
        ach = xarm.get_tcp_pose(arm=arm, base_frame=waist_frame, timeout=0.08)
        now = time.monotonic()
        if ach is not None:
            got_read = True
            cur_xyz = np.asarray([float(v) for v in ach["translation"]], dtype=float)
            s_meas = (
                float(np.dot(cur_xyz - start_xyz, direction)) if line_len > 1e-9 else line_len
            )
            s_best = max(s_best, max(0.0, min(line_len, s_meas)))  # monotonic
            end_dist = float(np.linalg.norm(cur_xyz - end_xyz))
            if end_dist < best_end_dist - 0.001:  # >1 mm improvement
                best_end_dist = end_dist
                last_improve_t = now
            if slerp is not None and ori_angle > 1e-6:
                cq = np.asarray([float(v) for v in ach["rotation"]], dtype=float)
                cq /= (np.linalg.norm(cq) + 1e-12)
                dq = max(-1.0, min(1.0, abs(float(np.dot(sq, cq)))))
                ori_frac_meas = min(1.0, (2.0 * math.acos(dq)) / ori_angle)
            if end_dist <= float(pos_tol_m):
                done = True
                break
            if (now - last_improve_t) > settle_s and end_dist <= reach_accept:
                settled = True
                break
        # Carrot: a fixed lookahead ahead of the best-measured progress, on the line.
        carrot_s = min(line_len, s_best + lookahead)
        frac_pos = (carrot_s / line_len) if line_len > 1e-6 else 1.0
        # Orientation carrot: leash it to a bounded lead over the ACTUAL achieved
        # orientation (not the position fraction). On a short path with a big
        # rotation, frac_pos jumps near 1.0 immediately and would command the full
        # reorientation in one step (the dis_ori tracking-limit failure). Leashing
        # keeps each step within ori_lead_rad so the wrist slews smoothly; for the
        # common small-rotation move frac_pos is the binding term (unchanged).
        if slerp is not None and ori_angle > 1e-6:
            frac = max(0.0, min(frac_pos, ori_frac_meas + ori_lead_rad / ori_angle))
        else:
            frac = frac_pos
        _publish(start_xyz + carrot_s * direction, frac)
        rclpy.spin_once(xarm, timeout_sec=dt)

    # Command the exact final pose so the controller's OTG converges position AND
    # orientation. We do NOT read TCP every hold cycle: a timeout=0.0 lookup right
    # after spin floods the logs with "TF查找超时 ... 超时0.0s" warnings. One read
    # with a real timeout at the end is enough to report the final distance.
    hold_cycles = max(1, int(round(rate * float(hold_sec))))
    # Seed the orientation ramp from a fresh read so any residual reorientation
    # finishes UNDER the same leash (smoothly) rather than jumping to frac=1.0 at
    # the end (which would re-trip the dis_ori tracking limit for a big rotation).
    hold_frac = 1.0
    if slerp is not None and ori_angle > 1e-6:
        _ah = xarm.get_tcp_pose(arm=arm, base_frame=waist_frame, timeout=0.2)
        if _ah is not None:
            _cq = np.asarray([float(v) for v in _ah["rotation"]], dtype=float)
            _cq /= (np.linalg.norm(_cq) + 1e-12)
            _dq = max(-1.0, min(1.0, abs(float(np.dot(sq, _cq)))))
            hold_frac = min(1.0, (2.0 * math.acos(_dq)) / ori_angle)
        _step = ori_lead_rad / ori_angle
        ramp_cycles = int(max(0.0, 1.0 - hold_frac) / max(1e-6, _step)) + 1
        hold_cycles = max(hold_cycles, ramp_cycles)
    cur_frac = hold_frac
    for _ in range(hold_cycles):
        if not rclpy.ok():
            break
        if slerp is not None and ori_angle > 1e-6:
            cur_frac = min(1.0, cur_frac + ori_lead_rad / ori_angle)
            _publish(end_xyz, cur_frac)
        else:
            _publish(end_xyz, 1.0)
        rclpy.spin_once(xarm, timeout_sec=dt)
    ach = xarm.get_tcp_pose(arm=arm, base_frame=waist_frame, timeout=0.2)
    if ach is not None:
        got_read = True
        end_dist = float(
            np.linalg.norm(np.asarray([float(v) for v in ach["translation"]]) - end_xyz)
        )

    reached = bool(got_read) and end_dist <= reach_accept
    if done:
        pass
    elif settled:
        _log(
            f"QP-Stream {label}: settled {end_dist*100:.1f}cm from target "
            f"(QP steady-state; exiting without deadline hover)"
        )
    else:
        _log(
            f"QP-Stream {label}: finished by deadline, {end_dist*100:.1f}cm from target"
            + ("" if got_read else " [no TCP reads -- TF stale]")
        )
    if not reached:
        _log(
            f"QP-Stream {label}: WARNING target not reached within {reach_accept*100:.1f}cm "
            f"(final {end_dist*100:.1f}cm)"
        )
    return reached


def _resolve_phase_backends(strategy: str) -> Dict[str, str]:
    if strategy in ("moveit", "moveit_direct"):
        return {"approach": "moveit", "grasp": "moveit", "lift": "moveit"}
    if strategy == "qpik":
        return {"approach": "qpik", "grasp": "qpik", "lift": "qpik"}
    if strategy in ("qp_stream", "qp_all"):
        # qp_all == whole pipeline on the streamed QP controller (no MoveIt at
        # all). Big transit moves use a faster otg step; the grasp insert uses
        # the fine step. See the per-phase wiring in main().
        return {"approach": "qp_stream", "grasp": "qp_stream", "lift": "qp_stream"}
    if strategy == "auto_stream":
        # MoveIt approach + streamed STRAIGHT-LINE Cartesian insertion/lift.
        # Single-goal QPIK lets the QP take a joint-space shortcut, so the TCP
        # bows off the line; streaming dense on-line setpoints keeps it straight.
        return {"approach": "moveit", "grasp": "qp_stream", "lift": "qp_stream"}
    if strategy == "auto_cartesian":
        # MoveIt approach + MoveIt compute_cartesian_path insertion/lift: a
        # geometrically straight TCP segment (Cartesian interpolation + IK per
        # waypoint). Implies --use-cartesian-grasp / --use-cartesian-lift.
        return {"approach": "moveit", "grasp": "moveit", "lift": "moveit"}
    # bottle_cup_pour_place-like hybrid: MoveIt approach + QPIK insertion/fine motion
    return {"approach": "moveit", "grasp": "qpik", "lift": "qpik"}


def _exec_pose_by_backend(
    backend: str,
    *,
    action: ActionCall,
    moveit: MoveitCall,
    topic_pub: TopicPublisher,
    xarm: XARM_manager,
    arm: str,
    pose7: List[float],
    waist_frame: str,
    vel_scale: float,
    acc_scale: float,
    label: str,
    use_cartesian_path: bool,
    qp_otg_p_step: float,
    qp_otg_r_step: float,
    qp_stream_duration: float,
    qp_stream_rate_hz: float,
    keep_current_orientation: bool,
    moveit_timeout_sec: float,
    stream_closed_loop: bool = False,
    stream_step_m: float = 0.005,
    stream_waypoint_tol_m: float = 0.004,
    stream_waypoint_timeout_s: float = 0.6,
    qp_speed_scale: float = 0.7,
    qp_hold_sec: float = 0.4,
    qp_lag_guard_m: float = 0.0,
    qp_stall_timeout_s: float = 2.0,
    qp_lookahead_m: float = 0.08,
    qp_pos_tol_m: float = 0.01,
    guard: Optional[JointLimitGuard] = None,
) -> bool:
    if backend == "moveit":
        return _moveit_to_pose7_with_fallback(
            moveit,
            xarm,
            arm,
            pose7,
            waist_frame,
            vel_scale,
            acc_scale,
            label,
            use_cartesian_path=use_cartesian_path,
            moveit_timeout_sec=moveit_timeout_sec,
        )
    if backend == "qpik":
        return _qpik_to_pose7(
            action,
            xarm,
            arm,
            pose7,
            waist_frame,
            label,
            otg_p_step=qp_otg_p_step,
            otg_r_step=qp_otg_r_step,
            keep_current_orientation=keep_current_orientation,
        )
    if backend == "qp_stream":
        return _qp_stream_to_pose7(
            topic_pub,
            xarm,
            arm,
            pose7,
            waist_frame,
            label,
            otg_p_step=qp_otg_p_step,
            otg_r_step=qp_otg_r_step,
            stream_duration_sec=qp_stream_duration,
            stream_rate_hz=qp_stream_rate_hz,
            keep_current_orientation=keep_current_orientation,
            closed_loop=stream_closed_loop,
            step_m=stream_step_m,
            waypoint_tol_m=stream_waypoint_tol_m,
            waypoint_timeout_s=stream_waypoint_timeout_s,
            speed_scale=qp_speed_scale,
            hold_sec=qp_hold_sec,
            lag_guard_m=qp_lag_guard_m,
            stall_timeout_s=qp_stall_timeout_s,
            lookahead_m=qp_lookahead_m,
            pos_tol_m=qp_pos_tol_m,
            guard=guard,
        )
    raise ValueError(f"Unknown backend: {backend}")


def _gripper_target_reached(
    pos_pct: float,
    target_pct: float,
    *,
    tolerance_pct: float = 5.0,
) -> bool:
    target = float(target_pct)
    pos = float(pos_pct)
    if abs(pos - target) <= float(tolerance_pct):
        return True
    # Closing: accept strong partial close when ZMQ feedback is laggy.
    if target >= 50.0 and pos >= max(0.0, target - 15.0):
        return True
    # Opening: accept near-open.
    if target <= 10.0 and pos <= 10.0:
        return True
    return False


def _gripper_open_if_needed(
    gripper: Any,
    target_open_pct: float,
    speed_pct: float,
    force_pct: float,
    *,
    wait_timeout_s: float,
    settle_s: float,
    already_open_tol_pct: float = 12.0,
) -> tuple[float, int, bool]:
    try:
        pos0 = float(gripper.get_current_position_percent())
    except Exception:
        pos0 = -1.0
    if pos0 >= 0 and pos0 <= float(already_open_tol_pct):
        _log(f"gripper already open enough: {pos0:.0f}% (skip open command)")
        return pos0, 3, True
    return _gripper_move_and_wait(
        gripper,
        target_open_pct,
        speed_pct,
        force_pct,
        label="open",
        wait_timeout_s=wait_timeout_s,
        settle_s=settle_s,
    )


def _gripper_move_and_wait(
    gripper: Any,
    target_pct: float,
    speed_pct: float,
    force_pct: float,
    *,
    label: str,
    wait_timeout_s: float,
    settle_s: float,
) -> tuple[float, int, bool]:
    """Move gripper with wait; on sparse ZMQ feedback, fall back to settle sleep."""
    try:
        pos0 = float(gripper.get_current_position_percent())
    except Exception:
        pos0 = -1.0
    _log(f"gripper {label}: start={pos0:.0f}% -> target={float(target_pct):.0f}%")

    p, st = gripper.move_and_wait_percent(
        float(target_pct),
        float(speed_pct),
        float(force_pct),
    )
    ok = _gripper_target_reached(float(p), float(target_pct)) or int(st) in {1, 2, 3}
    if ok:
        _log(f"gripper {label} result: pos={p}% status={st} ok=True")
        return float(p), int(st), True

    _log(
        f"gripper {label}: wait incomplete (pos={p}% status={st}); "
        f"retry with settle {settle_s:.1f}s"
    )
    try:
        gripper.move_percent(float(target_pct), float(speed_pct), float(force_pct))
    except Exception as e:  # noqa: BLE001
        _log(f"WARNING: gripper {label} move_percent failed: {e!r}")
        return float(p), int(st), False

    deadline = time.monotonic() + max(0.5, float(settle_s))
    last_p = float(p)
    while time.monotonic() < deadline:
        time.sleep(0.1)
        try:
            last_p = float(gripper.get_current_position_percent())
        except Exception:
            continue
        if _gripper_target_reached(last_p, float(target_pct)):
            _log(f"gripper {label} result (after settle): pos={last_p:.0f}% ok=True")
            return last_p, 3, True

    ok2 = _gripper_target_reached(last_p, float(target_pct))
    _log(f"gripper {label} result (after settle): pos={last_p:.0f}% ok={ok2}")
    return last_p, int(st), ok2


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--arm",
        choices=["left", "right", "auto"],
        default="auto",
        help="Which arm/gripper to grasp with. 'auto' (default) picks the arm "
        "from the detected object's Y position in the waist frame.",
    )
    p.add_argument(
        "--arm-select-boundary-y",
        type=float,
        default=0.0,
        help="Waist-frame Y (m) half-width for --arm auto when >0. Outside "
        "[-boundary,+boundary], +Y uses left and -Y uses right. Inside the band, "
        "the signed object long-axis yaw chooses the arm using "
        "--arm-select-long-axis-yaw-threshold-deg. Set 0 for the legacy "
        "single-boundary Y split.",
    )
    p.add_argument(
        "--arm-select-long-axis-yaw-threshold-deg",
        type=float,
        default=15.0,
        help="Signed long-axis yaw threshold used inside the center Y band. For "
        "0<=Y<=boundary, yaw<=threshold uses left else right. For "
        "-boundary<=Y<0, yaw<-threshold uses left else right.",
    )
    p.add_argument(
        "--arm-select-deadband-m",
        type=float,
        default=0.0,
        help="If |object_y - boundary| < this, use --arm-select-default instead "
        "of the sign, to avoid flip-flopping for near-center objects.",
    )
    p.add_argument(
        "--arm-select-default",
        choices=["left", "right"],
        default="right",
        help="Fallback arm when --arm auto cannot read object Y or signed long-axis yaw.",
    )
    p.add_argument(
        "--start-home",
        action="store_true",
        default=True,
        help="Move BOTH arms to their home posture before the detect->grasp "
        "cycle (on by default).",
    )
    p.add_argument(
        "--no-start-home",
        dest="start_home",
        action="store_false",
        help="Skip the pre-cycle move of both arms to home.",
    )
    p.add_argument(
        "--pre-cycle-move",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="After optional --start-home and before detection, move both arms to "
        "user-configured joint-space pre-cycle postures.",
    )
    p.add_argument(
        "--activate-grippers-on-start",
        action="store_true",
        default=True,
        help="Activate both grippers (open/close cycle) before the cycle starts "
        "(on by default; modbus/USB backend only).",
    )
    p.add_argument(
        "--no-activate-grippers-on-start",
        dest="activate_grippers_on_start",
        action="store_false",
        help="Skip activating the grippers at startup.",
    )
    p.add_argument("--prompt", action="append", default=[])
    p.add_argument(
        "--pipeline-version",
        choices=["current", "accelerated"],
        default="accelerated",
        help="Detection backend from grasp_pose_generation internals.",
    )
    p.add_argument("--detected-pose-json", default="", help="Optional existing detection json path.")
    p.add_argument("--base-url", default="http://10.20.0.24:939")
    p.add_argument(
        "--camera-yaml",
        default="poseestimator/camera_pose_config_dev29.yaml",
    )
    p.add_argument("--rgb-topic", default="/ob_camera_head/color/image_raw")
    p.add_argument("--depth-topic", default="/ob_camera_head/depth/image_raw")
    p.add_argument("--mode", choices=["segment", "yolo"], default="segment")
    p.add_argument("--waist-frame", default="waist_yaw_link")
    p.add_argument("--head-frame", default="head_roll_link")
    p.add_argument("--segment-confidence", type=float, default=0.3)
    p.add_argument("--tf-timeout", type=float, default=2.0)
    p.add_argument("--tf-retries", type=int, default=4)
    p.add_argument("--tf-warmup-sec", type=float, default=0.6)
    p.add_argument("--cam-timeout", type=float, default=5.0)
    p.add_argument(
        "--orientation-policy",
        choices=["detected", "current", "yaw_only", "grasp", "grasp_topdown", "grasp_side"],
        default="current",
        help="Detection-time orientation; motion uses --use-fixed-grasp-quat by default.",
    )
    p.add_argument("--grasp-yaw-offset-deg", type=float, default=0.0)
    p.add_argument("--max-grasp-yaw-delta-deg", type=float, default=30.0)
    p.add_argument(
        "--use-fixed-grasp-quat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use calibrated fixed TCP quat (bottle_cup_pour_place style).",
    )
    p.add_argument(
        "--grasp-quat",
        type=float,
        nargs=4,
        metavar=("QX", "QY", "QZ", "QW"),
        default=None,
        help="Override fixed grasp quaternion; default is arm-specific calibrated quat.",
    )
    p.add_argument(
        "--use-pour-place-offsets",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply pour_place centroid offsets (-0.25m X on right, etc.). "
        "Only for fixed-centroid targets like detect_pour_place; OFF for live detection.",
    )
    p.add_argument(
        "--grasp-x-offset",
        type=float,
        default=None,
        help="Extra grasp-target offset along the FIRST horizontal axis (m). With "
        "--grasp-offset-frame object (default) this is ALONG the object's long "
        "axis; with 'waist' it is waist_yaw_link +X. Overrides pour defaults.",
    )
    p.add_argument(
        "--grasp-y-offset",
        type=float,
        default=None,
        help="Extra grasp-target offset along the SECOND horizontal axis (m). With "
        "--grasp-offset-frame object (default) this is ACROSS the object (its short "
        "axis / jaw-gap direction); with 'waist' it is waist_yaw_link +Y.",
    )
    p.add_argument("--grasp-z-offset", type=float, default=None)
    p.add_argument(
        "--grasp-offset-frame",
        choices=["object", "waist"],
        default="waist",
        help="Frame the horizontal grasp offset (x/y) is applied in. 'waist' "
        "(default) applies the offset directly in waist_yaw_link X/Y. 'object' "
        "applies dx ALONG the detected object long axis and dy ACROSS it, so a "
        "FIXED offset grasps the same spot regardless of the object's heading -- "
        "but the offset direction then rotates with the object. Z is always "
        "vertical (waist).",
    )
    p.add_argument("--tcp-to-tip-x", type=float, default=0.0)
    p.add_argument("--tcp-to-tip-y", type=float, default=0.0)
    p.add_argument("--tcp-to-tip-z", type=float, default=-0.20)
    p.add_argument("--save-dir", default="")
    p.add_argument("--save-prefix", default="grasp_exec")
    p.add_argument("--json-out", default="")
    p.add_argument(
        "--handoff-out",
        default="/tmp/grasp_handoff.json",
        help="Write a small handoff file recording which arm holds the object so "
        "the placement phase (grasp_pose_place_execute) knows which gripper to use. "
        "Empty to disable. 'holding' is true only when the grasp succeeded, the "
        "gripper closed, and the object was NOT released at finish.",
    )

    # motion
    p.add_argument(
        "--motion-strategy",
        choices=[
            "moveit",
            "moveit_direct",
            "qpik",
            "qp_stream",
            "qp_all",
            "auto_hybrid",
            "auto_stream",
            "auto_cartesian",
        ],
        default="auto_hybrid",
        help="Motion backend strategy. auto_hybrid: MoveIt approach + QPIK insertion/lift "
        "(QPIK can bow off the straight line). auto_stream: MoveIt approach + streamed "
        "STRAIGHT-LINE Cartesian insertion/lift (recommended when the grasp insert must "
        "be a straight line). moveit_direct: no approach waypoint - one MoveIt move "
        "straight from the current pose to the final grasp point. qp_all: run the WHOLE "
        "pipeline (approach + grasp + lift) on the streamed QP controller - no MoveIt, so "
        "no RRTConnect 'big circle' and no flaky KDL goal-IK; big transit moves use "
        "--qp-transit-p-step, the grasp insert uses --qp-otg-p-step.",
    )
    p.add_argument("--vel-scale", type=float, default=0.8)
    p.add_argument("--acc-scale", type=float, default=0.8)
    p.add_argument(
        "--approach-dx",
        type=float,
        default=-0.10,
        help="Approach offset on grasp X in waist frame (default -0.10 m, pour_place pull-back).",
    )
    p.add_argument("--approach-dy", type=float, default=0.0)
    p.add_argument("--approach-dz", type=float, default=0.0)
    p.add_argument(
        "--approach-along-axis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Place the approach waypoint by backing off --approach-dist along the "
        "grasp approach axis (tool +Z), so a tilted grasp approaches from -X/+Z "
        "(human-like). Use --no-approach-along-axis to use waist-frame "
        "--approach-dx/dy/dz instead.",
    )
    p.add_argument(
        "--approach-dist",
        type=float,
        default=0.12,
        help="Distance (m) to back off the approach waypoint along the grasp approach "
        "axis when --approach-along-axis is set.",
    )
    p.add_argument(
        "--approach-clamp-lateral",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clamp the approach standoff's |Y| to the grasp |Y| so backing out along "
        "tool_z (which usually has a small outboard Y component from the arm's fixed "
        "grasp orientation) doesn't push the approach start FURTHER from the body "
        "center than the grasp. At far-outboard reaches that extra lateral push is "
        "what folds the elbow into its stop (left arm elbow_pitch -> -2.56 at approach "
        "Y=+0.276 vs grasp Y=+0.254) -- the approach start becomes unreachable even "
        "though the grasp is fine. The standoff only needs to be above/behind, so this "
        "backs out in X-Z at the grasp's lateral position. No-op for inboard approaches.",
    )
    p.add_argument(
        "--elbow-low-reorient",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reach-aware reorientation of the STANDARD grasp (not the elbow-high "
        "path). The fixed grasp orientation points the gripper sideways, so the wrist "
        "TCP lands further outboard than the object tip (TCP_Y = tip_Y + 0.266 * "
        "tool_z_Y). For a far-side object that puts the TCP past the elbow-low basin's "
        "reach and forces an elbow-high fallback -- even for a perpendicular object. "
        "When the would-be standard TCP |Y| >= --elbow-low-reorient-trigger-abs-y-m, "
        "rotate the grasp about waist X by up to --elbow-low-reorient-max-deg to flatten "
        "tool_z_Y so the wrist stays over the object, giving the standard basin a chance "
        "to reach it with a steeper (more top-down) approach instead of falling back. "
        "Experimental: the steeper orientation may misalign the jaws or hit the table; "
        "if it does, disable this and let the elbow-high fallback handle far-side "
        "objects. No-op for centered grasps (|TCP_Y| below the trigger).",
    )
    p.add_argument(
        "--elbow-low-reorient-trigger-abs-y-m",
        type=float,
        default=0.24,
        help="Standard grasp TCP |Y| (m, waist frame) at/above which the reach-aware "
        "reorientation fires. Matches the elbow-low basin's reach edge (~0.25, where "
        "elbow_pitch hits its -2.61 stop). Below this the standard grasp is used "
        "as-is.",
    )
    p.add_argument(
        "--elbow-low-reorient-max-deg",
        type=float,
        default=20.0,
        help="Cap (deg) on the waist-X reorientation. The angle needed to fully flatten "
        "tool_z_Y is atan2(tool_z_Y, tool_z_Z) (~18deg for the left arm's sideways "
        "grasp); 20 lets it fully flatten. Lower = less jaw misalignment but the TCP "
        "stays further outboard (may still fold elbow_pitch into its reach stop).",
    )
    p.add_argument(
        "--grasp-tilt-y-deg",
        type=float,
        default=15.0,
        help="Tilt the grasp orientation by this many degrees about the waist-frame Y "
        "axis (human-like nose-down grasp; 0 = parallel to ground).",
    )
    p.add_argument(
        "--adaptive-grasp-orientation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Adapt the grasp yaw to the object's long-axis heading. For elongated "
        "objects (e.g. a banana) lying 'parallel to the body' (long axis along the "
        "waist Y / left-right direction), apply an extra yaw about waist Z so the "
        "fingers close across the narrow dimension. 'Vertical to body' objects keep "
        "the fixed orientation. Only affects objects classified parallel.",
    )
    p.add_argument(
        "--parallel-extra-yaw-deg",
        type=float,
        default=15.0,
        help="Extra yaw (deg about waist Z) applied for 'parallel to body' objects. "
        "Positive for the right arm, negated for the left arm.",
    )
    p.add_argument(
        "--parallel-detect-threshold-deg",
        type=float,
        default=45.0,
        help="Object is classified 'parallel to body' when its long-axis angle from "
        "the waist X (forward) axis is >= this many degrees (0=forward/vertical, "
        "90=left-right/parallel).",
    )
    p.add_argument(
        "--continuous-grasp-orientation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Continuously rotate the fixed grasp quat by the object's measured "
        "long-axis yaw (object_long_axis_yaw_waist_deg) so the grasp tracks the "
        "object's heading instead of snapping to the binary parallel/vertical "
        "buckets. When set, this takes precedence over --adaptive-grasp-orientation.",
    )
    p.add_argument(
        "--continuous-grasp-yaw-gain",
        type=float,
        default=1.0,
        help="Scale factor applied to the object's long-axis yaw when "
        "--continuous-grasp-orientation is set (1.0 = full alignment).",
    )
    p.add_argument(
        "--continuous-grasp-max-yaw-deg",
        type=float,
        default=15.0,
        help="Clamp (deg) on the continuous yaw applied about waist Z when "
        "--continuous-grasp-orientation is set, to avoid unreachable wrist poses. "
        "Kept conservative (15) because larger long-axis alignment contorts the "
        "wrist near the reach limit and stalls the insert a few cm short; raise "
        "it only if your target poses stay well inside the arm's envelope. The "
        "diagonal schedule (--diagonal-schedule) raises this clamp automatically "
        "for diagonal objects while scaling the tilt down to keep the wrist in range.",
    )
    p.add_argument(
        "--diagonal-schedule",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scale the grasp tilt DOWN and the jaw-yaw clamp UP continuously with "
        "the detected object long-axis angle (between --diagonal-schedule-angle-start-deg "
        "and --diagonal-schedule-angle-end-deg). Fixes diagonal objects (~20-60deg): "
        "the base 45deg tilt + +/-15deg yaw can't align the jaws AND drives wrist_pitch "
        "past its +1.05 stop at far reaches. The schedule frees wrist_pitch (less tilt) "
        "and aligns the jaws (more yaw) as the object rotates from perpendicular toward "
        "parallel. Off below the start angle (perpendicular behaviour unchanged); the "
        "elbow-high path takes over at/above the end angle.",
    )
    p.add_argument(
        "--diagonal-schedule-angle-start-deg",
        type=float,
        default=15.0,
        help="Object long-axis angle from waist +X (deg) above which the diagonal "
        "schedule begins to take effect. Below this the standard 45deg tilt + "
        "+/-15deg yaw is used as-is (the proven perpendicular behaviour).",
    )
    p.add_argument(
        "--diagonal-schedule-angle-end-deg",
        type=float,
        default=60.0,
        help="Object long-axis angle from waist +X (deg) at which the schedule "
        "reaches its full tilt_scale/yaw_clamp. Defaults to 60 to match the "
        "elbow-high proactive threshold, so the schedule hands off smoothly to the "
        "elbow-high top-down path for parallel objects.",
    )
    p.add_argument(
        "--diagonal-schedule-tilt-scale-end",
        type=float,
        default=0.5,
        help="Tilt scale at the schedule end angle (1.0 at the start). 0.5 means the "
        "45deg tilt becomes 22.5deg for a near-parallel object, freeing wrist_pitch. "
        "Lower = less nose-down (easier on the wrist, shallower approach).",
    )
    p.add_argument(
        "--diagonal-schedule-yaw-clamp-end-deg",
        type=float,
        default=45.0,
        help="Jaw-yaw clamp (deg) at the schedule end angle (the base "
        "--continuous-grasp-max-yaw-deg at the start). 45 lets the jaws track a "
        "45deg object's long axis instead of capping at 15deg. Higher = better jaw "
        "alignment but more wrist demand; balanced against the reduced tilt.",
    )
    p.add_argument(
        "--reach-tilt-reduce",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="On the STANDARD (elbow-low) path, back the side-tilt off for a "
        "far-outboard object (large |Y|) so the elbow can extend to reach it "
        "instead of folding elbow_pitch into its -2.61 stop. This is the "
        "elbow-low way to reach far perpendicular objects; the elbow-high path "
        "is reserved for the parallel-axis bucket.",
    )
    p.add_argument(
        "--reach-tilt-start-abs-y-m",
        type=float,
        default=0.18,
        help="Object tip |Y| (m, waist frame) below which the full side-tilt is "
        "kept. Above this the tilt scales down toward --reach-tilt-scale-end.",
    )
    p.add_argument(
        "--reach-tilt-end-abs-y-m",
        type=float,
        default=0.28,
        help="Object tip |Y| (m) at which the reach tilt reduction reaches its "
        "full --reach-tilt-scale-end.",
    )
    p.add_argument(
        "--reach-tilt-scale-end",
        type=float,
        default=0.4,
        help="Side-tilt scale at --reach-tilt-end-abs-y-m (1.0 at the start). 0.4 "
        "turns the 45deg tilt into ~18deg for the farthest-outboard object.",
    )
    p.add_argument(
        "--grasp-reach-retry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If the approach fails to reach the target (e.g. a wrist joint hits "
        "its limit mid-path because full long-axis alignment is unreachable), "
        "retry once with a REDUCED adaptive/continuous yaw (see "
        "--grasp-retry-yaw-scale). This is a small, trackable orientation change "
        "that relieves the limiting joint. Useful with "
        "--continuous-grasp-orientation / --adaptive-grasp-orientation.",
    )
    p.add_argument(
        "--grasp-retry-yaw-scale",
        type=float,
        default=0.5,
        help="Scale (0..1) applied to the adaptive/continuous yaw on the reach "
        "retry. 0.5 = half the long-axis alignment; 0.0 = fall back to the fixed "
        "grasp orientation. Only used when --grasp-reach-retry is set.",
    )
    # --- Elbow-HIGH "top reconfiguration" (handles awkward object poses) ---
    p.add_argument(
        "--elbow-high-enable-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If the elbow-LOW approach + reduced-tilt retry still cannot hold the "
        "grasp orientation (wrist saturates because the redundant arm is in the "
        "elbow-low IK basin), reconfigure the arm into an elbow-HIGH seed posture "
        "(jointspace, via a transition waypoint) and retry the SAME grasp from "
        "there. QP then stays in the elbow-high basin where the wrist has room. "
        "Falls through to the best-effort grasp only if elbow-high also fails.",
    )
    p.add_argument(
        "--elbow-high-arms",
        type=str,
        default="left",
        help="Comma-separated arms allowed to use the elbow-high path (proactive "
        "AND fallback), e.g. 'left', 'left,right', 'right'. An arm NOT in this set "
        "uses the standard elbow-low path even for parallel-to-body objects. "
        "Default 'left' because only the left elbow-high seed was hand-taught; the "
        "right seed is a sagittal mirror and can drive the right wrist_roll past "
        "its tighter -1.3 stop at far-right reaches. Set 'left,right' once a right "
        "seed is taught/validated for far-right objects.",
    )
    p.add_argument(
        "--elbow-high-proactive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Choose the elbow-HIGH path UP FRONT (before the first approach) when the "
        "detected object long axis is ~parallel to the body (its angle from waist +X "
        "is >= --elbow-high-proactive-angle-min-deg). This is the PARALLEL half of the "
        "two-strategy axis split: elbow-high grasps a parallel object's fore-aft short "
        "axis (its seed jaws already point that way), while perpendicular/diagonal "
        "objects stay on the elbow-low path. Off by default (fallback-only).",
    )
    p.add_argument(
        "--elbow-high-proactive-angle-min-deg",
        type=float,
        default=60.0,
        help="Object long-axis angle from waist +X (deg) at/above which the object "
        "is treated as 'parallel to the body' and routed to the elbow-high path. "
        "This is the split point between the two strategies (below -> elbow-low, "
        "at/above -> elbow-high). 90deg = fully parallel. Defaults to 60 to MATCH "
        "--diagonal-schedule-angle-end-deg: diagonal objects (15-60deg) stay on the "
        "elbow-low path where the diagonal schedule handles them; only strongly "
        "parallel objects (>=60deg) go elbow-high, where the jaw-align yaw is small "
        "so wrist_roll stays in range (a diagonal object needs a large align yaw "
        "that saturates wrist_roll's cramped +1.3 stop). Used by both the proactive "
        "and the fallback elbow-high triggers.",
    )
    p.add_argument(
        "--elbow-high-deadband-deg",
        type=float,
        default=8.0,
        help="Deadband (deg) around a long-axis yaw of 0 (perpendicular) in the "
        "SIGN-aware strategy split. Each arm's elbow-LOW comfort wedge is on its "
        "own sign of the signed long-axis yaw (RIGHT: 0..+T, LEFT: -T..0), but "
        "within +/- this deadband of 0 BOTH arms stay on elbow-low regardless of "
        "sign -- a near-perpendicular object sits at yaw~=0 where the sign is "
        "physically meaningless and can flip run-to-run from detection noise. "
        "Set 0 for a hard split exactly at yaw=0.",
    )
    p.add_argument(
        "--elbow-high-clamp-margin-rad",
        type=float,
        default=0.15,
        help="When seeding the elbow-high posture, clamp every joint to at least "
        "this margin (rad) inside its limit. Protects against a hand-taught seed "
        "that sits at/over a joint limit (e.g. shoulder_yaw) faulting the motor.",
    )
    p.add_argument(
        "--elbow-high-ready-left-joints",
        type=float,
        nargs=7,
        default=list(_ELBOW_HIGH_READY_LEFT_JOINTS),
        help="Left-arm elbow-high seed posture (rad, 7 joints).",
    )
    p.add_argument(
        "--elbow-high-ready-right-joints",
        type=float,
        nargs=7,
        default=list(_ELBOW_HIGH_READY_RIGHT_JOINTS),
        help="Right-arm elbow-high seed posture (rad, 7 joints).",
    )
    p.add_argument(
        "--elbow-high-stage-left-joints",
        type=float,
        nargs=7,
        default=list(_ELBOW_HIGH_STAGE_LEFT_JOINTS),
        help="Left-arm STAGING posture (rad, 7 joints) visited from home BEFORE "
        "the elbow-high reconfigure (and passed back through on return). The "
        "transition waypoint is computed from this stage, not from home. The "
        "elbow-LOW path does not use it.",
    )
    p.add_argument(
        "--elbow-high-stage-right-joints",
        type=float,
        nargs=7,
        default=list(_ELBOW_HIGH_STAGE_RIGHT_JOINTS),
        help="Right-arm elbow-high staging posture (rad, 7 joints).",
    )
    p.add_argument(
        "--elbow-high-transition-left-joints",
        type=float,
        nargs="*",
        default=[],
        help="Left-arm intermediate jointspace waypoint (rad, 7 joints) bridging "
        "stage <-> elbow-high to shrink the Cartesian sweep. Empty = auto "
        "(elementwise midpoint of stage and ready, clamped).",
    )
    p.add_argument(
        "--elbow-high-transition-right-joints",
        type=float,
        nargs="*",
        default=[],
        help="Right-arm intermediate jointspace waypoint (rad, 7 joints). "
        "Empty = auto (midpoint of home and ready, clamped).",
    )
    p.add_argument(
        "--elbow-high-topdown",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="On the elbow-high path, grasp TOP-DOWN (gripper noses straight down) "
        "instead of the side/tilt orientation. The side-tilt needs a wrist_pitch "
        "outside limits in BOTH elbow branches; top-down keeps the wrist centred "
        "and makes the approach a small reorientation. Yaw still tracks the object "
        "long axis.",
    )
    p.add_argument(
        "--elbow-high-topdown-tilt-deg",
        type=float,
        default=90.0,
        help="Absolute approach-axis tilt (deg) for the top-down elbow-high grasp. "
        "90 = pure vertical (tool straight down). Lower it slightly for a near-"
        "vertical grasp with some body clearance.",
    )
    p.add_argument(
        "--elbow-high-topdown-max-yaw-deg",
        type=float,
        default=90.0,
        help="Yaw clamp (deg) about the (vertical) tool axis for the top-down "
        "elbow-high grasp, so the jaws can fully align across an object lying "
        "parallel to the body. Wider than the side-grasp clamp because yaw about "
        "a vertical tool is cheap.",
    )
    p.add_argument(
        "--elbow-high-topdown-retry-tilt-deg",
        type=float,
        default=55.0,
        help="Shallower top-down tilt (deg) used on the reach retry when a pure-"
        "vertical top-down saturates wrist_pitch at its lower stop. Less vertical "
        "frees the wrist and pulls the TCP target closer (more reachable).",
    )
    p.add_argument(
        "--elbow-high-orientation",
        choices=["seed", "topdown", "sidetilt"],
        default="seed",
        help="Grasp orientation strategy on the elbow-high path. 'seed' (default) "
        "anchors the grasp on the orientation the arm is ALREADY in at the seed, so "
        "the approach only has to translate and the wrist never leaves its range "
        "(most likely to succeed). 'topdown' forces the gripper straight down "
        "(needs wrist headroom). 'sidetilt' reuses the normal side/tilt grasp.",
    )
    p.add_argument(
        "--elbow-high-seed-yaw-max-deg",
        type=float,
        default=90.0,
        help="Max jaw-align yaw (deg, about waist Z) layered on top of the seed "
        "orientation in --elbow-high-orientation seed. With --elbow-high-align-jaws "
        "this is the window in which the jaw-symmetry search may place the yaw; "
        "90deg lets the jaws align to ANY object orientation (incl. parallel) since "
        "in the top-down posture the yaw maps onto wrist_roll, which has room.",
    )
    p.add_argument(
        "--elbow-high-align-jaws",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="On the elbow-high (top-down) path, align the jaws to the object's "
        "long axis by rotating about waist Z. Because the tool points ~down, this "
        "maps onto wrist_roll (which has range), and the parallel-jaw 180deg "
        "symmetry is used to pick the roll-feasible equivalent yaw -- so a parallel "
        "object can be grasped on its short axis. Disable to keep the old small "
        "fixed +/-yaw clamp.",
    )
    p.add_argument(
        "--elbow-high-align-wrist-margin-rad",
        type=float,
        default=0.08,
        help="Safety margin (rad) kept from the wrist_roll limits when the "
        "jaw-symmetry search picks the elbow-high align yaw.",
    )
    p.add_argument(
        "--elbow-high-jaw-flip-retry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If the elbow-high short-axis alignment saturates wrist_roll on the "
        "arm's CRAMPED stop (left upper +1.3 / right lower -1.3), retry the SAME "
        "grip from the 180deg-flipped jaw line, which rolls the wrist onto its "
        "ROOMY side (left down to -1.65 / right up to +1.65). Keeps full "
        "alignment (unlike the reduced-yaw retry). Fires before the reduced-yaw "
        "retry when the abort was a wrist_roll saturation. Default OFF: the flip "
        "needs a ~180deg reorientation the pure-pursuit streamer cannot turn "
        "through (it stalls) for diagonal objects; the yaw-backoff ladder is "
        "preferred instead.",
    )
    p.add_argument(
        "--elbow-high-roll-align-waist-x",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If the elbow-high short-axis alignment saturates wrist_roll on the "
        "arm's CRAMPED stop, re-aim the jaws to a FIXED waist heading "
        "(--elbow-high-jaw-axis-heading-deg, default waist +X = 0deg) instead of "
        "the object's diagonal short axis. Off the fore-aft seed this is only a "
        "small roll that stays in range. For a PARALLEL object the short axis is "
        "already waist X (no change); for a DIAGONAL object it trades exact "
        "short-axis alignment (up to ~45deg off) for reachability. Preferred "
        "wrist_roll recovery (tried before the yaw-backoff ladder / jaw-flip).",
    )
    p.add_argument(
        "--elbow-high-jaw-axis-heading-deg",
        type=float,
        default=0.0,
        help="Fixed jaw-line heading (deg in the waist XY plane) used by "
        "--elbow-high-roll-align-waist-x. 0 = waist +X (fore-aft, matches the "
        "elbow-high seed). The jaw line is 180deg-symmetric, so 0 and 180 are "
        "equivalent.",
    )
    p.add_argument(
        "--elbow-high-diagonal-use-waist-x",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="PROACTIVELY align the elbow-high jaws to the fixed waist heading "
        "(--elbow-high-jaw-axis-heading-deg) for DIAGONAL objects -- those whose "
        "|long-axis yaw| is below --elbow-high-proactive-angle-min-deg (i.e. NOT "
        "near-parallel). Such objects are routed to elbow-high by the sign rule, "
        "and their exact short axis needs a large yaw that pins wrist_roll; going "
        "straight to the waist heading skips that doomed short-axis attempt (and "
        "the big recovery reorientation). NEAR-PARALLEL objects keep exact "
        "short-axis alignment (which is ~= waist X anyway).",
    )
    p.add_argument(
        "--elbow-high-yaw-backoff",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If the elbow-high short-axis alignment saturates wrist_roll on the "
        "arm's CRAMPED stop, progressively BACK OFF the jaw-align yaw (scale the "
        "waist-Z alignment down through --elbow-high-yaw-backoff-scales) and "
        "re-approach until wrist_roll clears its stop. Each step is a small "
        "rotation (unlike the 180deg jaw-flip) so the streamer can track it; the "
        "cost is a few degrees of short-axis misalignment. Preferred over "
        "--elbow-high-jaw-flip-retry for diagonal objects.",
    )
    p.add_argument(
        "--elbow-high-yaw-backoff-scales",
        type=float,
        nargs="+",
        default=[0.7, 0.45, 0.25],
        help="Descending jaw-align yaw scales tried by the elbow-high wrist_roll "
        "backoff ladder (1.0 = full short-axis alignment, 0 = none). The first "
        "scale that keeps wrist_roll off its stop wins.",
    )
    p.add_argument(
        "--elbow-high-always",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Route EVERY grasp through the top-down elbow-high posture (for arms "
        "in --elbow-high-arms), regardless of object angle/reach. Useful for "
        "testing the elbow-high path as the single strategy. Costs extra motion "
        "time and leans on wrist_pitch margin for near objects.",
    )
    p.add_argument(
        "--elbow-high-qp-lock",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="On the elbow-high path, tighten the QP controller's joint limits "
        "into a window around the elbow-high seed (for the proximal joints) so the "
        "QP solver CANNOT let the elbow sag back down into the wrist's hard stop "
        "while it streams to the grasp. The window is restored to the true hard "
        "limits after the grasp. The joint-limit safety guard keeps watching the "
        "real hard limits regardless.",
    )
    p.add_argument(
        "--elbow-high-qp-lock-margin-rad",
        type=float,
        default=0.4,
        help="Half-width (rad) of the per-joint window around the elbow-high seed "
        "used by --elbow-high-qp-lock, on the FLEX side (the joint may move further "
        "into the elbow-high posture). Smaller = stays closer to the seed; larger = "
        "more reach but more drift. The DROP side is bounded separately by "
        "--elbow-high-qp-lock-drop-margin-rad so the elbow can't sag back down.",
    )
    p.add_argument(
        "--elbow-high-qp-lock-drop-margin-rad",
        type=float,
        default=0.15,
        help="Half-width (rad) of the per-joint window on the DROP side -- the "
        "direction that LOSES the elbow-high posture (elbow extending back down). "
        "Keep this SMALL so the elbow is actually held high during the grasp; the "
        "drop direction per joint is inferred from the seed sign (negative seed "
        "drops by increasing toward 0). 0 = symmetric (use the flex margin).",
    )
    p.add_argument(
        "--elbow-high-qp-lock-joints",
        type=str,
        default="shoulder_pitch,shoulder_roll,shoulder_yaw,elbow_pitch,elbow_yaw",
        help="Comma-separated joints to pin into the seed window for "
        "--elbow-high-qp-lock. Default pins the shoulder+elbow (keeps the elbow "
        "high) and leaves the wrists free to track the grasp orientation. Add "
        "'wrist_pitch' to also keep the wrist off its lower stop.",
    )
    p.add_argument(
        "--elbow-high-seed-tilt-up-deg",
        type=float,
        default=12.0,
        help="Tilt the seed-anchored elbow-high grasp UP by this many degrees "
        "(less nose-down) so the descent onto the object needs less wrist_pitch "
        "flexion and doesn't hit its lower -0.785 stop. The hand-taught seed points "
        "~64deg nose-down; at a forward object that drives wrist_pitch onto its stop "
        "mid-descent (the grasp aborts ~2cm short). 12deg backs the tool off to "
        "~52deg, which also pulls the TCP back (less forward reach). 0 = use the "
        "seed orientation as-is; lower it for a more top-down grasp.",
    )
    p.add_argument(
        "--elbow-high-seed-retry-tilt-up-deg",
        type=float,
        default=25.0,
        help="Larger seed tilt-up (deg) used on the reach retry once wrist_pitch "
        "has saturated the guard. A shallower tool frees the wrist and shortens "
        "the reach. Bigger than the first-build value so the retry meaningfully "
        "changes the wrist demand (the yaw-only retry does not touch wrist_pitch).",
    )
    p.add_argument(
        "--elbow-high-guard-margin-rad",
        type=float,
        default=0.01,
        help="Tighter joint-limit guard margin (rad) applied ONLY to the wrist "
        "joints ONLY on the elbow-high path. The nose-down grasp at a far object "
        "needs wrist_pitch close to its -0.785 operational stop; the default 0.1rad "
        "margin aborts at -0.685 before the arm can reach it. This lets the wrist "
        "use the headroom it physically has while every other joint keeps the full "
        "margin. The QP controller's own -0.785 lower limit is the hard backstop, so "
        "a small value (e.g. 0.01 -> wrist may travel to -0.775) is safe. Set 0 to "
        "disable (keep the global margin). If the grasp still aborts at ~-0.775, the "
        "descent physically needs the full stop and this configuration can't "
        "complete it -- re-teach the seed or use the other arm.",
    )
    p.add_argument(
        "--capture-elbow-high-seed",
        action="store_true",
        default=False,
        help="Record the arm's CURRENT joints (hand-drag it into an elbow-high "
        "posture with the gripper pointing straight down above the object first) as "
        "elbow_high_ready_{arm}_joints in config.yaml, then exit. Requires --arm "
        "left|right. No motion is commanded. Warns if any captured joint sits near "
        "a limit (a usable seed needs headroom on every joint).",
    )
    # --- Joint-limit protection (Phase 1/2/3) ---
    p.add_argument(
        "--joint-limit-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Real-time watchdog: during QP streaming, stop and hold BEFORE any "
        "arm joint reaches its hard limit (which would fault/'kill' the motor). "
        "The aborted phase then triggers the reduced-alignment retry.",
    )
    p.add_argument(
        "--joint-limit-margin-rad",
        type=float,
        default=0.10,
        help="Safety margin (rad) from each joint limit at which the watchdog "
        "stops motion. Larger = safer but more conservative.",
    )
    p.add_argument(
        "--joint-limit-reduce-tilt-on-retry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When a phase is aborted by the joint-limit guard, also scale down "
        "the nose-down grasp/lift tilt (not just the yaw) on the retry, to pull "
        "wrist_pitch back into range.",
    )
    p.add_argument(
        "--grasp-orient-miss-tol-m",
        type=float,
        default=0.055,
        help="Max allowed fingertip miss (m) caused by orientation tracking error "
        "at the approach. The fingertip sits ~|tcp_to_tip| along the tool axis, so "
        "if the wrist cannot hold the commanded tilt the fingers miss the object "
        "even though the TCP position converged. Exceeding this triggers the "
        "reduced-tilt retry instead of closing on air. 0 disables the check. "
        "NOTE: with the COMPLIANT insert (which yields to contact and tolerates a "
        "small orientation miss) this is intentionally looser than a rigid grasp "
        "would need, so a wrist that can only hold the tilt to within ~9deg "
        "(the right arm near its elbow limit) still proceeds into the insert "
        "instead of bailing to home.",
    )
    p.add_argument(
        "--joint-limit-allow-recovery",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recovery-aware guard: a joint inside the margin only aborts if it "
        "is moving TOWARD its limit (or stuck); if the (retry) command is pulling "
        "it back out, the motion continues. This lets the reduced-alignment retry "
        "actually escape a joint parked at the limit instead of re-aborting on its "
        "first check (the cause of 'reached the standoff then returned, no grasp').",
    )
    p.add_argument(
        "--joint-limit-recovery-grace-iters",
        type=int,
        default=12,
        help="How many streaming cycles a joint already parked inside the margin "
        "at phase start is allowed before the trend must show it moving away from "
        "the limit. Only used with --joint-limit-allow-recovery.",
    )
    p.add_argument(
        "--lift-z",
        type=float,
        default=0.12,
        help="Straight-up lift after grasp (m). Higher clears the object/table "
        "before the toward-body retract and the joint-space home move.",
    )
    p.add_argument(
        "--lift-tilt-y-deg",
        type=float,
        default=-15.0,
        help="While lifting, also rotate the wrist by this many degrees about the "
        "waist-frame Y axis (delta from the grasp orientation). Default -15 backs the "
        "nose-down grasp tilt off so the joint-space move to the pre-home waypoint has "
        "no orientation conflict. 0 = keep the grasp orientation during lift.",
    )
    p.add_argument(
        "--moveit-timeout-sec",
        type=float,
        default=120.0,
        help="Max wait for each MoveIt plan_and_execute (0 = no timeout).",
    )
    p.add_argument("--use-cartesian-approach", action="store_true")
    p.add_argument("--use-cartesian-lift", action="store_true")
    p.add_argument(
        "--grasp-hold-orientation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="During the grasp insert, hold the orientation the arm had after the "
        "approach (pour_place default). Use --no-grasp-hold-orientation to instead "
        "command the fixed/target grasp quaternion so the final gripper orientation "
        "is enforced (e.g. level) regardless of where the approach ended.",
    )
    p.add_argument(
        "--use-cartesian-grasp",
        action="store_true",
        help="When the grasp phase uses MoveIt, plan it with compute_cartesian_path "
        "(straight-line TCP). Implied by --motion-strategy auto_cartesian.",
    )
    p.add_argument("--skip-approach", action="store_true")
    p.add_argument("--skip-lift", action="store_true")
    p.add_argument(
        "--qp-otg-p-step",
        type=float,
        default=_POUR_PLACE_QPIK_OTG_P_STEP,
        help="QPIK translation OTG step (pour_place tuned: 0.0008).",
    )
    p.add_argument(
        "--qp-otg-r-step",
        type=float,
        default=_POUR_PLACE_QPIK_OTG_R_STEP,
    )
    p.add_argument("--qp-stream-duration", type=float, default=1.5)
    p.add_argument("--qp-stream-rate-hz", type=float, default=100.0)
    p.add_argument(
        "--stream-closed-loop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For qp_stream/auto_stream: publish each on-line waypoint and wait for the "
        "TCP to reach it before advancing, forcing a straight Cartesian line (default on).",
    )
    p.add_argument(
        "--stream-step-m",
        type=float,
        default=0.005,
        help="Closed-loop straight-line waypoint spacing in meters (default 5 mm).",
    )
    p.add_argument(
        "--stream-waypoint-tol-m",
        type=float,
        default=0.004,
        help="Closed-loop: distance to a waypoint to consider it reached (default 4 mm).",
    )
    p.add_argument(
        "--stream-waypoint-timeout-s",
        type=float,
        default=0.6,
        help="Closed-loop: max wait per waypoint before moving on (default 0.6 s).",
    )
    # qp_all / streamed-QP interpolation tuning (used by qp_stream/qp_all phases).
    p.add_argument(
        "--qp-transit-p-step",
        type=float,
        default=0.014,
        help="QP controller otg_p_step for BIG transit moves (raise/approach/lift/"
        "retract under qp_all). Larger = faster straight-line transit (default "
        "14 mm/cycle). Lower it if the arm protective-stops on fast transits.",
    )
    p.add_argument(
        "--qp-transit-r-step",
        type=float,
        default=0.03,
        help="QP controller otg_r_step (rotation) for big transit moves (default 30 mrad/cycle).",
    )
    p.add_argument(
        "--qp-speed-scale",
        type=float,
        default=0.7,
        help="Fraction of the controller's per-cycle otg cap at which the streamed "
        "setpoint advances (<1 so the arm trails by a small bounded lag and stays "
        "on the straight line). Default 0.7.",
    )
    p.add_argument(
        "--qp-stream-hold-sec",
        type=float,
        default=0.4,
        help="Seconds to keep publishing the final target so the QP OTG converges "
        "before the phase returns (default 0.4 s).",
    )
    p.add_argument(
        "--qp-lag-guard-m",
        type=float,
        default=0.04,
        help="If the achieved TCP trails the streamed setpoint by more than this, hold "
        "the setpoint (and abort after --qp-stall-timeout-s) to catch "
        "singularities/limits/obstacles. 0 disables the guard (default 4 cm).",
    )
    p.add_argument(
        "--qp-stall-timeout-s",
        type=float,
        default=2.0,
        help="Abort a streamed phase if the TCP makes no progress along the line for "
        "this many seconds (singularity/limit/obstacle). Default 2 s.",
    )
    p.add_argument(
        "--qp-lookahead-m",
        type=float,
        default=0.08,
        help="Default/global pure-pursuit lookahead (used by non-qp_all qp_stream "
        "phases). The streamed setpoint is kept this far ahead of the actual TCP "
        "along the straight line; must stay below the controller's dis_err_bound "
        "(0.2 m). Default 8 cm.",
    )
    p.add_argument(
        "--qp-transit-lookahead-m",
        type=float,
        default=0.13,
        help="Pure-pursuit lookahead for FAST transit moves (raise/approach/lift/"
        "retract under qp_all). Bigger keeps the carrot far ahead so the controller "
        "runs at full otg speed; default 13 cm (<0.2 m dis_err_bound).",
    )
    p.add_argument(
        "--qp-grasp-lookahead-m",
        type=float,
        default=0.03,
        help="Pure-pursuit lookahead for the DENSE grasp insert (approach->grasp). "
        "Small = tight, finely-interpolated straight line that decelerates cleanly "
        "into the object; default 3 cm.",
    )
    p.add_argument(
        "--qp-transit-hold-sec",
        type=float,
        default=0.1,
        help="Convergence hold after a fast transit move (default 0.1 s). The grasp "
        "insert keeps the longer --qp-stream-hold-sec for precision.",
    )
    p.add_argument(
        "--qp-transit-pos-tol-m",
        type=float,
        default=0.02,
        help="Position tolerance to declare a fast transit move 'arrived' and exit "
        "immediately (default 2 cm). Transit moves don't need grasp precision.",
    )
    p.add_argument(
        "--qp-grasp-pos-tol-m",
        type=float,
        default=0.01,
        help="Position tolerance for the dense grasp insert (default 1 cm).",
    )
    p.add_argument(
        "--qp-transit-raise-z",
        type=float,
        default=0.0,
        help="qp_all only: before the approach, stream straight up by this many meters "
        "(keeping orientation) to clear the table/body, then proceed to the approach "
        "waypoint. 0 disables (default).",
    )
    p.add_argument("--dry-run", action="store_true")
    # pure gripper (no tactile) options
    p.add_argument(
        "--close-gripper",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Close gripper after reaching grasp pose, before lift (default: on).",
    )
    p.add_argument(
        "--open-gripper-before-grasp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open gripper before arm approach (not during grasp insert).",
    )
    p.add_argument(
        "--gripper-close-delay-sec",
        type=float,
        default=0.3,
        help="Extra wait after arm reaches grasp pose, before closing gripper.",
    )
    p.add_argument(
        "--grasp-reach-timeout-sec",
        type=float,
        default=20.0,
        help="Max wait for TCP to reach grasp XYZ before closing gripper.",
    )
    p.add_argument(
        "--grasp-reach-tol-m",
        type=float,
        default=0.03,
        help="Position tolerance (m) to consider arm at grasp pose.",
    )
    p.add_argument(
        "--gripper-backend",
        choices=["modbus_rtu", "zmq"],
        default="modbus_rtu",
        help="modbus_rtu: direct USB/RS485 like bottle_cup_pour_place (no extra server). "
        "zmq: requires robotiq_node_zmq in a separate terminal.",
    )
    p.add_argument("--gripper-wait-timeout-s", type=float, default=8.0)
    p.add_argument("--gripper-settle-s", type=float, default=2.5)
    p.add_argument("--gripper-recv-timeout-ms", type=int, default=500)
    p.add_argument("--gripper-close-pct", type=float, default=100.0, help="Target close percent (0..100).")
    p.add_argument("--gripper-open-pct", type=float, default=0.0, help="Target open percent (0..100).")
    p.add_argument("--gripper-speed-pct", type=float, default=100.0)
    p.add_argument("--gripper-force-pct", type=float, default=60.0)
    p.add_argument("--gripper-server-ip", type=str, default="127.0.0.1")
    p.add_argument("--gripper-server-set-port", type=int, default=4244)
    p.add_argument("--gripper-server-get-port", type=int, default=4245)
    p.add_argument("--gripper-serial-port", type=str, default="auto")
    p.add_argument("--gripper-slave-id", type=int, default=9)
    p.add_argument(
        "--gripper-activate-on-connect",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Activate the gripper at connect so it is motion-ready before grasping. "
        "With --gripper-async-connect this overlaps the arm approach and also keeps the "
        "resetActivate finger-cycle AWAY from the object. --no-... defers to first move.",
    )
    p.add_argument(
        "--gripper-force-activate",
        action="store_true",
        help="Always run resetActivate() even if gripper already active.",
    )
    p.add_argument(
        "--gripper-async-connect",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Connect to the gripper in a background thread so its (intermittently "
        "~12 s) serial open overlaps the arm enable + approach instead of stalling "
        "the sequence. The gripper is opened after the approach, before the grasp "
        "insert. For instant connects on EVERY run, run a persistent "
        "robotiq_node_zmq server and use --gripper-backend zmq.",
    )
    p.add_argument(
        "--gripper-port-cache",
        type=str,
        default="/tmp/robotiq_gripper_port.txt",
        help="Cache file for the auto-detected gripper serial port. Avoids the slow "
        "all-ports scan (which also nudges the gripper) on every run. Empty to disable.",
    )
    # finish behaviour: release object + go back to a safe home posture
    p.add_argument(
        "--return-home",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After the grasp sequence, move the arm back to the home joint posture.",
    )
    p.add_argument(
        "--release-on-finish",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open the gripper (release) at the very end, AFTER returning home, so "
        "the object is dropped clear of the table.",
    )
    p.add_argument(
        "--retract-to-approach",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After lifting, retract (holding the object) before the joint-space "
        "home move so the arm clears the table. By default the retract target is a "
        "high pose pulled toward the body (see --retract-toward-body); disable that "
        "to fall back to the middle/approach waypoint.",
    )
    p.add_argument(
        "--retract-toward-body",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Make the post-lift retract go to a HIGH pose pulled toward the body "
        "(waist -X) instead of the approach waypoint, so the following joint-space "
        "home move stays high and tucked and does not sweep through the table.",
    )
    p.add_argument(
        "--retract-toward-body-m",
        type=float,
        default=0.18,
        help="How far to pull the retract pose toward the body along waist -X (m). "
        "Bridges the gap to the tucked home so the home move is short and high.",
    )
    p.add_argument(
        "--retract-toward-body-y-m",
        type=float,
        default=0.1,
        help="How far to shift the retract pose along waist -Y (m). A pure -X pull "
        "into the close-body gesture can force one joint near its limit (overmovement); "
        "adding -Y keeps the arm in a more natural configuration. Set 0 to disable.",
    )
    p.add_argument(
        "--retract-extra-z",
        type=float,
        default=0.05,
        help="Extra height (m) added on top of the lift for the toward-body retract "
        "pose, so the arm is well above the table before going home.",
    )
    p.add_argument(
        "--home-right-joints",
        type=float,
        nargs=7,
        default=list(_RIGHT_ARM_HOME_JOINTS),
        help="Right-arm home joints used by --start-home and --return-home.",
    )
    p.add_argument(
        "--home-left-joints",
        type=float,
        nargs=7,
        default=list(_LEFT_ARM_HOME_JOINTS),
        help="Left-arm home joints used by --start-home and --return-home.",
    )
    p.add_argument(
        "--pre-cycle-right-joints",
        type=float,
        nargs=7,
        default=list(_RIGHT_ARM_HOME_JOINTS),
        help="Right-arm joint posture visited after --start-home and before detection "
        "when --pre-cycle-move is enabled.",
    )
    p.add_argument(
        "--pre-cycle-left-joints",
        type=float,
        nargs=7,
        default=list(_LEFT_ARM_HOME_JOINTS),
        help="Left-arm joint posture visited after --start-home and before detection "
        "when --pre-cycle-move is enabled.",
    )
    p.add_argument(
        "--use-pre-home-joints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After the lift, move through a fixed JOINT-SPACE waypoint (a hand-taught "
        "high/tucked posture, clear of table + joint limits) before the joint-space "
        "home move. This replaces the Cartesian toward-body retract (more reliable, no "
        "IK / joint over-rotation). Disable to use the Cartesian retract instead.",
    )
    p.add_argument(
        "--pre-home-right-joints",
        type=float,
        nargs=7,
        default=[
            2.4318695068359375e-05,
            -0.6644778251647949,
            -0.09999608993530273,
            -2.258571147918701,
            0.3064851760864258,
            -0.6528887748718262,
            0.37276792526245117,
        ],
        help="Right-arm joint-space waypoint (rad) visited after the lift and before "
        "--return-home (hand-taught via read_right_joints_admittance.py).",
    )
    p.add_argument(
        "--pre-home-left-joints",
        type=float,
        nargs=7,
        # Sagittal mirror of --pre-home-right-joints (negate joints 1,2,4,6),
        # matching how LEFT/RIGHT home postures mirror. The previous default
        # [1.0,0.3,0.4,-2.3,0.45,-0.2,0.18] was an uncalibrated placeholder, so
        # the left arm took a wrong "middle" posture before homing.
        default=[
            2.4318695068359375e-05,
            0.6644778251647949,
            0.09999608993530273,
            -2.258571147918701,
            -0.3064851760864258,
            -0.6528887748718262,
            -0.37276792526245117,
        ],
        help="Left-arm joint-space waypoint (rad) visited after the lift and "
        "before --return-home (mirror of --pre-home-right-joints).",
    )
    p.add_argument(
        "--use-elbow-low-pre-home-joints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For the standard ELBOW-LOW path, move through its dedicated fixed "
        "joint-space waypoint after the lift and before --return-home. This is "
        "independent of --use-pre-home-joints, which is reserved for ELBOW-HIGH.",
    )
    p.add_argument(
        "--elbow-low-pre-home-right-joints",
        type=float,
        nargs=7,
        default=[
            0.0,
            -1.18,
            0.0,
            -1.3,
            -1.4,
            -0.13,
            0.18,
        ],
        help="Right-arm ELBOW-LOW joint-space waypoint (rad) visited after the "
        "lift and before --return-home.",
    )
    p.add_argument(
        "--elbow-low-pre-home-left-joints",
        type=float,
        nargs=7,
        default=[
            0.0,
            1.18,
            0.0,
            -1.3,
            1.4,
            -0.13,
            0.18,
        ],
        help="Left-arm ELBOW-LOW joint-space waypoint (rad) visited after the "
        "lift and before --return-home.",
    )

    # --- Force-compliant grasp insert (F/T admittance) -------------------
    # When enabled, the FINAL grasp insert is performed under wrist-F/T
    # admittance: soft along the insertion axis, stiff laterally, rigid wrist.
    # The descent stops the instant a contact force is sensed (table/object),
    # which prevents the recurring table collisions / joint faults. After the
    # gripper closes (while the arm is held compliant), the wrist is rotated
    # away from the table and the usual lift/retract/home continues.
    # v1: NO object-weight re-zero and NO compliant lift.
    p.add_argument(
        "--compliant-grasp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the F/T admittance compliant descend-to-contact for the final "
        "grasp insert (default on). --no-compliant-grasp falls back to the "
        "position-controlled insert.",
    )
    p.add_argument(
        "--ft-topic-left",
        default="/arm_6dof_left",
        help="Wrench topic for the LEFT wrist F/T sensor.",
    )
    p.add_argument(
        "--ft-topic-right",
        default="/arm_6dof_right",
        help="Wrench topic for the RIGHT wrist F/T sensor.",
    )
    p.add_argument(
        "--ft-calib-left",
        default="",
        help="LEFT F/T calibration JSON (default: ft_calibration/ft_calibration_left.json). "
        "Generate with: python3 -m compliant_grasp_execute.ft_calibration.calibrate_ft --arm left",
    )
    p.add_argument(
        "--ft-calib-right",
        default="",
        help="RIGHT F/T calibration JSON (default: ft_calibration/ft_calibration_right.json). "
        "Generate with: python3 -m compliant_grasp_execute.ft_calibration.calibrate_ft --arm right",
    )
    p.add_argument(
        "--compliant-contact-force-n",
        type=float,
        default=1.5,
        help="Resisting force along the insertion axis (N) that counts as contact "
        "and stops the compliant descent.",
    )
    p.add_argument(
        "--compliant-stall-window-s",
        type=float,
        default=0.7,
        help="Window (s) over which to measure insertion progress for STALL contact "
        "detection. A compliant arm yields on contact (low force) but stops moving; "
        "if it progresses less than --compliant-stall-eps-m in this window it is "
        "treated as contact (blocked by object/table).",
    )
    p.add_argument(
        "--compliant-stall-eps-m",
        type=float,
        default=0.004,
        help="Minimum TCP progress (m) along the insertion axis within the stall "
        "window; below this the arm is considered blocked (contact).",
    )
    p.add_argument(
        "--compliant-contact-debounce",
        type=int,
        default=3,
        help="Consecutive samples over the contact threshold before tripping.",
    )
    p.add_argument(
        "--compliant-min-insert-m",
        type=float,
        default=0.008,
        help="Minimum ACTUAL TCP travel (m) along the insertion axis before a "
        "contact trip is accepted (rejects wrist reaction / start-up transients).",
    )
    p.add_argument(
        "--compliant-overshoot-m",
        type=float,
        default=0.02,
        help="How far PAST the planned grasp depth the compliant descent may keep "
        "going to find contact.",
    )
    p.add_argument(
        "--compliant-max-insert-m",
        type=float,
        default=0.20,
        help="Safety ceiling on total compliant travel (m). Must be >= the approach "
        "standoff distance + overshoot, otherwise the gripper stops short of the "
        "object and grasps air.",
    )
    p.add_argument(
        "--compliant-final-descent-m",
        type=float,
        default=0.04,
        help="HYBRID insert: only the FINAL this-many metres of the approach->grasp "
        "traverse are done with the F/T admittance (descend-to-contact). The earlier, "
        "longer portion is driven by the position-controlled QP stream (smooth, "
        "accurate, handles lateral motion) and only the last cm near the table are "
        "compliant. Set to 0 to make the WHOLE insert compliant (old behaviour). This "
        "removes the shake seen when a low-tilt grasp makes the insertion nearly "
        "horizontal and the soft admittance spring has to drag the arm sideways.",
    )
    p.add_argument(
        "--compliant-insert-speed-mps",
        type=float,
        default=0.020,
        help="Equilibrium slew speed of the compliant descent (m/s).",
    )
    p.add_argument(
        "--compliant-max-lag-m",
        type=float,
        default=0.025,
        help="Lag throttle: HOLD the equilibrium slew while the TCP lags the "
        "commanded target by more than this (m) along the insertion axis. Keeps "
        "the target from racing ahead of a slow arm (the 'grasp air' failure).",
    )
    p.add_argument(
        "--compliant-lateral-stiffness",
        type=float,
        default=40.0,
        help="Admittance stiffness K (N/m) on axes ORTHOGONAL to the insertion "
        "direction (hold the planned line).",
    )
    p.add_argument(
        "--compliant-insertion-stiffness",
        type=float,
        default=20.0,
        help="Admittance stiffness K (N/m) on axes ALONG the insertion direction "
        "(compliance on contact).",
    )
    p.add_argument(
        "--compliant-soften-threshold",
        type=float,
        default=0.30,
        help="A waist axis is treated as 'along insertion' (and made soft) when its "
        "|direction component| exceeds this. Prevents a stiff axis from fighting a "
        "tilted descent and producing a false contact.",
    )
    p.add_argument(
        "--compliant-damping-ratio",
        type=float,
        default=1.4,
        help="Target damping ratio. B is set per-axis to "
        "damping_ratio * 2*sqrt(K*M) so every axis is near-critically damped "
        "(>=1 avoids free-space shaking and table bounce).",
    )
    p.add_argument(
        "--compliant-hold-stiffness",
        type=float,
        default=150.0,
        help="Isotropic stiffness K (N/m) for the post-contact HOLD while the "
        "gripper closes. Stiffer than the descent so the arm stays planted on the "
        "residual contact force instead of springing back (bounce).",
    )
    p.add_argument(
        "--compliant-damping",
        type=float,
        default=3.0,
        help="Absolute floor (N s/m) on the per-axis admittance damping B.",
    )
    p.add_argument(
        "--compliant-mass",
        type=float,
        default=0.1,
        help="Admittance mass M (kg) on all translation axes.",
    )
    p.add_argument(
        "--compliant-filter-alpha",
        type=float,
        default=0.35,
        help="F/T EMA filter (prev = a*new + (1-a)*prev). LOWER = more smoothing. "
        "Applied identically to BOTH arms. If one side shakes while the other is "
        "smooth, that side's sensor is noisier -- lower this (e.g. 0.25).",
    )
    p.add_argument(
        "--compliant-force-deadzone",
        type=float,
        default=0.8,
        help="Force deadzone (N): wrench below this is ignored so sensor noise "
        "does not drive the admittance. Must stay < --compliant-contact-force-n.",
    )
    p.add_argument(
        "--compliant-torque-deadzone",
        type=float,
        default=0.08,
        help="Torque deadzone (Nm) for noise rejection (wrist is held rigid).",
    )
    p.add_argument(
        "--compliant-control-rate-hz",
        type=float,
        default=100.0,
        help="Supervisory descent loop rate (Hz) for the compliant insert.",
    )
    p.add_argument(
        "--compliant-max-vel",
        type=float,
        default=0.20,
        help="HARD cap (m/s) on admittance output velocity -- a SAFETY BACKSTOP on a "
        "force-driven runaway, not the primary guard (that is the negative-lag "
        "runaway stop). MUST stay well above --compliant-insert-speed-mps (~0.02): "
        "the published carrot is v_cmd*lead_time and the QP only tracks a fraction of "
        "it, so a healthy descent needs v_cmd ~5x the slew. Capping too low starves "
        "the carrot and the arm creeps -> false stall (grasp air).",
    )
    p.add_argument(
        "--compliant-max-omega",
        type=float,
        default=0.5,
        help="HARD cap (rad/s) on admittance angular velocity (wrist is held rigid).",
    )
    p.add_argument(
        "--compliant-otg-p-step",
        type=float,
        default=0.008,
        help="QP controller position OTG step (m) used during the compliant insert. "
        "Too small a step stalls the descent.",
    )
    p.add_argument(
        "--compliant-otg-r-step",
        type=float,
        default=0.005,
        help="QP controller rotation OTG step (rad) used during the compliant insert.",
    )
    p.add_argument(
        "--compliant-loop-period",
        type=float,
        default=0.004,
        help="Admittance integration period (s).",
    )
    p.add_argument(
        "--compliant-trans-lead-time",
        type=float,
        default=0.12,
        help="Published position carrot = v_cmd * this (s). Must be >> the loop "
        "period: a v_cmd*loop_period carrot is sub-millimetre for slow compliant "
        "motion and the QP controller produces no motion (the 'arm barely moves' "
        "failure). ~0.10-0.15 s gives a trackable carrot.",
    )
    p.add_argument(
        "--table-clear-rotate-deg",
        type=float,
        default=-20.0,
        help="After the grasp closes, rotate the wrist about waist Y by this many "
        "degrees to tilt the gripper away from the table before the lift "
        "(same sign convention as --lift-tilt-y-deg; 0 disables).",
    )
    add_config_args(p, default_config_path(__file__))
    return p


def _detect_pose(args: argparse.Namespace, xarm: XARM_manager) -> Dict[str, Any]:
    if args.detected_pose_json:
        with open(args.detected_pose_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "pose_tcp_waist_yaw_link_pose7" not in data:
            raise ValueError(f"{args.detected_pose_json} missing pose_tcp_waist_yaw_link_pose7")
        return data

    if not args.prompt:
        raise ValueError("Provide --prompt (repeatable) when not using --detected-pose-json")

    camera_yaml = _resolve_camera_yaml(args.camera_yaml)
    if args.pipeline_version == "current":
        percep = PerceptionTool(base_url=args.base_url, camera_pose_file_path=camera_yaml)
    else:
        percep = _AcceleratedSegToPoseAdapter(base_url=args.base_url, camera_yaml=camera_yaml)

    res = get_object_pose_in_waist_yaw_link(
        xarm=xarm,
        percep=percep,
        prompts=list(args.prompt),
        rgb_topic=args.rgb_topic,
        depth_topic=args.depth_topic,
        mode=args.mode,
        waist_frame=args.waist_frame,
        head_frame=args.head_frame,
        tf_timeout=float(args.tf_timeout),
        tf_retries=int(args.tf_retries),
        tf_warmup_sec=float(args.tf_warmup_sec),
        cam_timeout=float(args.cam_timeout),
        save_dir=(args.save_dir or None),
        save_prefix=args.save_prefix,
        tcp_to_tip_offset_m=np.array(
            [args.tcp_to_tip_x, args.tcp_to_tip_y, args.tcp_to_tip_z], dtype=float
        ),
        orientation_policy=str(args.orientation_policy),
        # 'auto' is resolved after detection; use a concrete arm here. With the
        # default fixed grasp quat the detection-time orientation is overridden
        # anyway, and the object centroid (used for selection) is arm-independent.
        rotation_arm=(str(args.arm) if str(args.arm) in ("left", "right") else "right"),
        grasp_yaw_offset=np.radians(float(args.grasp_yaw_offset_deg)),
        max_grasp_yaw_delta=(
            np.radians(float(args.max_grasp_yaw_delta_deg))
            if float(args.max_grasp_yaw_delta_deg) > 0
            else None
        ),
        verbose=True,
        segment_confidence=float(args.segment_confidence),
    )
    if res is None:
        raise RuntimeError("Detection failed: get_object_pose_in_waist_yaw_link returned None")
    return res


def _object_y_in_waist(det: Dict[str, Any]) -> Optional[float]:
    """Return the detected object's Y coordinate in the waist frame, or None."""
    tip = det.get("pose_tip_waist_yaw_link_rpy")
    if isinstance(tip, dict) and "y" in tip:
        return float(tip["y"])
    pose7 = det.get("pose_tcp_waist_yaw_link_pose7")
    if isinstance(pose7, (list, tuple)) and len(pose7) >= 2:
        return float(pose7[1])
    return None


def _elbow_high_enabled_for_arm(args: argparse.Namespace, arm: str) -> bool:
    """Whether the elbow-high path (proactive + fallback) is allowed for this arm.

    ``--elbow-high-arms`` is a comma-separated allow-list (default 'left'): only
    the left elbow-high seed was hand-taught, and the right seed (a sagittal
    mirror) can drive the right ``wrist_roll`` past its tighter -1.3 lower stop
    at far-right reaches. An arm not in the list uses the standard elbow-low
    path even for parallel-to-body objects.
    """
    raw = str(getattr(args, "elbow_high_arms", "left"))
    allowed = {s.strip().lower() for s in raw.split(",") if s.strip()}
    return str(arm).strip().lower() in allowed


def _select_arm(det: Dict[str, Any], args: argparse.Namespace) -> str:
    """Resolve which arm to grasp with.

    If --arm is left/right, that is honored. For --arm auto and a positive
    ``--arm-select-boundary-y``, use a three-zone split:
      * Y < -boundary: right arm
      * Y > +boundary: left arm
      * 0 <= Y <= boundary: yaw <= threshold -> left, else right
      * -boundary <= Y < 0: yaw < -threshold -> left, else right

    A non-positive boundary keeps the legacy single-boundary Y split.
    """
    requested = str(args.arm).strip().lower()
    if requested in ("left", "right"):
        return requested

    y = _object_y_in_waist(det)
    if y is None:
        _log(
            "ARM-SELECT: could not read object Y in waist frame; "
            f"defaulting to '{args.arm_select_default}'"
        )
        return str(args.arm_select_default)

    boundary = float(args.arm_select_boundary_y)
    deadband = float(args.arm_select_deadband_m)
    band = abs(boundary)
    if band > 1e-9:
        if y < -band:
            chosen = "right"
            reason = f"y < -boundary {-band:.3f}"
        elif y > band:
            chosen = "left"
            reason = f"y > +boundary {band:.3f}"
        else:
            long_yaw = det.get("object_long_axis_yaw_waist_deg")
            try:
                ly = float(long_yaw)
            except (TypeError, ValueError):
                chosen = str(args.arm_select_default)
                reason = (
                    f"inside center band [-{band:.3f}, +{band:.3f}] but no valid "
                    f"signed long-axis yaw; defaulting to {chosen}"
                )
            else:
                yaw_threshold = abs(
                    float(getattr(args, "arm_select_long_axis_yaw_threshold_deg", 15.0))
                )
                if y >= 0.0:
                    chosen = "left" if ly <= yaw_threshold else "right"
                    rule = f"Y>=0, yaw {'<=' if ly <= yaw_threshold else '>'} +{yaw_threshold:.1f}deg"
                else:
                    chosen = "left" if ly < -yaw_threshold else "right"
                    rule = f"Y<0, yaw {'<' if ly < -yaw_threshold else '>='} -{yaw_threshold:.1f}deg"
                reason = (
                    f"inside center band [-{band:.3f}, +{band:.3f}], "
                    f"signed long-axis yaw {ly:+.1f}deg ({rule})"
                )
    elif abs(y - boundary) < deadband:
        chosen = str(args.arm_select_default)
        reason = f"within deadband {deadband:.3f}m of boundary {boundary:.3f}"
    else:
        chosen = "left" if y >= boundary else "right"
        reason = f"y {'>=' if y >= boundary else '<'} boundary {boundary:.3f}"
    _log(
        f"ARM-SELECT: object waist Y={y:+.3f}m ({reason}) -> '{chosen}' arm"
    )
    return chosen


def _startup_home_and_activate(
    args: argparse.Namespace, xarm: XARM_manager, action: ActionCall
) -> None:
    """Move BOTH arms to their home posture and activate both grippers.

    Runs once before the detect->grasp cycle so the robot always starts from a
    known, clear posture with the grippers ready. Failures are logged but do not
    abort the run.
    """
    if bool(args.start_home):
        _log("STARTUP: enabling arms and moving BOTH to home posture")
        try:
            xarm.xarm_deactivate_all_controller()
            xarm.hardware_arm_enable(True)
            xarm.hardware_arm_mode(3)
        except Exception as e:  # noqa: BLE001
            _log(f"WARNING: startup arm enable failed: {e!r}")
        for arm_name, joints in (
            ("left", list(args.home_left_joints)),
            ("right", list(args.home_right_joints)),
        ):
            try:
                _log(f"STARTUP: {arm_name} -> home joints={[f'{v:.3f}' for v in joints]}")
                if arm_name == "left":
                    action.jointspace_arm_L_controller(joints)
                else:
                    action.jointspace_arm_R_controller(joints)
            except Exception as e:  # noqa: BLE001
                _log(f"WARNING: startup {arm_name} home move failed: {e!r}")

    if bool(args.activate_grippers_on_start):
        if str(args.gripper_backend) != "modbus_rtu":
            _log(
                f"STARTUP: gripper backend is '{args.gripper_backend}', skipping "
                "USB activation (server/backend handles it)"
            )
            return
        _log("STARTUP: activating both grippers (open/close cycle)")
        try:
            _ensure_robotiq_usb_import_path()
            from robotiq_grippers import create_gripper, list_grippers
        except Exception as e:  # noqa: BLE001
            _log(f"WARNING: cannot import robotiq_grippers for startup activation: {e!r}")
            return
        for name in list_grippers():
            try:
                g = create_gripper(
                    name,
                    activate_on_connect=True,
                    force_activate=bool(args.gripper_force_activate),
                )
                _log(f"STARTUP: gripper '{name}' activated")
                try:
                    g.disconnect()
                except Exception:  # noqa: BLE001
                    pass
            except Exception as e:  # noqa: BLE001
                _log(f"WARNING: startup activation of gripper '{name}' failed: {e!r}")


def _resolve_gripper_serial_port(args: argparse.Namespace) -> str:
    """Resolve the gripper serial port, caching the auto-detected port.

    The Robotiq modbus driver's ``auto`` mode scans every serial port (each with
    a modbus timeout) and even nudges the gripper while probing, which is slow
    and runs on every invocation. We cache the resolved port path and reuse it
    so later runs skip the scan entirely.
    """
    requested = str(args.gripper_serial_port)
    if requested != "auto":
        return requested
    # Prefer the stable, name-based by-id port for the selected arm. Both
    # physical grippers answer Modbus at slave id 9, so `auto` scanning is
    # ambiguous (it keeps whichever port responds last) and ttyUSBx numbers
    # swap on replug. robotiq_grippers maps arm name -> FTDI by-id path, the
    # same single source of truth test_both_grippers.py uses.
    arm = str(getattr(args, "arm", "")).strip().lower()
    if arm in ("left", "right"):
        try:
            _ensure_robotiq_usb_import_path()
            from robotiq_grippers import resolve_port

            port = resolve_port(arm)
            _log(f"gripper: arm='{arm}' -> stable by-id port {port}")
            return port
        except Exception as exc:  # noqa: BLE001 - fall back to cache/auto-scan
            _log(
                f"gripper: could not resolve by-id port for arm='{arm}' ({exc}); "
                "falling back to cached port / auto-scan"
            )
    cache = str(getattr(args, "gripper_port_cache", "") or "")
    if cache and os.path.exists(cache):
        try:
            with open(cache, "r", encoding="utf-8") as f:
                cached = f.read().strip()
            if cached and os.path.exists(cached):
                _log(f"gripper: using cached serial port {cached} (skip auto-scan)")
                return cached
        except Exception:  # noqa: BLE001
            pass
    return "auto"


def _cache_gripper_serial_port(args: argparse.Namespace, gripper: Any) -> None:
    cache = str(getattr(args, "gripper_port_cache", "") or "")
    if not cache or str(args.gripper_serial_port) != "auto":
        return
    try:
        port = getattr(getattr(gripper, "_gripper", None), "portname", None)
        if port:
            with open(cache, "w", encoding="utf-8") as f:
                f.write(str(port))
            _log(f"gripper: cached serial port {port} -> {cache}")
    except Exception:  # noqa: BLE001
        pass


def _build_gripper(args: argparse.Namespace) -> Optional[Any]:
    """Create and return the gripper controller (or None on failure).

    Extracted so it can run in a background thread while the arm enables and
    approaches, hiding the (intermittently slow, ~12 s) serial open behind motion
    that has to happen anyway.
    """
    _ensure_robotiq_usb_import_path()
    from robotiq_api import create_gripper_controller

    t0 = time.monotonic()
    gripper = create_gripper_controller(
        backend=str(args.gripper_backend),
        serial_port=_resolve_gripper_serial_port(args),
        slave_id=int(args.gripper_slave_id),
        server_ip=str(args.gripper_server_ip),
        server_set_port=int(args.gripper_server_set_port),
        server_get_port=int(args.gripper_server_get_port),
        recv_timeout_ms=int(args.gripper_recv_timeout_ms),
        move_wait_timeout_s=float(args.gripper_wait_timeout_s),
        activate_on_connect=bool(args.gripper_activate_on_connect),
        force_activate=bool(args.gripper_force_activate),
    )
    _cache_gripper_serial_port(args, gripper)
    _log(
        f"gripper controller ready: backend={args.gripper_backend} "
        f"port={args.gripper_serial_port} slave={args.gripper_slave_id} "
        f"(wait_timeout={args.gripper_wait_timeout_s}s) "
        f"[connect took {time.monotonic() - t0:.2f}s]"
    )
    if str(args.gripper_backend) == "zmq":
        _log(
            "gripper: ZMQ backend -> persistent robotiq_node_zmq server holds the "
            "serial open + activation, so this connect is instant on every run"
        )
    return gripper


def _arm_limits(
    xarm: XARM_manager, arm: str, guard: Optional[JointLimitGuard]
) -> List[Tuple[float, float]]:
    """Per-joint (lower, upper) limits for ``arm`` (live via guard, else table)."""
    if guard is not None:
        try:
            return guard.limits_for(arm)
        except Exception:  # noqa: BLE001
            pass
    return fetch_arm_limits(xarm, arm)


def _capture_elbow_high_seed(
    args: argparse.Namespace,
    xarm: XARM_manager,
    parser: argparse.ArgumentParser,
    config_path: str,
) -> int:
    """Record the arm's CURRENT joints as the elbow-high seed and write config.

    Workflow: hand-drag the arm (gravity-comp script) into an elbow-high posture
    with the gripper pointing straight DOWN above where the object sits, then run
    ``--capture-elbow-high-seed --arm left|right``. The captured joints become
    ``elbow_high_ready_{arm}_joints`` in config.yaml. A good seed keeps margin on
    EVERY joint -- this prints a warning for any joint near a limit so you can
    re-drag before committing.
    """
    arm = str(args.arm)
    if arm not in ("left", "right"):
        _log("CAPTURE: --arm must be 'left' or 'right' (not 'auto') to capture a seed.")
        return 2
    joints: Optional[List[Any]] = None
    for _ in range(60):
        xarm.joint_state_update()
        joints = (
            xarm.xarm_left_arm_joint_angles()
            if arm == "left"
            else xarm.xarm_right_arm_joint_angles()
        )
        if joints is not None and all(j is not None for j in joints):
            break
        rclpy.spin_once(xarm, timeout_sec=0.05)
    if joints is None or any(j is None for j in joints):
        _log("CAPTURE: could not read joint angles from /joint_states; is the robot up?")
        return 1
    joints = [round(float(j), 4) for j in joints]
    limits = _arm_limits(xarm, arm, None)
    warns: List[str] = []
    for i in range(7):
        lo, hi = limits[i]
        d = min(abs(joints[i] - lo), abs(hi - joints[i]))
        if d < 0.15:
            warns.append(
                f"{ARM_JOINT_NAMES[i]}={joints[i]:+.3f} only {d:.3f}rad from "
                f"limit [{lo:.3f},{hi:.3f}]"
            )
    _log(f"CAPTURE: {arm} arm current joints = {joints}")
    if warns:
        _log(
            "CAPTURE: WARNING near-limit joint(s) -- a usable seed needs headroom on "
            "ALL joints (especially wrist_pitch for a downward grasp); re-drag and "
            "recapture: " + "; ".join(warns)
        )
    dest = f"elbow_high_ready_{arm}_joints"
    setattr(args, dest, joints)
    # Never persist the capture flag itself (else every later run would re-capture).
    args.capture_elbow_high_seed = False
    # The --arm flag on a capture command selects WHICH arm's seed to record; it
    # must NOT overwrite the grasp arm-selection policy in config.yaml (e.g.
    # clobbering `arm: auto` with `arm: left`, which then forces every later
    # grasp onto the left arm regardless of object position). Restore the arm
    # value the config file already had before the CLI --arm override, then dump.
    try:
        import yaml

        with open(config_path, "r", encoding="utf-8") as _f:
            _cfg_existing = yaml.safe_load(_f) or {}
        _prev_arm = _cfg_existing.get("arm", "auto")
    except Exception:  # noqa: BLE001
        _prev_arm = "auto"
    if str(_prev_arm).lower() != str(getattr(args, "arm", "")).lower():
        _log(
            f"CAPTURE: restoring config 'arm' to '{_prev_arm}' (the --arm {arm} flag "
            "only selects which seed to capture, not the default grasp arm)."
        )
    args.arm = str(_prev_arm)
    dump_config(parser, args, config_path)
    _log(
        f"CAPTURE: wrote {dest} to {config_path}. Re-run "
        "`python3 -m compliant_grasp_execute.main` to grasp with the new seed."
    )
    return 0


def _clamp_joints_to_limits(
    joints: List[float],
    limits: List[Tuple[float, float]],
    margin_rad: float,
    *,
    label: str = "",
) -> List[float]:
    """Pull each joint to within ``margin_rad`` of its limit.

    A hand-taught seed can sit AT (or slightly past) a hard limit -- e.g. the
    recorded elbow-high pose has shoulder_yaw at -2.964 vs the -2.96 limit. Using
    it verbatim as a QP seed makes the joint start at/over the limit and any
    motion needing more of that joint faults the motor. Clamping guarantees a
    usable margin. If the safe band is narrower than 2*margin the midpoint is
    used.
    """
    out: List[float] = []
    clamped: List[str] = []
    m = max(0.0, float(margin_rad))
    for i in range(7):
        lo, hi = limits[i]
        lo_s, hi_s = lo + m, hi - m
        if lo_s > hi_s:  # band too narrow for the margin -> center it
            lo_s = hi_s = 0.5 * (lo + hi)
        v = float(joints[i])
        cv = min(max(v, lo_s), hi_s)
        if abs(cv - v) > 1e-6:
            clamped.append(
                f"{ARM_JOINT_NAMES[i]} {v:+.3f}->{cv:+.3f} [{lo:.3f},{hi:.3f}]"
            )
        out.append(cv)
    if clamped:
        _log(
            f"elbow-high seed{(' ' + label) if label else ''}: clamped "
            f"{len(clamped)} joint(s) inward (margin {m:.2f}rad): "
            + "; ".join(clamped)
        )
    return out


def _elbow_high_ready_joints(
    args: argparse.Namespace,
    xarm: XARM_manager,
    arm: str,
    guard: Optional[JointLimitGuard],
) -> List[float]:
    """Clamped elbow-high seed posture for ``arm``."""
    raw = (
        list(args.elbow_high_ready_left_joints)
        if arm == "left"
        else list(args.elbow_high_ready_right_joints)
    )
    limits = _arm_limits(xarm, arm, guard)
    return _clamp_joints_to_limits(
        raw, limits, float(args.elbow_high_clamp_margin_rad), label=f"ready/{arm}"
    )


def _elbow_high_stage_joints(
    args: argparse.Namespace,
    xarm: XARM_manager,
    arm: str,
    guard: Optional[JointLimitGuard],
) -> List[float]:
    """Clamped elbow-high STAGING posture for ``arm``.

    Visited from home before the elbow-high reconfigure (and on the way back).
    """
    raw = (
        list(args.elbow_high_stage_left_joints)
        if arm == "left"
        else list(args.elbow_high_stage_right_joints)
    )
    limits = _arm_limits(xarm, arm, guard)
    return _clamp_joints_to_limits(
        raw, limits, float(args.elbow_high_clamp_margin_rad), label=f"stage/{arm}"
    )


def _elbow_high_transition_joints(
    args: argparse.Namespace,
    xarm: XARM_manager,
    arm: str,
    guard: Optional[JointLimitGuard],
    base_joints: List[float],
    ready_joints: List[float],
) -> List[float]:
    """Clamped intermediate waypoint bridging stage <-> elbow-high.

    Uses the configured waypoint if a full 7-joint list was given; otherwise the
    elementwise midpoint of ``base_joints`` (the stage) and ready (which, because
    jointspace interpolation is per-joint, keeps every intermediate value inside
    the box spanned by two in-limit endpoints -> no joint-limit fault, smaller
    sweep).
    """
    cfg = (
        list(args.elbow_high_transition_left_joints)
        if arm == "left"
        else list(args.elbow_high_transition_right_joints)
    )
    if len(cfg) == 7:
        raw = [float(v) for v in cfg]
    else:
        raw = [0.5 * (float(base_joints[i]) + float(ready_joints[i])) for i in range(7)]
    limits = _arm_limits(xarm, arm, guard)
    return _clamp_joints_to_limits(
        raw, limits, float(args.elbow_high_clamp_margin_rad), label=f"transition/{arm}"
    )


def _jointspace_move(
    action: ActionCall, arm: str, joints: List[float], label: str
) -> bool:
    """Blocking jointspace move of ``arm`` to ``joints`` (7), with logging."""
    _log(f"ELBOW-HIGH: {arm} -> {label} joints={[f'{v:.3f}' for v in joints]}")
    try:
        if arm == "left":
            res = action.jointspace_arm_L_controller([float(v) for v in joints])
        else:
            res = action.jointspace_arm_R_controller([float(v) for v in joints])
        _log(f"ELBOW-HIGH: {label} move result: {res}")
        return True
    except Exception as e:  # noqa: BLE001
        _log(f"WARNING: elbow-high {label} move failed: {e!r}")
        return False


def _move_to_elbow_high(
    action: ActionCall,
    xarm: XARM_manager,
    args: argparse.Namespace,
    arm: str,
    guard: Optional[JointLimitGuard],
) -> bool:
    """Reconfigure ``arm`` into the elbow-high seed via the transition waypoint.

    Jointspace, blocking. The QP/compliant phases that follow re-seed off this
    posture and therefore stay in the elbow-high IK basin. Endpoints are clamped
    so no joint starts at/over a limit.
    """
    stage = _elbow_high_stage_joints(args, xarm, arm, guard)
    ready = _elbow_high_ready_joints(args, xarm, arm, guard)
    transition = _elbow_high_transition_joints(args, xarm, arm, guard, stage, ready)
    _log(
        f"ELBOW-HIGH: reconfiguring {arm} arm into elbow-high seed "
        "(jointspace: home -> stage -> transition -> ready). NOTE: jointspace "
        "moves run at the controller's own speed; watch the first reconfiguration "
        "for table/body clearance."
    )
    # Ensure the position controller is active for the jointspace action.
    try:
        xarm.hardware_arm_mode(3)
    except Exception as e:  # noqa: BLE001
        _log(f"WARNING: could not set mode 3 before elbow-high move: {e!r}")
    # From home, first stage into the arm-up posture, then bridge into the
    # elbow-high basin via the transition waypoint, then the ready seed.
    ok = _jointspace_move(action, arm, stage, "stage")
    ok = _jointspace_move(action, arm, transition, "transition") and ok
    ok = _jointspace_move(action, arm, ready, "ready") and ok
    if guard is not None:
        guard.report(arm)
    return ok


def _object_needs_elbow_high(
    det: Dict[str, Any], args: argparse.Namespace, arm: str
) -> bool:
    """SIGN- and ARM-aware elbow-high decision from the object's long-axis yaw.

    The long-axis yaw in the waist frame (``object_long_axis_yaw_waist_deg``,
    call it ``ly``) is SIGNED and folded to (-90, +90]:
      * ``ly > 0`` -> the line leans toward +Y (robot's LEFT),
      * ``ly < 0`` -> the line leans toward -Y (robot's RIGHT).

    Because each arm's ``wrist_roll`` range is asymmetric, elbow-LOW only works
    when the lean matches the arm's comfortable side. So each arm keeps a
    ``T``-wide elbow-LOW comfort wedge on ITS sign; everything else goes
    elbow-high:
      * RIGHT arm elbow-low wedge:  ``-d < ly < +T``   (i.e. [0, T) plus a
        ``d`` deadband into the negatives),
      * LEFT  arm elbow-low wedge:  ``-T < ly < +d``   (mirror),
    where ``T = --elbow-high-proactive-angle-min-deg`` (60) and
    ``d = --elbow-high-deadband-deg``.

    The deadband ``d`` around ``ly = 0`` keeps NEAR-PERPENDICULAR objects on the
    elbow-low path for BOTH arms regardless of the (noisy) sign -- a perpendicular
    object sits at ly~=0 where the sign is physically meaningless and could flip
    run-to-run from detection noise.

    Returns True when the object should go ELBOW-HIGH.
    """
    T = float(args.elbow_high_proactive_angle_min_deg)
    d = float(getattr(args, "elbow_high_deadband_deg", 0.0))

    long_yaw = det.get("object_long_axis_yaw_waist_deg")
    if long_yaw is None:
        # No signed axis available: fall back to the unsigned magnitude test.
        ang = det.get("object_angle_from_waist_x_deg")
        try:
            return ang is not None and float(ang) >= T
        except Exception:  # noqa: BLE001
            return False
    try:
        ly = float(long_yaw)
    except Exception:  # noqa: BLE001
        return False

    if str(arm).strip().lower() == "right":
        elbow_low = (-d < ly < T)
    else:  # left (and default/mirror)
        elbow_low = (-T < ly < d)
    return not elbow_low


def _move_pre_home_joints(action: ActionCall, args: argparse.Namespace) -> bool:
    """Move the active arm to the hand-taught pre-home joint waypoint."""
    joints = (
        list(args.pre_home_left_joints)
        if str(args.arm) == "left"
        else list(args.pre_home_right_joints)
    )
    _log(f"PRE-HOME WAYPOINT: {args.arm} -> joints={[f'{v:.3f}' for v in joints]}")
    try:
        if str(args.arm) == "left":
            res = action.jointspace_arm_L_controller(joints)
        else:
            res = action.jointspace_arm_R_controller(joints)
        _log(f"pre-home waypoint result: {res}")
        return True
    except Exception as e:  # noqa: BLE001
        _log(f"WARNING: pre-home waypoint move failed: {e!r}")
        return False


def _move_elbow_low_pre_home_joints(
    action: ActionCall, args: argparse.Namespace
) -> bool:
    """Move the active arm to its dedicated elbow-low pre-home waypoint."""
    joints = (
        list(args.elbow_low_pre_home_left_joints)
        if str(args.arm) == "left"
        else list(args.elbow_low_pre_home_right_joints)
    )
    _log(
        f"ELBOW-LOW PRE-HOME WAYPOINT: {args.arm} -> "
        f"joints={[f'{v:.3f}' for v in joints]}"
    )
    try:
        if str(args.arm) == "left":
            res = action.jointspace_arm_L_controller(joints)
        else:
            res = action.jointspace_arm_R_controller(joints)
        _log(f"elbow-low pre-home waypoint result: {res}")
        return True
    except Exception as e:  # noqa: BLE001
        _log(f"WARNING: elbow-low pre-home waypoint move failed: {e!r}")
        return False


def _move_pre_cycle_joints(action: ActionCall, args: argparse.Namespace) -> bool:
    """Move both arms to the configured pre-detection joint postures."""
    if not bool(getattr(args, "pre_cycle_move", False)):
        return True
    _log("PRE-CYCLE: moving BOTH arms to configured pre-detection postures")
    ok = True
    for arm_name, joints in (
        ("left", list(args.pre_cycle_left_joints)),
        ("right", list(args.pre_cycle_right_joints)),
    ):
        try:
            _log(f"PRE-CYCLE: {arm_name} -> joints={[f'{v:.3f}' for v in joints]}")
            if arm_name == "left":
                res = action.jointspace_arm_L_controller(joints)
            else:
                res = action.jointspace_arm_R_controller(joints)
            _log(f"pre-cycle {arm_name} result: {res}")
        except Exception as e:  # noqa: BLE001
            ok = False
            _log(f"WARNING: pre-cycle {arm_name} move failed: {e!r}")
    return ok


def _return_home(action: ActionCall, args: argparse.Namespace) -> bool:
    """Move the active arm back to its home joint posture."""
    joints = (
        list(args.home_left_joints)
        if str(args.arm) == "left"
        else list(args.home_right_joints)
    )
    _log(f"RETURN HOME: {args.arm} -> joints={[f'{v:.3f}' for v in joints]}")
    try:
        if str(args.arm) == "left":
            res = action.jointspace_arm_L_controller(joints)
        else:
            res = action.jointspace_arm_R_controller(joints)
        _log(f"return home result: {res}")
        return True
    except Exception as e:  # noqa: BLE001
        _log(f"WARNING: return home failed: {e!r}")
        return False


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    _, config_path = apply_config_defaults(parser, argv)
    args = parser.parse_args(argv)
    if maybe_write_config(parser, args, config_path):
        return 0
    if not rclpy.ok():
        rclpy.init()
    xarm = XARM_manager()
    if bool(getattr(args, "capture_elbow_high_seed", False)):
        # Record the dragged posture as the elbow-high seed and exit BEFORE any
        # motion (the startup home move would destroy the hand-taught pose).
        rc = _capture_elbow_high_seed(args, xarm, parser, config_path)
        xarm.destroy_node()
        rclpy.shutdown()
        return rc
    action = ActionCall(xarm)
    moveit = MoveitCall(xarm)
    topic_pub = TopicPublisher(xarm)

    result: Dict[str, Any] = {
        "arm": args.arm,
        "pipeline_version": args.pipeline_version,
        "motion_strategy": args.motion_strategy,
        "dry_run": bool(args.dry_run),
        "motion": {},
    }
    try:
        # Pre-cycle reset: move BOTH arms to home and activate both grippers so
        # every run starts from a known, clear posture. Skipped in dry-run.
        if not args.dry_run:
            _startup_home_and_activate(args, xarm, action)
            result["motion"]["pre_cycle_move_ok"] = _move_pre_cycle_joints(
                action, args
            )

        det = _detect_pose(args, xarm)
        detected_pose7 = [float(v) for v in det["pose_tcp_waist_yaw_link_pose7"]]
        # Resolve which arm to grasp with (handles --arm auto from object Y).
        arm = _select_arm(det, args)
        args.arm = arm
        result["arm"] = arm
        grasp_pose7 = _build_grasp_pose7(det, args, arm)
        approach_pose7, lift_pose7, pre_home_pose7 = _derive_motion_poses(grasp_pose7, args)
        dx, dy, dz = _resolve_grasp_xyz_offset(args, str(args.arm))
        result["detection"] = det
        result["motion"]["detected_pose7"] = detected_pose7
        result["motion"]["approach_pose7"] = approach_pose7
        result["motion"]["grasp_pose7"] = grasp_pose7
        result["motion"]["lift_pose7"] = lift_pose7
        result["motion"]["pre_home_pose7"] = pre_home_pose7
        result["motion"]["grasp_xyz_offset_m"] = [dx, dy, dz]
        result["motion"]["grasp_quat"] = grasp_pose7[3:7]
        result["motion"]["use_fixed_grasp_quat"] = bool(args.use_fixed_grasp_quat)
        phase_backends = _resolve_phase_backends(str(args.motion_strategy))
        result["motion"]["phase_backends"] = phase_backends

        _log(
            f"detected class={det.get('class_name')} "
            f"detected_pose7={[f'{v:.4f}' for v in detected_pose7]} "
            f"-> motion grasp_pose7={[f'{v:.4f}' for v in grasp_pose7]}"
        )

        if args.dry_run:
            _log("dry-run: skip hardware enable and motion execution")
            result["motion"]["approach_ok"] = None
            result["motion"]["grasp_ok"] = None
            result["motion"]["lift_ok"] = None
            result["gripper"] = {"open_ok": None, "close_ok": None}
        else:
            _t_enable = time.monotonic()
            xarm.xarm_deactivate_all_controller()
            xarm.hardware_arm_enable(True)
            xarm.hardware_arm_mode(3)
            _log(f"arms enabled (mode 3) [took {time.monotonic() - _t_enable:.2f}s]")

            guard: Optional[JointLimitGuard] = None
            if bool(getattr(args, "joint_limit_guard", True)):
                guard = JointLimitGuard(
                    xarm,
                    margin_rad=float(getattr(args, "joint_limit_margin_rad", 0.10)),
                    enabled=True,
                    allow_recovery=bool(getattr(args, "joint_limit_allow_recovery", True)),
                    recovery_grace_iters=int(getattr(args, "joint_limit_recovery_grace_iters", 12)),
                )
                lims = guard.limits_for(args.arm)  # warm up / log source
                result["motion"]["joint_limits_used"] = {
                    n: list(lims[i]) for i, n in enumerate(ARM_JOINT_NAMES)
                }
                _log(
                    f"joint-limit guard ON (margin {guard.margin:.3f} rad) arm={args.arm}"
                )
            else:
                _log("joint-limit guard OFF (--no-joint-limit-guard)")

            gripper = None
            gripper_open_ok = None
            gripper_close_ok = None
            gripper_final_pct = None
            need_gripper = bool(args.close_gripper or args.open_gripper_before_grasp)
            _gripper_holder: Dict[str, Any] = {"g": None, "err": None}
            _gripper_thread: Optional[threading.Thread] = None
            if not need_gripper:
                _log(
                    "gripper actions disabled (--no-close-gripper and no open-before-grasp); "
                    "arm motion only"
                )
            elif bool(args.gripper_async_connect):
                def _gripper_connect_worker() -> None:
                    try:
                        _gripper_holder["g"] = _build_gripper(args)
                    except Exception as e:  # noqa: BLE001
                        _gripper_holder["err"] = e

                _gripper_thread = threading.Thread(
                    target=_gripper_connect_worker, daemon=True
                )
                _gripper_thread.start()
                _log(
                    "gripper: connecting in background (overlaps arm enable + approach "
                    "to hide the slow serial open; --no-gripper-async-connect for the "
                    "old blocking behaviour)"
                )
            else:
                try:
                    gripper = _build_gripper(args)
                except Exception as e:  # noqa: BLE001
                    _log(f"WARNING: gripper init failed, continue without gripper action: {e!r}")
                    gripper = None

            def _ensure_gripper_ready(timeout: Optional[float] = None) -> Optional[Any]:
                """Return the gripper, joining the background connect.

                timeout=None blocks until the connect finishes. timeout=0.0 is a
                non-blocking peek: it returns the gripper only if the background
                connect has already finished, else None (still connecting).
                """
                nonlocal gripper, _gripper_thread
                if gripper is not None:
                    return gripper
                if _gripper_thread is not None:
                    _gripper_thread.join(timeout=timeout)
                    if _gripper_thread.is_alive():
                        return None  # still connecting (non-blocking peek)
                    _gripper_thread = None
                    if _gripper_holder["err"] is not None:
                        _log(
                            "WARNING: gripper init failed, continue without gripper "
                            f"action: {_gripper_holder['err']!r}"
                        )
                        gripper = None
                    else:
                        gripper = _gripper_holder["g"]
                return gripper

            # Per-phase QP otg step: big transit moves (approach/lift) go fast,
            # the grasp insert stays at the fine step. Only qp_all opts the
            # transit phases into the faster step so other strategies are
            # unchanged.
            _qp_all = str(args.motion_strategy) == "qp_all"

            def _qp_p_step(phase: str) -> float:
                if _qp_all and phase in ("approach", "lift"):
                    return float(args.qp_transit_p_step)
                return float(args.qp_otg_p_step)

            def _qp_r_step(phase: str) -> float:
                if _qp_all and phase in ("approach", "lift"):
                    return float(args.qp_transit_r_step)
                return float(args.qp_otg_r_step)

            # Carrot lookahead, settle tolerance and convergence hold are also
            # phase-aware: the grasp insert stays DENSE (small lookahead + tight
            # tol + full hold) for a precise straight line; every transit move
            # (raise/approach/lift/retract) uses a big lookahead, loose tol and a
            # short hold so it runs at the controller's full speed and exits the
            # instant it arrives.
            def _qp_transit_phase(phase: str) -> bool:
                return _qp_all and phase in ("approach", "lift")

            def _qp_lookahead(phase: str) -> float:
                if phase == "grasp":
                    return float(args.qp_grasp_lookahead_m)
                if _qp_transit_phase(phase):
                    return float(args.qp_transit_lookahead_m)
                return float(args.qp_lookahead_m)

            def _qp_hold(phase: str) -> float:
                if _qp_transit_phase(phase):
                    return float(args.qp_transit_hold_sec)
                return float(args.qp_stream_hold_sec)

            def _qp_pos_tol(phase: str) -> float:
                if _qp_transit_phase(phase):
                    return float(args.qp_transit_pos_tol_m)
                return float(args.qp_grasp_pos_tol_m)

            approach_ok = True
            # Set True once the arm has been reconfigured into the elbow-high seed
            # (proactively or via the fallback). Used to (a) avoid re-trying it and
            # (b) route the post-grasp return home back through the transition
            # waypoint instead of sweeping straight from elbow-high to home.
            used_elbow_high = False
            # True once the grasp/approach poses have been switched to an
            # elbow-high-specific orientation (seed-anchored / top-down).
            eh_active = False
            # Orientation strategy for the elbow-high path:
            #   "seed"    -> anchor on the orientation the arm is ALREADY in at the
            #                seed (only translate; the wrist never leaves its range)
            #   "topdown" -> force the gripper straight down
            #   "sidetilt"-> reuse the normal side/tilt grasp orientation
            eh_mode = str(getattr(args, "elbow_high_orientation", "seed")).strip().lower()
            # Measured seed TCP orientation (quat xyzw), filled after reconfiguring.
            eh_seed_quat: Optional[List[float]] = None
            # Original QP joint limits to restore after the elbow-high grasp, set
            # once we tighten them into a window around the seed (see below).
            eh_qp_limits_orig: Optional[Tuple[List[float], List[float]]] = None

            def _lock_qp_to_elbow_high() -> None:
                """Tighten the QP solver's joint limits into a window around the
                elbow-high seed so it cannot drop the elbow (which would drive the
                wrist into its hard stop) while streaming to the grasp. The QP
                controller is re-activated AFTER the seed reconfigure in both the
                proactive and fallback flows, so it picks these up on activation.
                """
                nonlocal eh_qp_limits_orig
                if not bool(getattr(args, "elbow_high_qp_lock", True)):
                    return
                if eh_qp_limits_orig is not None:
                    return  # already locked this run
                seed = _read_arm_joints(xarm, args.arm)
                if seed is None:
                    _log("ELBOW-HIGH QP joint window: cannot read seed joints; skipping lock")
                    return
                lock_names = [
                    s.strip()
                    for s in str(
                        getattr(
                            args,
                            "elbow_high_qp_lock_joints",
                            "shoulder_pitch,shoulder_roll,shoulder_yaw,elbow_pitch,elbow_yaw",
                        )
                    ).split(",")
                    if s.strip()
                ]
                hard = _arm_limits(xarm, args.arm, guard)
                eh_qp_limits_orig = _set_qp_joint_window(
                    xarm,
                    args.arm,
                    seed,
                    float(getattr(args, "elbow_high_qp_lock_margin_rad", 0.6)),
                    lock_names,
                    hard,
                    drop_margin_rad=float(
                        getattr(args, "elbow_high_qp_lock_drop_margin_rad", 0.0)
                    ),
                )

            def _relax_wrist_guard_for_elbow_high() -> None:
                """Let wrist_pitch travel closer to its operational stop on the
                elbow-high path. The nose-down grasp at a far object NEEDS
                wrist_pitch ~ -0.71 (the orientation demands it; the elbow height
                does not change it), which is still well inside the -0.785 stop --
                but the default 0.1rad guard margin aborts at -0.685, before the
                arm can even reach the grasp. Give ONLY wrist_pitch a tighter
                margin here; every other joint keeps full protection.
                """
                if guard is None:
                    return
                m = float(getattr(args, "elbow_high_guard_margin_rad", 0.0))
                if m <= 0.0 or m >= float(guard.margin):
                    return
                guard.margin_overrides["wrist_pitch"] = m
                guard.margin_overrides["wrist_roll"] = m
                _log(
                    f"ELBOW-HIGH: wrist guard margin relaxed to {m:.3f}rad (from "
                    f"{float(guard.margin):.3f}) so the nose-down grasp can reach "
                    "wrist_pitch~-0.71 (still inside the -0.785 stop). Other joints "
                    "keep the full margin."
                )

            def _unlock_qp_from_elbow_high() -> None:
                nonlocal eh_qp_limits_orig
                if eh_qp_limits_orig is not None:
                    _restore_qp_joint_limits(xarm, args.arm, *eh_qp_limits_orig)
                    eh_qp_limits_orig = None
                if guard is not None and guard.margin_overrides:
                    guard.margin_overrides.clear()
                    _log("ELBOW-HIGH: wrist guard margin restored to full.")

            def _eh_build_grasp(
                extra_yaw_scale: float,
                jaw_flip: bool = False,
                jaw_target_heading_deg: Optional[float] = None,
            ) -> Optional[List[float]]:
                """Build a grasp pose7 for the active elbow-high orientation mode.

                Returns None for "sidetilt" (caller keeps the normal builder).

                ``jaw_flip`` selects the 180deg-flipped jaw-line equivalent (same
                short-axis grip, opposite wrist_roll side) -- used by the retry
                when the aligned grasp saturates wrist_roll on its cramped stop.

                When ``jaw_target_heading_deg`` is not given, the jaw target is
                chosen PROACTIVELY from the object angle: a DIAGONAL object (long
                axis NOT near-parallel to the body) needs a large short-axis yaw
                off the fore-aft seed that pins wrist_roll, so we align to the
                fixed waist heading straight away (no doomed short-axis attempt);
                a NEAR-PARALLEL object keeps exact short-axis alignment (which is
                ~= waist X anyway, a small roll).
                """
                _jtgt = jaw_target_heading_deg
                if _jtgt is None and bool(
                    getattr(args, "elbow_high_diagonal_use_waist_x", True)
                ):
                    _ly = det.get("object_long_axis_yaw_waist_deg")
                    _T = float(
                        getattr(args, "elbow_high_proactive_angle_min_deg", 60.0)
                    )
                    try:
                        if _ly is not None and abs(float(_ly)) < _T:
                            _jtgt = float(
                                getattr(args, "elbow_high_jaw_axis_heading_deg", 0.0)
                            )
                    except Exception:  # noqa: BLE001
                        pass
                if eh_mode == "seed" and eh_seed_quat is not None:
                    # Anchor on the seed orientation, but tilt it UP (less
                    # nose-down) so the descent doesn't drive wrist_pitch past its
                    # lower stop. Align the jaws to the object long axis about
                    # waist Z -- in this ~top-down posture that maps onto
                    # wrist_roll, which has room, so we can align to ANY object
                    # orientation (incl. parallel) by picking the roll-feasible
                    # 180deg-equivalent yaw (see _symmetric_jaw_yaw).
                    base_q = _reduce_approach_tilt(
                        eh_seed_quat, _seed_tilt_up_deg(args, guard)
                    )
                    _align = bool(getattr(args, "elbow_high_align_jaws", True))
                    _seed_joints = (
                        list(args.elbow_high_ready_left_joints)
                        if str(args.arm) == "left"
                        else list(args.elbow_high_ready_right_joints)
                    )
                    _roll0 = (
                        float(_seed_joints[6]) if len(_seed_joints) >= 7 else 0.0
                    )
                    _roll_lim = TIANYI2_ARM_LIMITS.get(
                        "left" if str(args.arm) == "left" else "right"
                    )[6]
                    return _build_grasp_pose7(
                        det,
                        args,
                        args.arm,
                        extra_yaw_scale=extra_yaw_scale,
                        tilt_override_deg=0.0,
                        max_yaw_override_deg=float(
                            getattr(args, "elbow_high_seed_yaw_max_deg", 90.0)
                        ),
                        base_quat_override=base_q,
                        jaw_yaw_symmetry=_align,
                        jaw_yaw_roll0=_roll0,
                        jaw_yaw_roll_limits=_roll_lim,
                        jaw_yaw_margin=float(
                            getattr(args, "elbow_high_align_wrist_margin_rad", 0.08)
                        ),
                        jaw_yaw_flip=bool(jaw_flip),
                        jaw_target_heading_deg=_jtgt,
                    )
                if eh_mode == "topdown":
                    # Full top-down on the first build (guard not yet tripped),
                    # shallower on a post-saturation retry (see helper).
                    return _build_topdown_grasp_pose7(
                        det,
                        args,
                        args.arm,
                        extra_yaw_scale=extra_yaw_scale,
                        tilt_deg=_topdown_retry_tilt_deg(args, guard),
                    )
                return None

            def _switch_to_elbow_high_poses(
                reason: str, seed_quat: Optional[List[float]] = None
            ) -> None:
                """Rebuild grasp/approach/lift/pre-home for the elbow-high path."""
                nonlocal grasp_pose7, approach_pose7, lift_pose7, pre_home_pose7
                nonlocal eh_active, eh_seed_quat
                eh_active = True
                if seed_quat is not None:
                    eh_seed_quat = [float(v) for v in seed_quat]
                if eh_mode == "seed" and eh_seed_quat is None:
                    _log(
                        "ELBOW-HIGH: WARNING could not read seed orientation (TF "
                        "stale); falling back to the side/tilt grasp orientation."
                    )
                g = _eh_build_grasp(1.0)
                if g is None:
                    _log(f"ELBOW-HIGH: using side/tilt grasp orientation ({reason}).")
                    return
                grasp_pose7 = g
                approach_pose7, lift_pose7, pre_home_pose7 = _derive_motion_poses(
                    grasp_pose7, args
                )
                result["motion"]["approach_pose7"] = approach_pose7
                result["motion"]["grasp_pose7"] = grasp_pose7
                result["motion"]["lift_pose7"] = lift_pose7
                result["motion"]["pre_home_pose7"] = pre_home_pose7
                result["motion"]["grasp_quat"] = grasp_pose7[3:7]
                result["motion"]["elbow_high_orientation"] = eh_mode
                _log(
                    f"ELBOW-HIGH: using {eh_mode.upper()} grasp orientation ({reason}); "
                    "rebuilt grasp/approach/lift poses. The seed-anchored mode keeps "
                    "the orientation the arm is already in, so the approach only "
                    "translates and the wrist stays in range."
                )

            skip_approach = bool(args.skip_approach) or (
                str(args.motion_strategy) == "moveit_direct"
            )
            if skip_approach:
                _log("ARM: skip approach waypoint (direct to grasp)")
            else:
                _log(f"ARM: approach via {phase_backends['approach']}")

            # TWO-STRATEGY axis split (see README "Elbow-high"): the elbow-LOW and
            # elbow-HIGH configs are complementary -- each grasps the short axis
            # for the object orientation the other cannot:
            #   * PERPENDICULAR / diagonal (angle-from-X < threshold): elbow-LOW.
            #     The fixed side-tilt already closes the jaws on the lateral short
            #     axis, and the wrist stays in range.
            #   * PARALLEL (angle-from-X >= threshold): elbow-HIGH. The taught
            #     seed's jaws point ~fore-aft (along waist X), which IS the short
            #     axis of a parallel object, so only a small jaw-align yaw is
            #     needed and wrist_roll stays near neutral (no saturation).
            # So elbow-high is selected by AXIS only (parallel bucket). The old
            # reach-based trigger is intentionally gone: it routed far-outboard
            # PERPENDICULAR objects into elbow-high, whose seed jaws are ~90deg
            # wrong for them -> wrist_roll saturates at its +1.3 stop and the grasp
            # fails. Far-outboard perpendicular reach is handled on the elbow-low
            # path instead (reach-based side-tilt reduction, see _reach_tilt_scale).
            # Only for arms whose elbow-high seed is enabled (--elbow-high-arms).
            _proactive_parallel = _object_needs_elbow_high(det, args, args.arm)
            _proactive_always = bool(getattr(args, "elbow_high_always", False))
            if (
                not skip_approach
                and _elbow_high_enabled_for_arm(args, args.arm)
                and (
                    _proactive_always
                    or (
                        bool(getattr(args, "elbow_high_proactive", False))
                        and _proactive_parallel
                    )
                )
            ):
                if _proactive_always:
                    _why = (
                        "elbow_high_always is set -- routing every grasp through "
                        "the elbow-high posture"
                    )
                else:
                    _why = (
                        f"object long-axis yaw (signed) "
                        f"{det.get('object_long_axis_yaw_waist_deg')}deg falls "
                        f"OUTSIDE the {args.arm} arm's elbow-low comfort wedge "
                        f"(T={float(args.elbow_high_proactive_angle_min_deg)}deg, "
                        f"deadband={float(getattr(args, 'elbow_high_deadband_deg', 0.0))}"
                        "deg); the lean is on the arm's cramped wrist_roll side or "
                        "the object is ~parallel, so route to elbow-high"
                    )
                _log(
                    f"ELBOW-HIGH (proactive): {_why}; reconfiguring to elbow-high "
                    "seed before approach."
                )
                used_elbow_high = _move_to_elbow_high(
                    action, xarm, args, args.arm, guard
                )
                result["motion"]["elbow_high_proactive_used"] = used_elbow_high
                if used_elbow_high:
                    _seed_q = _read_tcp_quat(
                        xarm, args.arm, args.waist_frame, grasp_pose7
                    )
                    _switch_to_elbow_high_poses("proactive", _seed_q)
                    _lock_qp_to_elbow_high()
                    _relax_wrist_guard_for_elbow_high()

            # qp_all big-move safety: lift straight up first to clear the
            # table/body, then stream to the approach waypoint.
            if (
                _qp_all
                and not skip_approach
                and float(args.qp_transit_raise_z) > 0.0
            ):
                cur_tcp = xarm.get_tcp_pose(
                    arm=args.arm, base_frame=args.waist_frame, timeout=2.0
                )
                if cur_tcp is not None:
                    raised = [
                        float(cur_tcp["translation"][0]),
                        float(cur_tcp["translation"][1]),
                        float(cur_tcp["translation"][2]) + float(args.qp_transit_raise_z),
                        *[float(v) for v in cur_tcp["rotation"]],
                    ]
                    _log(
                        f"ARM: qp_all transit raise +{args.qp_transit_raise_z*100:.0f}cm Z "
                        "before approach"
                    )
                    _qp_stream_to_pose7(
                        topic_pub,
                        xarm,
                        args.arm,
                        raised,
                        args.waist_frame,
                        "transit-raise",
                        otg_p_step=_qp_p_step("approach"),
                        otg_r_step=_qp_r_step("approach"),
                        stream_duration_sec=float(args.qp_stream_duration),
                        stream_rate_hz=float(args.qp_stream_rate_hz),
                        keep_current_orientation=True,
                        speed_scale=float(args.qp_speed_scale),
                        hold_sec=_qp_hold("approach"),
                        lag_guard_m=float(args.qp_lag_guard_m),
                        stall_timeout_s=float(args.qp_stall_timeout_s),
                        lookahead_m=_qp_lookahead("approach"),
                        pos_tol_m=_qp_pos_tol("approach"),
                        guard=guard,
                    )
            def _do_approach(pose7: List[float]) -> bool:
                return _exec_pose_by_backend(
                    phase_backends["approach"],
                    action=action,
                    moveit=moveit,
                    topic_pub=topic_pub,
                    xarm=xarm,
                    arm=args.arm,
                    pose7=pose7,
                    waist_frame=args.waist_frame,
                    vel_scale=float(args.vel_scale),
                    acc_scale=float(args.acc_scale),
                    label="approach",
                    use_cartesian_path=bool(args.use_cartesian_approach),
                    qp_otg_p_step=_qp_p_step("approach"),
                    qp_otg_r_step=_qp_r_step("approach"),
                    qp_stream_duration=float(args.qp_stream_duration),
                    qp_stream_rate_hz=float(args.qp_stream_rate_hz),
                    keep_current_orientation=False,
                    moveit_timeout_sec=float(args.moveit_timeout_sec),
                    qp_speed_scale=float(args.qp_speed_scale),
                    qp_hold_sec=_qp_hold("approach"),
                    qp_lag_guard_m=float(args.qp_lag_guard_m),
                    qp_stall_timeout_s=float(args.qp_stall_timeout_s),
                    qp_lookahead_m=_qp_lookahead("approach"),
                    qp_pos_tol_m=_qp_pos_tol("approach"),
                    guard=guard,
                )

            if not skip_approach:
                orient_tol_m = float(getattr(args, "grasp_orient_miss_tol_m", 0.04))
                _off_len = float(
                    np.linalg.norm(
                        [
                            float(args.tcp_to_tip_x),
                            float(args.tcp_to_tip_y),
                            float(args.tcp_to_tip_z),
                        ]
                    )
                )

                def _approach_and_check(pose7: List[float]):
                    """Approach + tracking, folding ORIENTATION error into success.

                    The TCP can converge on position while the wrist fails to hold
                    the commanded tilt. Because the fingertip is ~_off_len metres
                    out along the tool axis, that orientation miss throws the
                    fingers ~2*off*sin(err/2) off the object -> a grasp on air that
                    used to be reported as success. We treat a large fingertip miss
                    as a reach failure so the reduced-tilt retry fires.
                    """
                    ok = _do_approach(pose7)
                    trk = _tcp_tracking_error(
                        xarm, args.arm, args.waist_frame, pose7, "approach"
                    )
                    orient_failed = False
                    if ok and trk is not None and orient_tol_m > 0 and _off_len > 1e-6:
                        oe = math.radians(float(trk.get("orientation_error_deg", 0.0)))
                        miss = 2.0 * _off_len * math.sin(oe / 2.0)
                        if miss > orient_tol_m:
                            _log(
                                f"approach orientation miss {math.degrees(oe):.1f}deg -> "
                                f"~{miss*1000:.0f}mm fingertip offset > tol "
                                f"{orient_tol_m*1000:.0f}mm: TCP reached position but the wrist "
                                "could not hold the commanded tilt, so the fingers would miss "
                                "the object. Treating approach as failed to trigger the "
                                "reduced-tilt retry (prevents grasping air)."
                            )
                            ok = False
                            orient_failed = True
                    return ok, trk, orient_failed

                # Reach retry: if the approach could not reach the target (a wrist
                # joint hit its limit, OR the orientation could not be tracked),
                # retry with a REDUCED continuous/adaptive yaw and a shallower
                # tilt. A smaller, more reachable orientation is trackable (unlike
                # the 180deg gripper-symmetry flip the streaming controller cannot
                # track) and both relieves the limiting joint and lands the fingers.
                approach_ok, approach_trk, _orient_failed = _approach_and_check(approach_pose7)

                # ELBOW-HIGH jaw-flip retry (BEFORE the reduced-yaw retry): the
                # short-axis alignment can drive wrist_roll into the arm's CRAMPED
                # stop (left upper +1.3 / right lower -1.3) -- the QP then clamps
                # the wrist and can't reach the orientation. The identical grip is
                # reachable from the 180deg-flipped jaw line, which rolls the wrist
                # onto its ROOMY side (left down to -1.65 / right up to +1.65).
                # This keeps FULL short-axis alignment (unlike the reduced-yaw
                # retry, which just misaligns the jaws and does not free
                # wrist_roll). Only fires when the abort was actually a wrist_roll
                # saturation on the elbow-high aligned path.
                def _guard_saturated_joint() -> Optional[str]:
                    if guard is None or guard.last_event is None:
                        return None
                    c = guard.last_event.get("closest") or {}
                    return c.get("joint")

                # ELBOW-HIGH waist-axis align retry (preferred wrist_roll recovery):
                # the object's diagonal SHORT axis can need a large waist-Z yaw off
                # the fore-aft seed that pins wrist_roll on its CRAMPED stop. Instead
                # of chasing the short axis, re-aim the jaws to a FIXED waist heading
                # (default waist +X = 0deg). That is only a small roll from the seed
                # (which the arm can hold), so wrist_roll stays in range. For a
                # PARALLEL object the short axis already == waist X so this is a
                # no-op; for a DIAGONAL object it trades exact short-axis alignment
                # (up to ~45deg off) for reachability -- which is the whole point.
                if (
                    not approach_ok
                    and eh_active
                    and bool(getattr(args, "elbow_high_align_jaws", True))
                    and bool(getattr(args, "elbow_high_roll_align_waist_x", True))
                    and _guard_saturated_joint() == "wrist_roll"
                    and not result["motion"].get("elbow_high_waist_x_align_used")
                ):
                    _heading = float(
                        getattr(args, "elbow_high_jaw_axis_heading_deg", 0.0)
                    )
                    _log(
                        "ELBOW-HIGH: wrist_roll hit its CRAMPED stop aligning to the "
                        "object short axis; re-aiming the jaws to the FIXED waist "
                        f"heading {_heading:+.0f}deg instead (small roll from the "
                        "fore-aft seed -> stays in range; gives up exact short-axis "
                        "alignment for reachability)."
                    )
                    _eh_g = _eh_build_grasp(1.0, jaw_target_heading_deg=_heading)
                    if _eh_g is not None:
                        grasp_pose7 = _eh_g
                        approach_pose7, lift_pose7, pre_home_pose7 = (
                            _derive_motion_poses(grasp_pose7, args)
                        )
                        result["motion"]["approach_pose7"] = approach_pose7
                        result["motion"]["grasp_pose7"] = grasp_pose7
                        result["motion"]["lift_pose7"] = lift_pose7
                        result["motion"]["pre_home_pose7"] = pre_home_pose7
                        result["motion"]["grasp_quat"] = grasp_pose7[3:7]
                        result["motion"]["elbow_high_waist_x_align_used"] = True
                        result["motion"]["elbow_high_waist_x_heading_deg"] = _heading
                        if guard is not None:
                            guard.last_event = None
                        approach_ok, approach_trk, _orient_failed = (
                            _approach_and_check(approach_pose7)
                        )
                        result["motion"]["elbow_high_approach_ok"] = approach_ok

                # ELBOW-HIGH wrist_roll BACKOFF ladder (preferred over the 180deg
                # jaw-flip): the short-axis alignment can drive wrist_roll into the
                # arm's CRAMPED stop. The flip reaches the identical grip from the
                # opposite jaw line but needs a ~180deg reorientation the
                # pure-pursuit streamer cannot turn through (it stalls). Instead we
                # progressively BACK OFF the jaw-align yaw (extra_yaw_scale < 1),
                # which stays in the small-rotation basin and rolls wrist_roll back
                # toward its roomy seed value -- trading a few degrees of short-axis
                # alignment for reachability. Only fires when the abort was a
                # wrist_roll saturation on the elbow-high aligned path.
                if (
                    not approach_ok
                    and eh_active
                    and bool(getattr(args, "elbow_high_align_jaws", True))
                    and bool(getattr(args, "elbow_high_yaw_backoff", True))
                    and _guard_saturated_joint() == "wrist_roll"
                    and not result["motion"].get("elbow_high_yaw_backoff_used")
                ):
                    _scales = [
                        float(s)
                        for s in getattr(
                            args, "elbow_high_yaw_backoff_scales", [0.7, 0.45, 0.25]
                        )
                        if 0.0 <= float(s) < 1.0
                    ]
                    for _bscale in _scales:
                        _log(
                            "ELBOW-HIGH: wrist_roll hit its CRAMPED stop aligning to "
                            f"the short axis; backing off jaw-align yaw to scale "
                            f"{_bscale:.2f} (PARTIAL short-axis alignment that keeps "
                            "the wrist in range -- avoids the un-streamable 180deg "
                            "flip)."
                        )
                        _eh_g = _eh_build_grasp(_bscale)
                        if _eh_g is None:
                            break
                        grasp_pose7 = _eh_g
                        approach_pose7, lift_pose7, pre_home_pose7 = (
                            _derive_motion_poses(grasp_pose7, args)
                        )
                        result["motion"]["approach_pose7"] = approach_pose7
                        result["motion"]["grasp_pose7"] = grasp_pose7
                        result["motion"]["lift_pose7"] = lift_pose7
                        result["motion"]["pre_home_pose7"] = pre_home_pose7
                        result["motion"]["grasp_quat"] = grasp_pose7[3:7]
                        result["motion"]["elbow_high_yaw_backoff_used"] = True
                        result["motion"]["elbow_high_yaw_backoff_scale"] = _bscale
                        if guard is not None:
                            guard.last_event = None
                        approach_ok, approach_trk, _orient_failed = (
                            _approach_and_check(approach_pose7)
                        )
                        result["motion"]["elbow_high_approach_ok"] = approach_ok
                        if approach_ok:
                            break
                        # Stop the ladder if the block is no longer wrist_roll: a
                        # different joint / orientation miss won't be helped by
                        # further yaw backoff.
                        if _guard_saturated_joint() not in ("wrist_roll", None):
                            break

                if (
                    not approach_ok
                    and eh_active
                    and bool(getattr(args, "elbow_high_align_jaws", True))
                    and bool(getattr(args, "elbow_high_jaw_flip_retry", True))
                    and _guard_saturated_joint() == "wrist_roll"
                    and not result["motion"].get("elbow_high_jaw_flip_used")
                ):
                    _log(
                        "ELBOW-HIGH: wrist_roll hit its CRAMPED stop aligning to "
                        "the short axis; retrying the SAME grip from the "
                        "180deg-flipped jaw line (rolls the wrist onto its roomy "
                        "side) -- keeps full alignment."
                    )
                    _eh_g = _eh_build_grasp(1.0, jaw_flip=True)
                    if _eh_g is not None:
                        grasp_pose7 = _eh_g
                        approach_pose7, lift_pose7, pre_home_pose7 = (
                            _derive_motion_poses(grasp_pose7, args)
                        )
                        result["motion"]["approach_pose7"] = approach_pose7
                        result["motion"]["grasp_pose7"] = grasp_pose7
                        result["motion"]["lift_pose7"] = lift_pose7
                        result["motion"]["pre_home_pose7"] = pre_home_pose7
                        result["motion"]["grasp_quat"] = grasp_pose7[3:7]
                        result["motion"]["elbow_high_jaw_flip_used"] = True
                        if guard is not None:
                            guard.last_event = None
                        approach_ok, approach_trk, _orient_failed = (
                            _approach_and_check(approach_pose7)
                        )
                        result["motion"]["elbow_high_approach_ok"] = approach_ok

                retry_scale = float(getattr(args, "grasp_retry_yaw_scale", 0.5))
                if (
                    not approach_ok
                    and bool(getattr(args, "grasp_reach_retry", False))
                    and bool(getattr(args, "continuous_grasp_orientation", False)
                             or getattr(args, "adaptive_grasp_orientation", False))
                    and 0.0 <= retry_scale < 1.0
                ):
                    _log(
                        "approach failed to reach target; retrying with reduced "
                        f"long-axis alignment (yaw scale {retry_scale:.2f})"
                    )
                    # Reduce the nose-down tilt on retry when a wrist joint tripped
                    # the guard OR the orientation could not be tracked -- both are
                    # relieved by a shallower, more reachable tilt.
                    _retry_tilt_scale = (
                        retry_scale
                        if (
                            bool(getattr(args, "joint_limit_reduce_tilt_on_retry", True))
                            and (
                                (guard is not None and guard.last_event is not None)
                                or _orient_failed
                            )
                        )
                        else 1.0
                    )
                    _eh_g = _eh_build_grasp(retry_scale) if eh_active else None
                    if _eh_g is not None:
                        # Stay on the elbow-high orientation (seed-anchored / top-down);
                        # only the jaw-align yaw (and, for top-down, the tilt) is
                        # backed off here so the wrist stays in range.
                        _log(
                            f"ELBOW-HIGH {eh_mode} retry: yaw scale {retry_scale:.2f}"
                        )
                        grasp_pose7 = _eh_g
                    else:
                        grasp_pose7 = _build_grasp_pose7(
                            det,
                            args,
                            arm,
                            extra_yaw_scale=retry_scale,
                            extra_tilt_scale=_retry_tilt_scale,
                        )
                    approach_pose7, lift_pose7, pre_home_pose7 = _derive_motion_poses(
                        grasp_pose7, args
                    )
                    result["motion"]["approach_pose7"] = approach_pose7
                    result["motion"]["grasp_pose7"] = grasp_pose7
                    result["motion"]["lift_pose7"] = lift_pose7
                    result["motion"]["pre_home_pose7"] = pre_home_pose7
                    result["motion"]["grasp_quat"] = grasp_pose7[3:7]
                    result["motion"]["grasp_reach_retry_used"] = True
                    result["motion"]["grasp_reach_retry_yaw_scale"] = retry_scale
                    approach_ok, approach_trk, _orient_failed = _approach_and_check(approach_pose7)
                    # ELBOW-HIGH FALLBACK: the reduced-tilt retry is the best the
                    # elbow-LOW basin can do. It can fail in TWO ways, both meaning
                    # "the elbow-low IK branch cannot serve this pose":
                    #   (a) orientation-only miss (_orient_failed) -- TCP reached but
                    #       the wrist could not hold the tilt; OR
                    #   (b) a joint-limit guard ABORT -- a wrist joint saturated at
                    #       its hard stop mid-approach (guard.last_event set). NOTE
                    #       this does NOT set _orient_failed, because the orientation
                    #       check only runs when the approach returned ok. (b) is the
                    #       CLEAREST elbow-low-can't-do-it signal (e.g. wrist_pitch
                    #       pinned at its upper limit for an object lying parallel to
                    #       the body) and must trigger the reconfiguration too.
                    # Reconfigure into the elbow-HIGH seed and re-approach the SAME
                    # grasp orientation from there: QP re-seeds off the elbow-high
                    # posture and stays in that basin, where the wrist has room. Only
                    # if elbow-high ALSO fails does the best-effort path below run.
                    # Fallback to elbow-high is gated on the PARALLEL bucket (same
                    # axis split as the proactive trigger): the seed's fore-aft
                    # jaws only match a parallel object's short axis. A
                    # PERPENDICULAR object that failed elbow-low (e.g. far-outboard
                    # reach) would just saturate wrist_roll in elbow-high too, so we
                    # do NOT fall back for it -- it stays on the elbow-low
                    # best-effort path below.
                    _limit_aborted = guard is not None and guard.last_event is not None
                    if (
                        (not approach_ok)
                        and (_orient_failed or _limit_aborted)
                        and bool(getattr(args, "elbow_high_enable_fallback", True))
                        and _elbow_high_enabled_for_arm(args, args.arm)
                        and _object_needs_elbow_high(det, args, args.arm)
                        and not used_elbow_high
                    ):
                        _why = (
                            "wrist joint saturated at its limit"
                            if _limit_aborted
                            else "wrist could not hold the grasp orientation"
                        )
                        _log(
                            f"ELBOW-HIGH (fallback): elbow-low basin failed ({_why}) "
                            "on a ~parallel object; reconfiguring to the elbow-high "
                            "seed and re-approaching the grasp from there."
                        )
                        used_elbow_high = _move_to_elbow_high(
                            action, xarm, args, args.arm, guard
                        )
                        result["motion"]["elbow_high_fallback_used"] = used_elbow_high
                        if used_elbow_high:
                            # Anchor the grasp on the orientation the arm is already
                            # in at the seed (or top-down, per elbow_high_orientation)
                            # and re-derive the poses before re-approaching.
                            _seed_q = _read_tcp_quat(
                                xarm, args.arm, args.waist_frame, grasp_pose7
                            )
                            _switch_to_elbow_high_poses("fallback", _seed_q)
                            _lock_qp_to_elbow_high()
                            _relax_wrist_guard_for_elbow_high()
                            # Clear the stale elbow-low guard event so the elbow-high
                            # re-approach is judged on its own merits.
                            if guard is not None:
                                guard.last_event = None
                            approach_ok, approach_trk, _orient_failed = (
                                _approach_and_check(approach_pose7)
                            )
                            result["motion"]["elbow_high_approach_ok"] = approach_ok
                    # The reduced-tilt retry is the best orientation the arm can
                    # hold at this pose. If it STILL trips the orientation gate but
                    # the TCP DID reach the target position (an orientation-only
                    # miss, _orient_failed), do NOT abort the whole grasp -- proceed
                    # best-effort. Aborting guarantees no grasp; the compliant insert
                    # + overshoot still descends to contact, and a partial-tilt grasp
                    # often still catches the object. A true position/reach failure
                    # (arm could not get to the pose -> _orient_failed is False) is
                    # NOT overridden here, so we still bail out of genuinely
                    # unreachable targets.
                    if (not approach_ok) and _orient_failed:
                        _log(
                            "approach orientation still off after the reduced-tilt "
                            "retry, but the TCP reached the target position; proceeding "
                            "to grasp BEST-EFFORT (compliant descend-to-contact) instead "
                            "of returning empty-handed. (Root cause: this arm cannot "
                            "hold the commanded tilt at this far pose; see README tuning "
                            "notes on lowering grasp_tilt_y_deg if it recurs.)"
                        )
                        approach_ok = True
                        result["motion"]["approach_best_effort"] = True
                result["motion"]["approach_tcp_tracking"] = approach_trk
                _record_joint_limits(guard, args.arm, "approach", result)

            # Do NOT stall here waiting for the (slow ~12 s) serial connect. Peek
            # non-blocking: if the gripper is already connected, open it before the
            # insert; otherwise defer and let the dense grasp insert below overlap
            # the remaining connect time. We hard-wait for the gripper only right
            # before the close (by which point the connect is almost always done).
            if need_gripper:
                gripper = _ensure_gripper_ready(timeout=0.0)
            if gripper is None and need_gripper and args.open_gripper_before_grasp:
                _log(
                    "GRIPPER: still connecting -> deferring pre-grasp open; the grasp "
                    "insert will overlap the connect (fingers assumed already open). "
                    "Will hard-wait for the gripper before closing."
                )
            if gripper is not None and args.open_gripper_before_grasp:
                try:
                    _log("GRIPPER: open (after approach, before grasp insert)")
                    p, st, gripper_open_ok = _gripper_open_if_needed(
                        gripper,
                        float(args.gripper_open_pct),
                        float(args.gripper_speed_pct),
                        float(args.gripper_force_pct),
                        wait_timeout_s=float(args.gripper_wait_timeout_s),
                        settle_s=float(args.gripper_settle_s),
                    )
                except Exception as e:  # noqa: BLE001
                    gripper_open_ok = False
                    _log(f"WARNING: gripper open failed: {e!r}")

            # Live handle to the compliant (F/T admittance) insert. While set,
            # the arm is being held compliant at the contact pose so the gripper
            # can close without the arm fighting; it is stopped before the lift.
            compliant_handle: Optional[Any] = None

            def _do_compliant_grasp_insert(target_pose7: List[float]) -> bool:
                """F/T admittance descend-to-contact insert (replaces the
                position-controlled descent). Leaves the arm held compliant at
                the contact pose via ``compliant_handle`` for the gripper close."""
                nonlocal compliant_handle
                from compliant_grasp_execute.compliant_insert import (
                    CompliantInsertParams,
                    run_compliant_insert,
                )

                # Stop any prior compliant hold (e.g. from a previous attempt).
                if compliant_handle is not None:
                    try:
                        compliant_handle.stop()
                    except Exception:  # noqa: BLE001
                        pass
                    compliant_handle = None

                is_left = str(args.arm) == "left"
                cparams = CompliantInsertParams(
                    ft_topic=(args.ft_topic_left if is_left else args.ft_topic_right),
                    calib_path=(
                        (args.ft_calib_left or None) if is_left
                        else (args.ft_calib_right or None)
                    ),
                    contact_force_n=float(args.compliant_contact_force_n),
                    contact_debounce=int(args.compliant_contact_debounce),
                    min_insert_m=float(args.compliant_min_insert_m),
                    stall_window_s=float(args.compliant_stall_window_s),
                    stall_eps_m=float(args.compliant_stall_eps_m),
                    overshoot_m=float(args.compliant_overshoot_m),
                    max_insert_m=float(args.compliant_max_insert_m),
                    insert_speed_mps=float(args.compliant_insert_speed_mps),
                    max_lag_m=float(args.compliant_max_lag_m),
                    control_rate_hz=float(args.compliant_control_rate_hz),
                    max_vel=float(args.compliant_max_vel),
                    max_omega=float(args.compliant_max_omega),
                    loop_period=float(args.compliant_loop_period),
                    trans_lead_time=float(args.compliant_trans_lead_time),
                    otg_p_step=float(args.compliant_otg_p_step),
                    otg_r_step=float(args.compliant_otg_r_step),
                    lateral_stiffness=float(args.compliant_lateral_stiffness),
                    insertion_stiffness=float(args.compliant_insertion_stiffness),
                    soften_threshold=float(args.compliant_soften_threshold),
                    damping_ratio=float(args.compliant_damping_ratio),
                    damping=float(args.compliant_damping),
                    mass=float(args.compliant_mass),
                    hold_stiffness=float(args.compliant_hold_stiffness),
                    filter_alpha=float(args.compliant_filter_alpha),
                    force_deadzone=float(args.compliant_force_deadzone),
                    torque_deadzone=float(args.compliant_torque_deadzone),
                )
                # HYBRID insert: position-control (QP stream) the bulk of the
                # approach->grasp traverse, then hand over to the F/T admittance for
                # only the final short descent-to-contact. The compliant spring is a
                # descend-to-contact controller; asking it to drag the arm through a
                # long, near-horizontal traverse (which happens on a low-tilt grasp,
                # e.g. after the right arm's tilt retry) makes it shake. Position
                # control drives that traverse smoothly and accurately; we keep
                # compliance only for the last cm where the table-collision risk is.
                final_descent_m = float(getattr(args, "compliant_final_descent_m", 0.0))
                pre_grasp_pose7 = target_pose7
                if final_descent_m > 1e-4:
                    cur = xarm.get_tcp_pose(
                        arm=str(args.arm), base_frame=args.waist_frame, timeout=2.0
                    )
                    if cur is not None:
                        cur_xyz = np.asarray(
                            [float(v) for v in cur["translation"]], dtype=float
                        )
                        grasp_xyz = np.asarray(
                            [float(v) for v in target_pose7[:3]], dtype=float
                        )
                        insert_vec = grasp_xyz - cur_xyz
                        full_dist = float(np.linalg.norm(insert_vec))
                        # Only pre-descend if there is meaningfully more traverse than
                        # the compliant span; otherwise the whole move stays compliant.
                        if full_dist > final_descent_m + 0.01:
                            direction = insert_vec / full_dist
                            pre_xyz = grasp_xyz - final_descent_m * direction
                            pre_grasp_pose7 = [
                                float(pre_xyz[0]),
                                float(pre_xyz[1]),
                                float(pre_xyz[2]),
                                *[float(v) for v in target_pose7[3:7]],
                            ]
                            _log(
                                "ARM: HYBRID pre-descent (position QP-stream) -> "
                                f"{(full_dist - final_descent_m) * 100:.1f}cm, leaving "
                                f"final {final_descent_m * 100:.1f}cm for F/T compliant "
                                "descend-to-contact"
                            )
                            _exec_pose_by_backend(
                                "qp_stream",
                                action=action,
                                moveit=moveit,
                                topic_pub=topic_pub,
                                xarm=xarm,
                                arm=args.arm,
                                pose7=pre_grasp_pose7,
                                waist_frame=args.waist_frame,
                                vel_scale=float(args.vel_scale),
                                acc_scale=float(args.acc_scale),
                                label="pre-grasp",
                                use_cartesian_path=False,
                                # FAST profile: this pre-descent stops a safe
                                # `final_descent_m` ABOVE contact, so it runs at
                                # transit-class speed (big OTG step + lookahead +
                                # short hold) rather than the dense, slow grasp-insert
                                # params. Position tolerance stays tight so the
                                # hand-off to the compliant phase is accurate (the
                                # compliant span stays ~final_descent_m).
                                qp_otg_p_step=float(args.qp_transit_p_step),
                                qp_otg_r_step=float(args.qp_transit_r_step),
                                qp_stream_duration=float(args.qp_stream_duration),
                                qp_stream_rate_hz=float(args.qp_stream_rate_hz),
                                # Required kwarg of the shared helper; only used by
                                # the MoveIt branch, which qp_stream does NOT take.
                                moveit_timeout_sec=float(args.moveit_timeout_sec),
                                # Arm already holds the grasp orientation from the
                                # approach; keep it rigid through the descent.
                                keep_current_orientation=True,
                                qp_speed_scale=float(args.qp_speed_scale),
                                qp_hold_sec=float(args.qp_transit_hold_sec),
                                qp_lag_guard_m=float(args.qp_lag_guard_m),
                                qp_stall_timeout_s=float(args.qp_stall_timeout_s),
                                qp_lookahead_m=float(args.qp_lookahead_m),
                                qp_pos_tol_m=float(args.qp_grasp_pos_tol_m),
                                guard=guard,
                            )

                _log("ARM: COMPLIANT insert to grasp pose (F/T admittance descend-to-contact)")
                compliant_handle = run_compliant_insert(
                    xarm=xarm,
                    topic_pub=topic_pub,
                    arm=str(args.arm),
                    waist_frame=args.waist_frame,
                    tcp_frame=_arm_to_frame(str(args.arm)),
                    approach_pose7=pre_grasp_pose7,
                    grasp_pose7=target_pose7,
                    params=cparams,
                    guard=guard,
                    otg_p_step=float(args.compliant_otg_p_step),
                    otg_r_step=float(args.compliant_otg_r_step),
                )
                result["motion"]["compliant_insert"] = compliant_handle.result
                result["motion"]["grasp_tcp_tracking"] = _tcp_tracking_error(
                    xarm, args.arm, args.waist_frame, target_pose7, "grasp"
                )
                _record_joint_limits(guard, args.arm, "grasp", result)
                return bool(compliant_handle.ok)

            def _do_grasp_insert(target_pose7: List[float]) -> bool:
                if bool(getattr(args, "compliant_grasp", False)):
                    return _do_compliant_grasp_insert(target_pose7)
                _log("ARM: insert to grasp pose (QPIK/MoveIt per strategy)")
                # pour_place: QPIK insertion keeps orientation captured after approach.
                use_live_quat = str(args.motion_strategy) in {
                    "auto_hybrid",
                    "qpik",
                    "qp_stream",
                    "qp_all",
                    "auto_stream",
                } and bool(args.grasp_hold_orientation)
                grasp_use_cartesian = bool(args.use_cartesian_grasp) or (
                    str(args.motion_strategy) == "auto_cartesian"
                )
                ok = _exec_pose_by_backend(
                    phase_backends["grasp"],
                    action=action,
                    moveit=moveit,
                    topic_pub=topic_pub,
                    xarm=xarm,
                    arm=args.arm,
                    pose7=target_pose7,
                    waist_frame=args.waist_frame,
                    vel_scale=float(args.vel_scale),
                    acc_scale=float(args.acc_scale),
                    label="grasp",
                    use_cartesian_path=grasp_use_cartesian,
                    qp_otg_p_step=_qp_p_step("grasp"),
                    qp_otg_r_step=_qp_r_step("grasp"),
                    qp_stream_duration=float(args.qp_stream_duration),
                    qp_stream_rate_hz=float(args.qp_stream_rate_hz),
                    keep_current_orientation=use_live_quat,
                    moveit_timeout_sec=float(args.moveit_timeout_sec),
                    stream_closed_loop=bool(args.stream_closed_loop),
                    stream_step_m=float(args.stream_step_m),
                    stream_waypoint_tol_m=float(args.stream_waypoint_tol_m),
                    stream_waypoint_timeout_s=float(args.stream_waypoint_timeout_s),
                    qp_speed_scale=float(args.qp_speed_scale),
                    qp_hold_sec=_qp_hold("grasp"),
                    qp_lag_guard_m=float(args.qp_lag_guard_m),
                    qp_stall_timeout_s=float(args.qp_stall_timeout_s),
                    qp_lookahead_m=_qp_lookahead("grasp"),
                    qp_pos_tol_m=_qp_pos_tol("grasp"),
                    guard=guard,
                )
                # Settle briefly so QPIK/MoveIt finishes moving before we read.
                # Only when the insert reported success: on failure the QP stream
                # has already held to its own deadline at its closest point, so
                # blocking here for the full grasp_reach_timeout_sec waiting for a
                # tolerance the arm cannot meet just makes it look "stuck" and
                # delays the reduced-alignment retry below.
                if ok:
                    _wait_arm_near_pose7(
                        xarm,
                        args.arm,
                        args.waist_frame,
                        target_pose7,
                        pos_tol_m=float(args.grasp_reach_tol_m),
                        timeout_sec=float(args.grasp_reach_timeout_sec),
                    )
                result["motion"]["grasp_tcp_tracking"] = _tcp_tracking_error(
                    xarm, args.arm, args.waist_frame, target_pose7, "grasp"
                )
                _record_joint_limits(guard, args.arm, "grasp", result)
                return ok

            grasp_ok = False
            if approach_ok:
                grasp_ok = _do_grasp_insert(grasp_pose7)
                # Reach retry (insert phase): the approach can reach the standoff
                # point yet the final insert stalls a few cm short because the
                # contorted (tilt + full long-axis yaw) grasp pose pushes a wrist
                # joint to its limit. Same remedy as the approach retry: rebuild
                # with a REDUCED continuous/adaptive yaw, re-approach and re-insert.
                retry_scale = float(getattr(args, "grasp_retry_yaw_scale", 0.5))
                if (
                    not grasp_ok
                    and not bool(result["motion"].get("grasp_reach_retry_used"))
                    and bool(getattr(args, "grasp_reach_retry", False))
                    and bool(getattr(args, "continuous_grasp_orientation", False)
                             or getattr(args, "adaptive_grasp_orientation", False))
                    and 0.0 <= retry_scale < 1.0
                ):
                    _log(
                        "grasp insert failed to reach target; retrying with reduced "
                        f"long-axis alignment (yaw scale {retry_scale:.2f})"
                    )
                    _retry_tilt_scale = (
                        retry_scale
                        if (
                            guard is not None
                            and guard.last_event is not None
                            and bool(getattr(args, "joint_limit_reduce_tilt_on_retry", True))
                        )
                        else 1.0
                    )
                    _eh_g = _eh_build_grasp(retry_scale) if eh_active else None
                    if _eh_g is not None:
                        # Stay on the elbow-high orientation (seed-anchored / top-down);
                        # only the jaw-align yaw (and, for top-down, the tilt) is
                        # backed off here so the wrist stays in range.
                        _log(
                            f"ELBOW-HIGH {eh_mode} retry: yaw scale {retry_scale:.2f}"
                        )
                        grasp_pose7 = _eh_g
                    else:
                        grasp_pose7 = _build_grasp_pose7(
                            det,
                            args,
                            arm,
                            extra_yaw_scale=retry_scale,
                            extra_tilt_scale=_retry_tilt_scale,
                        )
                    approach_pose7, lift_pose7, pre_home_pose7 = _derive_motion_poses(
                        grasp_pose7, args
                    )
                    result["motion"]["approach_pose7"] = approach_pose7
                    result["motion"]["grasp_pose7"] = grasp_pose7
                    result["motion"]["lift_pose7"] = lift_pose7
                    result["motion"]["pre_home_pose7"] = pre_home_pose7
                    result["motion"]["grasp_quat"] = grasp_pose7[3:7]
                    result["motion"]["grasp_reach_retry_used"] = True
                    result["motion"]["grasp_reach_retry_yaw_scale"] = retry_scale
                    approach_ok = _do_approach(approach_pose7)
                    result["motion"]["approach_tcp_tracking"] = _tcp_tracking_error(
                        xarm, args.arm, args.waist_frame, approach_pose7, "approach"
                    )
                    _record_joint_limits(guard, args.arm, "approach", result)
                    if approach_ok:
                        grasp_ok = _do_grasp_insert(grasp_pose7)

            # The grasp insert above overlapped the background connect; now make
            # sure it finished before we try to close.
            if need_gripper and gripper is None and grasp_ok:
                gripper = _ensure_gripper_ready()
            if gripper is not None and args.close_gripper and grasp_ok:
                try:
                    # Compliant insert already stopped at the sensed contact pose
                    # (NOT the planned grasp pose) and is holding the arm compliant
                    # via the admittance handle, so skip the position-convergence
                    # wait; close directly while the arm stays soft.
                    if compliant_handle is None:
                        _wait_arm_near_pose7(
                            xarm,
                            args.arm,
                            args.waist_frame,
                            grasp_pose7,
                            pos_tol_m=float(args.grasp_reach_tol_m),
                            timeout_sec=float(args.grasp_reach_timeout_sec),
                        )
                    if float(args.gripper_close_delay_sec) > 0:
                        time.sleep(float(args.gripper_close_delay_sec))
                    _log("GRIPPER: close at grasp pose (after arm motion, not at approach)")
                    p, st, gripper_close_ok = _gripper_move_and_wait(
                        gripper,
                        float(args.gripper_close_pct),
                        float(args.gripper_speed_pct),
                        float(args.gripper_force_pct),
                        label="close",
                        wait_timeout_s=float(args.gripper_wait_timeout_s),
                        settle_s=float(args.gripper_settle_s),
                    )
                    gripper_final_pct = float(p)
                except Exception as e:  # noqa: BLE001
                    gripper_close_ok = False
                    _log(f"WARNING: gripper close failed: {e!r}")
            elif args.close_gripper and grasp_ok and gripper is None:
                _log(
                    "WARNING: --close-gripper requested but gripper controller is unavailable; "
                    "skipped close"
                )

            # Compliant grasp: the gripper is now closed on the object while the
            # arm is held compliant at the contact pose. Stop the admittance hold
            # and (v1 behaviour) rotate the wrist away from the table BEFORE the
            # usual position-controlled lift. No object-weight re-zero and no
            # compliant lift in this version.
            # Track whether the post-grasp table-clear rotation tilted the wrist
            # nose-up. If so, the lift must KEEP that orientation and rise straight
            # up -- otherwise the lift rotates to an orientation rebuilt from the
            # ORIGINAL grasp quat, undoing the table-clear and sweeping the fingertip
            # (~0.27m out) back down into the table -> end-joint fault.
            did_table_clear = False
            if compliant_handle is not None:
                try:
                    compliant_handle.stop()
                except Exception as e:  # noqa: BLE001
                    _log(f"WARNING: stopping compliant controller failed: {e!r}")
                _table_clear_deg = float(getattr(args, "table_clear_rotate_deg", 0.0))
                if grasp_ok and abs(_table_clear_deg) > 1e-6:
                    from compliant_grasp_execute.compliant_insert import (
                        build_table_clear_pose7,
                    )
                    cur = xarm.get_tcp_pose(
                        arm=args.arm, base_frame=args.waist_frame, timeout=2.0
                    )
                    if cur is not None:
                        cur_pose7 = [
                            float(cur["translation"][0]),
                            float(cur["translation"][1]),
                            float(cur["translation"][2]),
                            *[float(v) for v in cur["rotation"]],
                        ]
                        clear_pose7 = build_table_clear_pose7(cur_pose7, _table_clear_deg)
                        _log(
                            f"ARM: rotate wrist away from table {_table_clear_deg:+.0f}deg "
                            "about waist Y before lift"
                        )
                        _qp_stream_to_pose7(
                            topic_pub,
                            xarm,
                            args.arm,
                            clear_pose7,
                            args.waist_frame,
                            "table-clear",
                            otg_p_step=_qp_p_step("lift"),
                            otg_r_step=_qp_r_step("lift"),
                            stream_duration_sec=float(args.qp_stream_duration),
                            stream_rate_hz=float(args.qp_stream_rate_hz),
                            keep_current_orientation=False,
                            speed_scale=float(args.qp_speed_scale),
                            hold_sec=_qp_hold("lift"),
                            lag_guard_m=float(args.qp_lag_guard_m),
                            stall_timeout_s=float(args.qp_stall_timeout_s),
                            lookahead_m=_qp_lookahead("lift"),
                            pos_tol_m=_qp_pos_tol("lift"),
                            guard=guard,
                        )
                        did_table_clear = True
                    else:
                        _log("WARNING: could not read TCP for table-clear rotate; skipping")
                compliant_handle = None

            lift_ok = True
            if grasp_ok and not args.skip_lift:
                _log("ARM: lift after grasp")
                lift_ok = _exec_pose_by_backend(
                    phase_backends["lift"],
                    action=action,
                    moveit=moveit,
                    topic_pub=topic_pub,
                    xarm=xarm,
                    arm=args.arm,
                    pose7=lift_pose7,
                    waist_frame=args.waist_frame,
                    vel_scale=float(args.vel_scale),
                    acc_scale=float(args.acc_scale),
                    label="lift",
                    use_cartesian_path=bool(args.use_cartesian_lift)
                    or (str(args.motion_strategy) == "auto_cartesian"),
                    qp_otg_p_step=_qp_p_step("lift"),
                    qp_otg_r_step=_qp_r_step("lift"),
                    qp_stream_duration=float(args.qp_stream_duration),
                    qp_stream_rate_hz=float(args.qp_stream_rate_hz),
                    # Rotate to the (tilted) lift orientation during the lift unless
                    # no lift tilt was requested, in which case just hold orientation.
                    # BUT: if the table-clear rotate already tilted the wrist nose-up,
                    # KEEP that orientation and rise straight up. Rotating to a fresh
                    # lift tilt (rebuilt from the original grasp quat) would undo the
                    # table-clear and sweep the fingertip back down into the table.
                    keep_current_orientation=(
                        did_table_clear
                        or abs(float(args.lift_tilt_y_deg)) <= 1e-6
                    ),
                    moveit_timeout_sec=float(args.moveit_timeout_sec),
                    stream_closed_loop=bool(args.stream_closed_loop),
                    stream_step_m=float(args.stream_step_m),
                    stream_waypoint_tol_m=float(args.stream_waypoint_tol_m),
                    stream_waypoint_timeout_s=float(args.stream_waypoint_timeout_s),
                    qp_speed_scale=float(args.qp_speed_scale),
                    qp_hold_sec=_qp_hold("lift"),
                    qp_lag_guard_m=float(args.qp_lag_guard_m),
                    qp_stall_timeout_s=float(args.qp_stall_timeout_s),
                    qp_lookahead_m=_qp_lookahead("lift"),
                    qp_pos_tol_m=_qp_pos_tol("lift"),
                    guard=guard,
                )
            _record_joint_limits(guard, args.arm, "lift", result)
            # Retract straight back to the middle/approach waypoint while still
            # holding the object, so the arm clears the table before the
            # joint-space home move (which would otherwise sweep low).
            retract_ok = None
            if (
                grasp_ok
                and not args.skip_lift
                and bool(args.retract_to_approach)
                and not bool(args.use_pre_home_joints)
            ):
                if bool(args.retract_toward_body):
                    retract_target7 = pre_home_pose7
                    _log(
                        "ARM: retract HIGH + toward body "
                        f"(-X {args.retract_toward_body_m*100:.0f}cm, "
                        f"+Z {(float(args.lift_z)+float(args.retract_extra_z))*100:.0f}cm "
                        "above grasp) so the home move clears the table"
                    )
                else:
                    retract_target7 = approach_pose7
                    _log(
                        "ARM: retract to middle/approach waypoint "
                        "(straight line, holding object)"
                    )
                retract_ok = _exec_pose_by_backend(
                    phase_backends["lift"],
                    action=action,
                    moveit=moveit,
                    topic_pub=topic_pub,
                    xarm=xarm,
                    arm=args.arm,
                    pose7=retract_target7,
                    waist_frame=args.waist_frame,
                    vel_scale=float(args.vel_scale),
                    acc_scale=float(args.acc_scale),
                    label="retract",
                    use_cartesian_path=bool(args.use_cartesian_lift)
                    or (str(args.motion_strategy) == "auto_cartesian"),
                    qp_otg_p_step=_qp_p_step("lift"),
                    qp_otg_r_step=_qp_r_step("lift"),
                    qp_stream_duration=float(args.qp_stream_duration),
                    qp_stream_rate_hz=float(args.qp_stream_rate_hz),
                    keep_current_orientation=True,
                    moveit_timeout_sec=float(args.moveit_timeout_sec),
                    stream_closed_loop=bool(args.stream_closed_loop),
                    stream_step_m=float(args.stream_step_m),
                    stream_waypoint_tol_m=float(args.stream_waypoint_tol_m),
                    stream_waypoint_timeout_s=float(args.stream_waypoint_timeout_s),
                    qp_speed_scale=float(args.qp_speed_scale),
                    qp_hold_sec=_qp_hold("lift"),
                    qp_lag_guard_m=float(args.qp_lag_guard_m),
                    qp_stall_timeout_s=float(args.qp_stall_timeout_s),
                    qp_lookahead_m=_qp_lookahead("lift"),
                    qp_pos_tol_m=_qp_pos_tol("lift"),
                    guard=guard,
                )
                _record_joint_limits(guard, args.arm, "retract", result)

            # Restore the QP solver's joint limits (if we tightened them into the
            # elbow-high window) before any further motion, so subsequent runs /
            # manual control see the true hard limits again.
            _unlock_qp_from_elbow_high()

            # Return to the home posture BEFORE releasing, so the object is only
            # dropped once the arm is well clear of the table.
            home_ok = None
            if args.return_home:
                xarm.xarm_deactivate_all_controller()
                xarm.hardware_arm_enable(True)
                xarm.hardware_arm_mode(3)
                # If we reconfigured into the elbow-high basin, bridge back to the
                # elbow-low family through the transition waypoint FIRST. Going
                # straight from elbow-high to the home/pre-home posture would sweep
                # the forearm through a large arc (table/body collision risk).
                if used_elbow_high:
                    _stage = _elbow_high_stage_joints(args, xarm, str(args.arm), guard)
                    _ready = _elbow_high_ready_joints(args, xarm, str(args.arm), guard)
                    _trans = _elbow_high_transition_joints(
                        args, xarm, str(args.arm), guard, _stage, _ready
                    )
                    _log(
                        "ELBOW-HIGH: returning via transition -> stage waypoints "
                        "before home (unwinds elbow-high -> elbow-low without a "
                        "large sweep)."
                    )
                    _jointspace_move(action, str(args.arm), _trans, "transition(return)")
                    _jointspace_move(action, str(args.arm), _stage, "stage(return)")
                # Keep the two IK families' pre-home waypoints independent:
                # elbow-high uses the original pre_home_* after transition/stage;
                # elbow-low uses its own elbow_low_pre_home_* directly after lift.
                if (
                    grasp_ok
                    and not args.skip_lift
                    and bool(args.use_pre_home_joints)
                    and used_elbow_high
                ):
                    _move_pre_home_joints(action, args)
                elif (
                    grasp_ok
                    and not args.skip_lift
                    and bool(args.use_elbow_low_pre_home_joints)
                    and not used_elbow_high
                ):
                    _move_elbow_low_pre_home_joints(action, args)
                elif not used_elbow_high:
                    _log(
                        "ELBOW-LOW: dedicated pre-home waypoint disabled/unavailable; "
                        "returning HOME directly"
                    )
                home_ok = _return_home(action, args)

            # Release (open) only after we are home / clear of the table.
            released_ok = None
            if args.release_on_finish and gripper is not None:
                try:
                    _log("GRIPPER: release (open) after return home")
                    _, _, released_ok = _gripper_move_and_wait(
                        gripper,
                        float(args.gripper_open_pct),
                        float(args.gripper_speed_pct),
                        float(args.gripper_force_pct),
                        label="release",
                        wait_timeout_s=float(args.gripper_wait_timeout_s),
                        settle_s=float(args.gripper_settle_s),
                    )
                except Exception as e:  # noqa: BLE001
                    released_ok = False
                    _log(f"WARNING: gripper release failed: {e!r}")

            if gripper is not None:
                try:
                    gripper.disconnect()
                except Exception:
                    pass

            result["motion"]["retract_ok"] = retract_ok
            result["motion"]["return_home_ok"] = home_ok
            result["motion"]["released_on_finish_ok"] = released_ok
            result["motion"]["approach_ok"] = approach_ok
            result["motion"]["grasp_ok"] = grasp_ok
            result["motion"]["lift_ok"] = lift_ok
            result["gripper"] = {
                "open_ok": gripper_open_ok,
                "close_ok": gripper_close_ok,
                "final_pos_pct": gripper_final_pct,
                "backend": str(args.gripper_backend),
            }
            result["ok"] = bool(
                approach_ok
                and grasp_ok
                and lift_ok
                and (
                    (not args.close_gripper)
                    or (gripper_close_ok is True)
                )
            )

    finally:
        # Safety net: if an error left the QP joint limits tightened into the
        # elbow-high window, restore the true hard limits before we tear down ROS
        # (otherwise the controller would keep the narrow window for later runs).
        try:
            _eh_orig = eh_qp_limits_orig  # type: ignore[name-defined]
        except NameError:
            _eh_orig = None
        if _eh_orig is not None and rclpy.ok():
            try:
                _restore_qp_joint_limits(xarm, args.arm, *_eh_orig)
            except Exception:  # noqa: BLE001
                pass
        if rclpy.ok():
            rclpy.shutdown()

    safe = _json_safe(result)
    print(json.dumps(safe, indent=2, ensure_ascii=False))
    if args.json_out:
        out = os.path.abspath(args.json_out)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(safe, f, indent=2, ensure_ascii=False)
        _log(f"saved result json: {out}")

    # Handoff to the placement phase: record which arm now holds the object. The
    # object is "held" only if the grasp succeeded, the gripper actually closed,
    # and we did NOT release it at finish (run grasp with --no-release-on-finish
    # when a placement phase will follow).
    if getattr(args, "handoff_out", ""):
        motion = result.get("motion", {}) if isinstance(result, dict) else {}
        gripper_info = result.get("gripper", {}) if isinstance(result, dict) else {}
        grasp_ok = bool(motion.get("grasp_ok"))
        close_ok = gripper_info.get("close_ok") is True
        released = motion.get("released_on_finish_ok") is True
        holding = bool(grasp_ok and close_ok and not released)
        handoff = {
            "arm": result.get("arm"),
            "holding": holding,
            "object": list(args.prompt) if getattr(args, "prompt", None) else [],
            "grasp_ok": grasp_ok,
            "gripper_close_ok": close_ok,
            "released": released,
            "grasp_pose7": motion.get("grasp_pose7"),
            "waist_frame": str(args.waist_frame),
            "timestamp": time.time(),
        }
        try:
            hout = os.path.abspath(args.handoff_out)
            os.makedirs(os.path.dirname(hout) or ".", exist_ok=True)
            with open(hout, "w", encoding="utf-8") as f:
                json.dump(_json_safe(handoff), f, indent=2, ensure_ascii=False)
            _log(
                f"saved handoff: {hout} (arm={handoff['arm']} holding={holding})"
            )
        except Exception as e:  # noqa: BLE001
            _log(f"WARNING: failed to write handoff file: {e!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
