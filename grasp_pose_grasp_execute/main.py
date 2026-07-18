#!/usr/bin/env python3
"""
grasp_pose_grasp_execute
========================

Use detected grasp pose from grasp_pose_generation and move one arm to grasp.

Configuration
-------------
All tunable parameters live in ``config.yaml`` (next to this file) and are
auto-loaded on every run, so the common case is simply::

    python3 -m grasp_pose_grasp_execute.main

Edit ``config.yaml`` to change behaviour (object prompt, tilt, offsets, motion
strategy, ...). Any command-line flag still overrides the file for a one-off,
e.g. grasp a different object once::

    python3 -m grasp_pose_grasp_execute.main --prompt sponge

Precedence: CLI flag > config.yaml > built-in default. Regenerate/refresh the
file (e.g. after adding a new flag) with::

    python3 -m grasp_pose_grasp_execute.main --write-config

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

from grasp_pose_grasp_execute.config_io import (
    add_config_args,
    apply_config_defaults,
    default_config_path,
    maybe_write_config,
)
from grasp_pose_grasp_execute.joint_limits import ARM_JOINT_NAMES, JointLimitGuard
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
# are sent here before the detect->grasp cycle starts.
_LEFT_ARM_HOME_JOINTS = [
    0.7324762344360352,
    0.6585812568664551,
    0.24663496017456055,
    -2.487205982208252,
    -0.898409366607666,
    -0.43663787841796875,
    0.22499990463256836,
]
_RIGHT_ARM_HOME_JOINTS = [
    0.7324762344360352,
    -0.6585812568664551,
    -0.24663496017456055,
    -2.487205982208252,
    0.898409366607666,
    -0.43663787841796875,
    -0.22499990463256836,
]
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


def _build_grasp_pose7(
    det: Dict[str, Any],
    args: argparse.Namespace,
    arm: str,
    extra_yaw_scale: float = 1.0,
    extra_tilt_scale: float = 1.0,
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

    if args.use_fixed_grasp_quat:
        quat = [float(v) for v in _resolve_grasp_quat(args, arm)]
    else:
        quat = [float(v) for v in detected_tcp[3:7]]

    # Human-like tilt: rotate the grasp orientation about the waist-frame Y axis
    # so the gripper noses down instead of staying parallel to the ground.
    tilt_deg = float(getattr(args, "grasp_tilt_y_deg", 0.0)) * float(extra_tilt_scale)
    if abs(tilt_deg) > 1e-6:
        quat_tilted = (
            R.from_euler("y", tilt_deg, degrees=True) * R.from_quat(quat)
        ).as_quat()
        quat = [float(v) for v in quat_tilted]
        scale_str = (
            f" (tilt scale {float(extra_tilt_scale):.2f})"
            if abs(float(extra_tilt_scale) - 1.0) > 1e-6
            else ""
        )
        _log(f"applied grasp tilt {tilt_deg:+.1f}deg about waist Y{scale_str} -> quat={[f'{v:.4f}' for v in quat]}")

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
            max_yaw = float(getattr(args, "continuous_grasp_max_yaw_deg", 90.0))
            extra_yaw = float(np.clip(gain * float(long_yaw), -max_yaw, max_yaw))
            extra_yaw *= float(extra_yaw_scale)
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
                f"about waist Z (gain={gain:.2f}, clamp +/-{max_yaw:.0f}deg{scale_str}) -> quat="
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
    out[0] += dx
    out[1] += dy
    out[2] += dz
    return out


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
    while rclpy.ok() and time.monotonic() < deadline:
        # Joint-limit watchdog: stop BEFORE a joint reaches its hard stop.
        if guard is not None and guard.enabled:
            ev = guard.check_live(arm)
            if ev is not None and ev.get("should_abort"):
                b = ev["breached"][0]
                _log(
                    f"QP-Stream {label}: ABORT joint-limit guard - '{b['joint']}'"
                    f"={b['value']} within {guard.margin:.3f}rad of limit "
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
            if end_dist <= float(pos_tol_m):
                done = True
                break
            if (now - last_improve_t) > settle_s and end_dist <= reach_accept:
                settled = True
                break
        # Carrot: a fixed lookahead ahead of the best-measured progress, on the line.
        carrot_s = min(line_len, s_best + lookahead)
        frac = (carrot_s / line_len) if line_len > 1e-6 else 1.0
        _publish(start_xyz + carrot_s * direction, frac)
        rclpy.spin_once(xarm, timeout_sec=dt)

    # Command the exact final pose so the controller's OTG converges position AND
    # orientation. We do NOT read TCP every hold cycle: a timeout=0.0 lookup right
    # after spin floods the logs with "TF查找超时 ... 超时0.0s" warnings. One read
    # with a real timeout at the end is enough to report the final distance.
    hold_cycles = max(1, int(round(rate * float(hold_sec))))
    for _ in range(hold_cycles):
        if not rclpy.ok():
            break
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
        help="Waist-frame Y (m) boundary for --arm auto. object_y >= boundary "
        "-> left arm, else right arm (+Y is the robot's left, REP-103).",
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
        help="Arm to use when an --arm auto object falls inside the deadband.",
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
        help="Extra X offset on grasp target in waist_yaw_link (m). Overrides pour defaults.",
    )
    p.add_argument("--grasp-y-offset", type=float, default=None)
    p.add_argument("--grasp-z-offset", type=float, default=None)
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
        "it only if your target poses stay well inside the arm's envelope.",
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
        default=0.04,
        help="Max allowed fingertip miss (m) caused by orientation tracking error "
        "at the approach. The fingertip sits ~|tcp_to_tip| along the tool axis, so "
        "if the wrist cannot hold the commanded tilt the fingers miss the object "
        "even though the TCP position converged. Exceeding this triggers the "
        "reduced-tilt retry instead of closing on air. 0 disables the check.",
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


def _select_arm(det: Dict[str, Any], args: argparse.Namespace) -> str:
    """Resolve which arm to grasp with.

    If --arm is left/right, that is honored. For --arm auto, pick by the object's
    Y in the waist frame: +Y is the robot's left, -Y the right (REP-103). A
    deadband around the boundary falls back to --arm-select-default.
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
    if abs(y - boundary) < deadband:
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
            skip_approach = bool(args.skip_approach) or (
                str(args.motion_strategy) == "moveit_direct"
            )
            if skip_approach:
                _log("ARM: skip approach waypoint (direct to grasp)")
            else:
                _log(f"ARM: approach via {phase_backends['approach']}")

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

            def _do_grasp_insert(target_pose7: List[float]) -> bool:
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
                    keep_current_orientation=(abs(float(args.lift_tilt_y_deg)) <= 1e-6),
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

            # Return to the home posture BEFORE releasing, so the object is only
            # dropped once the arm is well clear of the table.
            home_ok = None
            if args.return_home:
                xarm.xarm_deactivate_all_controller()
                xarm.hardware_arm_enable(True)
                xarm.hardware_arm_mode(3)
                if grasp_ok and not args.skip_lift and bool(args.use_pre_home_joints):
                    _move_pre_home_joints(action, args)
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

