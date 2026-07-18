#!/usr/bin/env python3
"""
grasp_pose_place_execute
========================

python3 -m grasp_pose_place_execute.main

Placement phase. After ``grasp_pose_grasp_execute`` grasps an object and returns
to the home posture **still holding it** (run the grasp with
``--no-release-on-finish``), this module moves the SAME arm that is holding the
object from home to a target placement pose and releases it.

Communication with the grasp phase
----------------------------------
The grasp phase writes a small handoff file (``--handoff-out``, default
``/tmp/grasp_handoff.json``) recording which arm holds the object::

    {"arm": "left", "holding": true, "object": ["banana"], ...}

This module reads that file (``--handoff-in``) to decide which gripper to use.
``--arm left|right`` overrides the handoff.

Motion (mirrors the grasp path, but with a FIXED orientation)
-------------------------------------------------------------
1. (optional) move the holding arm to its home joint posture (known start)
2. approach / middle waypoint: backed off along the tool +Z approach axis
3. place pose: the target placement pose, raised a little (``--place-z-clearance``)
   so the gripper releases slightly above the table and does not collide with it
4. release (open) the gripper
5. lift straight up (``--lift-z`` + ``--lift-tilt-y-deg``) to clear the table
6. retract straight back to the middle waypoint
7. return to the home joint posture

Placement target
----------------
If ``--place-x/--place-y/--place-z`` are omitted, the target defaults to the
ORIGINAL detected grasp position from the handoff (``grasp_pose7``) -- i.e. the
object is placed back exactly where it was picked (known-reachable). When placing
back at the grasp pose the original grasp orientation is reused; pass
``--place-fixed-quat`` to use the arm's fixed grasp quat instead. Providing all
three ``--place-*`` coordinates overrides the target and uses a fixed orientation
(``--place-quat`` to override) with an optional nose-down tilt.

Configuration
-------------
All tunable parameters live in ``config.yaml`` (next to this file) and are
auto-loaded, so the common case (place back where it was grasped) is simply::

    python3 -m grasp_pose_place_execute.main

Edit ``config.yaml`` to change behaviour; any CLI flag overrides it for a
one-off. Precedence: CLI flag > config.yaml > built-in default. Refresh the file
with ``--write-config``; ignore it with ``--config ''``.

Example (explicit placement target, overriding the config)
----------------------------------------------------------
python3 -m grasp_pose_place_execute.main \
  --place-x 0.55 --place-y -0.20 --place-z 0.05 --place-tilt-y-deg 45
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np
import rclpy

from xarm_sdk import ActionCall, MoveitCall, TopicPublisher, XARM_manager

from grasp_pose_grasp_execute.config_io import (
    add_config_args,
    apply_config_defaults,
    default_config_path,
    maybe_write_config,
)

# Reuse the grasp phase's motion + gripper primitives so the two phases share a
# single, tested implementation.
from grasp_pose_grasp_execute.main import (
    _LEFT_ARM_HOME_JOINTS,
    _RIGHT_ARM_HOME_JOINTS,
    _build_gripper,
    _default_grasp_quat,
    _exec_pose_by_backend,
    _gripper_move_and_wait,
    _json_safe,
    _resolve_phase_backends,
    _tcp_tracking_error,
    _wait_arm_near_pose7,
)

_TAG = "[PLACE-EXEC]"


def _log(msg: str) -> None:
    print(f"{_TAG} {msg}", flush=True)


def _arm_to_frame(arm: str) -> str:
    return "left_tcp_link" if arm == "left" else "right_tcp_link"


def _load_handoff(args: argparse.Namespace) -> Dict[str, Any]:
    """Read the grasp handoff file if present, else return {}."""
    path = str(args.handoff_in or "")
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return dict(json.load(f))
    except Exception as e:  # noqa: BLE001
        _log(f"WARNING: failed to read handoff {path!r}: {e!r}")
        return {}


def _resolve_arm(args: argparse.Namespace, handoff: Dict[str, Any]) -> str:
    """Resolve which arm holds the object: --arm overrides, else the handoff."""
    requested = str(args.arm).strip().lower()
    if requested in ("left", "right"):
        _log(f"ARM: using --arm '{requested}' (overrides handoff)")
        return requested

    path = str(args.handoff_in or "")
    if not handoff:
        raise SystemExit(
            f"{_TAG} no --arm given and handoff file not found/readable: {path!r}. "
            "Run the grasp phase first (it writes --handoff-out), or pass --arm."
        )
    arm = str(handoff.get("arm") or "").strip().lower()
    holding = bool(handoff.get("holding"))
    obj = handoff.get("object")
    _log(f"ARM: handoff {path} -> arm='{arm}' holding={holding} object={obj}")
    if arm not in ("left", "right"):
        raise SystemExit(
            f"{_TAG} handoff file {path!r} has no valid 'arm' (got {arm!r})."
        )
    if not holding and bool(args.require_holding):
        raise SystemExit(
            f"{_TAG} handoff says the gripper is NOT holding an object "
            "(holding=false). Run grasp with --no-release-on-finish, or pass "
            "--no-require-holding to place anyway."
        )
    return arm


def _build_place_pose7(
    args: argparse.Namespace, arm: str, handoff: Dict[str, Any]
) -> List[float]:
    """Build the TCP placement pose7 in the waist frame, with a fixed orientation.

    Target position:
      * if --place-x/--place-y/--place-z are given, use them, and
      * otherwise fall back to the ORIGINAL detected grasp position from the
        handoff (``grasp_pose7``) -- i.e. place the object back where it was
        picked. This is the default convenience target for testing.

    The release point is raised by ``--place-z-clearance`` above the target so the
    object is dropped a short distance and the fingers clear the table.
    """
    from scipy.spatial.transform import Rotation as R

    explicit_xyz = all(
        v is not None for v in (args.place_x, args.place_y, args.place_z)
    )
    grasp_pose7 = handoff.get("grasp_pose7") if isinstance(handoff, dict) else None
    use_grasp_pose = (not explicit_xyz) and isinstance(grasp_pose7, (list, tuple)) and len(grasp_pose7) >= 7

    if explicit_xyz:
        base_xyz = [float(args.place_x), float(args.place_y), float(args.place_z)]
        src = "args --place-x/y/z"
    elif use_grasp_pose:
        base_xyz = [float(v) for v in grasp_pose7[:3]]
        src = "handoff grasp_pose7 (original detected grasp position)"
    else:
        raise SystemExit(
            f"{_TAG} no placement target: pass --place-x/--place-y/--place-z, or "
            "ensure the handoff has 'grasp_pose7' (run the grasp phase first)."
        )

    # Orientation: a FIXED placement orientation is used by default
    # (--place-fixed-quat, on) so the placement never depends on how the object
    # was grasped -- the arm always moves from home to the same, easy-to-reach
    # orientation. Explicit --place-quat wins; else the arm's calibrated grasp
    # quat (+ --place-tilt-y-deg). Only when --no-place-fixed-quat is set AND we
    # are placing back at the handoff grasp pose do we reuse the original
    # (known-reachable) grasp orientation.
    if getattr(args, "place_quat", None):
        quat = [float(v) for v in args.place_quat]
        apply_tilt = True
    elif use_grasp_pose and not bool(args.place_fixed_quat):
        quat = [float(v) for v in grasp_pose7[3:7]]
        apply_tilt = False  # grasp_pose7 already includes the grasp tilt
    else:
        quat = [float(v) for v in _default_grasp_quat(arm)]
        apply_tilt = True

    tilt_deg = float(args.place_tilt_y_deg)
    if apply_tilt and abs(tilt_deg) > 1e-6:
        quat = [
            float(v)
            for v in (R.from_euler("y", tilt_deg, degrees=True) * R.from_quat(quat)).as_quat()
        ]
        _log(
            f"applied place tilt {tilt_deg:+.1f}deg about waist Y -> "
            f"quat={[f'{v:.4f}' for v in quat]}"
        )

    x = base_xyz[0] + float(args.place_x_offset)
    y = base_xyz[1] + float(args.place_y_offset)
    z = base_xyz[2] + float(args.place_z_offset) + float(args.place_z_clearance)
    _log(
        f"place TCP target (waist) from {src}: x={x:.4f} y={y:.4f} z={z:.4f} "
        f"(+{float(args.place_z_clearance)*100:.0f}cm release clearance)"
    )
    return [x, y, z, float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]


def _warmup_tcp(
    xarm: XARM_manager,
    arm: str,
    waist_frame: str,
    *,
    attempts: int = 12,
    per_timeout: float = 0.5,
) -> Optional[Dict[str, Any]]:
    """Prime the TF buffer so the first streamed motion can read the TCP pose.

    The grasp phase warms TF during detection; placement jumps straight to motion,
    so a cold TransformListener makes the very first get_tcp_pose() time out
    (``TF查找超时 waist_yaw_link <- *_tcp_link``) and the approach aborts before
    moving. Retry with spinning until the transform is available.
    """
    for i in range(max(1, int(attempts))):
        cur = xarm.get_tcp_pose(arm=arm, base_frame=waist_frame, timeout=per_timeout)
        if cur is not None:
            _log(f"TF warmup ok: read TCP after {i + 1} attempt(s)")
            return cur
        for _ in range(10):
            rclpy.spin_once(xarm, timeout_sec=0.05)
    _log("WARNING: TF warmup failed; TCP still unreadable (approach will likely fail)")
    return None


def _derive_approach_pose7(place_pose7: List[float], args: argparse.Namespace) -> List[float]:
    """Middle/approach waypoint: backed off from the place pose along tool +Z."""
    from scipy.spatial.transform import Rotation as R

    approach = list(place_pose7)
    if bool(args.approach_along_axis):
        tool_z = R.from_quat([float(v) for v in place_pose7[3:7]]).as_matrix()[:, 2]
        xyz = np.asarray(place_pose7[:3], dtype=float) + float(args.approach_dist) * np.asarray(
            tool_z, dtype=float
        )
        # Add pure-vertical clearance on top of the along-axis backoff. At a 45deg
        # tilt the along-axis backoff only gains ~cos(45)*dist of height, which
        # leaves the standoff too close to the table; raise it straight up so the
        # intermediate (and the descent it starts from) clears the table.
        xyz[2] += float(args.approach_extra_z)
        approach[0], approach[1], approach[2] = float(xyz[0]), float(xyz[1]), float(xyz[2])
        _log(
            f"approach along place axis: dist={args.approach_dist:.3f}m "
            f"extra_z={args.approach_extra_z:.3f}m "
            f"tool_z=[{tool_z[0]:.3f},{tool_z[1]:.3f},{tool_z[2]:.3f}] "
            f"approach_xyz={[f'{v:.4f}' for v in approach[:3]]}"
        )
    else:
        approach[0] += float(args.approach_dx)
        approach[1] += float(args.approach_dy)
        approach[2] += float(args.approach_dz)
    return approach


def _approach_reorient_multihop(
    xarm: XARM_manager,
    args: argparse.Namespace,
    arm: str,
    approach_pose7: List[float],
    exec_fn: Any,
) -> bool:
    """Approach the standoff, splitting a large reorientation into several QP hops.

    The place uses a FIXED orientation (grasp quat + ``--place-tilt-y-deg``) that
    can sit ~70-80deg away from the home tool orientation. Handing the QP
    controller that full orientation in one shot exceeds its tracking limit
    (``目标超出跟踪限, dis_ori``), so it refuses to rotate and the approach stalls
    far short. Instead SLERP the orientation (and LERP the position) from the
    CURRENT tool pose to the approach pose in N steps so each hop's orientation
    change stays under ``--approach-max-reorient-deg`` -- each hop is individually
    within the tracking limit. Falls back to a single move if TCP is unreadable,
    the cap is disabled (<=0), or the gap already fits in one hop.
    """
    from scipy.spatial.transform import Rotation as R, Slerp

    max_deg = float(getattr(args, "approach_max_reorient_deg", 35.0))
    cur = xarm.get_tcp_pose(arm=arm, base_frame=args.waist_frame, timeout=2.0)
    if cur is None or max_deg <= 0.0:
        return exec_fn("approach", approach_pose7, keep_orientation=False)

    cur_xyz = np.asarray([float(v) for v in cur["translation"]], dtype=float)
    cur_quat = [float(v) for v in cur["rotation"]]
    tgt_xyz = np.asarray([float(v) for v in approach_pose7[:3]], dtype=float)
    tgt_quat = [float(v) for v in approach_pose7[3:7]]

    r_key = R.from_quat([cur_quat, tgt_quat])
    ang = float((r_key[1] * r_key[0].inv()).magnitude()) * 180.0 / np.pi
    if ang <= max_deg + 1e-3:
        return exec_fn("approach", approach_pose7, keep_orientation=False)

    n = int(np.ceil(ang / max_deg))
    _log(
        f"approach reorient {ang:.1f}deg > {max_deg:.0f}deg/hop -> split into "
        f"{n} QP hops (keeps each under the tracking limit)"
    )
    slerp = Slerp([0.0, 1.0], r_key)
    ok = True
    for i in range(1, n + 1):
        t = float(i) / float(n)
        q_i = slerp([t])[0].as_quat()
        xyz_i = (1.0 - t) * cur_xyz + t * tgt_xyz
        pose_i = [
            float(xyz_i[0]), float(xyz_i[1]), float(xyz_i[2]),
            float(q_i[0]), float(q_i[1]), float(q_i[2]), float(q_i[3]),
        ]
        _log(f"approach hop {i}/{n} (t={t:.2f})")
        ok = exec_fn("approach", pose_i, keep_orientation=False)
    return ok


def _derive_lift_pose7(place_pose7: List[float], args: argparse.Namespace) -> List[float]:
    """Lift waypoint used AFTER releasing: raise straight up by ``--lift-z`` and
    apply ``--lift-tilt-y-deg`` about waist Y so the gripper clears the table
    before the arm retracts / returns home (mirrors the grasp phase's lift)."""
    from scipy.spatial.transform import Rotation as R

    lift = list(place_pose7)
    lift[2] += float(args.lift_z)
    lift_tilt_deg = float(args.lift_tilt_y_deg)
    if abs(lift_tilt_deg) > 1e-6:
        lq = (
            R.from_euler("y", lift_tilt_deg, degrees=True)
            * R.from_quat([float(v) for v in place_pose7[3:7]])
        ).as_quat()
        lift[3:7] = [float(v) for v in lq]
        _log(
            f"applied lift tilt {lift_tilt_deg:+.1f}deg about waist Y -> "
            f"lift quat={[f'{v:.4f}' for v in lift[3:7]]}"
        )
    _log(
        f"lift after release: +{float(args.lift_z)*100:.0f}cm up -> "
        f"lift_xyz={[f'{v:.4f}' for v in lift[:3]]}"
    )
    return lift


def _jointspace_to(
    action: ActionCall, arm: str, joints: List[float], label: str
) -> bool:
    """Blocking jointspace move of ``arm`` to ``joints`` (7), with logging."""
    _log(f"{label}: {arm} -> joints={[f'{v:.3f}' for v in joints]}")
    try:
        if arm == "left":
            res = action.jointspace_arm_L_controller([float(v) for v in joints])
        else:
            res = action.jointspace_arm_R_controller([float(v) for v in joints])
        _log(f"{label} result: {res}")
        return True
    except Exception as e:  # noqa: BLE001
        _log(f"WARNING: {label} failed: {e!r}")
        return False


def _move_home(action: ActionCall, args: argparse.Namespace, arm: str, label: str) -> bool:
    """Move the given arm to its home joint posture."""
    joints = list(args.home_left_joints if arm == "left" else args.home_right_joints)
    return _jointspace_to(action, arm, joints, label)


def _place_mid_joints(args: argparse.Namespace, arm: str) -> Optional[List[float]]:
    """Fixed intermediate placement posture (7 joints) for ``arm``, or None.

    When set, the arm jointspace-moves to this ONE fixed, safe posture before the
    QP descent to the drop, and jointspace-returns to it after releasing (instead
    of the QP lift/retract, which loses tracking at a far/awkward place pose and
    drives the arm into the table). Empty/!=7 -> fall back to the along-axis
    derived approach waypoint.
    """
    raw = list(
        args.place_mid_left_joints if arm == "left" else args.place_mid_right_joints
    )
    if len(raw) == 7:
        return [float(v) for v in raw]
    return None


def _add_gripper_args(p: argparse.ArgumentParser) -> None:
    """Gripper args (names match grasp_pose_grasp_execute so _build_gripper works)."""
    p.add_argument(
        "--release-gripper",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open the gripper to release the object at the place pose (default on).",
    )
    p.add_argument(
        "--gripper-backend",
        choices=["modbus_rtu", "zmq"],
        default="modbus_rtu",
    )
    p.add_argument("--gripper-wait-timeout-s", type=float, default=8.0)
    p.add_argument("--gripper-settle-s", type=float, default=2.5)
    p.add_argument("--gripper-recv-timeout-ms", type=int, default=500)
    p.add_argument("--gripper-open-pct", type=float, default=0.0, help="Release/open percent.")
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
    )
    p.add_argument("--gripper-force-activate", action="store_true")
    p.add_argument(
        "--gripper-async-connect",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Connect to the gripper in a background thread so its slow serial open "
        "overlaps the arm approach instead of stalling the sequence.",
    )
    p.add_argument(
        "--gripper-port-cache",
        type=str,
        default="/tmp/robotiq_gripper_port.txt",
    )


def _add_motion_args(p: argparse.ArgumentParser) -> None:
    """Motion/QP args (names + defaults match grasp_pose_grasp_execute)."""
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
        default="qp_all",
    )
    p.add_argument("--vel-scale", type=float, default=0.8)
    p.add_argument("--acc-scale", type=float, default=0.8)
    p.add_argument("--moveit-timeout-sec", type=float, default=120.0)
    p.add_argument("--use-cartesian-approach", action="store_true")
    p.add_argument("--use-cartesian-lift", action="store_true")
    p.add_argument("--use-cartesian-grasp", action="store_true")
    p.add_argument("--qp-otg-p-step", type=float, default=0.0015)
    p.add_argument("--qp-otg-r-step", type=float, default=0.001)
    p.add_argument("--qp-stream-duration", type=float, default=1.5)
    p.add_argument("--qp-stream-rate-hz", type=float, default=100.0)
    p.add_argument("--stream-closed-loop", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--stream-step-m", type=float, default=0.005)
    p.add_argument("--stream-waypoint-tol-m", type=float, default=0.004)
    p.add_argument("--stream-waypoint-timeout-s", type=float, default=0.6)
    p.add_argument("--qp-transit-p-step", type=float, default=0.014)
    p.add_argument("--qp-transit-r-step", type=float, default=0.03)
    p.add_argument("--qp-speed-scale", type=float, default=0.7)
    p.add_argument("--qp-stream-hold-sec", type=float, default=0.4)
    p.add_argument("--qp-lag-guard-m", type=float, default=0.04)
    p.add_argument("--qp-stall-timeout-s", type=float, default=2.0)
    p.add_argument("--qp-lookahead-m", type=float, default=0.08)
    p.add_argument("--qp-transit-lookahead-m", type=float, default=0.13)
    p.add_argument("--qp-grasp-lookahead-m", type=float, default=0.03)
    p.add_argument("--qp-transit-hold-sec", type=float, default=0.1)
    p.add_argument("--qp-transit-pos-tol-m", type=float, default=0.02)
    p.add_argument("--qp-grasp-pos-tol-m", type=float, default=0.01)
    p.add_argument("--qp-transit-raise-z", type=float, default=0.0)


def _add_compliant_args(p: argparse.ArgumentParser) -> None:
    """F/T admittance compliant place-descent args.

    Mirrors ``compliant_grasp_execute`` (whose ``run_compliant_insert`` and F/T
    calibration this phase reuses) so the final set-down is compliant: the arm
    descends softly along the place axis and STOPS the instant the object /
    gripper touches the table, instead of position-driving into it and faulting
    a joint. Defaults match the proven grasp descent profile.
    """
    p.add_argument(
        "--compliant-place",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the F/T admittance compliant descend-to-contact for the final "
        "place set-down (soft along the place axis, rigid wrist, stops on table "
        "contact). Reuses compliant_grasp_execute's admittance stack + F/T "
        "calibration. --no-compliant-place falls back to the plain QP descent.",
    )
    p.add_argument(
        "--compliant-place-extra-drop-m",
        type=float,
        default=0.05,
        help="Deepen the compliant place target this far PAST the nominal place "
        "pose (along the descent axis) so the object is actually driven down to "
        "the table and the F/T contact stops it at the true surface (instead of "
        "stopping in the air at --place-z-clearance). Compliance keeps the "
        "contact gentle. Set 0 to descend only to the nominal place pose.",
    )
    p.add_argument(
        "--compliant-final-descent-m",
        type=float,
        default=0.06,
        help="HYBRID: position-control (QP) the bulk of the standoff->place "
        "traverse and hand over to the compliant admittance only for the FINAL "
        "this-many metres near the table (where the collision risk is). 0 = whole "
        "descent compliant.",
    )
    p.add_argument("--ft-topic-left", default="/arm_6dof_left",
                   help="Wrench topic for the LEFT wrist F/T sensor.")
    p.add_argument("--ft-topic-right", default="/arm_6dof_right",
                   help="Wrench topic for the RIGHT wrist F/T sensor.")
    p.add_argument(
        "--ft-calib-left",
        default="",
        help="LEFT F/T calibration JSON (default: compliant_grasp_execute/"
        "ft_calibration/ft_calibration_left.json).",
    )
    p.add_argument(
        "--ft-calib-right",
        default="",
        help="RIGHT F/T calibration JSON (default: compliant_grasp_execute/"
        "ft_calibration/ft_calibration_right.json).",
    )
    p.add_argument("--compliant-contact-force-n", type=float, default=1.5,
                   help="Resisting force along the place axis (N) counted as contact.")
    p.add_argument("--compliant-stall-window-s", type=float, default=0.7)
    p.add_argument("--compliant-stall-eps-m", type=float, default=0.004)
    p.add_argument("--compliant-contact-debounce", type=int, default=3)
    p.add_argument("--compliant-min-insert-m", type=float, default=0.008)
    p.add_argument("--compliant-overshoot-m", type=float, default=0.02)
    p.add_argument("--compliant-max-insert-m", type=float, default=0.25)
    p.add_argument("--compliant-insert-speed-mps", type=float, default=0.035)
    p.add_argument("--compliant-max-lag-m", type=float, default=0.035)
    p.add_argument("--compliant-lateral-stiffness", type=float, default=40.0)
    p.add_argument("--compliant-insertion-stiffness", type=float, default=20.0)
    p.add_argument("--compliant-soften-threshold", type=float, default=0.30)
    p.add_argument("--compliant-damping-ratio", type=float, default=1.4)
    p.add_argument("--compliant-hold-stiffness", type=float, default=150.0)
    p.add_argument("--compliant-damping", type=float, default=3.0)
    p.add_argument("--compliant-mass", type=float, default=0.1)
    p.add_argument("--compliant-filter-alpha", type=float, default=0.35)
    p.add_argument("--compliant-force-deadzone", type=float, default=0.8)
    p.add_argument("--compliant-torque-deadzone", type=float, default=0.08)
    p.add_argument("--compliant-control-rate-hz", type=float, default=100.0)
    p.add_argument("--compliant-max-vel", type=float, default=0.20)
    p.add_argument("--compliant-max-omega", type=float, default=0.5)
    p.add_argument("--compliant-otg-p-step", type=float, default=0.008)
    p.add_argument("--compliant-otg-r-step", type=float, default=0.005)
    p.add_argument("--compliant-loop-period", type=float, default=0.004)
    p.add_argument("--compliant-trans-lead-time", type=float, default=0.12)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Place a held object at a target pose.")

    # which arm holds the object
    p.add_argument(
        "--arm",
        choices=["left", "right", "auto"],
        default="auto",
        help="Arm holding the object. 'auto' reads it from --handoff-in.",
    )
    p.add_argument(
        "--handoff-in",
        default="/tmp/grasp_handoff.json",
        help="Handoff file written by the grasp phase (--handoff-out there).",
    )
    p.add_argument(
        "--require-holding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Abort if the handoff says the gripper is not holding an object.",
    )

    # placement target (TCP pose, waist_yaw_link frame). If omitted, defaults to
    # the original detected grasp position from the handoff (place back where
    # picked). Provide all three to override.
    p.add_argument("--place-x", type=float, default=None, help="Target TCP X in waist frame (m).")
    p.add_argument("--place-y", type=float, default=None, help="Target TCP Y in waist frame (m).")
    p.add_argument("--place-z", type=float, default=None, help="Target TCP Z in waist frame (m).")
    p.add_argument(
        "--place-z-clearance",
        type=float,
        default=0.05,
        help="Release the object this much ABOVE the target (grasp) Z so the "
        "gripper clears the table. The place uses a FIXED tilt orientation "
        "(--place-tilt-y-deg) that differs from how the object was grasped, so at "
        "45deg the object hangs several cm below the TCP; this clearance (plus tilt "
        "overshoot / z-undershoot margin) keeps the low end off the table. Lower it "
        "for a gentler drop if the fingers stay clear.",
    )
    p.add_argument("--place-x-offset", type=float, default=0.0)
    p.add_argument("--place-y-offset", type=float, default=0.0)
    p.add_argument("--place-z-offset", type=float, default=0.0)

    # fixed placement orientation (object orientation is irrelevant)
    p.add_argument(
        "--place-quat",
        type=float,
        nargs=4,
        metavar=("QX", "QY", "QZ", "QW"),
        default=None,
        help="Fixed placement TCP quaternion. Default = arm's calibrated grasp quat.",
    )
    p.add_argument(
        "--place-tilt-y-deg",
        type=float,
        default=45.0,
        help="Nose-down tilt about waist Y applied to the fixed orientation "
        "(matches the grasp posture; 0 = level).",
    )
    p.add_argument(
        "--place-fixed-quat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a FIXED placement orientation (the arm's calibrated grasp quat "
        "+ --place-tilt-y-deg, or --place-quat if given) for EVERY placement, "
        "instead of reusing the original grasp orientation from the handoff. "
        "Default on: placement no longer depends on how the object was grasped, "
        "so the arm always moves from home to the same, easy-to-reach place "
        "orientation. Pass --no-place-fixed-quat to reuse the grasp orientation.",
    )

    # approach / middle waypoint
    p.add_argument(
        "--approach-along-axis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Place the middle waypoint by backing off --approach-dist along the "
        "place approach axis (tool +Z). Else use waist-frame --approach-dx/dy/dz.",
    )
    p.add_argument("--approach-dist", type=float, default=0.12)
    p.add_argument(
        "--approach-extra-z",
        type=float,
        default=0.05,
        help="Extra pure-vertical clearance (m) added to the along-axis standoff "
        "so the intermediate pre-insert pose sits above the table. Kept small "
        "(the compliant F/T set-down stops on table contact, so the standoff does "
        "not need to be high): a tall+tilted+outreached standoff drives a wrist/"
        "elbow joint toward its limit.",
    )
    p.add_argument("--approach-dx", type=float, default=-0.10)
    p.add_argument("--approach-dy", type=float, default=0.0)
    p.add_argument("--approach-dz", type=float, default=0.0)
    p.add_argument(
        "--approach-max-reorient-deg",
        type=float,
        default=35.0,
        help="Max tool reorientation (deg) per QP approach hop. The placement "
        "uses a FIXED orientation (grasp quat + --place-tilt-y-deg) that can sit "
        "~70-80deg from the home posture; the QP-stream approach only ramps "
        "POSITION and rejects an orientation target that far away (目标超出跟踪限). "
        "If the gap exceeds this cap, the approach is split into ceil(gap/cap) "
        "SLERP hops so each stays under the QP tracking limit. Set 0 to disable "
        "(single-shot approach).",
    )
    # lift after release (clear the table before retract / return home)
    p.add_argument(
        "--lift-after-release",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After releasing, lift straight up (with --lift-tilt-y-deg) to clear "
        "the table before retracting / returning home (mirrors the grasp lift).",
    )
    p.add_argument(
        "--lift-z",
        type=float,
        default=0.12,
        help="Vertical distance to lift the gripper after releasing (m).",
    )
    p.add_argument(
        "--lift-tilt-y-deg",
        type=float,
        default=-15.0,
        help="Tilt about waist Y applied during the post-release lift "
        "(matches the grasp phase; 0 = keep the place orientation).",
    )
    p.add_argument(
        "--retract-to-approach",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After releasing, retract straight back to the middle waypoint.",
    )
    p.add_argument(
        "--post-release-tilt-up-deg",
        type=float,
        default=15.0,
        help="After opening the gripper, rotate the tool NOSE-UP this many degrees "
        "about waist Y (in place, same axis as the place tilt but opposite sign) "
        "BEFORE the retract. At the ~45deg nose-down touchdown pose the fingertips "
        "sit near the table, so opening then translating can scrape it; swinging "
        "the low fingertip up first clears the surface. 0 = disable.",
    )

    # finish behaviour
    p.add_argument(
        "--start-home",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Move the holding arm to its home posture before the placement move "
        "(known start; the arm should already be there after the grasp phase).",
    )
    p.add_argument(
        "--return-home",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After releasing, move the arm back to the home joint posture.",
    )

    p.add_argument("--waist-frame", default="waist_yaw_link")
    p.add_argument("--place-reach-tol-m", type=float, default=0.03)
    p.add_argument("--place-reach-timeout-sec", type=float, default=20.0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--json-out", default="")
    p.add_argument(
        "--handoff-out",
        default="/tmp/grasp_handoff.json",
        help="Update the handoff after placing (sets holding=false). Empty to skip.",
    )
    p.add_argument(
        "--home-left-joints",
        type=float,
        nargs=7,
        default=list(_LEFT_ARM_HOME_JOINTS),
    )
    p.add_argument(
        "--home-right-joints",
        type=float,
        nargs=7,
        default=list(_RIGHT_ARM_HOME_JOINTS),
    )
    p.add_argument(
        "--place-mid-left-joints",
        type=float,
        nargs="*",
        default=[],
        help="OPTIONAL fixed intermediate placement posture (rad, 7 joints) for "
        "the LEFT arm. If set, the arm jointspace-moves here (instead of the 45deg "
        "pre-insert standoff) before the descent and jointspace-returns here after "
        "releasing. Empty (default) = use the grasp-style 45deg-tilt standoff "
        "above the drop as the intermediate.",
    )
    p.add_argument(
        "--place-mid-right-joints",
        type=float,
        nargs="*",
        default=[],
        help="Optional fixed intermediate placement posture (rad, 7 joints) for "
        "the RIGHT arm (mirror of --place-mid-left-joints).",
    )

    _add_gripper_args(p)
    _add_motion_args(p)
    _add_compliant_args(p)
    add_config_args(p, default_config_path(__file__))
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    _, config_path = apply_config_defaults(parser, argv)
    args = parser.parse_args(argv)
    if maybe_write_config(parser, args, config_path):
        return 0
    handoff = _load_handoff(args)
    arm = _resolve_arm(args, handoff)
    args.arm = arm  # so the shared gripper port resolver picks the right by-id port

    if not rclpy.ok():
        rclpy.init()
    xarm = XARM_manager()
    action = ActionCall(xarm)
    moveit = MoveitCall(xarm)
    topic_pub = TopicPublisher(xarm)

    place_pose7 = _build_place_pose7(args, arm, handoff)
    approach_pose7 = _derive_approach_pose7(place_pose7, args)

    result: Dict[str, Any] = {
        "arm": arm,
        "motion_strategy": args.motion_strategy,
        "dry_run": bool(args.dry_run),
        "motion": {
            "place_pose7": place_pose7,
            "approach_pose7": approach_pose7,
        },
    }
    released_ok: Optional[bool] = None

    # qp_all: big transit moves (approach/retract) use the fast transit step; the
    # place insert stays at the fine grasp step. Mirrors the grasp phase wiring.
    _qp_all = str(args.motion_strategy) == "qp_all"

    def _transit(phase: str) -> bool:
        return _qp_all and phase in ("approach", "retract")

    def _qp_p_step(phase: str) -> float:
        return float(args.qp_transit_p_step) if _transit(phase) else float(args.qp_otg_p_step)

    def _qp_r_step(phase: str) -> float:
        return float(args.qp_transit_r_step) if _transit(phase) else float(args.qp_otg_r_step)

    def _qp_lookahead(phase: str) -> float:
        if phase == "place":
            return float(args.qp_grasp_lookahead_m)
        if _transit(phase):
            return float(args.qp_transit_lookahead_m)
        return float(args.qp_lookahead_m)

    def _qp_hold(phase: str) -> float:
        return float(args.qp_transit_hold_sec) if _transit(phase) else float(args.qp_stream_hold_sec)

    def _qp_pos_tol(phase: str) -> float:
        return float(args.qp_transit_pos_tol_m) if _transit(phase) else float(args.qp_grasp_pos_tol_m)

    def _exec(phase: str, pose7: List[float], *, keep_orientation: bool) -> bool:
        # "place" -> grasp backend; "lift"/"retract" -> lift backend; else approach.
        backend_phase = (
            "grasp" if phase == "place"
            else "lift" if phase in ("retract", "lift")
            else "approach"
        )
        phase_backends = _resolve_phase_backends(str(args.motion_strategy))
        return _exec_pose_by_backend(
            phase_backends[backend_phase],
            action=action,
            moveit=moveit,
            topic_pub=topic_pub,
            xarm=xarm,
            arm=arm,
            pose7=pose7,
            waist_frame=args.waist_frame,
            vel_scale=float(args.vel_scale),
            acc_scale=float(args.acc_scale),
            label=phase,
            use_cartesian_path=(
                bool(args.use_cartesian_approach)
                if phase == "approach"
                else bool(args.use_cartesian_grasp)
                if phase == "place"
                else bool(args.use_cartesian_lift)
            )
            or (str(args.motion_strategy) == "auto_cartesian"),
            qp_otg_p_step=_qp_p_step(phase),
            qp_otg_r_step=_qp_r_step(phase),
            qp_stream_duration=float(args.qp_stream_duration),
            qp_stream_rate_hz=float(args.qp_stream_rate_hz),
            keep_current_orientation=keep_orientation,
            moveit_timeout_sec=float(args.moveit_timeout_sec),
            stream_closed_loop=bool(args.stream_closed_loop),
            stream_step_m=float(args.stream_step_m),
            stream_waypoint_tol_m=float(args.stream_waypoint_tol_m),
            stream_waypoint_timeout_s=float(args.stream_waypoint_timeout_s),
            qp_speed_scale=float(args.qp_speed_scale),
            qp_hold_sec=_qp_hold(phase),
            qp_lag_guard_m=float(args.qp_lag_guard_m),
            qp_stall_timeout_s=float(args.qp_stall_timeout_s),
            qp_lookahead_m=_qp_lookahead(phase),
            qp_pos_tol_m=_qp_pos_tol(phase),
        )

    # Live handle to the compliant descent while it holds the arm compliant at
    # the touchdown pose (so the gripper can release without the arm fighting).
    compliant_handle: Optional[Any] = None

    def _do_compliant_place_insert(target_pose7: List[float]) -> bool:
        """F/T admittance descend-to-contact set-down (replaces the QP place
        descent). Descends softly along the place axis holding the current
        (place) orientation rigid and STOPS on table/object contact, then holds
        the arm compliant at the touchdown pose via ``compliant_handle`` so the
        gripper can release. The caller releases, then stops the handle."""
        nonlocal compliant_handle
        from compliant_grasp_execute.compliant_insert import (
            CompliantInsertParams,
            run_compliant_insert,
        )

        if compliant_handle is not None:
            try:
                compliant_handle.stop()
            except Exception:  # noqa: BLE001
                pass
            compliant_handle = None

        is_left = arm == "left"
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

        # Read the current TCP (the pre-place standoff, already at the place
        # orientation) to build the straight descent axis toward the target, and
        # to run the optional hybrid pre-descent.
        cur = xarm.get_tcp_pose(arm=arm, base_frame=args.waist_frame, timeout=2.0)
        cur_xyz = (
            np.asarray([float(v) for v in cur["translation"]], dtype=float)
            if cur is not None
            else np.asarray(target_pose7[:3], dtype=float)
        )
        target_xyz = np.asarray([float(v) for v in target_pose7[:3]], dtype=float)
        insert_vec = target_xyz - cur_xyz
        dist = float(np.linalg.norm(insert_vec))
        direction = insert_vec / dist if dist > 1e-6 else np.array([0.0, 0.0, -1.0])

        # Deepen the target PAST the nominal place pose along the descent axis so
        # the object is actually driven to the table and contact stops it at the
        # true surface (rather than halting in the air at --place-z-clearance).
        extra_drop = float(args.compliant_place_extra_drop_m)
        deep_xyz = target_xyz + max(0.0, extra_drop) * direction
        deep_target7 = [
            float(deep_xyz[0]), float(deep_xyz[1]), float(deep_xyz[2]),
            *[float(v) for v in target_pose7[3:7]],
        ]

        # HYBRID: QP-stream the bulk of the traverse, leave only the final span
        # for the compliant descent near the table.
        final_descent_m = float(args.compliant_final_descent_m)
        pre_target7 = target_pose7
        if final_descent_m > 1e-4 and dist > final_descent_m + 0.01:
            pre_xyz = target_xyz - final_descent_m * direction
            pre_target7 = [
                float(pre_xyz[0]), float(pre_xyz[1]), float(pre_xyz[2]),
                *[float(v) for v in target_pose7[3:7]],
            ]
            _log(
                "ARM: HYBRID pre-descent (QP, fast transit) -> "
                f"{(dist - final_descent_m) * 100:.1f}cm, leaving final "
                f"{final_descent_m * 100:.1f}cm for F/T compliant set-down"
            )
            # Use the FAST transit profile (big OTG step, long lookahead, short
            # hold) -- NOT the fine "place" grasp profile. This pre-descent stops
            # a safe --compliant-final-descent-m ABOVE contact, so it does not
            # need the slow, dense grasp steps; running it at transit speed is the
            # single biggest placement speed-up. (The "approach" phase label
            # selects the transit knobs under qp_all.)
            _exec("approach", pre_target7, keep_orientation=True)

        _log("ARM: COMPLIANT set-down (F/T admittance descend-to-contact)")
        compliant_handle = run_compliant_insert(
            xarm=xarm,
            topic_pub=topic_pub,
            arm=arm,
            waist_frame=args.waist_frame,
            tcp_frame=_arm_to_frame(arm),
            approach_pose7=pre_target7,
            grasp_pose7=deep_target7,
            params=cparams,
            guard=None,
            otg_p_step=float(args.compliant_otg_p_step),
            otg_r_step=float(args.compliant_otg_r_step),
        )
        result["motion"]["compliant_place"] = compliant_handle.result
        result["motion"]["place_tcp_tracking"] = _tcp_tracking_error(
            xarm, arm, args.waist_frame, target_pose7, "place"
        )
        return bool(compliant_handle.ok)

    try:
        if args.dry_run:
            _log("dry-run: skip hardware enable and motion execution")
            result["motion"]["approach_ok"] = None
            result["motion"]["place_ok"] = None
            result["motion"]["released_ok"] = None
        else:
            xarm.xarm_deactivate_all_controller()
            xarm.hardware_arm_enable(True)
            xarm.hardware_arm_mode(3)
            _log(f"arms enabled (mode 3); placing with '{arm}' arm")

            # Connect the gripper in the background so its slow serial open
            # overlaps the home/approach moves.
            gripper = None
            holder: Dict[str, Any] = {"g": None, "err": None}
            gthread: Optional[threading.Thread] = None
            if bool(args.release_gripper):
                if bool(args.gripper_async_connect):
                    def _connect() -> None:
                        try:
                            holder["g"] = _build_gripper(args)
                        except Exception as e:  # noqa: BLE001
                            holder["err"] = e

                    gthread = threading.Thread(target=_connect, daemon=True)
                    gthread.start()
                    _log("gripper: connecting in background (overlaps approach)")
                else:
                    try:
                        gripper = _build_gripper(args)
                    except Exception as e:  # noqa: BLE001
                        _log(f"WARNING: gripper init failed, will skip release: {e!r}")

            def _ensure_gripper() -> Optional[Any]:
                nonlocal gripper, gthread
                if gripper is not None:
                    return gripper
                if gthread is not None:
                    gthread.join()
                    gthread = None
                    if holder["err"] is not None:
                        _log(f"WARNING: gripper init failed, skip release: {holder['err']!r}")
                        gripper = None
                    else:
                        gripper = holder["g"]
                return gripper

            if bool(args.start_home):
                xarm.xarm_deactivate_all_controller()
                xarm.hardware_arm_enable(True)
                xarm.hardware_arm_mode(3)
                _move_home(action, args, arm, "START HOME")
                xarm.xarm_deactivate_all_controller()
                xarm.hardware_arm_enable(True)
                xarm.hardware_arm_mode(3)

            # Prime TF before any streamed motion: a cold TransformListener makes
            # the first get_tcp_pose() time out and the approach aborts before
            # moving (achieved == home). The grasp phase warms TF during
            # detection; we must do it explicitly here.
            _warmup_tcp(xarm, arm, args.waist_frame)

            # Optional safety raise before the big transit to the approach point.
            if _qp_all and float(args.qp_transit_raise_z) > 0.0:
                cur = xarm.get_tcp_pose(arm=arm, base_frame=args.waist_frame, timeout=2.0)
                if cur is not None:
                    raised = [
                        float(cur["translation"][0]),
                        float(cur["translation"][1]),
                        float(cur["translation"][2]) + float(args.qp_transit_raise_z),
                        *[float(v) for v in cur["rotation"]],
                    ]
                    _log(f"transit raise +{args.qp_transit_raise_z*100:.0f}cm before approach")
                    _exec("approach", raised, keep_orientation=True)

            # Intermediate waypoint: a FIXED, safe joint posture (jointspace,
            # reliable) if configured, else the old along-axis derived approach.
            mid_joints = _place_mid_joints(args, arm)
            if mid_joints is not None:
                _log("ARM: go to FIXED intermediate placement posture (jointspace)")
                xarm.xarm_deactivate_all_controller()
                xarm.hardware_arm_enable(True)
                xarm.hardware_arm_mode(3)
                approach_ok = _jointspace_to(
                    action, arm, mid_joints, "MID (fixed intermediate)"
                )
                xarm.xarm_deactivate_all_controller()
                xarm.hardware_arm_enable(True)
                xarm.hardware_arm_mode(3)
            else:
                _log("ARM: approach to middle waypoint")
                approach_ok = _approach_reorient_multihop(
                    xarm, args, arm, approach_pose7, _exec
                )
                result["motion"]["approach_tcp_tracking"] = _tcp_tracking_error(
                    xarm, arm, args.waist_frame, approach_pose7, "approach"
                )

            place_ok = False
            if approach_ok and bool(args.compliant_place):
                # Compliant set-down: descend softly along the place axis and
                # STOP on table/object contact so the object is never driven into
                # the table. The compliant descent holds the CURRENT orientation
                # rigid, so it must already be at the place orientation; from the
                # fixed mid posture the tool is not, so first QP to the standoff
                # (which carries the place orientation) before handing over.
                if mid_joints is not None:
                    _log("ARM: move to pre-place standoff (establish place orientation)")
                    _exec("approach", approach_pose7, keep_orientation=False)
                place_ok = _do_compliant_place_insert(place_pose7)
            elif approach_ok:
                _log("ARM: insert to place pose")
                # From the fixed mid posture the tool is NOT yet at the place
                # orientation, so let the descent establish it (reorient + move);
                # the derived-approach path already oriented, so it only translates.
                place_ok = _exec(
                    "place", place_pose7, keep_orientation=(mid_joints is None)
                )
                _wait_arm_near_pose7(
                    xarm,
                    arm,
                    args.waist_frame,
                    place_pose7,
                    pos_tol_m=float(args.place_reach_tol_m),
                    timeout_sec=float(args.place_reach_timeout_sec),
                )
                result["motion"]["place_tcp_tracking"] = _tcp_tracking_error(
                    xarm, arm, args.waist_frame, place_pose7, "place"
                )
            else:
                _log("approach failed; skipping place insert")

            # Release the object at the (raised) place pose.
            if bool(args.release_gripper) and place_ok:
                g = _ensure_gripper()
                if g is not None:
                    try:
                        _log("GRIPPER: release (open) at place pose")
                        _, _, released_ok = _gripper_move_and_wait(
                            g,
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
                else:
                    _log("WARNING: gripper unavailable; object NOT released")

            # Compliant set-down: the object has been released while the arm was
            # held compliant at the touchdown pose. Stop the admittance hold
            # BEFORE the retract so the QP controller regains full control.
            if compliant_handle is not None:
                try:
                    compliant_handle.stop()
                except Exception as e:  # noqa: BLE001
                    _log(f"WARNING: stopping compliant controller failed: {e!r}")
                compliant_handle = None

            # Nose-up in place before the retract. At the ~45deg nose-down touchdown
            # pose the low fingertip sits right at the table; opening the gripper and
            # then translating can scrape it. Rotate about waist Y (same axis as the
            # place tilt, opposite sign) so the low fingertip swings UP first, then
            # continue with the normal retract.
            tilt_up_deg = float(args.post_release_tilt_up_deg)
            if place_ok and abs(tilt_up_deg) > 1e-6:
                from scipy.spatial.transform import Rotation as R

                cur = xarm.get_tcp_pose(
                    arm=arm, base_frame=args.waist_frame, timeout=2.0
                )
                if cur is not None:
                    cur_xyz = [float(v) for v in cur["translation"]]
                    cur_quat = [float(v) for v in cur["rotation"]]
                else:
                    cur_xyz = [float(v) for v in place_pose7[:3]]
                    cur_quat = [float(v) for v in place_pose7[3:7]]
                up_quat = (
                    R.from_euler("y", -abs(tilt_up_deg), degrees=True)
                    * R.from_quat(cur_quat)
                ).as_quat()
                tilt_up_pose7 = [
                    cur_xyz[0], cur_xyz[1], cur_xyz[2], *[float(v) for v in up_quat]
                ]
                _log(
                    f"ARM: post-release nose-up {abs(tilt_up_deg):.0f}deg about waist "
                    "Y (lift fingertips off the table before retract)"
                )
                _exec("retract", tilt_up_pose7, keep_orientation=False)

            # After releasing, clear the table before the joint-space home move.
            lift_ok = None
            retract_ok = None
            if mid_joints is not None:
                # Jointspace BACK to the fixed intermediate posture (reliable, no
                # QP tracking). This replaces the QP lift/retract, which at a
                # far/awkward place pose loses orientation tracking and drives the
                # arm DOWN into the table.
                if place_ok:
                    _log(
                        "ARM: return to FIXED intermediate posture after release "
                        "(jointspace; replaces QP lift/retract)"
                    )
                    xarm.xarm_deactivate_all_controller()
                    xarm.hardware_arm_enable(True)
                    xarm.hardware_arm_mode(3)
                    retract_ok = _jointspace_to(
                        action, arm, mid_joints, "RETRACT to fixed intermediate"
                    )
            else:
                # Go straight back UP to the pre-insert standoff (the SAME
                # 45deg-tilt pose we descended from), KEEPING the current
                # orientation. This is simply the reverse of the descent -- up and
                # back along the tool axis -- so it clears the object and the table
                # WITHOUT a re-tilting lift, which at a far/awkward place pose loses
                # QP orientation tracking and drives the arm down into the table.
                if place_ok and bool(args.retract_to_approach):
                    _log("ARM: retract to pre-insert standoff (keep orientation)")
                    retract_ok = _exec("retract", approach_pose7, keep_orientation=True)
                    result["motion"]["retract_tcp_tracking"] = _tcp_tracking_error(
                        xarm, arm, args.waist_frame, approach_pose7, "retract"
                    )

            home_ok = None
            if bool(args.return_home):
                xarm.xarm_deactivate_all_controller()
                xarm.hardware_arm_enable(True)
                xarm.hardware_arm_mode(3)
                home_ok = _move_home(action, args, arm, "RETURN HOME")

            if gripper is not None:
                try:
                    gripper.disconnect()
                except Exception:
                    pass

            result["motion"]["approach_ok"] = approach_ok
            result["motion"]["place_ok"] = place_ok
            result["motion"]["released_ok"] = released_ok
            result["motion"]["lift_ok"] = lift_ok
            result["motion"]["retract_ok"] = retract_ok
            result["motion"]["return_home_ok"] = home_ok
            result["ok"] = bool(
                approach_ok
                and place_ok
                and ((not args.release_gripper) or (released_ok is True))
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

    # Mark the handoff as no longer holding once the object was released.
    if getattr(args, "handoff_out", "") and (released_ok is True):
        try:
            hpath = os.path.abspath(args.handoff_out)
            handoff: Dict[str, Any] = {}
            if os.path.exists(hpath):
                with open(hpath, "r", encoding="utf-8") as f:
                    handoff = json.load(f)
            handoff.update(
                {
                    "arm": arm,
                    "holding": False,
                    "released": True,
                    "placed": True,
                    "place_timestamp": time.time(),
                }
            )
            os.makedirs(os.path.dirname(hpath) or ".", exist_ok=True)
            with open(hpath, "w", encoding="utf-8") as f:
                json.dump(_json_safe(handoff), f, indent=2, ensure_ascii=False)
            _log(f"updated handoff: {hpath} (holding=false, placed=true)")
        except Exception as e:  # noqa: BLE001
            _log(f"WARNING: failed to update handoff file: {e!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
