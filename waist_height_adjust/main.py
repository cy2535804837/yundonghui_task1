"""
Adjust the robot's waist (torso) height to a commanded value.

The tianyi2 "body" is a 4-DOF chain
``[first_leg_pitch, second_leg_pitch, waist_pitch, waist_yaw]``. The SDK exposes
it in Cartesian space through ``ActionCall.endpose_body_controller`` with a
4-element target ordered ``[x, z, pitch, yaw]`` (metres / radians). The factory
zero pose is ``[0.05, 0.68, 0.0, 0.0]`` -- so ``z`` is the waist height.

This tool sets the waist HEIGHT (``z``) to ``--height`` while holding the other
body DOFs at safe defaults (forward offset ``--body-x``, ``--body-pitch``,
``--body-yaw``). It enables the leg + waist hardware, sends one blocking body
endpose command, and reports the body joint angles before / after.

Usage:
    # lower the waist to 0.60 m (uses config.yaml defaults for everything else)
    python3 -m waist_height_adjust.main --height 0.60

    # preview without moving
    python3 -m waist_height_adjust.main --height 0.60 --dry-run

All tunable parameters live in ``config.yaml`` next to this file. Precedence:
explicit CLI flag > config.yaml > built-in default.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, List, Optional

import rclpy

from xarm_sdk import ActionCall, XARM_manager

from grasp_pose_grasp_execute.config_io import (
    add_config_args,
    apply_config_defaults,
    default_config_path,
    maybe_write_config,
)

_TAG = "[WAIST-HEIGHT]"

# SDK factory zero pose for the body endpose controller: [x, z, pitch, yaw].
_BODY_ZERO_POSE = [0.05, 0.68, 0.0, 0.0]


def _log(msg: str) -> None:
    print(f"{_TAG} {msg}", flush=True)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 6)
    return obj


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="waist_height_adjust",
        description="Adjust the robot waist (torso) height to a commanded value.",
    )
    add_config_args(p, default_config_path(__file__))

    p.add_argument(
        "--height",
        type=float,
        default=_BODY_ZERO_POSE[1],
        help="Target waist height (m) = the body endpose z. Factory zero is "
        f"{_BODY_ZERO_POSE[1]} m.",
    )
    p.add_argument(
        "--min-height",
        type=float,
        default=0.45,
        help="Safety lower bound (m). The target is CLAMPED into "
        "[--min-height, --max-height] and a warning is logged if it was out of "
        "range. Widen this once the true mechanical range is confirmed.",
    )
    p.add_argument(
        "--max-height",
        type=float,
        default=0.90,
        help="Safety upper bound (m); see --min-height.",
    )
    p.add_argument(
        "--body-x",
        type=float,
        default=_BODY_ZERO_POSE[0],
        help="Body endpose forward offset x (m). Kept at the zero-pose default "
        f"({_BODY_ZERO_POSE[0]}) unless you know you need to shift the torso.",
    )
    p.add_argument(
        "--body-pitch",
        type=float,
        default=_BODY_ZERO_POSE[2],
        help="Body endpose pitch (rad). 0 = upright torso.",
    )
    p.add_argument(
        "--body-yaw",
        type=float,
        default=_BODY_ZERO_POSE[3],
        help="Body endpose yaw (rad). 0 = facing forward.",
    )
    p.add_argument(
        "--enable-hardware",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the leg + waist hardware before commanding the height "
        "(needed on real hardware; no-op in sim).",
    )
    p.add_argument(
        "--settle-sec",
        type=float,
        default=1.0,
        help="Seconds to wait after the move completes before reading back the "
        "final body joint angles.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved target and skip all motion / hardware calls.",
    )
    p.add_argument("--json-out", default="/tmp/waist_height.json")
    return p


def _read_body_joints(xarm: XARM_manager) -> Optional[List[float]]:
    """Best-effort read of [first_leg_pitch, second_leg_pitch, waist_pitch,
    waist_yaw]; spins a few times so the joint_states subscription is primed."""
    for _ in range(40):
        joints = xarm.xarm_body_joint_angles()
        if joints is not None and all(v is not None for v in joints):
            return [float(v) for v in joints]
        rclpy.spin_once(xarm, timeout_sec=0.05)
    return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    _, config_path = apply_config_defaults(parser, argv)
    args = parser.parse_args(argv)
    if maybe_write_config(parser, args, config_path):
        return 0

    lo, hi = float(args.min_height), float(args.max_height)
    if lo > hi:
        raise SystemExit(f"{_TAG} --min-height {lo} > --max-height {hi}")

    requested = float(args.height)
    height = max(lo, min(hi, requested))
    if abs(height - requested) > 1e-9:
        _log(
            f"WARNING: requested height {requested:.4f} m out of safe range "
            f"[{lo:.3f}, {hi:.3f}] -> clamped to {height:.4f} m"
        )

    target_pose = [
        float(args.body_x),
        float(height),
        float(args.body_pitch),
        float(args.body_yaw),
    ]
    _log(
        f"target body endpose [x, z(height), pitch, yaw] = "
        f"[{target_pose[0]:.4f}, {target_pose[1]:.4f}, "
        f"{target_pose[2]:.4f}, {target_pose[3]:.4f}]"
    )

    result: Dict[str, Any] = {
        "requested_height": requested,
        "commanded_height": height,
        "target_body_pose": target_pose,
        "dry_run": bool(args.dry_run),
        "ok": False,
    }

    if args.dry_run:
        _log("dry-run: not moving")
        result["ok"] = True
        print(json.dumps(_json_safe(result), indent=2, ensure_ascii=False))
        return 0

    rclpy.init()
    try:
        xarm = XARM_manager()
        action = ActionCall(xarm)

        xarm.xarm_deactivate_all_controller()

        if bool(args.enable_hardware):
            _log("enabling leg + waist hardware")
            xarm.hardware_leg_enable(True)
            xarm.hardware_waist_enable(True)

        before = _read_body_joints(xarm)
        result["body_joints_before"] = before
        _log(f"body joints before: {before}")

        # Snapshot the controller's warn/error code queue so we can tell which
        # NEW codes this command produced (e.g. 600101 = command over a joint
        # limit -> the height is outside the reachable range).
        try:
            codes_before = list(xarm.warn_and_error_code_queue)
        except Exception:  # noqa: BLE001
            codes_before = []

        _log("commanding body endpose (this blocks until the move completes)...")
        move_res = action.endpose_body_controller(target_pose)
        result["move_result"] = str(move_res)

        if float(args.settle_sec) > 0.0:
            time.sleep(float(args.settle_sec))
        # Let any error/warn messages for this command arrive.
        for _ in range(10):
            rclpy.spin_once(xarm, timeout_sec=0.02)

        after = _read_body_joints(xarm)
        result["body_joints_after"] = after
        _log(f"body joints after: {after}")

        try:
            codes_after = list(xarm.warn_and_error_code_queue)
        except Exception:  # noqa: BLE001
            codes_after = []
        new_codes = codes_after[len(codes_before):] if len(
            codes_after
        ) >= len(codes_before) else codes_after
        result["error_codes"] = new_codes

        # The action returns None when the goal was NOT accepted (validation /
        # controller rejection) OR no result came back. A rejected goal means the
        # arm did NOT move (before ~= after), which for this controller is almost
        # always an out-of-reach HEIGHT: the commanded z maps to a leg-pitch joint
        # past its limit (code 600101, "指令超上限位").
        result["ok"] = move_res is not None
        if move_res is None:
            moved = False
            if before is not None and after is not None:
                moved = any(
                    abs(a - b) > 1e-3 for a, b in zip(after, before)
                )
            hint = (
                "the body controller REJECTED the goal (the arm did not move). "
                f"This height ({height:.3f} m) is almost certainly OUTSIDE the "
                "reachable range: the commanded z maps to a leg-pitch joint past "
                "its limit"
            )
            if 600101 in new_codes:
                hint += " (error 600101 '指令超上限位' = command over the UPPER "
                hint += (
                    "joint limit -> the waist is being asked to go LOWER than it "
                    "mechanically can). Try a height closer to the ~0.68 m nominal "
                    "(e.g. raise it in 1-2 cm steps to find the floor)."
                )
            else:
                hint += (
                    f". Controller error codes: {new_codes or 'none captured'}. "
                    "Try a height closer to the ~0.68 m nominal."
                )
            result["hint"] = hint
            result["moved"] = moved
            _log(f"FAILED: {hint}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()

    safe = _json_safe(result)
    print(json.dumps(safe, indent=2, ensure_ascii=False))
    if args.json_out:
        try:
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(safe, f, indent=2, ensure_ascii=False)
            _log(f"saved result json: {args.json_out}")
        except OSError as e:
            _log(f"WARNING: could not write {args.json_out}: {e!r}")

    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
