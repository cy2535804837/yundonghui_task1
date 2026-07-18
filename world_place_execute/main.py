#!/usr/bin/env python3
"""
world_place_execute
===================

Bridge from a target placement point in the SLAM map/world frame to the existing
``grasp_pose_place_execute`` place node.

Interfaces
----------
Target-place HTTP interface (owned by the upstream system):

    GET <target_url>

Expected JSON:

    {
      "x": 1.23,
      "y": 0.45,
      "z": 0.75
    }

``x/y/z`` are metres. The coordinate frame is supplied by ``target_frame`` in
config/CLI unless the response optionally includes ``frame_id`` or ``frame``.
Optional ``yaw``/quaternion fields are recorded but not used for place posture.
The target frame must match the robot SLAM pose frame.

Robot SLAM pose HTTP interface:

    GET <robot_pose_url>

Expected JSON:

    {"x": 1.0, "y": 2.0, "yaw": 0.5}

``x/y`` are metres and ``yaw`` is radians in the same map/world frame.

Transform
---------
Default mode reads live TF for ``base_footprint -> waist_yaw_link`` and combines
it with the SLAM map pose:

    T_obj_waist = T_waist_base @ inv(T_world_base) @ T_obj_world

For offline/debug, ``--no-use-live-waist-tf`` falls back to the older
approximation that treats waist and mobile-base centre as coincident in X/Y/Yaw.

It then calls:

    python3 -m grasp_pose_place_execute.main --place-x ... --place-y ... --place-z ...

Run
---

    python3 -m world_place_execute.main --target-url http://host/api/place_target

Debug without HTTP:

    python3 -m world_place_execute.main --target-x 1.2 --target-y 0.3 --target-z 0.75 \\
      --robot-x 1.0 --robot-y 0.0 --robot-yaw 0.0 --dry-run --no-execute-place
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from compliant_grasp_execute.config_io import (
    add_config_args,
    apply_config_defaults,
    default_config_path,
    maybe_write_config,
)
from world_place_execute.frame_transforms import (
    object_world_to_waist_approx,
    object_world_to_waist_live,
    planar_map_delta_to_base,
)
from world_place_execute.robot_pose import get_robot_pose

_TAG = "[WORLD-PLACE]"
_REPO_ROOT = Path(__file__).resolve().parent.parent


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


def _http_get_json(url: str, timeout_sec: float, label: str) -> Dict[str, Any]:
    if not url:
        raise ValueError(f"{label} URL is empty")
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=float(timeout_sec)) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to GET {label} URL {url!r}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label} URL {url!r} did not return valid JSON: {raw[:200]!r}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{label} URL {url!r} returned non-object JSON: {type(data).__name__}")
    return data


def _float_field(data: Dict[str, Any], key: str, label: str) -> float:
    if key not in data:
        raise ValueError(f"{label} missing required field {key!r}: {data}")
    try:
        return float(data[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} field {key!r} must be a number, got {data[key]!r}") from exc


def _optional_float_field(data: Dict[str, Any], key: str) -> Optional[float]:
    if key not in data or data[key] is None:
        return None
    return float(data[key])


def _load_target(args: argparse.Namespace) -> Dict[str, Any]:
    if str(args.target_url or "").strip():
        data = _http_get_json(str(args.target_url), float(args.target_timeout_sec), "target")
        frame_id = str(data.get("frame_id") or data.get("frame") or args.target_frame)
        return {
            "frame_id": frame_id,
            "x": _float_field(data, "x", "target"),
            "y": _float_field(data, "y", "target"),
            "z": _float_field(data, "z", "target"),
            "yaw": _optional_float_field(data, "yaw"),
            "qx": _optional_float_field(data, "qx"),
            "qy": _optional_float_field(data, "qy"),
            "qz": _optional_float_field(data, "qz"),
            "qw": _optional_float_field(data, "qw"),
            "source": "http",
            "raw": data,
        }

    if args.target_x is None or args.target_y is None or args.target_z is None:
        raise SystemExit(
            f"{_TAG} provide --target-url, or provide all of --target-x --target-y --target-z"
        )
    return {
        "frame_id": str(args.target_frame),
        "x": float(args.target_x),
        "y": float(args.target_y),
        "z": float(args.target_z),
        "yaw": None if args.target_yaw is None else float(args.target_yaw),
        "qx": None,
        "qy": None,
        "qz": None,
        "qw": None,
        "source": "cli",
        "raw": None,
    }


def _target_pose_sequence(target: Dict[str, Any]) -> List[float]:
    quat_vals = [target.get(k) for k in ("qx", "qy", "qz", "qw")]
    if any(v is not None for v in quat_vals) and not all(v is not None for v in quat_vals):
        raise ValueError("target quaternion must provide all of qx/qy/qz/qw or none")
    if all(v is not None for v in quat_vals):
        return [
            float(target["x"]),
            float(target["y"]),
            float(target["z"]),
            float(target["qx"]),
            float(target["qy"]),
            float(target["qz"]),
            float(target["qw"]),
        ]
    return [float(target["x"]), float(target["y"]), float(target["z"])]


def _waist_pose7_to_target(
    waist_pose7: List[float],
    *,
    place_z_offset: float,
    diagnostics: Dict[str, float],
    mode: str,
) -> Dict[str, Any]:
    return {
        "place_x": float(waist_pose7[0]),
        "place_y": float(waist_pose7[1]),
        "place_z": float(waist_pose7[2]) + float(place_z_offset),
        "waist_pose7": [float(v) for v in waist_pose7],
        "transform_mode": mode,
        **diagnostics,
    }


def _build_place_argv(args: argparse.Namespace, waist_target: Dict[str, float], json_out: str) -> List[str]:
    argv: List[str] = [
        "--handoff-in",
        str(args.handoff_in),
        "--arm",
        str(args.arm),
        "--json-out",
        json_out,
        "--place-x",
        f"{waist_target['place_x']:.6f}",
        "--place-y",
        f"{waist_target['place_y']:.6f}",
        "--place-z",
        f"{waist_target['place_z']:.6f}",
    ]
    if args.place_z_clearance is not None:
        argv += ["--place-z-clearance", f"{float(args.place_z_clearance):.6f}"]
    if args.place_tilt_y_deg is not None:
        argv += ["--place-tilt-y-deg", f"{float(args.place_tilt_y_deg):.6f}"]
    if args.motion_strategy:
        argv += ["--motion-strategy", str(args.motion_strategy)]
    if args.require_holding is not None:
        argv += ["--require-holding" if bool(args.require_holding) else "--no-require-holding"]
    if args.dry_run:
        argv += ["--dry-run"]
    if args.place_extra_args:
        argv += list(args.place_extra_args)
    return argv


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="world_place_execute",
        description="Convert a map/world placement target to waist_yaw_link and call the place node.",
    )
    add_config_args(p, default_config_path(__file__))

    p.add_argument("--target-url", default="")
    p.add_argument("--target-timeout-sec", type=float, default=3.0)
    p.add_argument("--robot-pose-url", default="http://192.168.41.6:1448/api/core/slam/v1/localization/pose")
    p.add_argument("--robot-pose-timeout-sec", type=float, default=3.0)
    p.add_argument("--robot-pose-frame", default="map")
    p.add_argument("--robot-x", type=float, default=None)
    p.add_argument("--robot-y", type=float, default=None)
    p.add_argument("--robot-yaw", type=float, default=None)
    p.add_argument("--use-live-waist-tf", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--waist-frame", default="waist_yaw_link")
    p.add_argument("--base-frame", default="base_footprint")
    p.add_argument("--tf-timeout-sec", type=float, default=3.0)

    p.add_argument("--target-x", type=float, default=None)
    p.add_argument("--target-y", type=float, default=None)
    p.add_argument("--target-z", type=float, default=None)
    p.add_argument("--target-yaw", type=float, default=None)
    p.add_argument("--target-frame", default="map")
    p.add_argument("--target-quat", type=float, nargs=4, default=None, metavar=("QX", "QY", "QZ", "QW"))

    p.add_argument("--waist-z-in-map", type=float, default=0.0)
    p.add_argument("--place-z-offset", type=float, default=0.0)

    p.add_argument("--handoff-in", default="/tmp/grasp_handoff.json")
    p.add_argument("--arm", choices=["auto", "left", "right"], default="auto")
    p.add_argument("--motion-strategy", default=None)
    p.add_argument("--place-z-clearance", type=float, default=None)
    p.add_argument("--place-tilt-y-deg", type=float, default=None)
    p.add_argument("--require-holding", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--place-extra-args", nargs="*", default=[])
    p.add_argument("--timeout-sec", type=float, default=600.0)

    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--no-execute-place",
        action="store_true",
        help="Only fetch/convert/write JSON; do not launch grasp_pose_place_execute.",
    )
    p.add_argument("--json-out", default="/tmp/world_place_exec.json")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    _, config_path = apply_config_defaults(parser, argv)
    args = parser.parse_args(argv)
    if maybe_write_config(parser, args, config_path):
        return 0

    started = time.time()
    result: Dict[str, Any] = {
        "ok": False,
        "dry_run": bool(args.dry_run),
        "no_execute_place": bool(args.no_execute_place),
        "transform_mode": "live_tf" if bool(args.use_live_waist_tf) else "approx_base_equals_waist",
    }

    place_json_out = os.path.splitext(os.path.abspath(str(args.json_out)))[0] + ".place.json"
    ros_started = False
    try:
        target = _load_target(args)
        if args.target_quat is not None:
            target.update(
                {
                    "qx": float(args.target_quat[0]),
                    "qy": float(args.target_quat[1]),
                    "qz": float(args.target_quat[2]),
                    "qw": float(args.target_quat[3]),
                }
            )
        robot_pose = get_robot_pose(
            url=str(args.robot_pose_url),
            timeout=float(args.robot_pose_timeout_sec),
            frame_id=str(args.robot_pose_frame),
            direct_x=args.robot_x,
            direct_y=args.robot_y,
            direct_yaw=args.robot_yaw,
        )

        diagnostics = planar_map_delta_to_base(
            float(target["x"]),
            float(target["y"]),
            float(robot_pose["x"]),
            float(robot_pose["y"]),
            float(robot_pose["yaw"]),
        )
        target_pose_world = _target_pose_sequence(target)
        if bool(args.use_live_waist_tf):
            import rclpy
            from xarm_sdk import XARM_manager

            if not rclpy.ok():
                rclpy.init()
                ros_started = True
            xarm = XARM_manager()
            waist_pose7 = object_world_to_waist_live(
                target_pose_world,
                xarm,
                robot_base_world=robot_pose,
                target_frame_id=str(target["frame_id"]),
                waist_frame=str(args.waist_frame),
                base_frame=str(args.base_frame),
                tf_timeout=float(args.tf_timeout_sec),
            )
            mode = "live_tf"
        else:
            waist_pose7 = object_world_to_waist_approx(
                target_pose_world,
                robot_pose,
                target_frame_id=str(target["frame_id"]),
                waist_z_in_map=float(args.waist_z_in_map),
            )
            mode = "approx_base_equals_waist"

        waist_target = _waist_pose7_to_target(
            waist_pose7,
            place_z_offset=float(args.place_z_offset),
            diagnostics=diagnostics,
            mode=mode,
        )
        place_argv = _build_place_argv(args, waist_target, place_json_out)

        result.update(
            {
                "target": target,
                "robot_pose": robot_pose,
                "waist_target": waist_target,
                "frames": {
                    "target_frame": str(target["frame_id"]),
                    "robot_pose_frame": str(robot_pose["frame_id"]),
                    "base_frame": str(args.base_frame),
                    "waist_frame": str(args.waist_frame),
                },
                "place_cli_argv": place_argv,
                "place_json_out": place_json_out,
            }
        )
        _log(
            "converted target -> waist place "
            f"x={waist_target['place_x']:.4f} y={waist_target['place_y']:.4f} "
            f"z={waist_target['place_z']:.4f}"
        )

        if bool(args.no_execute_place):
            _log("--no-execute-place: not launching place node")
            result["ok"] = True
        else:
            cmd = [sys.executable, "-u", "-m", "grasp_pose_place_execute.main", *place_argv]
            result["place_cmd"] = cmd
            _log("launching place node: " + " ".join(cmd))
            proc = subprocess.run(
                cmd,
                cwd=str(_REPO_ROOT),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=float(args.timeout_sec),
                check=False,
            )
            result["place_exit_code"] = int(proc.returncode)
            result["place_output_tail"] = "\n".join(proc.stdout.splitlines()[-80:])
            place_result = None
            if os.path.exists(place_json_out):
                with open(place_json_out, "r", encoding="utf-8") as f:
                    place_result = json.load(f)
            result["place_result"] = place_result
            result["ok"] = bool(proc.returncode == 0 and isinstance(place_result, dict) and place_result.get("ok"))
    except subprocess.TimeoutExpired as exc:
        result["error"] = f"place node timed out after {float(args.timeout_sec):.0f}s"
        result["place_output_tail"] = (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else ""
        _log("ERROR: " + result["error"])
    except Exception as exc:  # noqa: BLE001
        result["error"] = repr(exc)
        _log("ERROR: " + repr(exc))
    finally:
        if ros_started:
            try:
                import rclpy

                if rclpy.ok():
                    rclpy.shutdown()
            except Exception:  # noqa: BLE001
                pass

    result["duration_sec"] = round(time.time() - started, 3)
    safe = _json_safe(result)
    print(json.dumps(safe, indent=2, ensure_ascii=False))
    if args.json_out:
        out = os.path.abspath(str(args.json_out))
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(safe, f, indent=2, ensure_ascii=False)
        _log(f"saved result json: {out}")
    return 0 if bool(result.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
