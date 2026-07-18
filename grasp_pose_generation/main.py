#!/usr/bin/env python3
"""
grasp_pose_generation/main.py
=============================

Dedicated CLI for grasp pose generation only (capture -> perception -> 3D pose
-> TF -> grasp TCP pose), independent from bottle_cup_pour_place pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import rclpy

from grasp_pose_generation.internal.object_pose_pipeline import (  # noqa: E402
    get_object_pose_in_waist_yaw_link,
)
from grasp_pose_generation.internal.perception_tools import PerceptionTool  # noqa: E402
from grasp_pose_generation.internal.pose_estimator import (  # noqa: E402
    PoseEstimator,
    extract_masks_from_results,
)
from xarm_sdk import XARM_manager  # noqa: E402

_TAG = "[GRASP-POSE]"
_STAGE_KEYS = ("capture_sec", "segmentation_sec", "pose3d_sec", "tf_sec")


def _log(msg: str) -> None:
    print(f"{_TAG} {msg}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--prompt",
        action="append",
        required=False,
        help="Open-vocab prompt (repeatable), e.g. --prompt bottle --prompt cup",
    )
    p.add_argument(
        "--pipeline-version",
        choices=["current", "accelerated", "compare"],
        default="current",
        help=(
            "current: tactile_grasp PerceptionTool pipeline; "
            "accelerated: detection_only/test_call segmentation style + existing 3D pose conversion; "
            "compare: run both and print timing comparison."
        ),
    )
    p.add_argument("--base-url", default="http://10.20.0.24:939")
    p.add_argument(
        "--camera-yaml",
        default="poseestimator/camera_pose_config_dev29.yaml",
        help="Path under tactile_grasp/ or absolute path.",
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
        default="detected",
    )
    p.add_argument("--arm", choices=["left", "right"], default="right")
    p.add_argument(
        "--grasp-yaw-offset-deg",
        type=float,
        default=0.0,
        help="Only for grasp/grasp_topdown/grasp_side policies.",
    )
    p.add_argument(
        "--max-grasp-yaw-delta-deg",
        type=float,
        default=30.0,
        help="Only for grasp/grasp_topdown/grasp_side policies; 0 disables clamp.",
    )
    p.add_argument("--tcp-to-tip-x", type=float, default=0.0)
    p.add_argument("--tcp-to-tip-y", type=float, default=0.0)
    p.add_argument("--tcp-to-tip-z", type=float, default=-0.20)
    p.add_argument("--save-dir", default="")
    p.add_argument("--save-prefix", default="grasp_pose")
    p.add_argument("--json-out", default="", help="Optional path to save result JSON.")
    p.add_argument("--retries", type=int, default=1, help="Detection retries.")
    p.add_argument("--retry-sleep", type=float, default=0.3)
    p.add_argument(
        "--bench-runs",
        type=int,
        default=1,
        help="Run N full calls and report latency stats.",
    )
    p.add_argument(
        "--sam3-upload-format",
        choices=["jpeg", "png"],
        default="jpeg",
        help="Forwarded as SAM3_UPLOAD_FORMAT env var before perception call.",
    )
    p.add_argument(
        "--sam3-jpeg-quality",
        type=int,
        default=85,
        help="Forwarded as SAM3_JPEG_QUALITY env var when upload format is jpeg.",
    )
    p.add_argument(
        "--doctor",
        action="store_true",
        help="Print environment/dependency checks and exit.",
    )
    # ---- perception accuracy validation (touch test) ----
    p.add_argument(
        "--touch-record",
        action="store_true",
        help="Record ground-truth object center from CURRENT arm pose: read live TCP, "
        "derive fingertip (tip) in waist frame, save to --ground-truth-json, then exit. "
        "Jog the fingertip to touch the object center before running this.",
    )
    p.add_argument(
        "--validate",
        action="store_true",
        help="Run N detections (N=--bench-runs), report median/std of detected object "
        "center (tip) in waist frame, and compare to --ground-truth-json if present.",
    )
    p.add_argument(
        "--ground-truth-json",
        default="/tmp/grasp_pose_touch_gt.json",
        help="Path for the recorded touch ground-truth tip position.",
    )
    p.add_argument(
        "--from-recorded-pose",
        default="",
        help="With --touch-record: read TCP from a handover.record_zero_stiff_poses "
        "JSON (drag-to-teach output) instead of the live arm. No ROS/arm needed.",
    )
    p.add_argument(
        "--recorded-pose-name",
        default="",
        help="Which pose entry name to use from --from-recorded-pose (default: last).",
    )
    return p


def _resolve_camera_yaml(path: str) -> str:
    if os.path.isabs(path):
        return path
    # Prefer local bundled copy first (single-folder deployment), then legacy sibling.
    local_candidate = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "assets",
        path,
    )
    if os.path.exists(local_candidate):
        return local_candidate
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "tactile_grasp",
        path,
    )


class _AcceleratedSegToPoseAdapter:
    """Adapter that plugs detection_only/test_call segmentation into the
    existing get_object_pose_in_waist_yaw_link contract.

    It implements a PerceptionTool-like ``perception_pipeline`` method:
      input:  rgb_image, depth_image, prompts, confidence
      output: (results_output, results_2d)
    """

    def __init__(self, *, base_url: str, camera_yaml: str) -> None:
        from grasp_pose_generation.internal.fast_seg_client import FastSegClient

        self._seg = FastSegClient(base_url=base_url)
        self._estimator = PoseEstimator.from_yaml(camera_yaml)
        self.last_stage_timing: Dict[str, float] = {}

    def perception_pipeline(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        prompts: List[str],
        confidence: float = 0.8,
    ):
        _ = confidence  # FastSeg client currently uses a fixed confidence in its method.
        self.last_stage_timing = {"segmentation_sec": 0.0, "pose3d_sec": 0.0}

        t_seg0 = time.perf_counter()
        results_2d = self._seg.perception_pipeline(rgb_image, prompts)
        self.last_stage_timing["segmentation_sec"] += time.perf_counter() - t_seg0
        if not isinstance(results_2d, dict) or not results_2d.get("results"):
            return None, results_2d

        masks = extract_masks_from_results(results_2d, depth_image)
        detections = list(results_2d.get("results") or [])
        results_output = []

        for i, det in enumerate(detections):
            mask = masks[i] if i < len(masks) else None
            if mask is None:
                continue
            t_pose0 = time.perf_counter()
            pose, info = self._estimator.estimate_pose(mask, depth_image)
            self.last_stage_timing["pose3d_sec"] += time.perf_counter() - t_pose0
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
        if not results_output:
            return None, results_2d
        return results_output, results_2d


def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy types so result can be JSON serialized."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _stage_stats(samples: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for k in _STAGE_KEYS:
        vals = [float(s.get(k, 0.0)) for s in samples]
        if not vals:
            vals = [0.0]
        out[k] = {
            "avg": float(np.mean(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    return out


def _run_once(
    *,
    xarm: XARM_manager,
    percep: Any,
    args: argparse.Namespace,
    run_idx: int,
    label: str = "run",
) -> Dict[str, Any]:
    stage_timing = {k: 0.0 for k in _STAGE_KEYS}
    # runtime patching to measure shared stages inside get_object_pose_in_waist_yaw_link
    import grasp_pose_generation.internal.object_pose_pipeline as _opw
    import xarm_sdk.tools as _xtools

    _orig_capture = _opw.capture_single_rgbd
    _orig_tf = _xtools.lookup_tf_once
    _orig_seg = None
    _orig_pose = None

    def _capture_timed(*aa, **kk):
        t = time.perf_counter()
        r = _orig_capture(*aa, **kk)
        stage_timing["capture_sec"] += time.perf_counter() - t
        return r

    def _tf_timed(*aa, **kk):
        t = time.perf_counter()
        r = _orig_tf(*aa, **kk)
        stage_timing["tf_sec"] += time.perf_counter() - t
        return r

    _opw.capture_single_rgbd = _capture_timed
    _xtools.lookup_tf_once = _tf_timed

    # current pipeline timing hooks
    if hasattr(percep, "segment_multi_target_image"):
        _orig_seg = percep.segment_multi_target_image

        def _seg_timed(*aa, **kk):
            t = time.perf_counter()
            r = _orig_seg(*aa, **kk)
            stage_timing["segmentation_sec"] += time.perf_counter() - t
            return r

        percep.segment_multi_target_image = _seg_timed
    if hasattr(percep, "estimator") and hasattr(percep.estimator, "estimate_pose"):
        _orig_pose = percep.estimator.estimate_pose

        def _pose_timed(*aa, **kk):
            t = time.perf_counter()
            r = _orig_pose(*aa, **kk)
            stage_timing["pose3d_sec"] += time.perf_counter() - t
            return r

        percep.estimator.estimate_pose = _pose_timed

    t0 = time.perf_counter()
    try:
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
            save_prefix=f"{args.save_prefix}_run{run_idx}",
            tcp_to_tip_offset_m=np.array(
                [args.tcp_to_tip_x, args.tcp_to_tip_y, args.tcp_to_tip_z], dtype=float
            ),
            orientation_policy=str(args.orientation_policy),
            rotation_arm=str(args.arm),
            grasp_yaw_offset=np.radians(float(args.grasp_yaw_offset_deg)),
            max_grasp_yaw_delta=(
                np.radians(float(args.max_grasp_yaw_delta_deg))
                if float(args.max_grasp_yaw_delta_deg) > 0.0
                else None
            ),
            verbose=True,
            segment_confidence=float(args.segment_confidence),
        )
        dt = time.perf_counter() - t0
    finally:
        _opw.capture_single_rgbd = _orig_capture
        _xtools.lookup_tf_once = _orig_tf
        if _orig_seg is not None:
            percep.segment_multi_target_image = _orig_seg
        if _orig_pose is not None:
            percep.estimator.estimate_pose = _orig_pose

    # accelerated adapter owns seg+pose timing internally
    if hasattr(percep, "last_stage_timing") and isinstance(percep.last_stage_timing, dict):
        stage_timing["segmentation_sec"] = float(
            percep.last_stage_timing.get("segmentation_sec", stage_timing["segmentation_sec"])
        )
        stage_timing["pose3d_sec"] = float(
            percep.last_stage_timing.get("pose3d_sec", stage_timing["pose3d_sec"])
        )

    if res is None:
        raise RuntimeError("get_object_pose_in_waist_yaw_link returned None")
    _log(
        f"{label} {run_idx}: total={dt:.3f}s class={res.get('class_name')} "
        f"tcp_pose7={res.get('pose_tcp_waist_yaw_link_pose7')}"
    )
    _log(
        f"{label} {run_idx} stages: "
        f"capture={stage_timing['capture_sec']:.3f}s "
        f"seg={stage_timing['segmentation_sec']:.3f}s "
        f"pose3d={stage_timing['pose3d_sec']:.3f}s "
        f"tf={stage_timing['tf_sec']:.3f}s"
    )
    out = dict(res)
    out["total_time_sec"] = float(dt)
    out["stage_timing_sec"] = {k: float(v) for k, v in stage_timing.items()}
    return out


def _fingertip_gt_from_tcp(
    tcp_xyz: np.ndarray,
    quat: List[float],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Derive fingertip (tip) ground truth in waist frame from a TCP pose.

    Mirrors tip_pose_to_tcp_pose: tcp = tip + R @ (-offset)  ->  tip = tcp + R @ offset.
    """
    from scipy.spatial.transform import Rotation as R

    tcp_xyz = np.asarray(tcp_xyz, dtype=float).reshape(3)
    quat = [float(v) for v in quat]
    offset = np.array(
        [float(args.tcp_to_tip_x), float(args.tcp_to_tip_y), float(args.tcp_to_tip_z)],
        dtype=float,
    )
    rot = R.from_quat(quat).as_matrix()
    tip_xyz = tcp_xyz + rot @ offset
    return {
        "tip_waist_m": [float(v) for v in tip_xyz.tolist()],
        "tcp_waist_pose7": [float(tcp_xyz[0]), float(tcp_xyz[1]), float(tcp_xyz[2]), *quat],
        "tcp_to_tip_offset_m": [float(v) for v in offset.tolist()],
        "arm": str(args.arm),
        "waist_frame": str(args.waist_frame),
    }


def _tcp_pose7_from_recorded(args: argparse.Namespace) -> tuple[np.ndarray, List[float], str]:
    """Read a TCP pose7 (in waist frame) from a record_zero_stiff_poses JSON."""
    with open(str(args.from_recorded_pose), "r", encoding="utf-8") as f:
        data = json.load(f)
    poses = data.get("poses") or []
    if not poses:
        raise ValueError(f"{args.from_recorded_pose}: no 'poses' entries")
    entry = None
    want = str(args.recorded_pose_name).strip().lower()
    if want:
        for e in poses:
            if str(e.get("name", "")).strip().lower() == want:
                entry = e
                break
        if entry is None:
            names = [str(e.get("name")) for e in poses]
            raise ValueError(f"pose name {args.recorded_pose_name!r} not found; have: {names}")
    else:
        entry = poses[-1]
    block = entry.get(str(args.arm))
    if not isinstance(block, dict) or "pose_xyz_quat" not in block:
        raise ValueError(
            f"recorded entry {entry.get('name')!r} has no '{args.arm}.pose_xyz_quat'"
        )
    p = [float(v) for v in block["pose_xyz_quat"]]
    if len(p) != 7:
        raise ValueError(f"pose_xyz_quat must be length 7, got {len(p)}")
    return np.array(p[:3], dtype=float), p[3:7], str(entry.get("name", "recorded"))


def _record_touch_gt(xarm: Optional[XARM_manager], args: argparse.Namespace) -> int:
    if args.from_recorded_pose:
        tcp_xyz, quat, src_name = _tcp_pose7_from_recorded(args)
        _log(f"touch-record source: recorded pose '{src_name}' from {args.from_recorded_pose}")
        gt = _fingertip_gt_from_tcp(tcp_xyz, quat, args)
    else:
        assert xarm is not None
        cur = xarm.get_tcp_pose(
            arm=str(args.arm), base_frame=str(args.waist_frame), timeout=float(args.tf_timeout)
        )
        if cur is None:
            _log("touch-record FAILED: could not read current TCP pose")
            return 1
        gt = _fingertip_gt_from_tcp(
            np.asarray(cur["translation"], dtype=float),
            [float(v) for v in cur["rotation"]],
            args,
        )
    gt["recorded_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out_path = os.path.abspath(str(args.ground_truth_json))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(gt), f, indent=2, ensure_ascii=False)
    _log(
        f"touch-record OK: fingertip (object center) in {args.waist_frame} = "
        f"{[f'{v:.4f}' for v in gt['tip_waist_m']]} m  -> saved {out_path}"
    )
    return 0


def _run_validate(xarm: XARM_manager, percep: Any, args: argparse.Namespace) -> Dict[str, Any]:
    n = max(1, int(args.bench_runs))
    tips: List[List[float]] = []
    for i in range(n):
        try:
            r = _run_once(xarm=xarm, percep=percep, args=args, run_idx=i + 1, label="validate")
        except Exception as e:  # noqa: BLE001
            _log(f"validate run {i + 1} failed: {e!r}")
            continue
        tip = r.get("pose_tip_waist_yaw_link_rpy")
        if isinstance(tip, dict):
            tips.append([float(tip["x"]), float(tip["y"]), float(tip["z"])])
    if not tips:
        raise RuntimeError("validate: no successful detections")

    arr = np.asarray(tips, dtype=float)
    median = np.median(arr, axis=0)
    std = np.std(arr, axis=0)
    out: Dict[str, Any] = {
        "mode": "validate",
        "n_detections": int(arr.shape[0]),
        "detected_tip_waist_median_m": [float(v) for v in median.tolist()],
        "detected_tip_waist_std_m": [float(v) for v in std.tolist()],
        "detected_tip_waist_all_m": [[float(v) for v in row] for row in arr.tolist()],
    }
    _log(
        f"validate: n={arr.shape[0]} median tip(waist)="
        f"[{median[0]:.4f}, {median[1]:.4f}, {median[2]:.4f}] m  "
        f"std=[{std[0]*1000:.1f}, {std[1]*1000:.1f}, {std[2]*1000:.1f}] mm"
    )

    gt_path = os.path.abspath(str(args.ground_truth_json))
    if os.path.exists(gt_path):
        with open(gt_path, "r", encoding="utf-8") as f:
            gt = json.load(f)
        gt_tip = np.asarray(gt.get("tip_waist_m", []), dtype=float).reshape(-1)
        if gt_tip.shape[0] == 3:
            err = median - gt_tip
            err_norm = float(np.linalg.norm(err))
            out["ground_truth_tip_waist_m"] = [float(v) for v in gt_tip.tolist()]
            out["error_m"] = [float(v) for v in err.tolist()]
            out["error_norm_m"] = err_norm
            _log(
                f"validate: ground-truth tip(waist)="
                f"[{gt_tip[0]:.4f}, {gt_tip[1]:.4f}, {gt_tip[2]:.4f}] m"
            )
            _log(
                f"validate: ERROR (detected - gt) = "
                f"[{err[0]*1000:.1f}, {err[1]*1000:.1f}, {err[2]*1000:.1f}] mm  "
                f"|err|={err_norm*1000:.1f} mm"
            )
        else:
            _log(f"validate: ground-truth file malformed (tip_waist_m): {gt_path}")
    else:
        _log(
            f"validate: no ground-truth file at {gt_path}; "
            f"run with --touch-record first to enable error comparison."
        )
    return out


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.doctor:
        _log("doctor report:")
        camera_yaml = os.path.abspath(_resolve_camera_yaml(str(args.camera_yaml)))
        _log(f"  package_root: {os.path.dirname(os.path.abspath(__file__))}")
        _log(f"  camera_yaml: {camera_yaml}  exists={os.path.exists(camera_yaml)}")
        try:
            import xarm_sdk  # noqa: F401
            _log("  xarm_sdk import: OK")
        except Exception as e:  # noqa: BLE001
            _log(f"  xarm_sdk import: FAIL ({e!r})")
        try:
            import cv_bridge  # noqa: F401
            _log("  cv_bridge import: OK")
        except Exception as e:  # noqa: BLE001
            _log(f"  cv_bridge import: FAIL ({e!r})")
        return 0

    if args.touch_record:
        if args.from_recorded_pose:
            return _record_touch_gt(None, args)
        if not rclpy.ok():
            rclpy.init()
        xarm = XARM_manager()
        try:
            return _record_touch_gt(xarm, args)
        finally:
            if rclpy.ok():
                rclpy.shutdown()

    if not args.prompt:
        raise SystemExit("Please provide at least one --prompt (or run with --doctor).")
    os.environ["SAM3_UPLOAD_FORMAT"] = str(args.sam3_upload_format)
    os.environ["SAM3_JPEG_QUALITY"] = str(int(args.sam3_jpeg_quality))
    camera_yaml = _resolve_camera_yaml(str(args.camera_yaml))

    if args.validate and str(args.pipeline_version) == "compare":
        raise SystemExit("--validate requires --pipeline-version current or accelerated (not compare).")

    if not rclpy.ok():
        rclpy.init()
    xarm = XARM_manager()

    def _run_pipeline(
        label: str, percep: Any
    ) -> tuple[List[float], List[Dict[str, float]], Dict[str, Any]]:
        latencies: List[float] = []
        stage_samples: List[Dict[str, float]] = []
        last: Optional[Dict[str, Any]] = None
        for i in range(max(1, int(args.bench_runs))):
            ok = False
            last_err: Optional[Exception] = None
            for k in range(max(1, int(args.retries))):
                try:
                    r = _run_once(
                        xarm=xarm, percep=percep, args=args, run_idx=i + 1, label=label
                    )
                    latencies.append(float(r["total_time_sec"]))
                    stage_samples.append(
                        {k: float(r.get("stage_timing_sec", {}).get(k, 0.0)) for k in _STAGE_KEYS}
                    )
                    last = r
                    ok = True
                    break
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    _log(
                        f"{label} {i + 1} attempt {k + 1}/{args.retries} failed: {e!r}"
                    )
                    if k < int(args.retries) - 1:
                        time.sleep(float(args.retry_sleep))
            if not ok:
                raise RuntimeError(
                    f"{label} run {i + 1}: all attempts failed; last error={last_err!r}"
                )
        assert last is not None
        return latencies, stage_samples, last

    final_res: Optional[Dict[str, Any]] = None
    try:
        mode = str(args.pipeline_version)
        if mode in ("current", "compare"):
            _log(
                f"init current PerceptionTool(base_url={args.base_url}, camera_yaml={camera_yaml})"
            )
            percep_current = PerceptionTool(
                base_url=str(args.base_url), camera_pose_file_path=camera_yaml
            )
        else:
            percep_current = None

        if mode in ("accelerated", "compare"):
            _log(
                f"init accelerated segmentation adapter(base_url={args.base_url}, camera_yaml={camera_yaml})"
            )
            percep_fast = _AcceleratedSegToPoseAdapter(
                base_url=str(args.base_url), camera_yaml=camera_yaml
            )
        else:
            percep_fast = None

        if args.validate:
            percep_validate = percep_current if mode == "current" else percep_fast
            final_res = _run_validate(xarm, percep_validate, args)
        elif mode == "current":
            latencies, stage_samples, final_res = _run_pipeline("current", percep_current)
            _log(
                "current latency stats: "
                f"n={len(latencies)} avg={float(np.mean(latencies)):.3f}s "
                f"min={float(np.min(latencies)):.3f}s max={float(np.max(latencies)):.3f}s"
            )
            _log(f"current stage avg: {_stage_stats(stage_samples)}")
            final_res["stage_stats_sec"] = _stage_stats(stage_samples)
        elif mode == "accelerated":
            latencies, stage_samples, final_res = _run_pipeline("accelerated", percep_fast)
            _log(
                "accelerated latency stats: "
                f"n={len(latencies)} avg={float(np.mean(latencies)):.3f}s "
                f"min={float(np.min(latencies)):.3f}s max={float(np.max(latencies)):.3f}s"
            )
            _log(f"accelerated stage avg: {_stage_stats(stage_samples)}")
            final_res["stage_stats_sec"] = _stage_stats(stage_samples)
        else:
            lat_cur, stage_cur, res_cur = _run_pipeline("current", percep_current)
            lat_fast, stage_fast, res_fast = _run_pipeline("accelerated", percep_fast)
            cur_avg = float(np.mean(lat_cur))
            fast_avg = float(np.mean(lat_fast))
            speedup = (cur_avg / fast_avg) if fast_avg > 1e-9 else float("inf")
            cur_stage_stats = _stage_stats(stage_cur)
            fast_stage_stats = _stage_stats(stage_fast)
            _log(
                "compare summary: "
                f"current avg={cur_avg:.3f}s, accelerated avg={fast_avg:.3f}s, "
                f"speedup={speedup:.3f}x"
            )
            final_res = {
                "pipeline_version": "compare",
                "current": {
                    "stats_sec": {
                        "n": len(lat_cur),
                        "avg": float(np.mean(lat_cur)),
                        "min": float(np.min(lat_cur)),
                        "max": float(np.max(lat_cur)),
                    },
                    "stage_stats_sec": cur_stage_stats,
                    "sample_result": res_cur,
                },
                "accelerated": {
                    "stats_sec": {
                        "n": len(lat_fast),
                        "avg": float(np.mean(lat_fast)),
                        "min": float(np.min(lat_fast)),
                        "max": float(np.max(lat_fast)),
                    },
                    "stage_stats_sec": fast_stage_stats,
                    "sample_result": res_fast,
                },
                "speedup_x_current_over_accelerated": float(speedup),
            }
    finally:
        if rclpy.ok():
            rclpy.shutdown()

    assert final_res is not None

    final_res_safe = _json_safe(final_res)
    print(json.dumps(final_res_safe, indent=2, ensure_ascii=False))
    if args.json_out:
        out_path = os.path.abspath(str(args.json_out))
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(final_res_safe, f, indent=2, ensure_ascii=False)
        _log(f"saved result JSON: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

