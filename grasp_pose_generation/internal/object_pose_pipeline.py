from __future__ import annotations

import json
import os
import time
from typing import Dict, Optional

import numpy as np
import rclpy

from .cam_to_waist import head_pose_to_waist
from .ros_rgbd_capture import capture_single_rgbd


def _pose7_to_list(pose: np.ndarray) -> list:
    pose = np.asarray(pose, dtype=float).reshape(-1)
    if pose.shape[0] != 7:
        raise ValueError(f"Expected pose length 7, got {pose.shape}")
    return [float(x) for x in pose.tolist()]


def _rpy_dict_to_matrix(p: Dict[str, float]) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    T = np.eye(4, dtype=float)
    T[:3, 3] = [float(p["x"]), float(p["y"]), float(p["z"])]
    T[:3, :3] = R.from_euler("xyz", [p["roll"], p["pitch"], p["yaw"]], degrees=False).as_matrix()
    return T


def _matrix_to_rpy_dict(T: np.ndarray) -> Dict[str, float]:
    from scipy.spatial.transform import Rotation as R

    pos = T[:3, 3]
    roll, pitch, yaw = R.from_matrix(T[:3, :3]).as_euler("xyz", degrees=False)
    return {
        "x": float(pos[0]),
        "y": float(pos[1]),
        "z": float(pos[2]),
        "roll": float(roll),
        "pitch": float(pitch),
        "yaw": float(yaw),
    }


def _quat_to_rpy(q: list[float]) -> tuple[float, float, float]:
    from scipy.spatial.transform import Rotation as R

    qx, qy, qz, qw = [float(x) for x in q]
    return tuple(R.from_quat([qx, qy, qz, qw]).as_euler("xyz", degrees=False).tolist())


def _pose7_from_xyz_quat(xyz: list[float], quat: list[float]) -> list[float]:
    x, y, z = [float(v) for v in xyz]
    qx, qy, qz, qw = [float(v) for v in quat]
    return [x, y, z, qx, qy, qz, qw]


def tip_pose_to_tcp_pose(
    tip_pose_waist_rpy: Dict[str, float],
    tcp_to_tip_offset_m: np.ndarray = np.array([0.0, 0.0, -0.20], dtype=float),
) -> Dict[str, float]:
    T_waist_tip = _rpy_dict_to_matrix(tip_pose_waist_rpy)
    T_tcp_tip = np.eye(4, dtype=float)
    T_tcp_tip[:3, 3] = np.asarray(tcp_to_tip_offset_m, dtype=float).reshape(3)
    T_waist_tcp = T_waist_tip @ np.linalg.inv(T_tcp_tip)
    return _matrix_to_rpy_dict(T_waist_tcp)


def _rpy_dict_to_pose7(p: Dict[str, float]) -> list:
    from scipy.spatial.transform import Rotation as R

    qx, qy, qz, qw = R.from_euler("xyz", [p["roll"], p["pitch"], p["yaw"]], degrees=False).as_quat()
    return [float(p["x"]), float(p["y"]), float(p["z"]), float(qx), float(qy), float(qz), float(qw)]


def _save_debug_overlay(
    rgb: np.ndarray,
    save_dir: str,
    prefix: str,
    boxes: list,
    labels: list[str],
    highlight_idx: int = 0,
    text_lines: Optional[list[str]] = None,
) -> str:
    from PIL import Image, ImageDraw, ImageFont

    os.makedirs(save_dir, exist_ok=True)
    img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    for i, box in enumerate(boxes):
        if box is None:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        color = (0, 255, 0) if i != highlight_idx else (255, 0, 0)
        w = 2 if i != highlight_idx else 4
        draw.rectangle([x1, y1, x2, y2], outline=color, width=w)
        label = labels[i] if i < len(labels) else f"obj{i}"
        draw.text((x1 + 4, max(0, y1 - 22)), label, fill=color, font=font)
    if text_lines:
        x, y = 10, 10
        for line in text_lines:
            draw.text((x, y), line, fill=(255, 255, 0), font=font)
            y += 22
    out_path = os.path.join(save_dir, f"{prefix}_overlay.png")
    img.save(out_path)
    return out_path


def get_object_pose_in_waist_yaw_link(
    xarm,
    percep,
    prompts: list[str],
    rgb_topic: str,
    depth_topic: str,
    mode: str = "segment",
    waist_frame: str = "waist_yaw_link",
    head_frame: str = "head_roll_link",
    tf_timeout: float = 3.0,
    tf_retries: int = 3,
    tf_warmup_sec: float = 0.5,
    cam_timeout: float = 3.0,
    save_dir: Optional[str] = None,
    save_prefix: str = "debug",
    tcp_to_tip_offset_m: np.ndarray = np.array([0.0, 0.0, -0.20], dtype=float),
    orientation_policy: str = "current",
    use_current_tcp_rotation: Optional[bool] = None,
    rotation_arm: str = "right",
    grasp_yaw_offset: float = 0.0,
    max_grasp_yaw_delta: Optional[float] = None,
    verbose: bool = True,
    segment_confidence: float = 0.8,
) -> Optional[Dict]:
    def _v(msg: str) -> None:
        if verbose:
            print(f"[object_pose] {msg}", flush=True)

    policy = str(orientation_policy or "current").strip().lower()
    if use_current_tcp_rotation is not None:
        policy = "current" if bool(use_current_tcp_rotation) else "detected"
    if policy not in {"detected", "current", "yaw_only", "grasp", "grasp_topdown", "grasp_side"}:
        raise ValueError(f"Invalid orientation_policy='{orientation_policy}'.")

    frame = capture_single_rgbd(xarm, rgb_topic=rgb_topic, depth_topic=depth_topic, timeout_sec=cam_timeout)
    if frame is None:
        xarm.get_logger().error(f"Failed to capture RGBD from {rgb_topic} and {depth_topic}")
        return None
    d = frame.depth_m
    valid = np.isfinite(d) & (d > 0.05) & (d < 10.0)
    _v(
        f"RGB {frame.rgb.shape}, depth {d.shape}, valid-depth pixels: {int(valid.sum())}/{valid.size} "
        f"(median valid depth {float(np.median(d[valid])) if valid.any() else float('nan'):.3f} m)"
    )

    mode = (mode or "").lower().strip()
    if mode == "yolo":
        obj = percep.perception_pipeline_yolo(frame.rgb, frame.depth_m)
        if not obj:
            return None
        center = np.asarray(obj["pose"], dtype=float).reshape(-1)
        pose_head = [float(center[0]), float(center[1]), float(center[2]), 0.0, 0.0, 0.0, 1.0]
        class_name = obj.get("class name")
        bbox_2d = obj.get("bbox_2d")
        bbox_3d = None
        grasp_axes_head = None
        all_boxes = [bbox_2d] if bbox_2d is not None else []
        all_labels = [str(class_name or "yolo")]
    else:
        results_output, results_2d = percep.perception_pipeline(
            frame.rgb, frame.depth_m, prompts, confidence=segment_confidence
        )
        if not results_output:
            return None
        obj = results_output[0]
        pose_head = _pose7_to_list(obj["pose"])
        class_name = obj.get("class_name")
        bbox_2d = obj.get("bbox_2d")
        bbox_3d = obj.get("bbox_3d")
        grasp_axes_head = obj.get("grasp_axes")
        all_boxes, all_labels = [], []
        if isinstance(results_2d, dict) and isinstance(results_2d.get("results"), list):
            for r in results_2d["results"]:
                all_boxes.append(r.get("bbox"))
                all_labels.append(str(r.get("label", "obj")))
        elif bbox_2d is not None:
            all_boxes, all_labels = [bbox_2d], [str(class_name or "obj")]

    from xarm_sdk.tools import lookup_tf_once

    warmup_end = time.time() + max(0.0, float(tf_warmup_sec))
    while rclpy.ok() and time.time() < warmup_end:
        rclpy.spin_once(xarm, timeout_sec=0.05)
    T_wh = None
    for attempt in range(max(1, int(tf_retries))):
        T_wh = lookup_tf_once(xarm, target_frame=waist_frame, source_frame=head_frame, timeout=tf_timeout)
        if T_wh is not None:
            break
        end = time.time() + min(0.5 * (2**attempt), 2.0)
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(xarm, timeout_sec=0.05)
    if T_wh is None:
        return None
    trans, rot = T_wh
    from scipy.spatial.transform import Rotation as R

    T = np.eye(4, dtype=float)
    T[:3, 3] = np.asarray(trans, dtype=float)
    T[:3, :3] = R.from_quat(np.asarray(rot, dtype=float)).as_matrix()
    pose_waist = head_pose_to_waist(pose_head, T_waist_head=T)

    current_tcp = None
    if policy in {"current", "grasp", "grasp_topdown", "grasp_side"}:
        current_tcp = xarm.get_tcp_pose(arm=rotation_arm, base_frame=waist_frame, timeout=tf_timeout)

    R_waist_head = T[:3, :3]

    # --- Object long-axis heading in the waist frame (for adaptive grasp
    # orientation). The PCA major axis points along the object's longest
    # dimension (e.g. a banana's length). Projected into the waist XY plane:
    #   angle_from_x ~ 0deg  -> long axis points forward (waist +X): "vertical to body"
    #   angle_from_x ~ 90deg -> long axis points left/right (waist Y): "parallel to body"
    object_long_axis_waist = None
    object_long_axis_yaw_waist_deg = None
    object_angle_from_waist_x_deg = None
    if isinstance(grasp_axes_head, dict) and grasp_axes_head.get("major_axis") is not None:
        major_head = np.asarray(grasp_axes_head["major_axis"], dtype=float).reshape(3)
        major_waist = R_waist_head @ major_head
        object_long_axis_waist = [float(v) for v in major_waist.tolist()]
        vx, vy = float(major_waist[0]), float(major_waist[1])
        if (vx * vx + vy * vy) > 1e-12:
            # A line axis has no direction sign, so fold the heading into (-90, 90].
            yaw = float(np.degrees(np.arctan2(vy, vx)))
            if yaw > 90.0:
                yaw -= 180.0
            elif yaw <= -90.0:
                yaw += 180.0
            object_long_axis_yaw_waist_deg = yaw
            object_angle_from_waist_x_deg = float(np.degrees(np.arctan2(abs(vy), abs(vx))))

    grasp_yaw_waist = None
    if policy in {"grasp", "grasp_topdown", "grasp_side"}:
        if grasp_axes_head is not None:
            minor_head = np.asarray(grasp_axes_head["minor_axis"], dtype=float)
            minor_waist = R_waist_head @ minor_head
            raw_yaw = float(np.arctan2(minor_waist[1], minor_waist[0]))
            candidates = [raw_yaw, raw_yaw + np.pi / 2, raw_yaw + np.pi, raw_yaw - np.pi / 2]
            cur_yaw = _quat_to_rpy(current_tcp["rotation"])[2] if current_tcp else float(pose_waist.get("yaw", 0.0))

            def _angle_dist(a: float, b: float) -> float:
                return abs(((a - b) + np.pi) % (2 * np.pi) - np.pi)

            grasp_yaw_waist = float(min(candidates, key=lambda c: _angle_dist(c, cur_yaw)))
        else:
            grasp_yaw_waist = float(pose_waist.get("yaw", 0.0))
        grasp_yaw_waist += float(grasp_yaw_offset)
        if max_grasp_yaw_delta is not None and current_tcp is not None:
            _, _, cur_yaw = _quat_to_rpy(current_tcp["rotation"])
            delta = ((grasp_yaw_waist - cur_yaw) + np.pi) % (2 * np.pi) - np.pi
            limit = float(max_grasp_yaw_delta)
            if abs(delta) > limit:
                grasp_yaw_waist = float(cur_yaw + np.clip(delta, -limit, limit))

    if current_tcp is not None and policy == "current":
        quat_cmd = current_tcp["rotation"]
        r_cmd, p_cmd, y_cmd = _quat_to_rpy(quat_cmd)
        tip_pose_for_tcp = {
            "x": float(pose_waist["x"]),
            "y": float(pose_waist["y"]),
            "z": float(pose_waist["z"]),
            "roll": float(r_cmd),
            "pitch": float(p_cmd),
            "yaw": float(y_cmd),
        }
        tcp_pose_waist = tip_pose_to_tcp_pose(tip_pose_for_tcp, tcp_to_tip_offset_m=tcp_to_tip_offset_m)
        tcp_pose_cmd_rpy = {**tcp_pose_waist, "roll": float(r_cmd), "pitch": float(p_cmd), "yaw": float(y_cmd)}
        tcp_pose_cmd_pose7 = _pose7_from_xyz_quat(
            [tcp_pose_cmd_rpy["x"], tcp_pose_cmd_rpy["y"], tcp_pose_cmd_rpy["z"]],
            quat_cmd,
        )
    elif policy == "grasp" and current_tcp is not None:
        quat_cur = current_tcp["rotation"]
        r_cur, p_cur, _ = _quat_to_rpy(quat_cur)
        tip_pose_for_tcp = {
            "x": float(pose_waist["x"]),
            "y": float(pose_waist["y"]),
            "z": float(pose_waist["z"]),
            "roll": float(r_cur),
            "pitch": float(p_cur),
            "yaw": float(grasp_yaw_waist),
        }
        tcp_pose_cmd_rpy = tip_pose_to_tcp_pose(tip_pose_for_tcp, tcp_to_tip_offset_m=tcp_to_tip_offset_m)
        tcp_pose_cmd_pose7 = _rpy_dict_to_pose7(tcp_pose_cmd_rpy)
    elif policy == "yaw_only":
        tip_pose_for_tcp = {
            "x": float(pose_waist["x"]),
            "y": float(pose_waist["y"]),
            "z": float(pose_waist["z"]),
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": float(pose_waist["yaw"]),
        }
        tcp_pose_cmd_rpy = tip_pose_to_tcp_pose(tip_pose_for_tcp, tcp_to_tip_offset_m=tcp_to_tip_offset_m)
        tcp_pose_cmd_pose7 = _rpy_dict_to_pose7(tcp_pose_cmd_rpy)
    elif policy == "grasp_topdown":
        tip_pose_for_tcp = {
            "x": float(pose_waist["x"]),
            "y": float(pose_waist["y"]),
            "z": float(pose_waist["z"]),
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": float(grasp_yaw_waist),
        }
        tcp_pose_cmd_rpy = tip_pose_to_tcp_pose(tip_pose_for_tcp, tcp_to_tip_offset_m=tcp_to_tip_offset_m)
        tcp_pose_cmd_pose7 = _rpy_dict_to_pose7(tcp_pose_cmd_rpy)
    elif policy == "grasp_side":
        tip_pose_for_tcp = {
            "x": float(pose_waist["x"]),
            "y": float(pose_waist["y"]),
            "z": float(pose_waist["z"]),
            "roll": 0.0,
            "pitch": float(-np.pi / 2),
            "yaw": float(grasp_yaw_waist),
        }
        tcp_pose_cmd_rpy = tip_pose_to_tcp_pose(tip_pose_for_tcp, tcp_to_tip_offset_m=tcp_to_tip_offset_m)
        tcp_pose_cmd_pose7 = _rpy_dict_to_pose7(tcp_pose_cmd_rpy)
    else:
        tip_pose_for_tcp = pose_waist
        tcp_pose_cmd_rpy = tip_pose_to_tcp_pose(tip_pose_for_tcp, tcp_to_tip_offset_m=tcp_to_tip_offset_m)
        tcp_pose_cmd_pose7 = _rpy_dict_to_pose7(tcp_pose_cmd_rpy)

    current_tcp_pose7 = None
    if current_tcp is not None:
        ct, cr = current_tcp["translation"], current_tcp["rotation"]
        current_tcp_pose7 = [float(ct[0]), float(ct[1]), float(ct[2]), float(cr[0]), float(cr[1]), float(cr[2]), float(cr[3])]

    out = {
        "class_name": class_name,
        "pose_head_roll_link": pose_head,
        "pose_tip_waist_yaw_link_rpy": pose_waist,
        "pose_tip_used_for_tcp_conversion_rpy": tip_pose_for_tcp,
        "pose_tcp_waist_yaw_link_rpy": tcp_pose_cmd_rpy,
        "pose_tcp_waist_yaw_link_pose7": tcp_pose_cmd_pose7,
        "current_tcp_pose7": current_tcp_pose7,
        "tcp_minus_tip_waist_m": [
            float(tcp_pose_cmd_rpy["x"] - tip_pose_for_tcp["x"]),
            float(tcp_pose_cmd_rpy["y"] - tip_pose_for_tcp["y"]),
            float(tcp_pose_cmd_rpy["z"] - tip_pose_for_tcp["z"]),
        ],
        "tcp_to_tip_offset_m": [float(x) for x in np.asarray(tcp_to_tip_offset_m, dtype=float).reshape(3).tolist()],
        "orientation_policy": policy,
        "use_current_tcp_rotation": bool(policy == "current"),
        "rotation_source_arm": str(rotation_arm),
        "bbox_2d": bbox_2d,
        "bbox_3d": bbox_3d,
        "object_long_axis_waist": object_long_axis_waist,
        "object_long_axis_yaw_waist_deg": object_long_axis_yaw_waist_deg,
        "object_angle_from_waist_x_deg": object_angle_from_waist_x_deg,
    }
    if save_dir:
        def _json_safe(v):
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, (np.floating, np.integer)):
                return v.item()
            if isinstance(v, dict):
                return {k: _json_safe(val) for k, val in v.items()}
            if isinstance(v, (list, tuple)):
                return [_json_safe(x) for x in v]
            return v

        ts = time.strftime("%Y%m%d_%H%M%S")
        prefix = f"{save_prefix}_{ts}"
        text_lines = [
            f"class: {class_name}",
            f"tip waist: x={pose_waist['x']:.3f} y={pose_waist['y']:.3f} z={pose_waist['z']:.3f}",
            f"tip rpy: r={pose_waist['roll']:.3f} p={pose_waist['pitch']:.3f} y={pose_waist['yaw']:.3f}",
            f"tcp waist: x={tcp_pose_cmd_rpy['x']:.3f} y={tcp_pose_cmd_rpy['y']:.3f} z={tcp_pose_cmd_rpy['z']:.3f}",
        ]
        overlay_path = _save_debug_overlay(
            rgb=frame.rgb,
            save_dir=save_dir,
            prefix=prefix,
            boxes=all_boxes,
            labels=all_labels,
            highlight_idx=0,
            text_lines=text_lines,
        )
        json_path = os.path.join(save_dir, f"{prefix}_result.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(out), f, ensure_ascii=False, indent=2)
        out["debug_overlay_path"] = overlay_path
        out["debug_json_path"] = json_path
    return out

