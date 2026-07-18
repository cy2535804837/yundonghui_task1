from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


@dataclass
class RgbdFrame:
    rgb: np.ndarray
    depth_m: np.ndarray


def _depth_to_meters(depth: np.ndarray, encoding: str) -> np.ndarray:
    enc = (encoding or "").lower()
    if enc == "16uc1":
        return (depth.astype(np.float32) / 1000.0).astype(np.float32)
    if enc == "32fc1":
        return depth.astype(np.float32)
    if enc in ("mono16",):
        return (depth.astype(np.float32) / 1000.0).astype(np.float32)
    raise ValueError(f"Unsupported depth encoding: {encoding}")


def capture_single_rgbd(
    node: Node,
    rgb_topic: str,
    depth_topic: str,
    timeout_sec: float = 3.0,
) -> Optional[RgbdFrame]:
    try:
        from cv_bridge import CvBridge
    except Exception as e:
        raise ImportError(
            "cv_bridge is required. Install: sudo apt install ros-humble-cv-bridge python3-opencv"
        ) from e

    bridge = CvBridge()
    rgb_msg: Optional[Image] = None
    depth_msg: Optional[Image] = None

    def _rgb_cb(msg: Image):
        nonlocal rgb_msg
        rgb_msg = msg

    def _depth_cb(msg: Image):
        nonlocal depth_msg
        depth_msg = msg

    # Camera image topics (e.g. the Orbbec driver) publish with SENSOR_DATA
    # (BEST_EFFORT) QoS. A RELIABLE subscriber will not match a BEST_EFFORT
    # publisher, so use BEST_EFFORT here -- it is compatible with both reliable
    # and best-effort publishers.
    rgb_sub = node.create_subscription(Image, rgb_topic, _rgb_cb, qos_profile_sensor_data)
    depth_sub = node.create_subscription(Image, depth_topic, _depth_cb, qos_profile_sensor_data)
    start = node.get_clock().now()
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.05)
        if rgb_msg is not None and depth_msg is not None:
            break
        if (node.get_clock().now() - start).nanoseconds / 1e9 > timeout_sec:
            break

    node.destroy_subscription(rgb_sub)
    node.destroy_subscription(depth_sub)
    if rgb_msg is None or depth_msg is None:
        return None

    rgb_bgr = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
    rgb = rgb_bgr[..., ::-1].copy()
    depth_np = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
    depth_m = _depth_to_meters(np.asarray(depth_np), depth_msg.encoding)
    return RgbdFrame(rgb=rgb, depth_m=depth_m)

