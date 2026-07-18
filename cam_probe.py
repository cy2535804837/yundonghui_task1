#!/usr/bin/env python3
"""Diagnose RGBD capture for the grasp pipeline.

Run on the SAME machine where you run grasp_pose_grasp_execute (here: ubuntu).

It does two independent things:
  1) Counts raw messages on the color + depth topics using BEST_EFFORT
     (sensor_data) QoS, so we can see per-topic whether frames arrive.
  2) Calls the real capture_single_rgbd() used by the pipeline, so we know if
     the actual code path succeeds.

Usage:
    python3 cam_probe.py
    python3 cam_probe.py --rgb-topic /camera/color/image_raw \
                         --depth-topic /camera/depth/image_raw --secs 3
"""
from __future__ import annotations

import argparse
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from grasp_pose_generation.internal.ros_rgbd_capture import capture_single_rgbd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb-topic", default="/ob_camera_head/color/image_raw")
    ap.add_argument("--depth-topic", default="/ob_camera_head/depth/image_raw")
    ap.add_argument("--secs", type=float, default=3.0)
    args = ap.parse_args()

    rclpy.init()
    node = Node("cam_probe")

    counts = {"rgb": 0, "depth": 0}
    info = {"rgb": "", "depth": ""}

    def _mk(key):
        def _cb(msg: Image):
            counts[key] += 1
            if not info[key]:
                info[key] = f"{msg.width}x{msg.height} enc={msg.encoding}"
        return _cb

    node.create_subscription(Image, args.rgb_topic, _mk("rgb"), qos_profile_sensor_data)
    node.create_subscription(Image, args.depth_topic, _mk("depth"), qos_profile_sensor_data)

    print(f"[probe] counting messages for {args.secs:.1f}s (BEST_EFFORT)...")
    print(f"[probe]   rgb   topic: {args.rgb_topic}")
    print(f"[probe]   depth topic: {args.depth_topic}")
    t0 = time.monotonic()
    while rclpy.ok() and (time.monotonic() - t0) < args.secs:
        rclpy.spin_once(node, timeout_sec=0.05)

    print(f"[probe] rgb   frames: {counts['rgb']:3d}  {info['rgb']}")
    print(f"[probe] depth frames: {counts['depth']:3d}  {info['depth']}")

    print("[probe] measuring time-to-first-frame for FRESH subscriptions...")
    fresh = {"rgb": None, "depth": None}
    t_start = time.monotonic()

    def _mk_fresh(key):
        def _cb(_msg: Image):
            if fresh[key] is None:
                fresh[key] = time.monotonic() - t_start
        return _cb

    fr = node.create_subscription(Image, args.rgb_topic, _mk_fresh("rgb"), qos_profile_sensor_data)
    fd = node.create_subscription(Image, args.depth_topic, _mk_fresh("depth"), qos_profile_sensor_data)
    while rclpy.ok() and (time.monotonic() - t_start) < 10.0:
        rclpy.spin_once(node, timeout_sec=0.02)
        if fresh["rgb"] is not None and fresh["depth"] is not None:
            break
    node.destroy_subscription(fr)
    node.destroy_subscription(fd)
    print(f"[probe] fresh-sub first rgb   after: {fresh['rgb']}")
    print(f"[probe] fresh-sub first depth after: {fresh['depth']}")

    print("[probe] now calling the real capture_single_rgbd()...")
    frame = capture_single_rgbd(
        node, rgb_topic=args.rgb_topic, depth_topic=args.depth_topic, timeout_sec=args.secs
    )
    if frame is None:
        print("[probe] capture_single_rgbd -> None (FAILED)")
    else:
        print(
            f"[probe] capture_single_rgbd -> OK rgb={frame.rgb.shape} "
            f"depth={frame.depth_m.shape}"
        )

    node.destroy_node()
    rclpy.shutdown()

    ok = counts["rgb"] > 0 and counts["depth"] > 0 and frame is not None
    print(f"[probe] RESULT: {'OK' if ok else 'PROBLEM'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
