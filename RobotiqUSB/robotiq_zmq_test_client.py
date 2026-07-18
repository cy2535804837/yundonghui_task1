#!/usr/bin/env python3
"""Simple ZMQ test client for robotiq_node_zmq.py.

Usage example:
python3 robotiq_zmq_test_client.py \
  --server_ip 127.0.0.1 \
  --server_set_port 4254 \
  --server_get_port 4255 \
  --target_joint 0.5 \
  --speed 255 \
  --force 255
"""

from __future__ import annotations

import argparse
import json
import time

import zmq


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Robotiq ZMQ server")
    parser.add_argument("--server_ip", type=str, default="127.0.0.1")
    parser.add_argument("--server_set_port", type=int, default=4244)
    parser.add_argument("--server_get_port", type=int, default=4245)
    parser.add_argument("--target_joint", type=float, default=0.1, help="0.0 ~ 1.0")
    parser.add_argument("--speed", type=int, default=255, help="0 ~ 255")
    parser.add_argument("--force", type=int, default=255, help="0 ~ 255")
    parser.add_argument("--target_joint_percent", type=float, default=None, help="0 ~ 100 (overrides --target_joint)")
    parser.add_argument("--speed_percent", type=float, default=None, help="0 ~ 100 (overrides --speed)")
    parser.add_argument("--force_percent", type=float, default=None, help="0 ~ 100 (overrides --force)")
    parser.add_argument("--read_seconds", type=float, default=3.0, help="How long to read state updates")
    parser.add_argument("--recv_timeout_ms", type=int, default=500, help="SUB recv timeout in milliseconds")
    args = parser.parse_args()

    def pct_to_255(value: float) -> int:
        return max(0, min(255, int(round(float(value) * 255.0 / 100.0))))

    target_joint = args.target_joint
    speed = args.speed
    force = args.force

    if args.target_joint_percent is not None:
        if not (0.0 <= args.target_joint_percent <= 100.0):
            raise ValueError("--target_joint_percent must be in [0, 100]")
        target_joint = args.target_joint_percent / 100.0
    if args.speed_percent is not None:
        if not (0.0 <= args.speed_percent <= 100.0):
            raise ValueError("--speed_percent must be in [0, 100]")
        speed = pct_to_255(args.speed_percent)
    if args.force_percent is not None:
        if not (0.0 <= args.force_percent <= 100.0):
            raise ValueError("--force_percent must be in [0, 100]")
        force = pct_to_255(args.force_percent)

    if not (0.0 <= target_joint <= 1.0):
        raise ValueError("--target_joint must be in [0.0, 1.0]")
    if not (0 <= speed <= 255):
        raise ValueError("--speed must be in [0, 255]")
    if not (0 <= force <= 255):
        raise ValueError("--force must be in [0, 255]")

    context = zmq.Context()
    push_socket = context.socket(zmq.PUSH)
    sub_socket = context.socket(zmq.SUB)

    try:
        push_addr = f"tcp://{args.server_ip}:{args.server_set_port}"
        sub_addr = f"tcp://{args.server_ip}:{args.server_get_port}"

        push_socket.connect(push_addr)
        sub_socket.connect(sub_addr)
        sub_socket.setsockopt(zmq.SUBSCRIBE, b"")
        sub_socket.setsockopt(zmq.RCVTIMEO, args.recv_timeout_ms)

        # Give SUB a brief moment to finish subscription setup.
        time.sleep(0.1)

        command = {
            "target_joint": float(target_joint),
            "speed": int(speed),
            "force": int(force),
        }
        push_socket.send_json(command)
        print(f"Sent command: {command}")

        deadline = time.time() + args.read_seconds
        got_any_state = False
        while time.time() < deadline:
            try:
                parts = sub_socket.recv_multipart()
                if not parts:
                    continue
                state = json.loads(parts[0].decode("utf-8"))
                print(f"State: {state}")
                got_any_state = True
            except zmq.Again:
                # Timeout: keep waiting until deadline.
                pass

        if not got_any_state:
            print("No state message received during read window.")
    finally:
        push_socket.close(0)
        sub_socket.close(0)
        context.term()


if __name__ == "__main__":
    main()
