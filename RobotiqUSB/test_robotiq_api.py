#!/usr/bin/env python3
"""Simple test script for robotiq_api.py.

Examples:
  # ZMQ backend (default)
  python3 test_robotiq_api.py --backend zmq --server_ip 127.0.0.1 --server_set_port 4244 --server_get_port 4245

  # Modbus backend
  python3 test_robotiq_api.py --backend modbus_rtu --serial_port /dev/ttyUSB0 --slave_id 9
"""

from __future__ import annotations

import argparse
import time

from robotiq_api import create_gripper_controller


def main() -> None:
    parser = argparse.ArgumentParser(description="Test unified Robotiq API")
    parser.add_argument("--backend", choices=["zmq", "modbus_rtu"], default="zmq")

    # ZMQ options
    parser.add_argument("--server_ip", type=str, default="127.0.0.1")
    parser.add_argument("--server_set_port", type=int, default=4244)
    parser.add_argument("--server_get_port", type=int, default=4245)
    parser.add_argument("--recv_timeout_ms", type=int, default=200)
    parser.add_argument("--move_wait_timeout_s", type=float, default=4.0)
    parser.add_argument("--target_tolerance_pct", type=float, default=2.0)

    # Modbus options
    parser.add_argument("--serial_port", type=str, default="auto")
    parser.add_argument("--slave_id", type=int, default=9)
    parser.add_argument("--activate_on_connect", action="store_true")

    # Test motion options
    parser.add_argument("--target_pct", type=float, default=50.0)
    parser.add_argument("--speed_pct", type=float, default=80.0)
    parser.add_argument("--force_pct", type=float, default=80.0)
    args = parser.parse_args()

    controller = create_gripper_controller(
        backend=args.backend,
        serial_port=args.serial_port,
        slave_id=args.slave_id,
        server_ip=args.server_ip,
        server_set_port=args.server_set_port,
        server_get_port=args.server_get_port,
        recv_timeout_ms=args.recv_timeout_ms,
        move_wait_timeout_s=args.move_wait_timeout_s,
        target_tolerance_pct=args.target_tolerance_pct,
        activate_on_connect=args.activate_on_connect,
    )

    try:
        print(f"[INFO] backend={args.backend}")
        pos0 = controller.get_current_position_percent()
        print(f"[INFO] current position: {pos0}%")

        ok, req = controller.move_percent(args.target_pct, args.speed_pct, args.force_pct)
        print(f"[INFO] move_percent -> ok={ok}, requested={req}%")

        final_pos, status = controller.move_and_wait_percent(args.target_pct, args.speed_pct, args.force_pct)
        print(f"[INFO] move_and_wait_percent -> final={final_pos}%, status={status}")

        time.sleep(0.1)
        pos1 = controller.get_current_position_percent()
        print(f"[INFO] readback position: {pos1}%")

        stopped = controller.stop()
        print(f"[INFO] stop -> {stopped}")
    finally:
        controller.disconnect()
        print("[INFO] disconnected")


if __name__ == "__main__":
    main()

