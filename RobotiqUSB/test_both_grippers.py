#!/usr/bin/env python3
"""Test both Robotiq grippers by name, one after the other.

For each gripper it: connects, activates (open/close cycle), then runs through a
short sequence of target positions, reporting the final position and motion
status for each move.

Examples:
  # Test every configured gripper with the default motion sequence
  python3 test_both_grippers.py

  # Test only one gripper
  python3 test_both_grippers.py --grippers left

  # Custom positions / speed / force
  python3 test_both_grippers.py --positions 0 50 100 0 --speed_pct 60 --force_pct 40

Motion status codes (gOBJ):
  3 = reached requested position (no object)
  1/2 = stopped on an object before target
  0 = still moving / timed out
"""

from __future__ import annotations

import argparse
import time

from robotiq_grippers import create_gripper, get_gripper_info, list_grippers, resolve_port


def test_one(name: str, positions, speed_pct: float, force_pct: float) -> bool:
    """Run the motion sequence on a single named gripper. Returns True on success."""
    info = get_gripper_info(name)
    print(f"\n{'=' * 70}")
    print(f"GRIPPER '{name}'  (slave {info.slave_id})")
    print(f"  {info.description}")

    try:
        port = resolve_port(name)
    except FileNotFoundError as exc:
        print(f"  [SKIP] {exc}")
        return False

    print(f"  port: {port}")
    print(f"{'=' * 70}")

    controller = create_gripper(name, activate_on_connect=True)
    try:
        start = controller.get_current_position_percent()
        print(f"  [INFO] start position: {start}%")
        for target in positions:
            final, status = controller.move_and_wait_percent(target, speed_pct, force_pct)
            note = {3: "reached", 2: "stopped on object (closing)",
                    1: "stopped on object (opening)", 0: "timed out"}.get(status, "?")
            print(f"  [MOVE] -> {target:>3}%   final={final:>3}%  status={status} ({note})")
            time.sleep(0.3)
        controller.stop()
        print(f"  [OK] '{name}' test complete")
        return True
    except Exception as exc:  # noqa: BLE001 - report and continue to next gripper
        print(f"  [FAIL] '{name}': {type(exc).__name__}: {exc}")
        return False
    finally:
        controller.disconnect()
        print(f"  [INFO] '{name}' disconnected")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test multiple Robotiq grippers by name")
    parser.add_argument(
        "--grippers",
        nargs="+",
        default=list_grippers(),
        help=f"Gripper names to test (default: all = {list_grippers()})",
    )
    parser.add_argument(
        "--positions",
        type=float,
        nargs="+",
        default=[0, 50, 100, 0],
        help="Sequence of target positions in percent (default: 0 50 100 0)",
    )
    parser.add_argument("--speed_pct", type=float, default=60.0)
    parser.add_argument("--force_pct", type=float, default=40.0)
    args = parser.parse_args()

    results = {}
    for name in args.grippers:
        results[name] = test_one(name, args.positions, args.speed_pct, args.force_pct)

    print(f"\n{'=' * 70}\nSUMMARY")
    for name, ok in results.items():
        print(f"  {name:6s}: {'PASS' if ok else 'FAIL/SKIP'}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
