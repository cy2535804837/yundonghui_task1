"""
重力补偿拖动示教示例 (Gravity-compensation drag / teach)

本示例将机械臂切换到 mode 2（重力补偿），此时电机仅补偿自身重力，
你可以用手自由拖动机械臂；脚本会以固定频率持续打印当前关节角度
（可选打印速度/力矩，并可记录为 CSV 便于离线复现示教轨迹）。

运行模式说明 (见 7_enable_hardware.py):
- mode 0: 力位混合（默认，安全）
- mode 1: 电机速度环
- mode 2: 重力补偿            <-- 本脚本默认使用，双臂均可拖动
- mode 3: 位置环
- mode 4: 左手柔顺右手位置     <-- 只拖左臂
- mode 5: 左手位置右手柔顺     <-- 只拖右臂

用法:
    cd /home/ubuntu/niu
    python3 -m xarm_sdk.demo.tianyi.11_gravity_compensation_drag
    # 或直接运行
    python3 xarm_sdk/demo/tianyi/11_gravity_compensation_drag.py

    # 只拖右臂、20Hz 打印、并记录到 CSV
    python3 xarm_sdk/demo/tianyi/11_gravity_compensation_drag.py \
        --mode 5 --print-arm right --rate 20 --record /tmp/teach.csv

安全提示:
- 进入重力补偿前请先扶住机械臂。
- 退出时脚本会切回 mode 0（力位混合），机械臂会保持当前位置；
  此过程请继续扶稳，避免姿态突变。
- 仅在真实模式 (run_type == "real") 下硬件操作才会生效。

按 Ctrl+C 退出。
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from typing import List, Optional

import rclpy

from xarm_sdk import XARM_manager


# 单臂 7 个关节的简短标签（与 config.joints_name 顺序一致）
_ARM_JOINT_LABELS = [
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow_pitch",
    "elbow_yaw",
    "wrist_pitch",
    "wrist_roll",
]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="重力补偿拖动示教并打印关节角度")
    p.add_argument(
        "--mode", type=int, default=2, choices=[2, 4, 5],
        help="拖动模式: 2=双臂重力补偿(默认), 4=只拖左臂, 5=只拖右臂",
    )
    p.add_argument(
        "--print-arm", choices=["both", "left", "right"], default="both",
        help="打印哪条手臂的关节 (默认 both)",
    )
    p.add_argument("--rate", type=float, default=10.0, help="打印频率 Hz (默认 10)")
    p.add_argument("--degrees", action="store_true", help="以角度(°)打印, 默认弧度")
    p.add_argument("--print-velocity", action="store_true", help="额外打印关节速度")
    p.add_argument("--print-effort", action="store_true", help="额外打印关节力矩")
    p.add_argument("--record", default="", help="将关节角度记录到该 CSV 文件路径")
    p.add_argument(
        "--no-enable", action="store_true",
        help="跳过硬件使能(若机械臂已经使能)",
    )
    p.add_argument(
        "--keep-mode-on-exit", action="store_true",
        help="退出时不切回 mode 0 (默认会切回力位混合以保持位置)",
    )
    return p


def _fmt(vals: Optional[List[Optional[float]]], to_deg: bool) -> str:
    if vals is None:
        return "<no data>"
    import math

    out = []
    for v in vals:
        if v is None:
            out.append("  n/a ")
        else:
            x = math.degrees(v) if to_deg else v
            out.append(f"{x:7.3f}")
    return "[" + ", ".join(out) + "]"


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    unit = "°" if args.degrees else "rad"

    rclpy.init()
    xarm = XARM_manager()

    print("=" * 64)
    print("重力补偿拖动示教 (Gravity-compensation drag teach)")
    print("=" * 64)
    print(f"robot_type: {xarm.robot_type}   run_type: {xarm.run_type}")

    if xarm.run_type != "real":
        print("\n[警告] 当前为仿真模式 (run_type != 'real')，硬件操作不会生效。")
        print("       仍会尝试读取并打印关节状态。")

    # ---------- 进入重力补偿 ----------
    if xarm.run_type == "real":
        if not args.no_enable:
            print("\n[步骤1] 使能双臂硬件 ...")
            xarm.hardware_arm_enable(True)
            print("        ✓ 双臂已使能")
        mode_desc = {2: "双臂重力补偿", 4: "左臂柔顺(可拖)/右臂位置", 5: "左臂位置/右臂柔顺(可拖)"}
        print(f"\n[步骤2] 设置机械臂模式 mode {args.mode} ({mode_desc[args.mode]}) ...")
        print("        >> 进入前请先扶住机械臂 <<")
        xarm.hardware_arm_mode(args.mode)
        print("        ✓ 已进入拖动模式，现在可以用手拖动机械臂")
    else:
        print("\n[仿真] 跳过使能与模式设置")

    # ---------- CSV 记录 ----------
    csv_file = None
    csv_writer = None
    if args.record:
        csv_file = open(args.record, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        header = ["timestamp"]
        header += [f"L_{n}" for n in _ARM_JOINT_LABELS]
        header += [f"R_{n}" for n in _ARM_JOINT_LABELS]
        csv_writer.writerow(header)
        print(f"\n[记录] 关节角度将写入: {args.record}")

    # 先阻塞式更新一次，确保有数据
    xarm.joint_state_update()

    period = 1.0 / max(0.1, float(args.rate))
    print(f"\n开始打印关节角度 (单位 {unit}, 频率 {args.rate} Hz)。按 Ctrl+C 退出。\n")

    try:
        while rclpy.ok():
            t0 = time.time()
            # 处理订阅回调以刷新 latest_joint_state
            rclpy.spin_once(xarm, timeout_sec=period)

            left = xarm.xarm_left_arm_joint_angles()
            right = xarm.xarm_right_arm_joint_angles()
            stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

            if args.print_arm in ("both", "left"):
                print(f"[{stamp}] L pos {unit}: {_fmt(left, args.degrees)}")
                if args.print_velocity:
                    print(f"            L vel    : {_fmt(xarm.xarm_left_arm_joint_velocities(), args.degrees)}")
                if args.print_effort:
                    print(f"            L effort : {_fmt(xarm.xarm_left_arm_joint_efforts(), False)}")
            if args.print_arm in ("both", "right"):
                print(f"[{stamp}] R pos {unit}: {_fmt(right, args.degrees)}")
                if args.print_velocity:
                    print(f"            R vel    : {_fmt(xarm.xarm_right_arm_joint_velocities(), args.degrees)}")
                if args.print_effort:
                    print(f"            R effort : {_fmt(xarm.xarm_right_arm_joint_efforts(), False)}")

            if csv_writer is not None:
                row = [time.time()]
                row += list(left) if left is not None else [None] * 7
                row += list(right) if right is not None else [None] * 7
                csv_writer.writerow(row)

            # 维持目标频率
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        print("\n\n[退出] 收到 Ctrl+C")
    finally:
        if csv_file is not None:
            csv_file.close()
            print(f"[记录] 已保存: {args.record}")

        # # 安全退出：切回 mode 0（力位混合），机械臂保持当前位置
        # if xarm.run_type == "real" and not args.keep_mode_on_exit:
        #     print(">> 请继续扶稳机械臂，正在切回 mode 0 (力位混合) ...")
        #     try:
        #         xarm.hardware_arm_mode(0)
        #         print("   ✓ 已切回 mode 0，机械臂保持当前位置（电机仍使能）")
        #     except Exception as e:  # noqa: BLE001
        #         print(f"   [警告] 切回 mode 0 失败: {e!r}")

        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
