"""
双臂 XARM QPIK 测试脚本 (8_qpik_test.py)

功能概述：
    本脚本用于测试双臂机器人在不同参考系下的末端位姿逆解（QPIK）控制。
    依次完成：初始化与回零 -> 左臂三种 QPIK 场景 -> 右臂三种 QPIK 场景。

测试场景：
    1. 左臂：base 系下目标位姿、waist_yaw_link 系下目标位姿、带 offset 的 waist 系目标
    2. 右臂：同上三种场景（镜像 y 与四元数）

依赖：ROS2、xarm_sdk、geometry_msgs.msg.Pose
"""

import rclpy
from xarm_sdk import XARM_manager, ActionCall, MoveitCall, TopicPublisher, ParamConfiger
from geometry_msgs.msg import Pose


if __name__ == "__main__":
    # ---------- 初始化 ----------
    rclpy.init()
    xarm_manager = XARM_manager()

    action_call = ActionCall(xarm_manager)
    topic_publisher = TopicPublisher(xarm_manager)
    moveit_call = MoveitCall(xarm_manager)
    param_configer = ParamConfiger(xarm_manager)
    # 先关闭所有控制器，再由后续 action 按需启用
    xarm_manager.xarm_deactivate_all_controller()

    # ---------- 双臂运动到预备工作位置（关节空间） ----------
    # 左臂关节角 [rad]，右臂 y 方向镜像
    action_call.jointspace_arm_L_controller([0.0, 1.18, 0.0, -1.3, 0.0, -0.13, 0.18])
    action_call.jointspace_arm_R_controller([0.0, -1.18, 0.0, -1.3, 0.0, -0.13, 0.18])

    # ========== 左臂 QPIK 测试 ==========
    input("按回车继续：左臂 QPIK，from_frame=base, to_frame=left_tcp_link")

    target_pose = Pose()
    # 目标位置（米），相对于 base
    target_pose.position.x = 0.4
    target_pose.position.y = 0.5
    target_pose.position.z = 1.0
    # 目标姿态（四元数，需已归一化）
    target_pose.orientation.x = -0.5
    target_pose.orientation.y = 0.5
    target_pose.orientation.z = 0.5
    target_pose.orientation.w = -0.5

    action_call.endpose_single_arm_qpik_L_controller(
        target_pose,
        from_frame="base",
        to_frame="left_tcp_link",
    )

    input("按回车继续：左臂 QPIK，from_frame=waist_yaw_link（默认 to_frame）")

    target_pose.position.x = 0.4
    target_pose.position.y = 0.5
    target_pose.position.z = 0.32  # 相对 waist 的较低高度

    action_call.endpose_single_arm_qpik_L_controller(
        target_pose,
        from_frame="waist_yaw_link",
    )

    input("按回车继续：左臂 QPIK，waist_yaw_link 系 + offset [x,y,z,r,p,y]")

    action_call.endpose_single_arm_qpik_L_controller(
        target_pose,
        from_frame="waist_yaw_link",
        offset=[0.0, 0.0, -0.1, 0.0, 0.0, 0.0],  
    )

    # ========== 右臂 QPIK 测试 ==========
    input("按回车继续：右臂 QPIK，from_frame=base, to_frame=right_tcp_link")

    target_pose.position.x = 0.4
    target_pose.position.y = -0.5  # 右臂侧，y 为负
    target_pose.position.z = 1.0
    target_pose.orientation.x = -0.5
    target_pose.orientation.y = -0.5
    target_pose.orientation.z = 0.5
    target_pose.orientation.w = 0.5

    action_call.endpose_single_arm_qpik_R_controller(
        target_pose,
        from_frame="base",
        to_frame="right_tcp_link",
    )

    input("按回车继续：右臂 QPIK，from_frame=waist_yaw_link")

    target_pose.position.x = 0.4
    target_pose.position.y = -0.5
    target_pose.position.z = 0.32

    action_call.endpose_single_arm_qpik_R_controller(
        target_pose,
        from_frame="waist_yaw_link",
    )

    input("按回车继续：右臂 QPIK，waist_yaw_link 系 + offset")

    action_call.endpose_single_arm_qpik_R_controller(
        target_pose,
        from_frame="waist_yaw_link",
        offset=[0.0, 0.0, -0.1, 0.0, 0.0, 0.0],
    )

    rclpy.shutdown()
