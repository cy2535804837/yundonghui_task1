"""
MoveIt! 运动规划控制示例

本示例演示如何使用 MoveitCall 类进行基于 MoveIt! 的运动规划控制。

功能演示:
1. 双臂随机运动（同步和异步）
2. 单臂关节空间控制（左臂/右臂，同步/异步）
3. 双臂关节空间协同控制
4. 单臂末端轨迹控制（waypoints）
5. 双臂末端轨迹协同控制


前置条件:
- 已安装并启动 XARM 环境
- 已启动 ROS2 和相关控制器

"""

import rclpy
from xarm_sdk import XARM_manager
from xarm_sdk import MoveitCall, ActionCall


def main():
    """
    主函数：演示 MoveIt! 运动规划控制的各种功能
    """
    # 初始化 ROS2
    rclpy.init()
    
    # 创建管理器和控制接口
    xarm_manager = XARM_manager()
    moveit_call = MoveitCall(xarm_manager)
    action_call = ActionCall(xarm_manager)
    
    print("=" * 60)
    print("MoveIt! 运动规划控制示例")
    print("=" * 60)
    
    # ========== 第三部分：末端轨迹控制（Waypoints）==========
    print("\n\n第三部分：末端轨迹控制（Waypoints）")
    print("-" * 60)
    print("说明: 通过指定末端位姿序列，让机器人沿着轨迹运动")
    
    # 定义路点（位置 + 姿态四元数）
    right_waypoints = [
        [0.2998808398681925, -0.16124054291397036, 0.13272045818504208, 0.4708, 0.5384, -0.5371, -0.4473]
    ]

    right_waypoint_2 = [
        [0.35, -0.16124054291397036, 0.13272045818504208, 0.4708, 0.5384, -0.5371, -0.4473]
    ]

    
    # 示例 7: 右臂末端轨迹控制（同步）
    print("\n示例 7: 右臂末端轨迹控制（同步）")
    print(f"右臂路点数量: {len(right_waypoints)}")
    json_data = moveit_call.build_right_arm_waypoints_json(right_waypoints)
    print(f"生成的 JSON 数据: {json_data}")
    result, error_code = moveit_call.arm_waypoints(json_data)
    print(f"结果: {result}, 错误码: {error_code}")


    json_data = moveit_call.build_right_arm_waypoints_json(right_waypoint_2)
    print(f"生成的 JSON 数据: {json_data}")
    result, error_code = moveit_call.arm_waypoints(json_data)
    print(f"结果: {result}, 错误码: {error_code}")

    
    # ========== 总结 ==========
    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)
    print("\n重要提示:")
    print("- MoveIt! 会自动进行路径规划和碰撞检测")
    print("- 返回值: (result, error_code)")
    print("  - result: True 表示成功，False 表示失败")
    print("  - error_code: 错误代码，0 表示无错误")
    print("- 同步方法会阻塞直到运动完成")
    print("- 异步方法返回 Future，需要使用 spin_until_future_complete 等待")
    print("\n参考文档: README.md")
    
    # 关闭 ROS2
    rclpy.shutdown()


if __name__ == "__main__":
    main()
