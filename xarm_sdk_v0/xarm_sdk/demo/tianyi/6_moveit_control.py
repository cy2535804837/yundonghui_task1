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
    
    # ========== 第一部分：双臂随机运动 ==========
    print("\n第一部分：双臂随机运动")
    print("-" * 60)
    
    # 示例 1: 双臂随机运动（同步版本）
    print("\n示例 1: 双臂随机运动（同步）")
    print("说明: 双臂同时移动到随机位置，阻塞直到运动完成")
    result, error_code = moveit_call.dual_arm_random_run()
    print(f"结果: {result}, 错误码: {error_code}")

    
    # 示例 2: 双臂随机运动（异步版本）
    print("\n示例 2: 双臂随机运动（异步）")
    print("说明: 使用异步方式进行双臂随机运动，可通过 Future 获取结果")
    future = moveit_call.dual_arm_random_run_async()
    rclpy.spin_until_future_complete(xarm_manager, future)
    result, error_code = future.result()
    print(f"结果: {result}, 错误码: {error_code}")

    # ========== 第二部分：关节空间控制 ==========
    print("\n\n第二部分：关节空间控制")
    print("-" * 60)
    
    # 示例 3: 左臂关节空间控制（同步）
    print("\n示例 3: 左臂关节空间控制（同步）")
    print("说明: 控制左臂到指定关节角度，使用 MoveIt! 规划路径")
    left_angles = [0.0, 1.18, 0.0, -1.3, 0.0, -0.13, 0.18]
    print(f"目标关节角度: {left_angles}")
    result, error_code = moveit_call.left_arm_joint_angles(left_angles)
    print(f"结果: {result}, 错误码: {error_code}")

    
    # 示例 4: 右臂关节空间控制（异步）
    print("\n示例 4: 右臂关节空间控制（异步）")
    print("说明: 异步控制右臂到指定关节角度")
    right_angles = [0.0, -1.18, 0.0, -1.3, 0.0, -0.13, 0.18]
    print(f"目标关节角度: {right_angles}")
    future = moveit_call.right_arm_joint_angles_async(right_angles)
    rclpy.spin_until_future_complete(xarm_manager, future)
    result, error_code = future.result()
    print(f"结果: {result}, 错误码: {error_code}")

    # 示例 5: 双臂关节空间协同控制（异步）
    print("\n示例 5: 双臂关节空间协同控制（异步）")
    print("说明: 同时控制左右臂到指定关节角度，双臂协同运动")
    left_angles_dual = [1.0, 1.18, 0.0, -1.3, 0.0, -0.13, 0.18]
    right_angles_dual = [-1.0, -1.18, 0.0, -1.3, 0.0, -0.13, 0.18]
    print(f"左臂目标: {left_angles_dual}")
    print(f"右臂目标: {right_angles_dual}")
    future = moveit_call.dual_arm_joint_angles_async(left_angles_dual, right_angles_dual)
    rclpy.spin_until_future_complete(xarm_manager, future)
    result, error_code = future.result()
    print(f"结果: {result}, 错误码: {error_code}")
    
    # ========== 第三部分：末端轨迹控制（Waypoints）==========
    print("\n\n第三部分：末端轨迹控制（Waypoints）")
    print("-" * 60)
    print("说明: 通过指定末端位姿序列，让机器人沿着轨迹运动")
    
    # 定义路点（位置 + 姿态四元数）
    # 格式: [x, y, z, qx, qy, qz, qw]
    left_waypoints = [
        [0.26954124421585335, 0.3736418954013253, 0.10457946313346808, 
         -0.1529562230220876, -0.6448902522919694, -0.1707295374594559, 0.7290901051149286]
    ]
    right_waypoints = [
        [0.26954124421585335, -0.3736418954013253, 0.10457946313346808, 
         0.1529562230220876, -0.6448902522919694, 0.1707295374594559, 0.7290901051149286]
    ]
    
    # 示例 6: 左臂末端轨迹控制（异步）
    print("\n示例 6: 左臂末端轨迹控制（异步）")
    print(f"左臂路点数量: {len(left_waypoints)}")
    json_data = moveit_call.build_left_arm_waypoints_json(left_waypoints)
    print(f"生成的 JSON 数据: {json_data}")
    future = moveit_call.arm_waypoints_async(json_data)
    rclpy.spin_until_future_complete(xarm_manager, future)
    result, error_code = future.result()
    print(f"结果: {result}, 错误码: {error_code}")

    
    # 示例 7: 右臂末端轨迹控制（同步）
    print("\n示例 7: 右臂末端轨迹控制（同步）")
    print(f"右臂路点数量: {len(right_waypoints)}")
    json_data = moveit_call.build_right_arm_waypoints_json(right_waypoints)
    print(f"生成的 JSON 数据: {json_data}")
    result, error_code = moveit_call.arm_waypoints(json_data)
    print(f"结果: {result}, 错误码: {error_code}")

    # 示例 8: 双臂末端轨迹协同控制（异步）
    print("\n示例 8: 双臂末端轨迹协同控制（异步）")
    print("说明: 双臂同时沿着各自的轨迹运动")
    print(f"左臂路点数量: {len(left_waypoints)}")
    print(f"右臂路点数量: {len(right_waypoints)}")
    json_data = moveit_call.build_dual_arm_waypoints_json(left_waypoints, right_waypoints)
    print(f"生成的 JSON 数据: {json_data}")
    future = moveit_call.arm_waypoints_async(json_data)
    rclpy.spin_until_future_complete(xarm_manager, future)
    result, error_code = future.result()
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
