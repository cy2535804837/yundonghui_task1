"""
XArm基于Topic的运动控制示例

本示例演示了如何使用TopicPublisher类通过ROS2 Topic发布关节命令来控制机器人：
1. 关节空间Topic控制
   - 通过发布关节命令到Topic来控制左右臂
   - 通过发布关节命令到Topic来控制身体、头部、腰部
   - 演示实时轨迹跟踪（正弦/余弦运动）

Topic控制 vs Action控制:
    - Topic控制：异步、高频、实时性好，适合连续轨迹跟踪
    - Action控制：同步、等待完成，适合点到点运动
    - Topic控制需要手动激活控制器，Action控制会自动切换控制器

TopicPublisher类特性:
    - 通过ROS2 Topic发布关节命令，实现实时控制
    - 支持高频发布（如100Hz），适合轨迹跟踪
    - 需要手动激活相应的控制器
    - 使用单例Publisher，避免重复创建资源浪费

使用方法:
    # 确保ROS2环境已配置并source
    source ~/XARM/install/setup.bash
    
    # 运行示例
    python3 demo/3_basic_move_topic.py


注意事项:
    - Topic控制是异步的，不会等待运动完成
    - 需要手动激活相应的控制器才能接收Topic命令
"""

import time
import rclpy
from xarm_sdk import XARM_manager
from xarm_sdk import ActionCall
from xarm_sdk import TopicPublisher

import numpy as np


def main():
    """
    主函数：演示XArm基于Topic的运动控制功能
    
    本函数展示了以下功能：
    1. 初始化XARM_manager、ActionCall和TopicPublisher
    2. 使用ActionCall将机器人移动到初始位置
    3. 激活控制器并切换到Topic控制模式
    4. 通过Topic发布关节命令实现实时轨迹跟踪：
       - 左右臂正弦/余弦运动（10秒）
       - 身体和头部运动（10秒）
       - 腰部俯仰运动（10秒）
       - 腰部摆动运动（10秒）
    
    控制流程:
        - 首先使用ActionCall将机器人移动到安全初始位置
        - 激活相应的控制器
        - 通过TopicPublisher发布关节命令实现连续运动
        - 每个运动阶段持续10秒，使用正弦/余弦函数生成轨迹
    
    Returns:
        None
    """
    # ========== 步骤1: 初始化ROS2和XARM_manager ==========
    # 初始化ROS2节点系统
    rclpy.init()
    
    # 创建XARM_manager实例
    xarm_manager = XARM_manager()
    
    # 创建ActionCall实例（用于初始定位）
    action_call = ActionCall(xarm_manager)
    
    # 创建TopicPublisher实例（用于Topic控制）
    topic_publisher = TopicPublisher(xarm_manager)

    # ========== 步骤2: 停用所有控制器并移动到初始位置 ==========
    # 停用所有控制器，确保从干净的状态开始
    xarm_manager.xarm_deactivate_all_controller()
    xarm_manager.get_logger().info("已停用所有控制器")

    # 使用ActionCall将机器人移动到零位（安全位置）
    xarm_manager.get_logger().info("移动机器人到零位...")
    action_call.jointspace_arm_L_controller([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    action_call.jointspace_arm_R_controller([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # 移动到初始工作位置
    xarm_manager.get_logger().info("移动机器人到初始工作位置...")
    action_call.jointspace_arm_L_controller([0.0, 1.18, 0.0, -1.3, 0.0, -0.13, 0.18])
    action_call.jointspace_arm_R_controller([0.0, -1.18, 0.0, -1.3, 0.0, -0.13, 0.18])

    # ========== 步骤3: 左右臂Topic控制 - 正弦/余弦运动 ==========
    # 激活左右臂控制器，准备接收Topic命令
    xarm_manager.xarm_activate_controller(['jointspace_arm_R_controller', 'jointspace_arm_L_controller'])
    xarm_manager.get_logger().info("已激活左右臂控制器，开始Topic控制...")

    # 构造初始关节位置
    target_positions_L = [0.0, 1.18, 0.0, -1.3, 0.0, -0.13, 0.18]
    target_positions_R = [0.0, -1.18, 0.0, -1.3, 0.0, -0.13, 0.18]

    # 实时轨迹跟踪：左右臂第一关节做正弦/余弦运动
    # 左臂第一关节：sin(t)，右臂第一关节：cos(t)
    # 频率：100Hz（每0.01秒发布一次），持续10秒
    xarm_manager.get_logger().info("左右臂开始正弦/余弦运动（10秒）...")
    start_time = time.time()
    t = 0.0
    while t <= 10.0:
        # 更新第一关节角度（正弦/余弦运动）
        target_positions_L[0] = np.sin(t)
        target_positions_R[0] = np.cos(t)
        
        # 发布关节命令到Topic
        topic_publisher.publish_jointspace_commands_L(target_positions_L)
        topic_publisher.publish_jointspace_commands_R(target_positions_R)
        
        # 控制发布频率（100Hz）
        time.sleep(0.01)
        t = time.time() - start_time
    
    # 回到初始位置
    xarm_manager.get_logger().info("左右臂返回初始工作位置...")
    action_call.jointspace_arm_L_controller([0.0, 1.18, 0.0, -1.3, 0.0, -0.13, 0.18])
    action_call.jointspace_arm_R_controller([0.0, -1.18, 0.0, -1.3, 0.0, -0.13, 0.18])

    # ========== 步骤4: 身体和头部Topic控制 ==========
    # 停用左右臂控制器，激活身体和头部控制器
    xarm_manager.xarm_deactivate_all_controller()
    xarm_manager.xarm_activate_controller('jointspace_body_controller')
    xarm_manager.xarm_activate_controller('jointspace_head_controller')
    xarm_manager.get_logger().info("已激活身体和头部控制器，开始Topic控制...")

    # 身体和头部实时运动
    # 头部三个关节都做正弦/余弦运动
    xarm_manager.get_logger().info("身体和头部开始运动（10秒）...")
    start_time = time.time()
    t = 0.0
    while t <= 10.0:
        # 身体关节位置（保持不动）
        body_target_positions = [0.0, 0.0, 0.0, 0.0]
        body_target_positions[3] = 0.0
        topic_publisher.publish_jointspace_commands_body(body_target_positions)

        # 头部关节位置（正弦/余弦运动，幅度0.3弧度）
        head_target_positions = [0.0, 0.0, 0.0]
        head_target_positions[0] = np.sin(t) * 0.3
        head_target_positions[1] = np.cos(t) * 0.3
        head_target_positions[2] = np.sin(t) * 0.3
        topic_publisher.publish_jointspace_commands_head(head_target_positions)
        
        time.sleep(0.01)
        t = time.time() - start_time

    # ========== 步骤5: 腰部俯仰Topic控制 ==========
    # 停用身体和头部控制器，激活腰部俯仰控制器
    xarm_manager.xarm_deactivate_all_controller()
    xarm_manager.xarm_activate_controller('jointspace_waist_pitch_controller')
    xarm_manager.get_logger().info("已激活腰部俯仰控制器，开始Topic控制...")

    # 腰部俯仰运动（正弦运动，幅度0.3弧度）
    xarm_manager.get_logger().info("腰部俯仰开始运动（10秒）...")
    start_time = time.time()
    t = 0.0
    while t <= 10.0:
        waist_pitch_target_positions = [0.0]
        waist_pitch_target_positions[0] = np.sin(t) * 0.3
        topic_publisher.publish_jointspace_commands_waist_pitch(waist_pitch_target_positions)
        time.sleep(0.01)
        t = time.time() - start_time

    # ========== 步骤6: 腰部摆动Topic控制 ==========
    # 停用腰部俯仰控制器，激活腰部摆动控制器
    xarm_manager.xarm_deactivate_all_controller()
    xarm_manager.xarm_activate_controller('jointspace_waist_yaw_controller')
    xarm_manager.get_logger().info("已激活腰部摆动控制器，开始Topic控制...")

    # 腰部摆动运动（正弦运动，幅度0.3弧度）
    xarm_manager.get_logger().info("腰部摆动开始运动（10秒）...")
    start_time = time.time()
    t = 0.0
    while t <= 10.0:
        waist_yaw_target_positions = [np.sin(t) * 0.3]
        topic_publisher.publish_jointspace_commands_waist_yaw(waist_yaw_target_positions)
        time.sleep(0.01)
        t = time.time() - start_time

    
    # 使用action回到初始位置
    xarm_manager.xarm_deactivate_all_controller()
    xarm_manager.get_logger().info("使用action控制各关节回到初始位置...")
    action_call = ActionCall(xarm_manager)
    # 身体、头部
    action_call.jointspace_body_controller([0.0, 0.0, 0.0, 0.0])
    action_call.jointspace_head_controller([0.0, 0.0, 0.0])
    # 左右手臂
    action_call.jointspace_arm_L_controller([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    action_call.jointspace_arm_R_controller([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # ========== 清理 ==========
    xarm_manager.get_logger().info("Topic控制演示完成")
    
    # 关闭ROS2节点系统
    rclpy.shutdown()


if __name__ == "__main__":
    main()
