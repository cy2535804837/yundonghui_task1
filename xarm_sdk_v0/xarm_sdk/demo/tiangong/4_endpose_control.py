"""
XArm笛卡尔空间Topic控制示例

本示例演示了如何使用TopicPublisher类通过ROS2 Topic发布笛卡尔空间位置命令来控制机器人末端执行器：
1. 笛卡尔空间Topic控制
   - 通过发布ArmTargetPose消息到Topic来控制左右臂末端位置
   - 演示实时轨迹跟踪（圆形轨迹运动）
   - 左右臂同步运动

笛卡尔空间Topic控制:
    - 通过发布 /endposetarget_L 和 /endposetarget_R 话题控制左右臂
    - 消息类型：eai_manipulator_msgs/msg/ArmTargetPose
    - 包含位置（x, y, z）和姿态（四元数）
    - 支持实时高频更新，适合连续轨迹跟踪


使用方法:
    # 确保ROS2环境已配置并source
    source ~/XARM/install/setup.bash
    
    # 运行示例
    python3 demo/4_endpose_control.py

注意事项:
    - Topic控制是异步的，不会等待运动完成
    - 需要手动激活相应的控制器才能接收Topic命令
    - 发布频率建议控制在100Hz以内（0.01秒间隔）
    - 姿态四元数需要归一化（x²+y²+z²+w²=1）
"""

import time
import rclpy
import numpy as np
from xarm_sdk import XARM_manager
from xarm_sdk import ActionCall
from geometry_msgs.msg import Pose
from xarm_sdk import TopicPublisher
from xarm_sdk import ParamConfiger
from xarm_sdk import lookup_tf_once

def quaternion_to_euler(x, y, z, w):
    """
    将四元数转换为欧拉角（roll, pitch, yaw）
    欧拉角单位为弧度
    欧拉角顺序: roll (x轴), pitch (y轴), yaw (z轴)
    """
    # roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    # pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)

    # yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    return roll_x, pitch_y, yaw_z

def euler_to_quaternion(roll_x, pitch_y, yaw_z):
    """
    将欧拉角转换为四元数
    四元数顺序: x, y, z, w
    """
    w = np.cos(roll_x / 2) * np.cos(pitch_y / 2) * np.cos(yaw_z / 2) + np.sin(roll_x / 2) * np.sin(pitch_y / 2) * np.sin(yaw_z / 2)
    x = np.sin(roll_x / 2) * np.cos(pitch_y / 2) * np.cos(yaw_z / 2) - np.cos(roll_x / 2) * np.sin(pitch_y / 2) * np.sin(yaw_z / 2)
    y = np.cos(roll_x / 2) * np.sin(pitch_y / 2) * np.cos(yaw_z / 2) + np.sin(roll_x / 2) * np.cos(pitch_y / 2) * np.sin(yaw_z / 2)
    z = np.cos(roll_x / 2) * np.cos(pitch_y / 2) * np.sin(yaw_z / 2) - np.sin(roll_x / 2) * np.sin(pitch_y / 2) * np.cos(yaw_z / 2)
    return x, y, z, w

def main():
    """
    主函数：演示XArm笛卡尔空间Topic控制功能
    
    本函数展示了以下功能：
    1. 初始化XARM_manager、ActionCall和TopicPublisher
    2. 使用ActionCall将机器人移动到初始位置
    3. 激活笛卡尔空间控制器并切换到Topic控制模式
    4. 通过Topic发布笛卡尔空间位置命令实现实时轨迹跟踪：
       - 左右臂末端沿圆形轨迹运动（20秒）
       - 左臂：y方向正弦运动，z方向余弦运动
       - 右臂：y方向反向正弦运动，z方向余弦运动
    
    控制流程:
        - 首先使用ActionCall将机器人移动到安全初始位置
        - 激活笛卡尔空间控制器（endpose_single_arm_qp_L_controller和endpose_single_arm_qp_R_controller）
        - 通过TopicPublisher发布笛卡尔空间位置命令实现连续运动
        - 运动持续20秒，使用正弦/余弦函数生成圆形轨迹
    
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

    param_configer = ParamConfiger(xarm_manager)
    
    # ========== 步骤2: 停用所有控制器并移动到初始位置 ==========
    # 停用所有控制器，确保从干净的状态开始
    xarm_manager.xarm_deactivate_all_controller()
    xarm_manager.get_logger().info("已停用所有控制器")

    # 使用ActionCall将机器人移动到初始位置
    xarm_manager.get_logger().info("移动机器人到初始位置...")
    
    
    # 移动左右臂到零位
    action_call.jointspace_arm_L_controller([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    action_call.jointspace_arm_R_controller([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # 移动到工作位置
    xarm_manager.get_logger().info("移动机器人到工作位置...")
    action_call.jointspace_arm_L_controller([0.0, 1.18, 0.0, -1.3, 0.0, -0.13, 0.18])
    action_call.jointspace_arm_R_controller([0.0, -1.18, 0.0, -1.3, 0.0, -0.13, 0.18])
    
    tf_result = lookup_tf_once(xarm_manager, "waist_yaw_link", "left_tcp_link")
    if tf_result is not None:
        translation, rotation = tf_result
    else:
        xarm_manager.get_logger().error("TF变换读取失败: waist_yaw_link <- left_tcp_link")
        translation, rotation = None, None
    # ========== 步骤3: 激活笛卡尔空间控制器 ==========
    # 激活左右臂笛卡尔空间控制器，准备接收Topic命令
    xarm_manager.xarm_activate_controller(['endpose_single_arm_qp_L_controller', 'endpose_single_arm_qp_R_controller'])
    xarm_manager.get_logger().info("已激活左右臂笛卡尔空间控制器，开始Topic控制...")
    
    # ========== 步骤4: 设置初始目标位姿 ==========
    # 创建左右臂的目标位姿对象
    target_pose_L = Pose()
    target_pose_R = Pose()
    
    # 设置左臂初始位置（单位：米）
    target_pose_L.position.x = 0.3
    target_pose_L.position.y = 0.6
    target_pose_L.position.z = 0.3
    # 设置左臂初始姿态（四元数，已归一化）
    x, y, z, w = euler_to_quaternion(np.pi/2, 0.0, -np.pi/2)
    target_pose_L.orientation.x = x
    target_pose_L.orientation.y = y
    target_pose_L.orientation.z = z
    target_pose_L.orientation.w = w
    roll_x, pitch_y, yaw_z = quaternion_to_euler(target_pose_L.orientation.x, target_pose_L.orientation.y, target_pose_L.orientation.z, target_pose_L.orientation.w)
    print(f"左臂初始欧拉角: roll_x={roll_x}, pitch_y={pitch_y}, yaw_z={yaw_z}")

    # 设置右臂初始位置（单位：米）
    target_pose_R.position.x = 0.3
    target_pose_R.position.y = -0.6  # 负值表示在机器人右侧
    target_pose_R.position.z = 0.3
    # 设置右臂初始姿态（四元数，已归一化）
    x, y, z, w = euler_to_quaternion(-np.pi/2, 0.0, np.pi/2)
    target_pose_R.orientation.x = x
    target_pose_R.orientation.y = y
    target_pose_R.orientation.z = z
    target_pose_R.orientation.w = w
    roll_x, pitch_y, yaw_z = quaternion_to_euler(target_pose_R.orientation.x, target_pose_R.orientation.y, target_pose_R.orientation.z, target_pose_R.orientation.w)
    print(f"右臂初始欧拉角: roll_x={roll_x}, pitch_y={pitch_y}, yaw_z={yaw_z}")

    # ========== 步骤5: 笛卡尔空间轨迹跟踪 ==========
    # 实时轨迹跟踪：左右臂末端沿圆形轨迹运动
    # 左臂：y方向正弦运动（0.4 ± 0.1），z方向余弦运动（0.1 ± 0.1）
    # 右臂：y方向反向正弦运动（-0.4 ∓ 0.1），z方向余弦运动（0.1 ± 0.1）
    # 频率：100Hz（每0.01秒发布一次），持续20秒
    xarm_manager.get_logger().info("左右臂开始圆形轨迹运动（20秒）...")
    start_time = time.time()
    while time.time() - start_time < 3.0:
        target_pose_L.position.y = 0.6 - (time.time() - start_time) / 3.0 * 0.2
        target_pose_R.position.y = -0.6 + (time.time() - start_time) / 3.0 * 0.2
        topic_publisher.publish_endposetarget_L(target_pose_L, from_frame = "waist_yaw_link")
        topic_publisher.publish_endposetarget_R(target_pose_R, from_frame = "waist_yaw_link")
        rclpy.spin_once(xarm_manager, timeout_sec=0.01)

    start_time = time.time()
    while time.time() - start_time < 20.0:
        current_time = time.time() - start_time
        
        # 更新左臂位置（圆形轨迹）
        target_pose_L.position.y = 0.4 + np.sin(current_time) * 0.1
        target_pose_L.position.z = 0.3 + np.cos(current_time) * 0.1
        
        # 更新右臂位置（圆形轨迹，y方向反向）
        target_pose_R.position.y = -0.4 - np.sin(current_time) * 0.1
        target_pose_R.position.z = 0.3 + np.cos(current_time) * 0.1
        
        # 发布笛卡尔空间位置命令到Topic
        topic_publisher.publish_endposetarget_L(target_pose_L, from_frame = "waist_yaw_link")
        topic_publisher.publish_endposetarget_R(target_pose_R, from_frame = "waist_yaw_link")
        
        # 控制发布频率（100Hz）
        rclpy.spin_once(xarm_manager, timeout_sec=0.01)
    

    # ========== 清理 ==========
    xarm_manager.get_logger().info("笛卡尔空间Topic控制演示完成")

    # 使用action返回零点
    xarm_manager.xarm_deactivate_all_controller()
    xarm_manager.get_logger().info("使用action控制各关节回到零点...")
    action_call = ActionCall(xarm_manager)
    # 左右手臂
    action_call.jointspace_arm_L_controller([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    action_call.jointspace_arm_R_controller([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # 关闭ROS2节点系统
    rclpy.shutdown()


if __name__ == "__main__":
    main()
