
import time
import rclpy
import numpy as np
from xarm_sdk import XARM_manager
from xarm_sdk import ActionCall
from geometry_msgs.msg import Pose
from xarm_sdk import TopicPublisher


def main():
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

    # 使用ActionCall将机器人移动到初始位置
    xarm_manager.get_logger().info("移动机器人到初始位置...")
    
    # 移动身体和头部到零位
    action_call.jointspace_body_controller([0.0, 0.0, 0.0, 0.0])
    action_call.jointspace_head_controller([0.0, 0.0, 0.0])
    
    # 移动左右臂到零位
    action_call.jointspace_arm_L_controller([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    action_call.jointspace_arm_R_controller([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # 测试
    xarm_manager.xarm_activate_controller(['jointspace_single_arm_qpik_L_controller', 'jointspace_single_arm_qpik_R_controller'])
    arm_L= [0.0, 1.18, 0.0, -1.3, 0.0, -0.13, 0.18]
    arm_R= [0.0, -1.18, 0.0, -1.3, 0.0, -0.13, 0.18]
    start_time = time.time()
    while time.time() - start_time < 2.0:
        arm_L[1] = 1.18 + np.sin(time.time())
        arm_R[1] = -1.18 - np.sin(time.time())
        topic_publisher.publish_jointspace_commands_L(arm_L)
        topic_publisher.publish_jointspace_commands_R(arm_R)
        time.sleep(0.01)
    input("按Enter键继续...")  # 等待用户确认后继续执行，确保控制器已激活

    action_call.jointspace_arm_L_controller([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    action_call.jointspace_arm_R_controller([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # 测试
    xarm_manager.xarm_activate_controller(['jointspace_single_arm_qpik_L_controller', 'jointspace_single_arm_qpik_R_controller'])
    arm_L= [0.0, 1.18, 0.0, -1.3, 0.0, -0.13, 0.18]
    arm_R= [0.0, -1.18, 0.0, -1.3, 0.0, -0.13, 0.18]
    start_time = time.time()
    while time.time() - start_time < 2.0:
        arm_L[1] = 1.18 + np.sin(time.time())
        arm_R[1] = -1.18 - np.sin(time.time())
        topic_publisher.publish_jointspace_commands_L(arm_L)
        topic_publisher.publish_jointspace_commands_R(arm_R)
        time.sleep(0.01)

    
    # 关闭ROS2节点系统
    rclpy.shutdown()


if __name__ == "__main__":
    main()
