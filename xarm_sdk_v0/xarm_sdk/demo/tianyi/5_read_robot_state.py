import time
import rclpy
from xarm_sdk import XARM_manager, action_caller
from xarm_sdk import ActionCall
from eai_manipulator_msgs.action import JointSpace, EndPosSingleTarget
from geometry_msgs.msg import Pose
from xarm_sdk import TopicPublisher
from xarm_sdk import topic_subscriber
from sensor_msgs.msg import JointState


if __name__ == "__main__":
    rclpy.init()
    xarm_manager = XARM_manager()
    action_call = ActionCall(xarm_manager)


    topic_publisher = TopicPublisher(xarm_manager)
    xarm_manager.xarm_deactivate_all_controller()

    xarm_manager.hardware_arm_enable(True)
    xarm_manager.hardware_arm_mode(3)

    # 使用ActionCall将机器人移动到零位（安全位置）
    # xarm_manager.get_logger().info("移动机器人到零位...")
    # action_call.jointspace_arm_L_controller([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # action_call.jointspace_arm_R_controller([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # # 用joint_state_update阻塞式更新一次
    # xarm_manager.joint_state_update()
    # left_arm_joint_angles = xarm_manager.xarm_left_arm_joint_angles()
    # right_arm_joint_angles = xarm_manager.xarm_right_arm_joint_angles()
    # # 应该全是0
    # print("左臂关节角度：", left_arm_joint_angles)
    # print("右臂关节角度：", right_arm_joint_angles)

    # 使用ActionCall将机器人移动到初始工作位置
    xarm_manager.get_logger().info("移动机器人到初始工作位置...")

    # left_target_positions = [0.0, 1.18, 0.0, -1.3, 0.0, -0.13, 0.18]
    # right_target_positions = [0.0, -1.18, 0.0, -1.3, 0.0, -0.13, 0.18]

    # action_call.jointspace_dual_arm_controller(left_target_positions, right_target_positions)

    xarm_manager.get_logger().info("移动机器人到初始工作位置...")
    action_call.jointspace_arm_L_controller([0.0, 1.18, 0.0, -1.3, 0.0, -0.13, 0.18])
    # action_call.jointspace_arm_R_controller([0.0, -1.18, 0.0, -1.3, 0.0, -0.13, 0.18])

    xarm_manager.hardware_arm_mode(2)
    # # 用joint_state_update阻塞式更新一次
    # xarm_manager.joint_state_update()
    # left_arm_joint_angles = xarm_manager.xarm_left_arm_joint_angles()
    # right_arm_joint_angles = xarm_manager.xarm_right_arm_joint_angles()
    # # 应该是上面这些值
    # print("左臂关节角度：", left_arm_joint_angles)
    # print("右臂关节角度：", right_arm_joint_angles)

    # xarm_manager.get_logger().info("左右臂开始圆形轨迹运动（20秒）...")

    # # 创建左右臂的目标位姿对象
    # target_pose_L = Pose()
    # target_pose_R = Pose()
    
    # # 设置左臂初始位置（单位：米）
    # target_pose_L.position.x = 0.4
    # target_pose_L.position.y = 0.5
    # target_pose_L.position.z = 1.0
    # # 设置左臂初始姿态（四元数，已归一化）
    # target_pose_L.orientation.x = 0.5133696794509888
    # target_pose_L.orientation.y = -0.5630228519439697
    # target_pose_L.orientation.z = -0.47574561834335327
    # target_pose_L.orientation.w = 0.4394572377204895

    # # 设置右臂初始位置（单位：米）
    # target_pose_R.position.x = 0.4
    # target_pose_R.position.y = -0.5  # 负值表示在机器人右侧
    # target_pose_R.position.z = 1.0
    # # 设置右臂初始姿态（四元数，已归一化）
    # target_pose_R.orientation.x = -0.4877980947494507
    # target_pose_R.orientation.y = -0.4983730912208557
    # target_pose_R.orientation.z = 0.512197732925415
    # target_pose_R.orientation.w = 0.5013290047645569

    # xarm_manager.xarm_activate_controller(['endpose_single_arm_qp_L_controller', 'endpose_single_arm_qp_R_controller'])
    # xarm_manager.get_logger().info("已激活左右臂笛卡尔空间控制器，开始Topic控制...")
    # import numpy as np
    start_time = time.time()
    while time.time() - start_time < 20.0:
    #     current_time = time.time()
        
    #     # 更新左臂位置（圆形轨迹）
    #     target_pose_L.position.y = 0.5 + np.sin(current_time) * 0.1
    #     target_pose_L.position.z = 1.0 + np.cos(current_time) * 0.1
        
    #     # 更新右臂位置（圆形轨迹，y方向反向）
    #     target_pose_R.position.y = -0.5 - np.sin(current_time) * 0.1
    #     target_pose_R.position.z = 1.0 + np.cos(current_time) * 0.1
        
    #     # 发布笛卡尔空间位置命令到Topic
    #     topic_publisher.publish_endposetarget_L(target_pose_L)
    #     topic_publisher.publish_endposetarget_R(target_pose_R)
        
    #     rclpy.spin_once(xarm_manager, timeout_sec=0.01)
    #     left_arm_joint_angles = xarm_manager.xarm_left_arm_joint_angles()
    #     right_arm_joint_angles = xarm_manager.xarm_right_arm_joint_angles()
        
    #     print("左臂关节角度：", left_arm_joint_angles)
    #     print("右臂关节角度：", right_arm_joint_angles)
    

    # xarm_manager.get_logger().info("移动机器人到零位...")
    # left_target_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # right_target_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # action_call.jointspace_dual_arm_controller(left_target_positions, right_target_positions)
        xarm_manager.joint_state_update()
        left_arm_joint_angles = xarm_manager.xarm_left_arm_joint_angles()
        right_arm_joint_angles = xarm_manager.xarm_right_arm_joint_angles()
        
        print("左臂关节角度：", left_arm_joint_angles)
        print("右臂关节角度：", right_arm_joint_angles)


