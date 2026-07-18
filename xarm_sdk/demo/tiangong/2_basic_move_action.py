"""
XArm基本运动控制示例

本示例演示了如何使用ActionCall类进行基本的机械臂运动控制：
1. 关节空间控制（Joint Space Control）
   - 控制左臂和右臂到指定的关节角度位置
2. 笛卡尔空间控制（Cartesian Space Control）
   - 控制左臂和右臂到指定的笛卡尔坐标位置
   - 演示立方体轨迹跟踪运动

关节空间控制:
    - 直接指定每个关节的目标角度（弧度）
    - 左臂和右臂各有7个关节
    - 控制精度高，但需要知道目标关节角度

笛卡尔空间控制:
    - 指定末端执行器的目标位置和姿态（Pose）
    - 位置由x、y、z坐标表示
    - 姿态由四元数（quaternion）表示：x、y、z、w
    - 更直观，但需要逆运动学求解

ActionCall类特性:
    - 自动处理控制器切换，无需手动激活/停用控制器
    - 封装了ROS2 Action调用，简化了使用流程
    - 支持反馈回调函数，可以实时监控运动状态
    - ActionCall类中的函数均为同步函数，会等待动作完成后再返回

使用方法:
    # 确保ROS2环境已配置并source
    source ~/XARM/install/setup.bash
    
    # 运行示例
    python3 demo/2_basic_move.py

前置要求:
    - ROS2 Humble (或其他ROS2版本)
    - 已安装xarm_sdk包: pip install -e .
    - ROS2硬件节点正在运行 (EAIHardware相关服务)
    - controller_manager节点正在运行
    - 相应的控制器节点正在运行

注意事项:
    - 运动轨迹中的坐标值需要根据实际机器人配置调整
    - 姿态四元数需要归一化（x²+y²+z²+w²=1）
    - 建议在仿真环境中先测试，确认无误后再在真实机器人上运行
    - 运动过程中请保持安全距离，避免碰撞
"""

import rclpy
import time
from xarm_sdk import XARM_manager
from xarm_sdk import ActionCall
from geometry_msgs.msg import Pose


def main():
    """
    主函数：演示XArm基本运动控制功能
    
    本函数展示了以下功能：
    1. 初始化XARM_manager和ActionCall
    2. 关节空间控制：控制左右臂到指定关节角度
    3. 笛卡尔空间控制：控制左右臂到指定位置和姿态
    4. 轨迹跟踪：让机械臂末端沿立方体轨迹运动
    
    运动流程:
        - 首先将左右臂移动到初始关节位置
        - 然后移动到初始笛卡尔位置
        - 左臂沿立方体轨迹运动3圈
        - 右臂移动到初始位置
        - 右臂沿立方体轨迹运动3圈
    
    Returns:
        None
    """
    # ========== 步骤1: 初始化ROS2和XARM_manager ==========
    # 初始化ROS2节点系统
    rclpy.init()
    
    # 创建XARM_manager实例
    # XARM_manager负责管理控制器和硬件资源
    xarm_manager = XARM_manager()

    # 创建ActionCall实例
    # ActionCall封装了运动控制相关的Action调用，简化使用流程
    action_call = ActionCall(xarm_manager)
    
    # 停用所有控制器，确保从干净的状态开始
    xarm_manager.xarm_deactivate_all_controller()
    xarm_manager.get_logger().info("已停用所有控制器，准备开始运动控制演示")
    
    # ========== 步骤2: 关节空间控制 ==========
    # ActionCall中已经封装了控制器切换，所以不需要手动切换控制器
    # 调用控制方法时会自动激活相应的控制器
    
    # 控制左臂到指定的关节空间位置
    # 参数：7个关节的目标角度（弧度）
    # [肩部旋转, 肩部俯仰, 肘部, 前臂旋转, 腕部俯仰, 腕部旋转, 末端旋转]
    xarm_manager.get_logger().info("控制左臂到关节空间位置...")
    action_call.jointspace_arm_L_controller([0.0, 1.18, 0.0, -1.3, 0.0, -0.13, 0.18])

    xarm_manager.joint_state_update()
    left_arm_joint_angles = xarm_manager.xarm_left_arm_joint_angles()
    print("左臂关节角度：", left_arm_joint_angles)
    
    # 控制右臂到指定的关节空间位置
    # 注意：右臂的关节角度通常与左臂对称
    xarm_manager.get_logger().info("控制右臂到关节空间位置...")
    action_call.jointspace_arm_R_controller([0.0, -1.18, 0.0, -1.3, 0.0, -0.13, 0.18])
    xarm_manager.joint_state_update()
    right_arm_joint_angles = xarm_manager.xarm_right_arm_joint_angles()
    print("右臂关节角度：", right_arm_joint_angles)
    
    
    
    # 关闭ROS2节点系统
    rclpy.shutdown()


if __name__ == "__main__":
    main()

