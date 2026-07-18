"""
XArm控制器切换和硬件配置示例

本示例演示了如何使用XARM_manager进行以下操作：
1. 停用所有控制器
2. 查询控制器的状态（激活、未激活、未配置）
3. 激活和停用单个或多个控制器
4. 处理控制器资源冲突
5. 配置未配置的控制器

控制器资源冲突说明:
    - 每个控制器都占用特定的硬件资源（如left_arm、right_arm、head、body等）
    - 不能同时激活占用相同硬件资源的控制器
    - 例如：empty_L_controller和jointspace_arm_L_controller都占用left_arm资源，不能同时激活
    - xarm_activate_controller()方法会自动处理资源冲突，先停用冲突的控制器再激活新控制器

控制器状态:
    - active: 控制器已激活，正在运行
    - inactive: 控制器已配置但未激活
    - unconfigured: 控制器未配置，需要先配置才能激活

使用方法:
    # 确保ROS2环境已配置并source
    source ~/XARM/install/setup.bash
    
    # 运行示例
    python3 demo/1_contorller_switch.py

前置要求:
    - ROS2 Humble (或其他ROS2版本)
    - 已安装xarm_sdk包: pip install -e .
    - ROS2硬件节点正在运行 (EAIHardware相关服务)
    - controller_manager节点正在运行

注意事项:
    - 本示例仅演示控制器切换，不涉及实际的运动控制
"""

from re import S
import sys
import rclpy
from xarm_sdk import XARM_manager


def main():
    """
    主函数：演示XArm控制器的控制器切换流程
    """
    # 初始化ROS2
    rclpy.init()

    xarm_manager = XARM_manager()
    
    # 停用所有控制器
    xarm_manager.xarm_deactivate_all_controller()

    # 获取未配置的控制器
    unconfigured_controllers = xarm_manager.xarm_find_unconfigured_controllers()
    if len(unconfigured_controllers) > 0:
        # 正常情况下，不应该有未配置的控制器，如果有，再次配置所有控制器
        xarm_manager.get_logger().warn(f"warning: unconfigured_controllers: {unconfigured_controllers}")
        xarm_manager.configure_all_controllers()


    # 激活jointspace_arm_L_controller控制器
    xarm_manager.xarm_activate_controller('jointspace_arm_L_controller')

    # 获取激活的控制器
    activate_controllers = xarm_manager.xarm_find_active_controllers()
    xarm_manager.get_logger().info(f"activate_controllers: {activate_controllers}")

    # 获取未激活的控制器
    inactive_controllers = xarm_manager.xarm_find_inactive_controllers()
    xarm_manager.get_logger().info(f"inactive_controllers: {inactive_controllers}")
    
    # 高层次接口xarm_activate_controller可以自动解决资源冲突，在不同控制器之间直接切换
    xarm_manager.xarm_activate_controller('empty_L_controller')
    xarm_manager.xarm_activate_controller('endpose_single_arm_qp_L_controller')

    # 用完之后，停用控制器
    xarm_manager.xarm_deactivate_controller('endpose_single_arm_qp_L_controller')
    
    # 也可以关闭所有控制器
    xarm_manager.xarm_deactivate_all_controller()
    
    # 激活控制器时可以激活多个控制器
    xarm_manager.xarm_activate_controller(['empty_R_controller', 'endpose_single_arm_qp_L_controller'])

    # 但多个之间不能有资源冲突, 以下代码会报警
    # res = xarm_manager.xarm_activate_controller(['empty_L_controller', 'endpose_single_arm_qp_L_controller'])
    # if not res:
    #     xarm_manager.get_logger().error(f"error: activate_controllers: ['empty_L_controller', 'endpose_single_arm_qp_L_controller'] failed")

    
    xarm_manager.xarm_deactivate_all_controller()



    rclpy.shutdown()


if __name__ == "__main__":
    main()