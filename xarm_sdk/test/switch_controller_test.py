# 服务调用 压力测试
from math import sqrt
import rclpy
from xarm_sdk import XARM_manager
from xarm_sdk import MoveitCall, ActionCall, TopicPublisher
from xarm_sdk import ParamConfiger
from eai_manipulator_msgs.action import EndPosSingleTarget
import time
import numpy as np
'''
    启动前需要先启动XARM
    并且source XARM环境
    source ~/XARM/install/setup.bash
'''

def check_controller_active(xarm_manager, controller_name):
    active_controllers = xarm_manager.xarm_find_active_controllers()
    if controller_name in active_controllers:
        return True
    else:
        return False

def check_controller_inactive(xarm_manager, controller_name):
    inactive_controllers = xarm_manager.xarm_find_inactive_controllers()
    if controller_name in inactive_controllers:
        return True
    else:
        return False


def main():    # 初始化ROS2
    rclpy.init()
    
    # XARM 管理节点s
    xarm_manager = XARM_manager()

    # XARM 三件套
    action_call = ActionCall(xarm_manager)
    topic_publisher = TopicPublisher(xarm_manager)
    moveit_call = MoveitCall(xarm_manager)
    param_configer = ParamConfiger(xarm_manager)
    xarm_manager.xarm_deactivate_all_controller()

    controller_list = xarm_manager.xarm_find_inactive_controllers()
    # INSERT_YOUR_CODE
    # 去除endpose_single_arm_qp2_R_controller

    if "endpose_single_arm_qp2_R_controller" in controller_list:
        controller_list.remove("endpose_single_arm_qp2_R_controller")
    if "endpose_single_arm_qp2_L_controller" in controller_list:
        controller_list.remove("endpose_single_arm_qp2_L_controller")
    if "identify_load_L_controller" in controller_list:
        controller_list.remove("identify_load_L_controller")
    if "identify_load_R_controller" in controller_list:
        controller_list.remove("identify_load_R_controller")
    if "jointspace_dual_arm_controller" in controller_list:
        controller_list.remove("jointspace_dual_arm_controller")

    input("Press Enter to continue...")

    start_time = time.time()
    success_count = 0

    
    loop_count = 0

    while time.time() - start_time < 100:

        for controller_name in controller_list:
            xarm_manager.xarm_activate_controller(controller_name)
            res = check_controller_active(xarm_manager, controller_name)
            if not res:
                raise Exception(f"{controller_name} is not active")
            
            time.sleep(0.01)
            xarm_manager.xarm_deactivate_controller(controller_name)
            res = check_controller_inactive(xarm_manager, controller_name)
            if not res:
                raise Exception(f"{controller_name} is not inactive")
            
            success_count += 1
            time.sleep(0.01)
        loop_count += 1
        print(f"Loop count: {loop_count}, Success count: {success_count}, Time: {time.time() - start_time}")

if __name__ == "__main__":
    main()
