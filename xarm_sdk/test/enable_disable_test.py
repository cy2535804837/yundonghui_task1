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

def check_arm_enable(debug_info):
    import re
    match = re.search(r"\[arm_enable:\s*(\d+)\]", debug_info)
    if match:
        status = match.group(1)
        if status == "1":
            return True
        else:
            return False
    else:
        return False

def check_arm_disable(debug_info):
    import re
    match = re.search(r"\[arm_enable:\s*(\d+)\]", debug_info)
    if match:
        status = match.group(1)
        if status == "0":
            return True
        else:
            return False
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



    input("Press Enter to continue...")
    start_time = time.time()
    success_count = 0
    while time.time() - start_time < 100:   
        xarm_manager.hardware_arm_enable(True)
        debug_info = xarm_manager.hardware_debug()
        if not check_arm_enable(debug_info):
            raise Exception("Arm is not enabled")
        time.sleep(0.01)
        xarm_manager.hardware_arm_enable(False)
        debug_info = xarm_manager.hardware_debug()
        if not check_arm_disable(debug_info):
            raise Exception("Arm is not disabled")
        time.sleep(0.01)
        success_count += 1
        print(f"Success count: {success_count}, Time: {time.time() - start_time}")

if __name__ == "__main__":
    main()
