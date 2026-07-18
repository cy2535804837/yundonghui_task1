from email import message
import select
from typing import Any, List, Union, Dict, Optional, Callable
from functools import wraps
import rclpy
from rclpy.node import Node
from controller_manager_msgs.srv import (
    ConfigureController,
    ListControllers,
    ListControllerTypes,
    ListHardwareComponents,
    ListHardwareInterfaces,
    LoadController,
    ReloadControllerLibraries,
    SetHardwareComponentState,
    SwitchController,
    UnloadController,
)

from rcl_interfaces.srv import GetParameters
from rcl_interfaces.msg import ParameterType

import argparse
import time
from std_srvs.srv import SetBool
from eai_manipulator_msgs.srv import Info, Mode
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from .config import *
from .tools import *
from eai_manipulator_msgs.msg import WarnAndErrorInfo
from collections import deque


def require_real_mode(func: Callable) -> Callable:
    """
    装饰器：只在run_type为'real'时执行硬件操作
    
    如果run_type不是'real'，则记录警告并返回适当的默认值。
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # 检查run_type是否为'real'
        if not hasattr(self, 'run_type') or self.run_type != 'real':
            run_type_str = getattr(self, 'run_type', 'None')
            self.get_logger().warn(
                f"硬件操作 '{func.__name__}' 被跳过，因为 run_type='{run_type_str}' 不是 'real'"
            )
            # 根据函数返回类型返回适当的默认值
            if func.__name__ == 'hardware_debug':
                return None  # hardware_debug返回信息，非real模式返回None
            else:
                return False  # 其他硬件使能/模式设置方法返回False表示未执行
        return func(self, *args, **kwargs)
    return wrapper


def update_state_after(func: Callable) -> Callable:
    """
    装饰器：在函数执行后自动调用self.update_controller_state方法
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        if hasattr(self, "update_controller_state") and callable(getattr(self, "update_controller_state", None)):
            self.update_controller_state()
        return result
    return wrapper




class XARM_manager(Node):
    def __init__(self):
        super().__init__('XARM_manager')
        self.node = self
        self.controller_manager = 'controller_manager'
        self.controller_hardware_resources = controller_hardware_resources      

        # self.check_XARM_running()
        self.run_type = get_node_parameter(self.node, self.controller_manager, 'run_type')
        self.robot_type = get_node_parameter(self.node, self.controller_manager, 'robot_type')

        self.get_logger().info(f"XARM_manager initialized, robot_type: {self.robot_type}, run_type: {self.run_type}")

        self.controller_state = {}
        self.time_measurement = False
        self.jointstates_topic = jointstates_topic[self.robot_type]
        self.joints_name = joints_name[self.robot_type]
        
        # 存储最新的关节状态
        self.latest_joint_state = None
        
        # 订阅关节状态话题
        # 使用适合传感器数据的QoS配置：BestEffort可靠性，Volatile持久性
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )
        self.joint_state_subscription = self.create_subscription(
            JointState,
            self.jointstates_topic,
            self.joint_state_callback,
            qos_profile
        )
        self.get_logger().info(f"已订阅关节状态话题: {self.jointstates_topic}")

        # 创建一个长度为10的FIFO错误码队列
        self.warn_and_error_code_queue = deque(maxlen=10)

        self._subscribe_warn_and_error_info()


    # 订阅/manipulator/warn_and_error_info，类型为eai_manipulator_msgs/msg/WarnAndErrorInfo

    def warn_and_error_callback(self, msg: WarnAndErrorInfo):
        """
        机械臂警告和错误信息回调
        Args:
            msg: WarnAndErrorInfo 消息对象
        """
        if msg.level == 'warn':
            self.warn_and_error_code_queue.append(msg.code)
            print(f"收到警告信息: source={msg.source}, code={msg.code}, info={msg.info}")
        if msg.level == 'error':
            print(f"收到错误信息: source={msg.source}, code={msg.code}, info={msg.info}")
            self.warn_and_error_code_queue.append(msg.code)

    def _subscribe_warn_and_error_info(self):
        """
        内部函数: 订阅/manipulator/warn_and_error_info
        """
        qos_profile = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )
        self.create_subscription(
            WarnAndErrorInfo,
            '/manipulator/warn_and_error_info',
            self.warn_and_error_callback,
            qos_profile
        )
        self.get_logger().info(f"已订阅警告和错误信息话题: /manipulator/warn_and_error_info")

    def check_XARM_running(self):
        # 检查 /controller_manager/list_controllers 服务是否存在
        service_list = self.get_service_names_and_types()
        service_exists = any(
            srv[0] == '/controller_manager/list_controllers' for srv in service_list
        )
        if not service_exists:
            self.get_logger().warn(
                "服务 /controller_manager/list_controllers 不存在，XARM未启动。")
            raise RuntimeError("XARM_manager 初始化失败, XARM未启动。")
        
        self.get_logger().info("XARM已启动。")
        
    def joint_state_callback(self, msg: JointState):
        """
        关节状态话题回调函数
        
        当接收到关节状态消息时，更新最新的关节状态数据
        
        Args:
            msg: sensor_msgs.msg.JointState 消息对象
        """
        self.latest_joint_state = msg
    
    # 同步获取最新的关节状态消息,用于在节点的回调没被调用时以同步的方式获取最新的关节状态消息
    def joint_state_update(self) -> Optional[JointState]:
        """
        同步（阻塞）方式获取最新的关节状态消息，并更新latest_joint_state属性。

        该方法会调用topic_subscriber同步订阅关节状态topic（一次性），
        如果接收到消息，则将其保存到self.latest_joint_state。
        最终返回self.latest_joint_state，不论是否收到了新消息。

        Returns:
            Optional[JointState]: 最新的关节状态消息（可能为None）
        """
        msg = topic_subscriber(self, self.jointstates_topic, JointState)
        if msg is not None:
            # 如果成功收到新的关节状态消息，则更新保存
            self.latest_joint_state = msg
        # 返回当前保存的关节状态消息（None表示未能读取到）
        return self.latest_joint_state

    def get_latest_joint_state(self) -> Optional[JointState]:
        """
        获取最新的关节状态
        
        Returns:
            最新的 JointState 消息对象，如果还没有收到消息则返回 None
        """
        return self.latest_joint_state

    def read_joint_angles(self) -> Optional[List[float]]:
        joint_positions_ordered = []
        msg = self.get_latest_joint_state()
        if msg is not None:
            joint_name_to_pos = {name: pos for name, pos in zip(msg.name, msg.position)}
            for joint in self.joints_name:
                joint_positions_ordered.append(joint_name_to_pos.get(joint, None))
            return joint_positions_ordered
        else:
            self.get_logger().warn("未能读取到关节状态消息，无法提取关节角")
            return None

    def read_joint_velocities(self) -> Optional[List[float]]:
        joint_velocities_ordered = []
        msg = self.get_latest_joint_state()
        if msg is not None:
            joint_name_to_vel = {name: vel for name, vel in zip(msg.name, msg.velocity)}
            for joint in self.joints_name:
                joint_velocities_ordered.append(joint_name_to_vel.get(joint, None))
            return joint_velocities_ordered
        else:
            self.get_logger().warn("未能读取到关节状态消息，无法提取关节速度")
            return None

    def read_joint_efforts(self) -> Optional[List[float]]:
        joint_efforts_ordered = []
        msg = self.get_latest_joint_state()
        if msg is not None:
            joint_name_to_effort = {name: effort for name, effort in zip(msg.name, msg.effort)}
            for joint in self.joints_name:
                joint_efforts_ordered.append(joint_name_to_effort.get(joint, None))
            return joint_efforts_ordered
        else:
            self.get_logger().warn("未能读取到关节状态消息，无法提取关节力矩")
            return None

    def xarm_left_arm_joint_angles(self) -> Optional[List[float]]:
        joints = self.read_joint_angles()
        if joints is not None:
            return joints[:7]
        else:
            self.get_logger().warn("未能读取到关节状态消息，无法提取关节角")
            return None
    
    def xarm_left_arm_joint_velocities(self) -> Optional[List[float]]:
        velocities = self.read_joint_velocities()
        if velocities is not None:
            return velocities[:7]
        else:
            self.get_logger().warn("未能读取到关节状态消息，无法提取关节速度")
            return None

    def xarm_left_arm_joint_efforts(self) -> Optional[List[float]]:
        efforts = self.read_joint_efforts()
        if efforts is not None:
            return efforts[:7]
        else:
            self.get_logger().warn("未能读取到关节状态消息，无法提取关节力矩")
            return None
    
    def xarm_right_arm_joint_angles(self) -> Optional[List[float]]:
        joints = self.read_joint_angles()
        if joints is not None:
            return joints[7:14]
        else:
            self.get_logger().warn("未能读取到关节状态消息，无法提取关节角")
            return None
    
    def xarm_right_arm_joint_velocities(self) -> Optional[List[float]]:
        velocities = self.read_joint_velocities()
        if velocities is not None:
            return velocities[7:14]
        else:
            self.get_logger().warn("未能读取到关节状态消息，无法提取关节速度")
            return None

    def xarm_right_arm_joint_efforts(self) -> Optional[List[float]]:
        efforts = self.read_joint_efforts()
        if efforts is not None:
            return efforts[7:14]
        else:
            self.get_logger().warn("未能读取到关节状态消息，无法提取关节力矩")
            return None

    
    @require_robot_type(["tianyi2", "tianyi_max"])
    def xarm_body_joint_angles(self) -> Optional[List[float]]:
        joints = self.read_joint_angles()
        return joints[14:18]
    
    @require_robot_type(["tianyi2", "tianyi_max"])
    def xarm_head_joint_angles(self) -> Optional[List[float]]:
        joints = self.read_joint_angles()
        return joints[18:21]

    
    @measure_time
    def update_controller_state(self):
        response = list_controllers(self, self.controller_manager)
        self.controller_state = {c.name: c.state for c in response.controller}
    
    @measure_time
    def query_controllers(self, state='active'):
        response = list_controllers(self, self.controller_manager)
        self.controller_state = {c.name: c.state for c in response.controller}
        if not response.controller:
            print("No controllers are currently loaded!")
            return None
        if state == 'active':
            return [c for c in response.controller if c.state == 'active']
        elif state == 'inactive':
            return [c for c in response.controller if c.state == 'inactive']
        elif state == 'unconfigured':
            return [c for c in response.controller if c.state == 'unconfigured']
        elif state == 'all':
            return response.controller
        else:
            return None
    @measure_time
    def find_active_controllers(self):
        return self.query_controllers(state='active')
    @measure_time
    def find_inactive_controllers(self):
        return self.query_controllers(state='inactive')
    @measure_time
    def find_unconfigured_controllers(self):
        return self.query_controllers(state='unconfigured')
    @measure_time
    def find_all_controllers(self):
        return self.query_controllers(state='all')
    
    
    @measure_time
    def xarm_find_active_controllers(self):
        active_controllers = self.find_active_controllers()
        return [c.name for c in active_controllers if c.name != 'joint_state_broadcaster']
    @measure_time
    def xarm_find_inactive_controllers(self):
        inactive_controllers = self.find_inactive_controllers()
        return [c.name for c in inactive_controllers if c.name != 'joint_state_broadcaster']
    @measure_time
    def xarm_find_unconfigured_controllers(self):
        unconfigured_controllers = self.find_unconfigured_controllers()
        return [c.name for c in unconfigured_controllers if c.name != 'joint_state_broadcaster']

    @measure_time
    def switch_controllers(self, activate_controllers, deactivate_controllers, strict=False, activate_asap=False, timeout=5.0):
        # 如果activate_controllers不是列表，则转为列表
        if not isinstance(activate_controllers, list):
            activate_controllers = [activate_controllers]
        # 如果deactivate_controllers不是列表，则转为列表
        if not isinstance(deactivate_controllers, list):
            deactivate_controllers = [deactivate_controllers]
        response = switch_controllers(
            self,
            self.controller_manager,
            deactivate_controllers,
            activate_controllers,
            strict,
            activate_asap,
            timeout=timeout,
        )
        return response.ok
    @measure_time
    def activate_controller(self, controller_name):
        return self.switch_controllers(activate_controllers=controller_name, deactivate_controllers=[], strict=False, activate_asap=False, timeout=5.0)
    @measure_time
    def deactivate_controller(self, controller_name):
        return self.switch_controllers(activate_controllers=[], deactivate_controllers=controller_name, strict=False, activate_asap=False, timeout=5.0)

    @measure_time
    def xarm_activate_controller(self, controller_name):
        # 如果controller_name不是list，则转为list
        if not isinstance(controller_name, list):
            controller_name = [controller_name]
        
        # 检查控制器是否存在
        # all_controllers = self.find_all_controllers()
        
        
        # unconfigured_controllers = [c.name for c in all_controllers if c.state == 'unconfigured']
        # activated_controllers = [c.name for c in all_controllers if c.state == 'active']
        

        # exist_controllers = []
        # for c in controller_name:
        #     if not any(ctrl.name == c for ctrl in all_controllers):
        #         self.get_logger().warn(f"Controller '{c}' not found in all controllers.")
        #         continue
        #     exist_controllers.append(c)
        
        # # 检查exist_controllers非空
        # if not exist_controllers:
        #     self.get_logger().warn('No valid controllers to activate.')
        #     return False
        
        

        # # 检查exist_controllers之间的资源是否冲突
        # for c in exist_controllers:
        #     resources = self.controller_hardware_resources[c]
        #     for other_c in exist_controllers:
        #         if other_c == c:
        #             continue
        #         other_resources = self.controller_hardware_resources[other_c]
                
        #         # 检查资源是否冲突
        #         if set(resources) & set(other_resources):
        #             self.get_logger().warn(f"Controller '{c}' and controller '{other_c}' have conflicting resources: {set(resources) & set(other_resources)}")
        #             return False

        activate_controllers(
            self,
            self.controller_manager,
            [],
            controller_name,
            False,
            False,
            5.0,
        )
        

        # # 全都不冲突，查找当前控制器中需要关闭的控制器
        # need_deactivated_controllers = []
        # for c in exist_controllers:
        #     # 查找activated_controllers中与c资源冲突的控制器
        #     resources = self.controller_hardware_resources[c]
        #     for active_c in activated_controllers:
        #         if active_c == 'joint_state_broadcaster':
        #             continue
        #         if active_c == c:
        #             continue
        #         active_resources = self.controller_hardware_resources.get(active_c)
        #         if set(active_resources) & set(resources):
        #             need_deactivated_controllers.append(active_c)

        # for c in need_deactivated_controllers:
        #     if c in unconfigured_controllers:
        #         response = configure_controller(
        #             self,
        #             self.controller_manager,
        #             c
        #             )
        #         if not response.ok:
        #             self.get_logger().error(f'Failed to configure controller: {c}')
        #             return False

        
        # if need_deactivated_controllers:
        #     # need_deactivated_controllers去重
        #     need_deactivated_controllers = list(set[Any](need_deactivated_controllers))
        #     self.deactivate_controller(need_deactivated_controllers)   
        #     self.get_logger().info(f"Deactivated controllers: {need_deactivated_controllers}")

        # # 去除activated_controllers中存在的部分
        # need_activated_controllers = [c for c in exist_controllers if c not in activated_controllers]
        # if need_activated_controllers:
        #     self.activate_controller(need_activated_controllers)
        #     self.get_logger().info(f"Activated controllers: {need_activated_controllers}")
        return True
    
    # 初始化时调用，配置所有控制器
    @measure_time
    def configure_all_controllers(self):
        controllers = self.query_controllers(state='all')
        all_ok = True
        for c in controllers:
            if c.state == 'unconfigured':
                response = configure_controller(
                    self,
                    self.controller_manager,
                    c.name
                    )
                if not response.ok:
                    all_ok = False
                    self.get_logger().error(f'Failed to configure controller: {c.name}')
        return all_ok

    @measure_time
    def xarm_deactivate_controller(self, controller_name):
        if not isinstance(controller_name, list):
            controller_name = [controller_name]
        return self.deactivate_controller(controller_name)

    @measure_time
    def xarm_deactivate_all_controller(self):
        
        controllers = self.query_controllers(state='all')
        for c in controllers:
            if c.state == 'unconfigured':
                response = configure_controller(
                    self,
                    self.controller_manager,
                    c.name
                    )
                if not response.ok:
                    all_ok = False
                    self.get_logger().error(f'Failed to configure controller: {c.name}')
        
        deactivate_controllers = [c.name for c in controllers if c.state == 'active'and c.name != 'joint_state_broadcaster']

        self.deactivate_controller(deactivate_controllers)

        self.get_logger().info(f"Deactivated controllers: {deactivate_controllers}")
        return True

    
    @require_real_mode
    def hardware_arm_enable(self, enable=True):
        """
        使能/不使能机械臂硬件
        
        注意：仅在run_type='real'时才会实际执行硬件操作
        """
        # ros2 service call /EAIHardware/set_arm_enable std_srvs/srv/SetBool data:\ false\ 
        request = SetBool.Request()
        request.data = enable
        return service_caller(self, '/EAIHardware/set_arm_enable', SetBool, request)
    
    @require_real_mode
    def hardware_head_enable(self, enable=True):
        """
        启用/禁用头部硬件
        
        注意：仅在run_type='real'时才会实际执行硬件操作
        """
        request = SetBool.Request()
        request.data = enable
        return service_caller(self, '/EAIHardware/set_head_enable', SetBool, request)
    
    @require_real_mode
    def hardware_leg_enable(self, enable=True):
        """
        启用/禁用腿部硬件
        
        注意：仅在run_type='real'时才会实际执行硬件操作
        """
        request = SetBool.Request()
        request.data = enable
        return service_caller(self, '/EAIHardware/set_leg_enable', SetBool, request)
    
    @require_real_mode
    def hardware_waist_enable(self, enable=True):
        """
        启用/禁用腰部硬件
        
        注意：仅在run_type='real'时才会实际执行硬件操作
        """
        request = SetBool.Request()
        request.data = enable
        return service_caller(self, '/EAIHardware/set_waist_enable', SetBool, request)
    
    @require_real_mode
    def hardware_all_enable(self, enable=True):
        """
        启用/禁用所有硬件
        
        注意：仅在run_type='real'时才会实际执行硬件操作
        """
        request = SetBool.Request()
        request.data = enable
        return service_caller(self, '/EAIHardware/set_all_enable', SetBool, request)

    @require_real_mode
    def hardware_arm_mode(self, mode):
        """
        设置机械臂模式
        
        注意：仅在run_type='real'时才会实际执行硬件操作
        """
        request = Mode.Request()
        request.mode = mode
        return service_caller(self, '/EAIHardware/set_arm_mode', Mode, request)

    @require_real_mode
    def hardware_debug(self):
        """
        获取硬件调试信息
        
        注意：仅在run_type='real'时才会实际执行硬件操作
        """
        request = Info.Request()
        response = service_caller(self, '/EAIHardware/debug', Info, request)
        return response.info
    
    def get_node_parameter(self, node_name: str, parameter_name: str, timeout_sec: float = 5.0) -> Optional[Any]:
        """
        读取指定节点的指定参数
        
        Args:
            node_name: 节点名称（不含命名空间，如 'controller_manager'）
            parameter_name: 参数名称
            timeout_sec: 超时时间（秒）
        
        Returns:
            参数值，如果参数不存在或读取失败则返回None
        
        Example:
            # 读取controller_manager节点的use_sim_time参数
            value = xarm_manager.get_node_parameter('controller_manager', 'use_sim_time')
        """
        # 构建参数服务名称
        service_name = f'/{node_name}/get_parameters'
        
        # 创建服务客户端
        client = self.create_client(GetParameters, service_name)
        
        if not client.wait_for_service(timeout_sec=timeout_sec):
            self.get_logger().error(f'Parameter service {service_name} is not available')
            return None
        
        # 创建请求
        request = GetParameters.Request()
        request.names = [parameter_name]
        
        # 调用服务
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        
        if not future.done():
            self.get_logger().error(f'Failed to get parameter {parameter_name} from {node_name}: timeout')
            return None
        
        response = future.result()
        
        if not response or len(response.values) == 0:
            self.get_logger().warn(f'Parameter {parameter_name} not found in node {node_name}')
            return None
        
        # 解析参数值
        param_value = response.values[0]
        return self._parse_parameter_value(param_value)
    
    def get_node_parameters(self, node_name: str, parameter_names: List[str], timeout_sec: float = 5.0) -> Dict[str, Any]:
        """
        读取指定节点的多个参数
        
        Args:
            node_name: 节点名称（不含命名空间，如 'controller_manager'）
            parameter_names: 参数名称列表
            timeout_sec: 超时时间（秒）
        
        Returns:
            参数字典，键为参数名，值为参数值。如果参数不存在，则对应的值为None
        
        Example:
            # 读取多个参数
            params = xarm_manager.get_node_parameters('controller_manager', ['use_sim_time', 'update_rate'])
        """
        # 构建参数服务名称
        service_name = f'/{node_name}/get_parameters'
        
        # 创建服务客户端
        client = self.create_client(GetParameters, service_name)
        
        if not client.wait_for_service(timeout_sec=timeout_sec):
            self.get_logger().error(f'Parameter service {service_name} is not available')
            return {name: None for name in parameter_names}
        
        # 创建请求
        request = GetParameters.Request()
        request.names = parameter_names
        
        # 调用服务
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        
        if not future.done():
            self.get_logger().error(f'Failed to get parameters from {node_name}: timeout')
            return {name: None for name in parameter_names}
        
        response = future.result()
        
        if not response:
            return {name: None for name in parameter_names}
        
        # 解析所有参数值
        result = {}
        for i, name in enumerate(parameter_names):
            if i < len(response.values):
                result[name] = self._parse_parameter_value(response.values[i])
            else:
                result[name] = None
        
        return result
    
    def _parse_parameter_value(self, param_value) -> Any:
        """
        解析ROS2参数值
        
        Args:
            param_value: rcl_interfaces.msg.ParameterValue对象
        
        Returns:
            Python原生类型的参数值
        """
        if param_value.type == ParameterType.PARAMETER_NOT_SET:
            return None
        elif param_value.type == ParameterType.PARAMETER_BOOL:
            return param_value.bool_value
        elif param_value.type == ParameterType.PARAMETER_INTEGER:
            return param_value.integer_value
        elif param_value.type == ParameterType.PARAMETER_DOUBLE:
            return param_value.double_value
        elif param_value.type == ParameterType.PARAMETER_STRING:
            return param_value.string_value
        elif param_value.type == ParameterType.PARAMETER_BYTE_ARRAY:
            return list(param_value.byte_array_value)
        elif param_value.type == ParameterType.PARAMETER_BOOL_ARRAY:
            return list(param_value.bool_array_value)
        elif param_value.type == ParameterType.PARAMETER_INTEGER_ARRAY:
            return list(param_value.integer_array_value)
        elif param_value.type == ParameterType.PARAMETER_DOUBLE_ARRAY:
            return list(param_value.double_array_value)
        elif param_value.type == ParameterType.PARAMETER_STRING_ARRAY:
            return list(param_value.string_array_value)
        else:
            self.get_logger().warn(f'Unknown parameter type: {param_value.type}')
            return None
    
