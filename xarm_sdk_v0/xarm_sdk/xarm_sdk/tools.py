"""
ROS2 Action调用工具模块

提供用于调用ROS2 Action服务器的辅助函数
"""
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
from typing import Optional, Callable, Any, Type
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from functools import wraps
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
import time

class SingletonActionClient:
    """
    单例ActionClient调用器类

    该类用于为同一Node下、同一action_name+action_type的ActionClient创建单例。
    每个 (node, action_type, fully_qualified_action_name) 只有一个ActionClient，避免重复创建和资源浪费。

    使用方法：
        提供 Node 实例、action_type 以及 action 的完全限定名称来获取客户端。
        若已存在对应客户端，则直接返回，否则新建。

    属性:
        _clients (dict): 存储 (node, action_type, fully_qualified_action_name) 到 ActionClient 的映射。

    方法:
        __new__ : 创建或返回已存在的ActionClient。
    """
    _clients = {}

    def __new__(cls, node, action_type, fully_qualified_action_name):
        """
        创建或返回 ActionClient。

        参数:
            node: rclpy.node.Node 实例
            action_type: ROS2 Action类型
            fully_qualified_action_name: action的完全限定名称（字符串）

        返回:
            rclpy.action.ActionClient 实例
        """
        key = (node, action_type, fully_qualified_action_name)
        if key not in cls._clients:
            cls._clients[key] = ActionClient(node, action_type, fully_qualified_action_name)
            node.get_logger().debug(
                f"创建新的ActionClient: {fully_qualified_action_name}，所属节点: {node.get_name()}，类型: {action_type.__name__}"
            )
        else:
            node.get_logger().debug(
                f"返回已存在的ActionClient: {fully_qualified_action_name}，所属节点: {node.get_name()}，类型: {action_type.__name__}"
            )
        return cls._clients[key]

class SingletonServiceCaller:
    """
    控制器管理器服务调用的单例类。

    该类用于为给定的服务名称创建（或复用）服务客户端。
    如果对应的服务客户端已存在，则返回已有客户端；
    否则创建新的客户端并进行复用，避免为同一服务名称重复创建多个客户端资源。

    需要三个参数：Node对象、服务类型、完整限定服务名称，来唯一确定并创建服务客户端。
    """

    _clients = {}

    def __new__(cls, node, service_type, fully_qualified_service_name):
        if (node, fully_qualified_service_name) not in cls._clients:
            cls._clients[(node, fully_qualified_service_name)] = node.create_client(
                service_type, fully_qualified_service_name
            )
            node.get_logger().debug(
                f"Creating a new service client : {fully_qualified_service_name} with node : {node.get_name()}"
            )

        node.get_logger().debug(
            f"Returning the existing service client : {fully_qualified_service_name} for node : {node.get_name()}"
        )
        return cls._clients[(node, fully_qualified_service_name)]
        

def action_caller(
    node: Node,
    action_name: str,
    action_type: Type[Any],
    goal_msg: Any,
    goal_response_timeout: float = 3.0,
    result_timeout: float = 60.0,
    feedback_callback: Optional[Callable] = None,
    max_attempts: int = 3,
) -> Optional[Any]:
    """
    Abstraction of an action call using ROS 2 actions (send_goal).

    Sends a goal to a given action server, optionally waits for result/feedback, retries as necessary.

    Args:
        node: Node object to be associated with
        action_name: Action server name (topic, e.g. '/my_action')
        action_type: Action type (e.g. MyAction)
        goal_msg: Goal message to send (instance of action_type.Goal)
        goal_response_timeout: Timeout for receiving the goal response (seconds)
        result_timeout: Timeout for receiving the final result (seconds)
        feedback_callback: Optional, called with feedback (feedback_msg)
        max_attempts: How many attempts to send/complete the goal
    
    Returns:
        The result message of the action or None if failure.
    """

    namespace = "" if node.get_namespace() == "/" else node.get_namespace()
    fully_qualified_action_name = (
        f"{namespace}/{action_name}" if not action_name.startswith("/") else action_name
    )
    client = SingletonActionClient(node, action_type, fully_qualified_action_name)

    # Wait for server
    if not client.wait_for_server(timeout_sec=goal_response_timeout):
        node.get_logger().error(f"Action server {fully_qualified_action_name} not available after {goal_response_timeout}s.")
        return None

    attempt = 0
    while attempt < max_attempts:
        attempt += 1

        node.get_logger().info(
            f"Sending goal to action server {fully_qualified_action_name} (attempt {attempt}/{max_attempts}):\n{goal_msg}"
        )

        future_goal = client.send_goal_async(goal_msg, feedback_callback=feedback_callback)
        rclpy.spin_until_future_complete(node, future_goal, timeout_sec=goal_response_timeout)

        goal_handle = future_goal.result()
        if not goal_handle or not goal_handle.accepted:
            node.get_logger().warning(
                f"Goal was not accepted by action server {fully_qualified_action_name} (try {attempt})."
            )
            continue

        node.get_logger().info(f"Goal accepted. Waiting for result...")

        future_result = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(node, future_result, timeout_sec=result_timeout)

        result = future_result.result()
        if hasattr(result, "result"):
            node.get_logger().info(f"Action result received from {fully_qualified_action_name}")
            node.get_logger().info(f"Action result: {result.result}")
            return result.result
        else:
            node.get_logger().warning(
                f"No result received from action server {fully_qualified_action_name} (try {attempt})."
            )

    node.get_logger().error(
        f"Could not get action result from {fully_qualified_action_name} after {max_attempts} attempts."
    )
    return None

class ServiceNotFoundError(Exception):
    pass


def service_caller(
    node,
    service_name,
    service_type,
    request,
    service_timeout=0.0,
    call_timeout=10.0,
    max_attempts=3,
):
    """
    Abstraction of a service call.

    Has an optional timeout to find the service, receive the answer to a call
    and a mechanism to retry a call of no response is received.

    @param node Node object to be associated with
    @type rclpy.node.Node
    @param service_name Service URL
    @type str
    @param request The request to be sent
    @type service request type
    @param service_timeout Timeout (in seconds) to wait until the service is available. 0 means
    waiting forever, retrying every 10 seconds.
    @type float
    @param call_timeout Timeout (in seconds) for getting a response
    @type float
    @param max_attempts Number of attempts until a valid response is received. With some
    middlewares it can happen, that the service response doesn't reach the client leaving it in
    a waiting state forever.
    @type int
    @return The service response

    """
    namespace = "" if node.get_namespace() == "/" else node.get_namespace()
    fully_qualified_service_name = (
        f"{namespace}/{service_name}" if not service_name.startswith("/") else service_name
    )
    cli = SingletonServiceCaller(node, service_type, fully_qualified_service_name)

    while not cli.service_is_ready():
        node.get_logger().info(
            f"waiting for service {fully_qualified_service_name} to become available..."
        )
        if service_timeout:
            if not cli.wait_for_service(service_timeout):
                raise ServiceNotFoundError(
                    f"Could not contact service {fully_qualified_service_name}"
                )
        elif not cli.wait_for_service(10.0):
            node.get_logger().warn(f"Could not contact service {fully_qualified_service_name}")

    node.get_logger().debug(f"requester: making request: {request}\n")
    future = None
    for attempt in range(max_attempts):
        future = cli.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=call_timeout)
        if future.result() is None:
            node.get_logger().warning(
                f"Failed getting a result from calling {fully_qualified_service_name} in "
                f"{call_timeout}. (Attempt {attempt+1} of {max_attempts}.)"
            )
        else:
            return future.result()
    raise RuntimeError(
        f"Could not successfully call service {fully_qualified_service_name} after {max_attempts} attempts."
    )




def require_robot_type(allow_types):
    """
    装饰器：仅在 self.node.robot_type 属于 allow_types 时才允许函数执行，
    否则记录警告并返回 None。

    allow_types: 可为字符串（单一类型）或字符串列表（多个允许类型）
    """
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            # 获取 robot_type
            robot_type = getattr(self.node, 'robot_type', None)
            if isinstance(allow_types, str):
                allow_list = [allow_types]
            else:
                allow_list = list(allow_types)
            if robot_type not in allow_list:
                if hasattr(self.node, 'get_logger'):
                    self.node.get_logger().warn(
                        f"调用函数 '{func.__name__}' 被跳过，因为 robot_type='{robot_type}' 不在允许范围 {allow_list}"
                    )
                return None
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


class SingletonPublisher:
    """
    单例Publisher调用器类

    该类用于为同一Node下、同一topic_name+msg_type的Publisher创建单例。
    每个 (node, msg_type, topic_name, qos_profile) 只有一个Publisher，避免重复创建和资源浪费。

    使用方法：
        提供 Node 实例、msg_type、topic_name 以及可选的 qos_profile 来获取发布者。
        若已存在对应发布者，则直接返回，否则新建。

    属性:
        _publishers (dict): 存储 (node, msg_type, topic_name, qos_key) 到 Publisher 的映射。

    方法:
        __new__ : 创建或返回已存在的Publisher。
    """
    _publishers = {}

    def __new__(cls, node, msg_type, topic_name, qos_profile=None):
        """
        创建或返回 Publisher。

        参数:
            node: rclpy.node.Node 实例
            msg_type: ROS2消息类型
            topic_name: topic的完全限定名称（字符串）
            qos_profile: QoS配置（可选）

        返回:
            rclpy.publisher.Publisher 实例
        """
        # 将QoS配置转换为可哈希的键
        if qos_profile is None:
            qos_key = "default"
        else:
            # 使用QoS的关键属性作为键
            qos_key = (
                qos_profile.depth,
                qos_profile.durability.kind if hasattr(qos_profile.durability, 'kind') else None,
                qos_profile.reliability.kind if hasattr(qos_profile.reliability, 'kind') else None,
                qos_profile.history.kind if hasattr(qos_profile.history, 'kind') else None,
            )
        
        key = (node, msg_type, topic_name, qos_key)
        if key not in cls._publishers:
            cls._publishers[key] = node.create_publisher(msg_type, topic_name, qos_profile or QoSProfile(depth=10))
            node.get_logger().debug(
                f"创建新的Publisher: {topic_name}，所属节点: {node.get_name()}，类型: {msg_type.__name__}"
            )
        else:
            node.get_logger().debug(
                f"返回已存在的Publisher: {topic_name}，所属节点: {node.get_name()}，类型: {msg_type.__name__}"
            )
        return cls._publishers[key]


def topic_publisher(
    node,
    topic_name: str,
    msg_type,
    msg_data,
    qos_profile=None,
    latch=False,
):
    """
    工具函数：发布一条消息到指定topic（使用单例Publisher）。

    Args:
        node: rclpy Node对象
        topic_name (str): Topic名称
        msg_type: 消息类型（如 std_msgs.msg.String）
        msg_data: 要发布的消息对象，必须是msg_type实例
        qos_profile: 可选，QoS配置
        latch (bool): 是否保持最新消息，常用于消息发布后新订阅者能获取最后一条（需要Middleware支持）

    Usage:
        from std_msgs.msg import String
        msg = String()
        msg.data = "hello"
        topic_publisher(node, "/chatter", String, msg)

    注意:
        该函数使用单例模式管理Publisher，相同参数的调用会复用同一个Publisher实例。
    """
    # 处理QoS配置
    if qos_profile is None:
        qos_profile = QoSProfile(depth=10)
    if latch:
        qos_profile.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL

    # 使用单例Publisher
    publisher = SingletonPublisher(node, msg_type, topic_name, qos_profile)
    publisher.publish(msg_data)
    node.get_logger().debug(f"Published message to topic {topic_name}")



def measure_time(func: Callable) -> Callable:
    """
    装饰器：根据self.time_measurement决定是否测量方法的运行时间，并通过self.get_logger().info输出耗时
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # 判断是否需要测量时间
        do_measure = getattr(self, "time_measurement", False)
        if do_measure:
            start_time = time.time()
            result = func(self, *args, **kwargs)
            end_time = time.time()
            elapsed = end_time - start_time
            if hasattr(self, "get_logger"):
                self.get_logger().info(f"方法 '{func.__name__}' 执行耗时: {elapsed:.4f} 秒")
            return result
        else:
            return func(self, *args, **kwargs)
    return wrapper


# 实现一个通用的topic订阅工具：同步接收ROS2消息、可设定超时
from threading import Lock
from collections import deque


class SingletonSubscriber:
    """
    单例Subscriber调用器类

    该类用于为同一Node下、同一topic_name+msg_type的Subscriber创建单例。
    每个 (node, msg_type, topic_name, qos_profile) 只有一个Subscriber，避免重复创建和资源浪费。
    订阅者会持续接收消息并存储在消息队列中。

    使用方法：
        提供 Node 实例、msg_type、topic_name 以及可选的 qos_profile 来获取订阅者。
        若已存在对应订阅者，则直接返回，否则新建。

    属性:
        _subscribers (dict): 存储 (node, msg_type, topic_name, qos_key) 到 SubscriberWrapper 的映射。
    """
    _subscribers = {}
    _lock = Lock()  # 用于线程安全的访问

    def __new__(cls, node, msg_type, topic_name, qos_profile=None):
        """
        创建或返回 SubscriberWrapper。

        参数:
            node: rclpy.node.Node 实例
            msg_type: ROS2消息类型
            topic_name: topic的完全限定名称（字符串）
            qos_profile: QoS配置（可选）

        返回:
            SubscriberWrapper 实例
        """
        # 将QoS配置转换为可哈希的键
        if qos_profile is None:
            qos_key = "default"
        else:
            # 使用QoS的关键属性作为键
            qos_key = (
                qos_profile.depth,
                qos_profile.durability.kind if hasattr(qos_profile.durability, 'kind') else None,
                qos_profile.reliability.kind if hasattr(qos_profile.reliability, 'kind') else None,
                qos_profile.history.kind if hasattr(qos_profile.history, 'kind') else None,
            )
        
        key = (node, msg_type, topic_name, qos_key)
        
        with cls._lock:
            if key not in cls._subscribers:
                cls._subscribers[key] = SubscriberWrapper(node, msg_type, topic_name, qos_profile)
                node.get_logger().debug(
                    f"创建新的Subscriber: {topic_name}，所属节点: {node.get_name()}，类型: {msg_type.__name__}"
                )
            else:
                node.get_logger().debug(
                    f"返回已存在的Subscriber: {topic_name}，所属节点: {node.get_name()}，类型: {msg_type.__name__}"
                )
            return cls._subscribers[key]


class SubscriberWrapper:
    """
    订阅者包装类，管理订阅者和消息队列
    """
    def __init__(self, node, msg_type, topic_name, qos_profile=None):
        """
        初始化订阅者包装类
        
        参数:
            node: rclpy.node.Node 实例
            msg_type: ROS2消息类型
            topic_name: topic名称
            qos_profile: QoS配置
        """
        self.node = node
        self.msg_type = msg_type
        self.topic_name = topic_name
        self.qos_profile = qos_profile or QoSProfile(depth=10)
        self.latest_msg = None
        self.msg_queue = deque(maxlen=100)  # 最多保存100条消息
        self.lock = Lock()
        
        # 创建订阅者
        self.subscription = node.create_subscription(
            msg_type,
            topic_name,
            self._callback,
            self.qos_profile
        )
    
    def _callback(self, msg):
        """
        内部回调函数，接收消息并更新队列
        """
        with self.lock:
            self.latest_msg = msg
            self.msg_queue.append(msg)
    
    def get_latest(self):
        """
        获取最新的消息
        
        Returns:
            最新的消息对象，如果没有消息则返回None
        """
        with self.lock:
            return self.latest_msg
    
    def wait_for_message(self, timeout=2.0):
        """
        等待接收一条消息（同步）
        
        该方法会等待订阅者接收到新消息。如果已经有消息，会立即返回最新的消息。
        如果没有消息，会等待直到收到消息或超时。
        
        Args:
            timeout: 超时时间（秒）
        
        Returns:
            接收到的消息对象，如果超时则返回None
        """
        # 先检查是否已有消息，如果有则立即返回
        with self.lock:
            if self.latest_msg is not None:
                return self.latest_msg
            initial_queue_size = len(self.msg_queue)
        
        # 如果没有消息，等待新消息
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            # 处理ROS2回调（这会触发_callback函数）
            rclpy.spin_once(self.node, timeout_sec=0.01)
            
            # 检查是否收到新消息
            with self.lock:
                # 如果队列大小增加或latest_msg不为None，说明收到了新消息
                if len(self.msg_queue) > initial_queue_size or self.latest_msg is not None:
                    return self.latest_msg
        
        # 超时处理
        if hasattr(self.node, "get_logger"):
            self.node.get_logger().warn(f"订阅{self.topic_name}超时({timeout}s)，未收到消息")
        return None


def topic_subscriber(node, topic_name, msg_type, timeout=2.0, qos_profile=None):
    """
    同步订阅一次topic，返回接收到的msg对象，支持超时（使用单例Subscriber）。
    
    Args:
        node: rclpy Node对象
        topic_name (str): topic名称
        msg_type: 消息类型
        timeout (float): 等待超时（秒）
        qos_profile: QoS配置
    
    Returns:
        接收到的msg对象，如果超时则返回None
    
    注意:
        该函数使用单例模式管理Subscriber，相同参数的调用会复用同一个Subscriber实例。
    """
    # 使用单例订阅者
    subscriber_wrapper = SingletonSubscriber(node, msg_type, topic_name, qos_profile)
    
    # 等待消息
    return subscriber_wrapper.wait_for_message(timeout=timeout)


def get_node_parameter(node, node_name, parameter_name):
    """
    读取指定节点的指定参数（同步接口，简化调用，内部使用 service_caller）
    
    Args:
        node: rclpy Node 实例
        node_name (str): 目标节点名称（不带命名空间）
        parameter_name (str): 参数名称
        timeout_sec (float): 超时时间（秒）
    
    Returns:
        参数值（自动类型推断），若获取失败返回None
    """
    from rcl_interfaces.srv import GetParameters

    # 构建服务名
    service_name = f'/{node_name}/get_parameters'
    # 构造请求
    request = GetParameters.Request()
    request.names = [parameter_name]

    # 调用service_caller
    # service_caller应接受：node, service_name, service_type, request
    response = service_caller(
        node=node,
        service_name=service_name,
        service_type=GetParameters,
        request=request,
    )

    if not response or len(response.values) == 0:
        if hasattr(node, "get_logger"):
            node.get_logger().warn(f'Parameter {parameter_name} not found in node {node_name}')
        return None

    # 解析参数值
    def parse_param_value(pval):
        t = pval.type
        TYPE_NOT_SET = 0
        TYPE_BOOL = 1
        TYPE_INTEGER = 2
        TYPE_DOUBLE = 3
        TYPE_STRING = 4
        TYPE_BYTE_ARRAY = 5
        TYPE_BOOL_ARRAY = 6
        TYPE_INTEGER_ARRAY = 7
        TYPE_DOUBLE_ARRAY = 8
        TYPE_STRING_ARRAY = 9

        if t == TYPE_BOOL:
            return pval.bool_value
        elif t == TYPE_INTEGER:
            return pval.integer_value
        elif t == TYPE_DOUBLE:
            return pval.double_value
        elif t == TYPE_STRING:
            return pval.string_value
        elif t == TYPE_BYTE_ARRAY:
            return list(pval.byte_array_value)
        elif t == TYPE_BOOL_ARRAY:
            return list(pval.bool_array_value)
        elif t == TYPE_INTEGER_ARRAY:
            return list(pval.integer_array_value)
        elif t == TYPE_DOUBLE_ARRAY:
            return list(pval.double_array_value)
        elif t == TYPE_STRING_ARRAY:
            return list(pval.string_array_value)
        else:
            return None

    return parse_param_value(response.values[0])

def list_node_parameters(node, node_name):
    """
    列出指定节点的所有参数
    """
    from rcl_interfaces.srv import ListParameters
    service_name = f'/{node_name}/list_parameters'
    request = ListParameters.Request()
    response = service_caller(
        node=node,
        service_name=service_name,
        service_type=ListParameters,
        request=request,
    )

    if not response.result or len(response.result.names) == 0:
        if hasattr(node, "get_logger"):
            node.get_logger().warn(f'Parameters not found in node {node_name}')
        return {}
    
    return response.result.names

def set_node_parameter(node, node_name, parameter_name, value):
    """
    设置指定节点的指定参数（同步接口，简化调用，内部使用 service_caller）

    Args:
        node: rclpy Node 实例
        node_name (str): 目标节点名称（不带命名空间）
        parameter_name (str): 参数名称
        value: 需要设置的参数值
        timeout_sec (float): 超时时间，单位秒

    Returns:
        bool，表示是否设置成功
    """
    from rcl_interfaces.srv import SetParameters
    from rcl_interfaces.msg import Parameter, ParameterValue

    service_name = f'/{node_name}/set_parameters'
    request = SetParameters.Request()
    param = Parameter()
    param.name = parameter_name

    # 推断类型并赋值
    TYPE_BOOL = 1
    TYPE_INTEGER = 2
    TYPE_DOUBLE = 3
    TYPE_STRING = 4
    TYPE_BYTE_ARRAY = 5
    TYPE_BOOL_ARRAY = 6
    TYPE_INTEGER_ARRAY = 7
    TYPE_DOUBLE_ARRAY = 8
    TYPE_STRING_ARRAY = 9

    if isinstance(value, bool):
        param.value.type = TYPE_BOOL
        param.value.bool_value = value
    elif isinstance(value, int):
        param.value.type = TYPE_INTEGER
        param.value.integer_value = value
    elif isinstance(value, float):
        param.value.type = TYPE_DOUBLE
        param.value.double_value = value
    elif isinstance(value, str):
        param.value.type = TYPE_STRING
        param.value.string_value = value
    elif isinstance(value, list):  # 支持基本数组类型
        if all(isinstance(v, bool) for v in value):
            param.value.type = TYPE_BOOL_ARRAY
            param.value.bool_array_value = value
        elif all(isinstance(v, int) for v in value):
            param.value.type = TYPE_INTEGER_ARRAY
            param.value.integer_array_value = value
        elif all(isinstance(v, float) for v in value):
            param.value.type = TYPE_DOUBLE_ARRAY
            param.value.double_array_value = value
        elif all(isinstance(v, str) for v in value):
            param.value.type = TYPE_STRING_ARRAY
            param.value.string_array_value = value
        elif all(isinstance(v, bytes) for v in value):
            param.value.type = TYPE_BYTE_ARRAY
            param.value.byte_array_value = value
        else:
            if hasattr(node, "get_logger"):
                node.get_logger().error(f'Unsupported list type for parameter: {parameter_name}')
            return False
    else:
        if hasattr(node, "get_logger"):
            node.get_logger().error(f'Unsupported parameter type for {parameter_name}')
        return False

    request.parameters = [param]

    response = service_caller(
        node=node,
        service_name=service_name,
        service_type=SetParameters,
        request=request,
    )

    if not response or not hasattr(response, "results") or len(response.results) == 0:
        if hasattr(node, "get_logger"):
            node.get_logger().error(f'Failed to set parameter {parameter_name} for node {node_name}')
        return False

    result = response.results[0]
    if hasattr(result, "successful") and result.successful:
        return True
    else:
        if hasattr(node, "get_logger"):
            node.get_logger().warn(
                f"Failed to set parameter {parameter_name} on node {node_name}: "
                f"{getattr(result, 'reason', 'unknown reason')}"
            )
        return False


# tf2单例全局变量
_TF2_SINGLETONS = {}

def lookup_tf_once(node, target_frame, source_frame, timeout=3.0):
    """
    订阅一次TF，获取target_frame相对于source_frame的变换。

    Args:
        node: 当前ROS2节点
        target_frame (str): 目标坐标系
        source_frame (str): 源坐标系
        timeout (float): 等待TF消息的超时时间（秒）

    Returns:
        tuple: (translation, rotation) 如果成功，translation和rotation都是[x, y, z]和[x, y, z, w]形式的list
        None: 如果超时或查询失败
    """
    import time
    from tf2_ros.buffer import Buffer
    from tf2_ros.transform_listener import TransformListener
    from rclpy.duration import Duration
    import rclpy


    # tf2_ros.Buffer的确会随着TF消息不断积累历史数据（即累积buffer），
    # 但只会缓存在tf2_ros包实现的历史窗口（一般默认10s，详见TF2文档）。
    # 这里本函数采用全局单例 (_TF2_SINGLETONS) 仅与node相关，因此每个node最多只有一个buffer实例，
    # 不会无限膨胀，仅受TF2内置buffer历史窗口和订阅消息量影响。

    node_id = id(node)
    if node_id not in _TF2_SINGLETONS:
        # 如果TF2 Buffer未初始化/未缓存，则创建新实例（每个node只会有一个buffer）
        tf_buffer = Buffer()
        tf_listener = TransformListener(tf_buffer, node, spin_thread=False)
        _TF2_SINGLETONS[node_id] = (tf_buffer, tf_listener)
    else:
        tf_buffer, tf_listener = _TF2_SINGLETONS[node_id]

    start_time = time.time()
    while rclpy.ok() and (time.time() - start_time) < timeout:
        try:
            trans = tf_buffer.lookup_transform(
                target_frame=target_frame,
                source_frame=source_frame,
                time=rclpy.time.Time(),
                timeout=Duration(seconds=0.2),
            )
            t = trans.transform.translation
            r = trans.transform.rotation
            translation = [t.x, t.y, t.z]
            rotation = [r.x, r.y, r.z, r.w]
            return translation, rotation
        except Exception:
            # spin时进行sleep
            rclpy.spin_once(node, timeout_sec=0.01)
    if hasattr(node, "get_logger"):
        node.get_logger().warn(
            f"TF查找超时: {target_frame} <- {source_frame}, 超时{timeout}s"
        )
    return None


def list_controllers(node, controller_manager_name, service_timeout=0.0, call_timeout=10.0):
    # call_timeout kept at 10 s. The normal case responds in <100 ms and returns
    # immediately; the larger timeout only matters when the controller_manager
    # response is slow/dropped on first call, where 2 s was not enough on some
    # machines and caused all retries to fail.
    request = ListControllers.Request()
    return service_caller(
        node,
        f"{controller_manager_name}/list_controllers",
        ListControllers,
        request,
        service_timeout,
        call_timeout,
    )

def switch_controllers(
    node,
    controller_manager_name,
    deactivate_controllers,
    activate_controllers,
    strict,
    activate_asap,
    timeout,
    call_timeout=10.0,
):
    request = SwitchController.Request()
    request.activate_controllers = activate_controllers
    request.deactivate_controllers = deactivate_controllers
    if strict:
        request.strictness = SwitchController.Request.STRICT
    else:
        request.strictness = SwitchController.Request.BEST_EFFORT
    request.activate_asap = activate_asap
    request.timeout = rclpy.duration.Duration(seconds=timeout).to_msg()
    return service_caller(
        node,
        f"{controller_manager_name}/switch_controller",
        SwitchController,
        request,
        call_timeout=call_timeout,
    )


def configure_controller(
    node, controller_manager_name, controller_name, service_timeout=0.0, call_timeout=10.0
):
    request = ConfigureController.Request()
    request.name = controller_name
    return service_caller(
        node,
        f"{controller_manager_name}/configure_controller",
        ConfigureController,
        request,
        service_timeout,
        call_timeout,
    )

