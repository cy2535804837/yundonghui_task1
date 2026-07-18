# xarm_sdk

XArm机器人SDK - 用于控制和管理XArm机器人的Python库

## 简介

`xarm_sdk` 是一个基于Ros2的Python SDK，用于控制和管理XARM机器人系统。它提供了简洁的API接口，支持关节空间控制、笛卡尔空间控制、控制器管理等功能。

## 安装

### 使用pip安装

```bash
pip install .
```

### 开发模式安装

```bash
git clone <repository-url>
cd xarm_sdk
pip install -e .
```

## 前置要求

安装msg_deb下的两个包

## 快速开始

### 1. 控制器切换示例

```python
import rclpy
from xarm_sdk import XARM_manager

rclpy.init()
xarm_manager = XARM_manager()

# 停用所有控制器
xarm_manager.xarm_deactivate_all_controller()

# 激活控制器（自动处理资源冲突）
xarm_manager.xarm_activate_controller('jointspace_arm_L_controller')

# 查询激活的控制器
active = xarm_manager.xarm_find_active_controllers()
print(f"激活的控制器: {active}")

rclpy.shutdown()
```

### 2. Action 方式运动控制

```python
import rclpy
from xarm_sdk import XARM_manager, ActionCall
from geometry_msgs.msg import Pose

rclpy.init()
xarm_manager = XARM_manager()
action_call = ActionCall(xarm_manager)

# 关节空间控制（带自动激活控制器）
action_call.jointspace_arm_L_controller([0.0, 1.18, 0.0, -1.3, 0.0, -0.13, 0.18])

# 笛卡尔空间控制
target_pose = Pose()
target_pose.position.x = 0.4
target_pose.position.y = 0.5
target_pose.position.z = 1.0
target_pose.orientation.w = 1.0  # 四元数需要归一化

action_call.endpose_single_arm_qp_L_controller(target_pose)

rclpy.shutdown()
```

### 3. Topic 方式运动控制

```python
import rclpy
from xarm_sdk import XARM_manager, TopicPublisher
from geometry_msgs.msg import Pose

rclpy.init()
xarm_manager = XARM_manager()
topic_publisher = TopicPublisher(xarm_manager)

# 激活相应的控制器
xarm_manager.xarm_activate_controller('endpose_single_arm_qp_L_controller')

# 通过 Topic 发送目标位姿（高频控制）
target_pose = Pose()
target_pose.position.x = 0.4
target_pose.position.y = 0.5
target_pose.position.z = 1.0
target_pose.orientation.w = 1.0

# 持续发送命令（例如在循环中）
import time
for _ in range(100):
    topic_publisher.publish_endposetarget_L(target_pose)
    time.sleep(0.01)  # 100Hz

rclpy.shutdown()
```

### 4. 读取机器人状态

```python
import rclpy
from xarm_sdk import XARM_manager

rclpy.init()
xarm_manager = XARM_manager()

# 方法1：阻塞式更新关节状态
xarm_manager.joint_state_update()
left_joint_angles = xarm_manager.xarm_left_arm_joint_angles()
right_joint_angles = xarm_manager.xarm_right_arm_joint_angles()
print(f"左臂关节角度: {left_joint_angles}")
print(f"右臂关节角度: {right_joint_angles}")

# 方法2：使用实时订阅
joint_state = xarm_manager.get_latest_joint_state()
if joint_state:
    print(f"关节名称: {joint_state.name}")
    print(f"关节位置: {joint_state.position}")
    print(f"关节速度: {joint_state.velocity}")

rclpy.shutdown()
```

## API文档

### XARM_manager

控制器和硬件资源管理器，是整个SDK的核心节点。

**控制器管理方法：**

- `xarm_activate_controller(controller_name)` - 激活控制器（自动处理资源冲突）
- `xarm_deactivate_controller(controller_name)` - 停用控制器
- `xarm_deactivate_all_controller()` - 停用所有控制器
- `xarm_find_active_controllers()` - 查询激活的控制器
- `xarm_find_inactive_controllers()` - 查询未激活的控制器
- `xarm_find_unconfigured_controllers()` - 查询未配置的控制器

**硬件管理方法：**

- `hardware_arm_enable(enable)` - 使能/禁用机械臂硬件
- `hardware_debug()` - 获取硬件调试信息

**状态读取方法：**

- `joint_state_update(timeout)` - 阻塞式更新关节状态（返回 JointState 消息）
- `get_latest_joint_state()` - 获取最新的关节状态（实时订阅，非阻塞，需要spin管理器节点）
- `xarm_left_arm_joint_angles()` - 获取左臂关节角度
- `xarm_right_arm_joint_angles()` - 获取右臂关节角度
- `xarm_left_arm_joint_velocities()` - 获取左臂关节速度
- `xarm_right_arm_joint_velocities()` - 获取右臂关节速度
- `xarm_left_arm_joint_efforts()` - 获取左臂关节力矩
- `xarm_right_arm_joint_efforts()` - 获取右臂关节力矩

### ActionCall

运动控制接口，封装了ROS2 Action调用，适用于需要完成反馈的运动任务。

**关节空间控制方法：**

- `jointspace_arm_L_controller(target_positions, feedback_callback)` - 控制左臂
- `jointspace_arm_R_controller(target_positions, feedback_callback)` - 控制右臂
- `jointspace_dual_arm_controller(left_positions, right_positions, feedback_callback)` - 单个action控制双臂
- `jointspace_head_controller(target_positions, feedback_callback)` - 控制头部（3个关节）
- `jointspace_body_controller(target_positions, feedback_callback)` - 控制身体（4个关节）
- `jointspace_waist_pitch_controller(target_positions, feedback_callback)` - 控制腰部俯仰（1个关节）
- `jointspace_waist_yaw_controller(target_positions, feedback_callback)` - 控制腰部摆动（1个关节）

**笛卡尔空间控制方法：**

- `endpose_single_arm_qp_L_controller(target_pose, ...)` - 控制左臂到指定位姿（QP 算法）
- `endpose_single_arm_qp_R_controller(target_pose, ...)` - 控制右臂到指定位姿（QP 算法）
- `endpose_single_arm_qpik_L_controller(target_pose, from_frame, to_frame, offset)` - 左臂 QPIK 末端位姿控制（支持多参考系与末端偏移）
- `endpose_single_arm_qpik_R_controller(target_pose, from_frame, to_frame, offset)` - 右臂 QPIK 末端位姿控制（支持多参考系与末端偏移）

**参数说明：**

- `target_positions`: 关节角度列表（弧度）
- `target_pose`: 目标位姿（geometry_msgs.msg.Pose）
- `from_frame`: 目标位姿所在坐标系（如 `"base"`、`"waist_yaw_link"`，默认 `"base"`）
- `to_frame`: 末端连杆坐标系（如 `"left_tcp_link"`、`"right_tcp_link"`，左/右臂有默认值）
- `offset`: 末端偏移 `[x, y, z, rx, ry, rz]`（米/弧度），仅 QPIK 接口支持
- `feedback_callback`: 可选的反馈回调函数

### TopicPublisher

Topic 发布接口，适用于高频实时控制（如轨迹跟踪）。

**关节空间 Topic 发布方法：**

- `publish_jointspace_commands_L(target_positions)` - 发布左臂关节命令
- `publish_jointspace_commands_R(target_positions)` - 发布右臂关节命令
- `publish_jointspace_commands_Dual(left_positions, right_positions)` - 发布双臂关节命令
- `publish_jointspace_commands_body(target_positions)` - 发布身体关节命令
- `publish_jointspace_commands_head(target_positions)` - 发布头部关节命令
- `publish_jointspace_commands_waist_pitch(target_positions)` - 发布腰部俯仰命令
- `publish_jointspace_commands_waist_yaw(target_positions)` - 发布腰部摆动命令

**笛卡尔空间 Topic 发布方法：**

- `publish_endposetarget_L(target_pose)` - 发布左臂末端位姿命令
- `publish_endposetarget_R(target_pose)` - 发布右臂末端位姿命令

**参数说明：**

- `target_positions`: 关节角度列表（弧度）
- `target_pose`: 目标位姿（geometry_msgs.msg.Pose）

**注意：** Topic 方式需要先手动激活相应的控制器。

### MoveitCall

MoveIt! 运动规划接口，基于 MoveIt! 进行路径规划，支持自动避障和碰撞检测。

**双臂随机运动方法：**

- `dual_arm_random_run()` - 双臂随机运动（同步，阻塞）
- `dual_arm_random_run_async()` - 双臂随机运动（异步，返回 Future）

**单臂关节空间控制方法：**

- `left_arm_joint_angles(angles, vel_scale, acc_scale)` - 左臂关节空间控制（同步）
- `left_arm_joint_angles_async(angles, vel_scale, acc_scale)` - 左臂关节空间控制（异步）
- `right_arm_joint_angles(angles, vel_scale, acc_scale)` - 右臂关节空间控制（同步）
- `right_arm_joint_angles_async(angles, vel_scale, acc_scale)` - 右臂关节空间控制（异步）

**双臂关节空间控制方法：**

- `dual_arm_joint_angles(left_angles, right_angles, vel_scale, acc_scale)` - 双臂协同控制（同步）
- `dual_arm_joint_angles_async(left_angles, right_angles, vel_scale, acc_scale)` - 双臂协同控制（异步）

**末端轨迹控制方法：**

- `arm_waypoints(json_data)` - 末端轨迹控制（同步）
- `arm_waypoints_async(json_data)` - 末端轨迹控制（异步）

**辅助方法（构建 waypoints JSON）：**

- `build_left_arm_waypoints_json(waypoints)` - 构建左臂路点 JSON 数据
- `build_right_arm_waypoints_json(waypoints)` - 构建右臂路点 JSON 数据
- `build_dual_arm_waypoints_json(left_waypoints, right_waypoints)` - 构建双臂路点 JSON 数据

**参数说明：**

- `angles`: 关节角度列表（7个关节，弧度）
- `vel_scale`: 速度缩放系数（默认 0.1）
- `acc_scale`: 加速度缩放系数（默认 0.1）
- `waypoints`: 路点列表，每个路点格式为 `[x, y, z, qx, qy, qz, qw]`（位置 + 姿态四元数）
- `json_data`: MoveIt! 服务所需的 JSON 格式数据

**返回值：**

- 同步方法：返回 `(result, error_code)` 元组
  - `result`: `True` 表示成功，`False` 表示失败
  - `error_code`: 错误代码，`0` 表示无错误
- 异步方法：返回 `rclpy.task.Future` 对象
  - 使用 `rclpy.spin_until_future_complete()` 等待完成
  - `future.result()` 返回 `(result, error_code)` 元组

**MoveIt! vs Action/Topic：**

- **MoveIt!**: 基于运动规划，自动避障和碰撞检测，适合复杂场景
- **Action**: 直接控制，执行速度快，有完成反馈，适合简单点对点运动
- **Topic**: 高频实时控制，适合轨迹跟踪和遥操作

### 工具函数

**action_caller** - 底层 Action 调用函数（通常通过 ActionCall 类使用）

**topic_publisher** - 底层 Topic 发布函数（通常通过 TopicPublisher 类使用）

**topic_subscriber** - Topic 订阅工具函数

```python
from xarm_sdk import topic_subscriber
from sensor_msgs.msg import JointState

# 同步订阅一次消息（带超时）
joint_state = topic_subscriber(node, "/joint_states", JointState, timeout=2.0)
```

**单例模式：** 所有 Action Client、Publisher 和 Subscriber 都使用单例模式管理，相同参数的调用会自动复用实例，提升性能和资源利用率。

## 示例代码

示例代码按机器人类型分类，位于 `demo/` 目录下：
- `demo/tianyi/` - 天轶2 型机器人示例
- `demo/tiangong/` - 天工2 系列机器人示例

### 天轶2 示例 (demo/tianyi/)

#### 1. 控制器切换示例 (`1_contorller_switch.py`)

演示如何管理控制器：
- 停用所有控制器
- 查询控制器状态
- 激活和停用控制器
- 处理控制器资源冲突

```bash
python3 demo/tianyi/1_contorller_switch.py
```

#### 2. Action 方式运动控制 (`2_basic_move_action.py`)

演示使用 Action 方式控制机器人：
- 关节空间控制（单臂、双臂、身体、头部）
- 笛卡尔空间控制
- 轨迹跟踪（立方体轨迹）
- Action 完成反馈

```bash
python3 demo/tianyi/2_basic_move_action.py
```

#### 3. Topic 方式运动控制 (`3_basic_move_topic.py`)

演示使用 Topic 方式控制机器人：
- 高频关节空间控制（100Hz）
- Topic 实时发布

```bash
python3 demo/tianyi/3_basic_move_topic.py
```

#### 4. 笛卡尔空间 Topic 控制 (`4_endpose_control.py`)

演示使用 Topic 方式进行笛卡尔空间控制：
- 高频末端位姿控制
- 圆形轨迹跟踪
- 双臂协同笛卡尔运动

```bash
python3 demo/tianyi/4_endpose_control.py
```

#### 5. 机器人状态读取 (`5_read_robot_state.py`)

演示如何读取机器人状态：
- 阻塞式关节状态更新
- 实时关节状态订阅
- 获取左右臂关节角度、速度、力矩
- 在运动过程中读取实时状态

```bash
python3 demo/tianyi/5_read_robot_state.py
```

#### 6. MoveIt! 运动规划控制 (`6_moveit_control.py`)

演示如何使用 MoveIt! 进行运动规划：
- 双臂随机运动（同步和异步）
- 单臂关节空间控制（左臂/右臂，同步/异步）
- 双臂关节空间协同控制
- 单臂末端轨迹控制（waypoints）
- 双臂末端轨迹协同控制
- MoveIt! 自动路径规划和避障

```bash
python3 demo/tianyi/6_moveit_control.py
```

#### 7. 硬件使能与模式配置 (`7_enable_hardware.py`)

演示机器人硬件使能与运行模式配置：
- 硬件使能/去使能（双臂、头部、腿部、腰部）
- 硬件运行模式配置（位置环、力位混合、重力补偿等）
- 硬件调试信息获取
- 真实模式与仿真模式下的行为差异

```bash
python3 demo/tianyi/7_enable_hardware.py
```

#### 8. QPIK 末端位姿逆解测试 (`8_qpik_test.py`)

演示不同参考系下的 QPIK 末端位姿控制：
- 左臂/右臂在 base 系下的目标位姿控制（可指定 to_frame）
- 在 waist_yaw_link 系下的目标位姿控制
- 带末端偏移 `offset [x,y,z,rx,ry,rz]` 的 QPIK 控制
- 分步回车交互，便于逐项验证

```bash
python3 demo/tianyi/8_qpik_test.py
```

### 天工2 示例 (demo/tiangong/)

天工2 系列的示例代码，使用方法类似。

## 支持的机器人类型

- `tianyi2` - 天翼2型机器人
- `tiangong2pro` - 天工2 Pro型机器人
- `tiangong2dex` - 天工2 Dex型机器人

## 控制器资源说明

### 硬件资源

每个控制器占用特定的硬件资源：
- `left_arm` - 左臂资源（7个关节）
- `right_arm` - 右臂资源（7个关节）
- `head` - 头部资源（3个关节：yaw, pitch, roll）
- `body` - 身体资源（4个关节：腿部2个 + 腰部2个）

**重要提示：** 不能同时激活占用相同硬件资源的控制器。`xarm_activate_controller()` 方法会自动处理资源冲突。

### 可用控制器

**左臂控制器：**
- `jointspace_arm_L_controller` - 关节空间控制
- `jointspace_single_arm_qpik_L_controller` - 关节空间控制（QPIK算法）
- `endpose_single_arm_qp_L_controller` - 笛卡尔空间控制（QP算法）
- `endpose_single_arm_qpik_L_controller` - 笛卡尔空间控制（QPIK算法）
- `moveit_left_arm_controller` - MoveIt! 运动规划

**右臂控制器：**
- `jointspace_arm_R_controller` - 关节空间控制
- `jointspace_single_arm_qpik_R_controller` - 关节空间控制（QPIK算法）
- `endpose_single_arm_qp_R_controller` - 笛卡尔空间控制（QP算法）
- `endpose_single_arm_qpik_R_controller` - 笛卡尔空间控制（QPIK算法）
- `moveit_right_arm_controller` - MoveIt! 运动规划

**双臂控制器：**
- `jointspace_dual_arm_controller` - 双臂关节空间协同控制
- `endpose_dual_arm_qp_controller` - 双臂笛卡尔空间协同控制
- `moveit_dual_arm_controller` - 双臂 MoveIt! 运动规划

**头部控制器：**
- `jointspace_head_controller` - 头部关节空间控制
- `endpose_head_controller` - 头部笛卡尔空间控制

**身体控制器：**
- `jointspace_body_controller` - 身体关节空间控制
- `jointspace_waist_pitch_controller` - 腰部俯仰控制
- `jointspace_waist_yaw_controller` - 腰部摆动控制
- `endpose_body_controller` - 身体笛卡尔空间控制

## 控制方式对比

### Action 控制

**优点：**
- 自动激活控制器
- 提供完成反馈
- 适合需要确认到达的运动任务
- 编程简单直观

**缺点：**
- 不适合高频控制
- 不适合实时更改目标
- 无路径规划

**适用场景：** 点对点运动、需要反馈确认的任务

### Topic 控制

**优点：**
- 适合高频控制（可达 100Hz+）
- 适合实时轨迹跟踪
- 延迟最低

**缺点：**
- 需要手动激活控制器
- 无完成反馈
- 需要持续发送命令
- 无路径规划

**适用场景：** 轨迹跟踪、实时控制、遥操作

### MoveIt! 控制

**优点：**
- 自动路径规划
- 碰撞检测和避障
- 适合复杂环境
- 支持路点（waypoints）控制

**缺点：**
- 规划时间较长
- 需要场景模型配置
- 资源占用较高

**适用场景：** 复杂环境运动、需要避障的任务、多路点轨迹规划

### 控制方式选择建议

| 场景 | 推荐方式 | 原因 |
|------|---------|------|
| 简单点对点运动 | Action | 简单可靠，有反馈 |
| 高频实时控制 | Topic | 延迟低，频率高 |
| 复杂环境避障 | MoveIt! | 自动规划，安全 |
| 轨迹跟踪 | Topic | 实时性好 |
| 多路点规划 | MoveIt! | 支持复杂路径 |

## 重要提示

### 1. 初始化顺序

```python
import rclpy
from xarm_sdk import XARM_manager, ActionCall, TopicPublisher

# 必须先初始化 ROS2
rclpy.init()

# 创建管理器节点
xarm_manager = XARM_manager()

# 创建控制接口
action_call = ActionCall(xarm_manager)
topic_publisher = TopicPublisher(xarm_manager)
```

### 2. 控制器切换

- **Action 方式**：会自动激活所需控制器
- **Topic 方式**：需要手动激活控制器

```python
# Topic 方式需要手动激活
xarm_manager.xarm_activate_controller('endpose_single_arm_qp_L_controller')
topic_publisher.publish_endposetarget_L(target_pose)
```

### 3. 资源冲突处理

激活新控制器时，SDK 会自动停用占用相同硬件资源的控制器：

```python
# 如果 jointspace_arm_L_controller 已激活
# 以下调用会自动停用它并激活新控制器
xarm_manager.xarm_activate_controller('endpose_single_arm_qp_L_controller')
```

### 4. 关节状态读取

两种方式读取关节状态：

```python
# 方式1：阻塞式更新（等待新消息）
xarm_manager.joint_state_update(timeout=2.0)
angles = xarm_manager.xarm_left_arm_joint_angles()

# 方式2：非阻塞获取（返回已订阅的最新状态）
joint_state = xarm_manager.get_latest_joint_state()
```


### 5. MoveIt! 使用示例

```python
import rclpy
from xarm_sdk import XARM_manager, MoveitCall

rclpy.init()
xarm_manager = XARM_manager()
moveit_call = MoveitCall(xarm_manager)

# 示例 1: 双臂随机运动（同步）
result, error_code = moveit_call.dual_arm_random_run()
print(f"结果: {result}, 错误码: {error_code}")

# 示例 2: 左臂关节空间控制（异步）
left_angles = [0.0, 1.18, 0.0, -1.3, 0.0, -0.13, 0.18]
future = moveit_call.left_arm_joint_angles_async(left_angles, vel_scale=0.1)
rclpy.spin_until_future_complete(xarm_manager, future)
result, error_code = future.result()
print(f"结果: {result}, 错误码: {error_code}")

# 示例 3: 末端轨迹控制
waypoints = [[0.27, 0.37, 0.10, -0.15, -0.64, -0.17, 0.73]]
json_data = moveit_call.build_left_arm_waypoints_json(waypoints)
result, error_code = moveit_call.arm_waypoints(json_data)
print(f"结果: {result}, 错误码: {error_code}")

rclpy.shutdown()
```

## 更新日志


### v0.1.0
- ✨ 初始版本发布
- ✨ 支持 Action、Topic 和 MoveIt! 三种控制方式
- ✨ 实现单例模式优化（ActionClient、Publisher、Subscriber）
- ✨ 添加关节状态订阅功能
- ✨ 提供完整的 MoveIt! 运动规划接口
- 📝 完善文档和示例代码（6 个详细示例）
