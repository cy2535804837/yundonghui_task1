from typing import List, Optional, Callable

from .tools import (
    action_caller,
    measure_time,
    require_robot_type,
    topic_publisher,
)
from .XARM_manager import XARM_manager

from eai_manipulator_msgs.action import JointSpace, EndPosSingleTarget
from eai_manipulator_msgs.msg import ArmTargetPose
from eai_manipulator_msgs.msg import DualArmTargetPose
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64MultiArray
from cm_msgs.srv import CommonStreamRos2
import json
import time
import rclpy
import asyncio
from rclpy.task import Future as RclpyFuture
from .tools import *


class ActionCall:
    def __init__(self, node: XARM_manager):
        self.node = node
        self.time_measurement = False

    def get_logger(self):
        """获取日志记录器，委托给node的get_logger方法"""
        return self.node.get_logger()

    # 关节空间控制

    
    @measure_time
    @require_robot_type(["tianyi2", "tianyi_max", "tiangong2pro", "tiangong2dex"])
    def jointspace_arm_L_controller(self, target_positions: List[float], feedback_callback: Optional[Callable] = None):
        """
        控制左臂到指定的关节空间位置
        
        Args:
            target_positions: 目标关节位置列表（7个关节）
            feedback_callback: 可选的反馈回调函数
        """
        # 参数验证
        if not isinstance(target_positions, list) or len(target_positions) != 7:
            self.node.get_logger().error("target_positions must be a list of length 7.")
            return None
        
        self.node.xarm_activate_controller('jointspace_arm_L_controller')

        # 创建目标消息
        goal_msg = JointSpace.Goal()
        goal_msg.target_positions = target_positions

        # 调用action
        return action_caller(
            node=self.node,
            action_name='/jointspace_arm_L_controller/jointspace',
            action_type=JointSpace,
            goal_msg=goal_msg,
            feedback_callback=feedback_callback,
        )

    @measure_time
    @require_robot_type(["tianyi2", "tianyi_max", "tiangong2pro", "tiangong2dex"])
    def jointspace_arm_R_controller(self, target_positions: List[float], feedback_callback: Optional[Callable] = None):
        """
        控制右臂到指定的关节空间位置

        Args:
            target_positions: 目标关节位置列表（7个关节）
            feedback_callback: 可选的反馈回调函数
        """
        # 参数验证
        if not isinstance(target_positions, list) or len(target_positions) != 7:
            self.node.get_logger().error("target_positions must be a list of length 7.")
            return None

        self.node.xarm_activate_controller('jointspace_arm_R_controller')

        # 创建目标消息
        goal_msg = JointSpace.Goal()
        goal_msg.target_positions = target_positions

        # 调用action
        return action_caller(
            node=self.node,
            action_name='/jointspace_arm_R_controller/jointspace',
            action_type=JointSpace,
            goal_msg=goal_msg,
            feedback_callback=feedback_callback,
        )



    
    # @measure_time
    # @require_robot_type(["tianyi2", "tianyi_max", "tiangong2pro", "tiangong2dex"])
    # def jointspace_dual_arm_controller(self, left_target_positions: List[float], right_target_positions: List[float], feedback_callback: Optional[Callable] = None):
    #     """
    #     控制右臂到指定的关节空间位置

    #     Args:
    #         target_positions: 目标关节位置列表（7个关节）
    #         feedback_callback: 可选的反馈回调函数
    #     """
    #     # 参数验证
    #     if not isinstance(left_target_positions, list) or len(left_target_positions) != 7 or not isinstance(right_target_positions, list) or len(right_target_positions) != 7:
    #         self.node.get_logger().error("target_positions must be a list of length 7.")
    #         return None

    #     self.node.xarm_activate_controller('jointspace_dual_arm_controller')

    #     # 创建目标消息
    #     goal_msg = JointSpace.Goal()
    #     goal_msg.target_positions = left_target_positions + right_target_positions

    #     # 调用action
    #     return action_caller(
    #         node=self.node,
    #         action_name='/jointspace_dual_arm_controller/jointspace',
    #         action_type=JointSpace,
    #         goal_msg=goal_msg,
    #         feedback_callback=feedback_callback,
    #     )
    

    @measure_time
    @require_robot_type(["tianyi2", "tianyi_max"])
    def jointspace_body_controller(self, target_positions: List[float], feedback_callback: Optional[Callable] = None):
        """
        控制身体到指定的关节空间位置（4个关节）

        Args:
            target_positions: 目标关节位置列表（应为4个关节）
            feedback_callback: 可选的反馈回调函数
        """
        # 参数验证
        if not isinstance(target_positions, list) or len(target_positions) != 4:
            self.node.get_logger().error("target_positions must be a list of length 4.")
            return None

        self.node.xarm_activate_controller('jointspace_body_controller')

        # 创建目标消息
        goal_msg = JointSpace.Goal()
        goal_msg.target_positions = target_positions

        # 调用action
        return action_caller(
            node=self.node,
            action_name='/jointspace_body_controller/jointspace',
            action_type=JointSpace,
            goal_msg=goal_msg,
            feedback_callback=feedback_callback,
        )

    @measure_time
    @require_robot_type(["tianyi2", "tianyi_max"])
    def jointspace_head_controller(self, target_positions: List[float], feedback_callback: Optional[Callable] = None):
        """
        控制头部到指定的关节空间位置（3自由度）

        Args:
            target_positions: 目标关节位置列表（应为3个关节）
            feedback_callback: 可选的反馈回调函数
        """
        if not isinstance(target_positions, list) or len(target_positions) != 3:
            self.node.get_logger().error("target_positions must be a list of length 3.")
            return None

        self.node.xarm_activate_controller('jointspace_head_controller')

        goal_msg = JointSpace.Goal()
        goal_msg.target_positions = target_positions

        return action_caller(
            node=self.node,
            action_name='/jointspace_head_controller/jointspace',
            action_type=JointSpace,
            goal_msg=goal_msg,
            feedback_callback=feedback_callback,
        )

    @measure_time
    @require_robot_type(["tianyi2", "tianyi_max"])
    def jointspace_waist_pitch_controller(self, target_positions: List[float], feedback_callback: Optional[Callable] = None):
        """
        控制腰部俯仰到指定的关节空间位置（通常为1自由度）

        参数:
            target_positions (List[float]): 目标关节位置（长度通常为1的列表）
            feedback_callback (Optional[Callable]): 可选的反馈回调函数
        """
        if not isinstance(target_positions, list) or len(target_positions) != 1:
            self.node.get_logger().error("target_positions must be a list of length 1.")
            return None

        self.node.xarm_activate_controller('jointspace_waist_pitch_controller')

        goal_msg = JointSpace.Goal()
        goal_msg.target_positions = target_positions

        return action_caller(
            node=self.node,
            action_name='/jointspace_waist_pitch_controller/jointspace',
            action_type=JointSpace,
            goal_msg=goal_msg,
            feedback_callback=feedback_callback,
        )

    @measure_time
    @require_robot_type(["tianyi2", "tianyi_max"])
    def jointspace_waist_yaw_controller(self, target_positions: List[float], feedback_callback: Optional[Callable] = None):
        """
        控制腰部摆动到指定的关节空间位置（1自由度）

        Args:
            target_positions: 目标关节位置列表（应为1个关节）
            feedback_callback: 可选的反馈回调函数
        """
        if not isinstance(target_positions, list) or len(target_positions) != 1:
            self.node.get_logger().error("target_positions must be a list of length 1.")
            return None

        self.node.xarm_activate_controller('jointspace_waist_yaw_controller')

        goal_msg = JointSpace.Goal()
        goal_msg.target_positions = target_positions

        return action_caller(
            node=self.node,
            action_name='/jointspace_waist_yaw_controller/jointspace',
            action_type=JointSpace,
            goal_msg=goal_msg,
            feedback_callback=feedback_callback,
        )


    # 笛卡尔空间控制
    @measure_time
    @require_robot_type(["tianyi2", "tianyi_max", "tiangong2pro", "tiangong2dex"])
    def endpose_single_arm_qp_L_controller(
        self, 
        target_pose: Pose, 
        from_frame: str = "base_footprint", 
        to_frame: str = "left_tcp_link", 
        offset_x: float = 0.0, 
        offset_y: float = 0.0, 
        offset_z: float = 0.0, 
        feedback_callback: Optional[Callable] = None
    ):
        """
        控制左臂到指定的笛卡尔空间位置
        
        Args:
            target_pose: 目标位姿（geometry_msgs.msg.Pose）
            feedback_callback: 可选的反馈回调函数
        """
        self.node.xarm_activate_controller('endpose_single_arm_qp_L_controller')
    
        # 构造目标消息
        endpose_goal = EndPosSingleTarget.Goal()
        
        # 填写目标的笛卡尔坐标与姿态
        endpose_goal.target.from_frame = from_frame
        endpose_goal.target.to_frame = to_frame
        endpose_goal.target.offset_x = offset_x
        endpose_goal.target.offset_y = offset_y
        endpose_goal.target.offset_z = offset_z
        endpose_goal.target.target = target_pose

        # 调用action
        return action_caller(
            node=self.node,
            action_name='/endpose_single_arm_qp_L_controller/endPosSingleTarget',
            action_type=EndPosSingleTarget,
            goal_msg=endpose_goal,
            feedback_callback=feedback_callback,
        )

    @measure_time
    @require_robot_type(["tianyi2", "tianyi_max", "tiangong2pro", "tiangong2dex"])
    def endpose_single_arm_qp_R_controller(
        self, 
        target_pose: Pose, 
        from_frame: str = "base_footprint", 
        to_frame: str = "right_tcp_link", 
        offset_x: float = 0.0, 
        offset_y: float = 0.0, 
        offset_z: float = 0.0, 
        feedback_callback: Optional[Callable] = None
    ):
        """
        控制左臂到指定的笛卡尔空间位置
        
        Args:
            target_pose: 目标位姿（geometry_msgs.msg.Pose）
            feedback_callback: 可选的反馈回调函数
        """
        self.node.xarm_activate_controller('endpose_single_arm_qp_R_controller')
    
        # 构造目标消息
        endpose_goal = EndPosSingleTarget.Goal()
        
        # 填写目标的笛卡尔坐标与姿态
        endpose_goal.target.from_frame = from_frame
        endpose_goal.target.to_frame = to_frame
        endpose_goal.target.offset_x = offset_x
        endpose_goal.target.offset_y = offset_y
        endpose_goal.target.offset_z = offset_z
        endpose_goal.target.target = target_pose

        # 调用action
        return action_caller(
            node=self.node,
            action_name='/endpose_single_arm_qp_R_controller/endPosSingleTarget',
            action_type=EndPosSingleTarget,
            goal_msg=endpose_goal,
            feedback_callback=feedback_callback,
        )


    # 笛卡尔空间控制
    @measure_time
    @require_robot_type(["tianyi2", "tianyi_max", "tiangong2pro", "tiangong2dex"])
    def endpose_single_arm_qpik_L_controller(
        self, 
        target_pose: Pose, 
        from_frame: str = "base", 
        to_frame: str = "left_tcp_link", 
        offset = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        feedback_callback: Optional[Callable] = None
    ):
        """
        控制左臂到指定的笛卡尔空间位置
        
        Args:
            target_pose: 目标位姿（geometry_msgs.msg.Pose）
            feedback_callback: 可选的反馈回调函数
        """
        set_node_parameter(self.node, 'endpose_single_arm_qpik_L_controller', 'from_frame', from_frame)
        set_node_parameter(self.node, 'endpose_single_arm_qpik_L_controller', 'to_frame', to_frame)
        set_node_parameter(self.node, 'endpose_single_arm_qpik_L_controller', 'EE_offset', offset)

        self.node.xarm_activate_controller('endpose_single_arm_qpik_L_controller')
    
        # 构造目标消息
        endpose_goal = EndPosSingleTarget.Goal()
        
        # 填写目标的笛卡尔坐标与姿态
        endpose_goal.target.from_frame = from_frame
        endpose_goal.target.to_frame = to_frame
        endpose_goal.target.offset_x = 0.0
        endpose_goal.target.offset_y = 0.0
        endpose_goal.target.offset_z = 0.0
        endpose_goal.target.target = target_pose

        # 调用action
        return action_caller(
            node=self.node,
            action_name='/endpose_single_arm_qpik_L_controller/endPosSingleTarget',
            action_type=EndPosSingleTarget,
            goal_msg=endpose_goal,
            feedback_callback=feedback_callback,
        )

    @measure_time
    @require_robot_type(["tianyi2", "tianyi_max", "tiangong2pro", "tiangong2dex"])
    def endpose_single_arm_qpik_R_controller(
        self, 
        target_pose: Pose, 
        from_frame: str = "base", 
        to_frame: str = "right_tcp_link", 
        offset = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        feedback_callback: Optional[Callable] = None
    ):
        """
        控制左臂到指定的笛卡尔空间位置
        
        Args:
            target_pose: 目标位姿（geometry_msgs.msg.Pose）
            feedback_callback: 可选的反馈回调函数
        """

        set_node_parameter(self.node, "endpose_single_arm_qpik_R_controller", "from_frame", from_frame)
        set_node_parameter(self.node, "endpose_single_arm_qpik_R_controller", "to_frame", to_frame)
        set_node_parameter(self.node, 'endpose_single_arm_qpik_R_controller', 'EE_offset', offset)
        
        self.node.xarm_activate_controller('endpose_single_arm_qpik_R_controller')
    
        # 构造目标消息
        endpose_goal = EndPosSingleTarget.Goal()
        
        # 填写目标的笛卡尔坐标与姿态
        endpose_goal.target.from_frame = from_frame
        endpose_goal.target.to_frame = to_frame
        endpose_goal.target.offset_x = 0.0
        endpose_goal.target.offset_y = 0.0
        endpose_goal.target.offset_z = 0.0
        endpose_goal.target.target = target_pose

        # 调用action
        return action_caller(
            node=self.node,
            action_name='/endpose_single_arm_qpik_R_controller/endPosSingleTarget',
            action_type=EndPosSingleTarget,
            goal_msg=endpose_goal,
            feedback_callback=feedback_callback,
        )

    
    @measure_time
    @require_robot_type(["tianyi2", "tianyi_max"])
    def endpose_body_controller(self, target_pose: List[float], feedback_callback: Optional[Callable] = None):
        """
        控制身体到指定的笛卡尔空间位置
        天轶的零位在 [0.05, 0.68, 0.0, 0.0]
    
        """
        if not isinstance(target_pose, list) or len(target_pose) != 4:
            self.node.get_logger().error("target_pose长度必须为4，顺序为: x, z, pitch, yaw")
            return None
        self.node.xarm_activate_controller('endpose_body_controller')
        endpose_goal = JointSpace.Goal()
        endpose_goal.target_positions = target_pose
        return action_caller(
            node=self.node,
            action_name='/endpose_body_controller/jointspace',
            action_type=JointSpace,
            goal_msg=endpose_goal,
            feedback_callback=feedback_callback,
        )
    

class TopicPublisher:
    def __init__(self, node: XARM_manager):
        self.node = node
        self.time_measurement = False

    def get_logger(self):
        """获取日志记录器，委托给node的get_logger方法"""
        return self.node.get_logger()

    @require_robot_type(["tianyi2", "tianyi_max", "tiangong2pro", "tiangong2dex"])
    def publish_jointspace_commands_L(self, target_positions: List[float]):
        """
        发布左臂到指定的关节空间位置
        """
        # 判断长度7
        if len(target_positions) != 7:
            self.node.get_logger().error("target_positions长度必须为7")
            return None
        joint_commands = Float64MultiArray()
        joint_commands.data = target_positions
        topic_publisher(
            self.node,                          
            "/jointspace_commands_L",          
            Float64MultiArray,                 
            joint_commands                     
        )
    
    @require_robot_type(["tianyi2", "tianyi_max", "tiangong2pro", "tiangong2dex"])
    def publish_jointspace_commands_R(self, target_positions: List[float]):
        """
        发布右臂到指定的关节空间位置
        """
        # 判断长度7
        if len(target_positions) != 7:
            self.node.get_logger().error("target_positions长度必须为7，右臂")
            return None
        joint_commands = Float64MultiArray()
        joint_commands.data = target_positions
        topic_publisher(
            self.node,                          
            "/jointspace_commands_R",          
            Float64MultiArray,                 
            joint_commands                     
        )
    
    @require_robot_type(["tianyi2", "tianyi_max", "tiangong2pro", "tiangong2dex"])
    def publish_jointspace_commands_Dual(self, target_positions_L: List[float], target_positions_R: List[float]):
        """
        同时发布左右臂到指定的关节空间位置（/jointspace_commands_Dual 话题）
        消息格式：Float64MultiArray，前7位为左臂，后7位为右臂
        """
        if len(target_positions_L) != 7 or len(target_positions_R) != 7:
            self.node.get_logger().error("target_positions_L和target_positions_R的长度都必须为7")
            return None
        joint_commands = Float64MultiArray()
        joint_commands.data = target_positions_L + target_positions_R
        topic_publisher(
            self.node,
            "/jointspace_commands_Dual",
            Float64MultiArray,
            joint_commands
        )



        # INSERT_YOUR_CODE

    @require_robot_type(["tianyi2", "tianyi_max"])
    def publish_jointspace_commands_body(self, target_positions: List[float]):
        """
        发布身体（body）关节的目标位置到 /jointspace_commands_body 话题
        """
        if len(target_positions) != 4:
            self.node.get_logger().error("target_positions长度必须为4，body")
            return None
        joint_commands = Float64MultiArray()
        joint_commands.data = target_positions
        topic_publisher(
            self.node,
            "/jointspace_commands_body",
            Float64MultiArray,
            joint_commands
        )

    @require_robot_type(["tianyi2", "tianyi_max"])
    def publish_jointspace_commands_body3joints(self, target_positions: List[float]):
        """
        发布身体（body）关节的目标位置到 /jointspace_commands_body3joints 话题
        """
        if len(target_positions) != 3:
            self.node.get_logger().error("target_positions长度必须为3，body3joints")
            return None
        joint_commands = Float64MultiArray()
        joint_commands.data = target_positions
        topic_publisher(
            self.node,
            "/jointspace_commands_body3joints",
            Float64MultiArray,
            joint_commands
        )
    @require_robot_type(["tianyi2", "tianyi_max"])
    def publish_jointspace_commands_head(self, target_positions: List[float]):
        """
        发布头部（head）关节的目标位置到 /jointspace_commands_head 话题
        """
        if len(target_positions) != 3:
            self.node.get_logger().error("target_positions长度必须为3，head")
            return None
        joint_commands = Float64MultiArray()
        joint_commands.data = target_positions
        topic_publisher(
            self.node,
            "/jointspace_commands_head",
            Float64MultiArray,
            joint_commands
        )
    
    @require_robot_type(["tianyi2", "tianyi_max"])
    def publish_jointspace_commands_waist_pitch(self, target_positions: List[float]):
        """
        发布腰部 pitch 关节的目标位置到 /jointspace_commands_waist_pitch 话题
        """
        if len(target_positions) != 1:
            self.node.get_logger().error("target_positions长度必须为1，waist_pitch (dof 1)")
            return None
        joint_commands = Float64MultiArray()
        joint_commands.data = target_positions
        topic_publisher(
            self.node,
            "/jointspace_commands_waist_pitch",
            Float64MultiArray,
            joint_commands
        )
    
    @require_robot_type(["tianyi2", "tianyi_max"])
    def publish_jointspace_commands_waist_yaw(self, target_positions: List[float]):
        """
        发布腰部 yaw 关节的目标位置到 /jointspace_commands_waist_yaw 话题
        """
        if len(target_positions) != 1:
            self.node.get_logger().error("target_positions长度必须为1，waist_yaw (dof 1)")
            return None
        joint_commands = Float64MultiArray()
        joint_commands.data = target_positions
        topic_publisher(
            self.node,
            "/jointspace_commands_waist_yaw",
            Float64MultiArray,
            joint_commands
        )

    @require_robot_type(["tianyi2", "tianyi_max", "tiangong2pro", "tiangong2dex"])
    def publish_endposetarget_L(
        self,
        target_pose: Pose,
        from_frame: str = "base_footprint",
        to_frame: str = "left_tcp_link",
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        offset_z: float = 0.0
    ):
        """
        发布左臂到指定的笛卡尔空间位置到 /endposetarget_L 话题
        
        Args:
            target_pose: 目标位姿（geometry_msgs.msg.Pose）
            from_frame: 起始坐标系（默认："base_footprint"）
            to_frame: 目标坐标系（默认："left_tcp_link"）
            offset_x: x方向偏移量（默认：0.0）
            offset_y: y方向偏移量（默认：0.0）
            offset_z: z方向偏移量（默认：0.0）
        """
        arm_target_pose = ArmTargetPose()
        arm_target_pose.from_frame = from_frame
        arm_target_pose.to_frame = to_frame
        arm_target_pose.offset_x = offset_x
        arm_target_pose.offset_y = offset_y
        arm_target_pose.offset_z = offset_z
        arm_target_pose.target = target_pose
        
        topic_publisher(
            self.node,
            "/endposetarget_L",
            ArmTargetPose,
            arm_target_pose
        )

    @require_robot_type(["tianyi2", "tianyi_max", "tiangong2pro", "tiangong2dex"])
    def publish_endposetarget_R(
        self,
        target_pose: Pose,
        from_frame: str = "base_footprint",
        to_frame: str = "right_tcp_link",
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        offset_z: float = 0.0
    ):
        """
        发布右臂到指定的笛卡尔空间位置到 /endposetarget_R 话题

        Args:
            target_pose: 目标位姿（geometry_msgs.msg.Pose）
            from_frame: 起始坐标系（默认："base_footprint"）
            to_frame: 目标坐标系（默认："right_tcp_link"）
            offset_x: x方向偏移量（默认：0.0）
            offset_y: y方向偏移量（默认：0.0）
            offset_z: z方向偏移量（默认：0.0）
        """
        arm_target_pose = ArmTargetPose()
        arm_target_pose.from_frame = from_frame
        arm_target_pose.to_frame = to_frame
        arm_target_pose.offset_x = offset_x
        arm_target_pose.offset_y = offset_y
        arm_target_pose.offset_z = offset_z
        arm_target_pose.target = target_pose

        topic_publisher(
            self.node,
            "/endposetarget_R",
            ArmTargetPose,
            arm_target_pose
        )
    def publish_endposetarget_dual(
        self,
        target_pose_l: Pose,
        target_pose_r: Pose,
        from_frame_l: str = "base_footprint",
        to_frame_l: str = "left_tcp_link",
        from_frame_r: str = "base_footprint",
        to_frame_r: str = "right_tcp_link",
        offset_x_l: float = 0.0,
        offset_y_l: float = 0.0,
        offset_z_l: float = 0.0,
        offset_x_r: float = 0.0,
        offset_y_r: float = 0.0,
        offset_z_r: float = 0.0,
        is_dual_arm_syn: bool = False
    ):
        """
        发布双臂末端目标位姿到 /endposetarget_dual 话题

        Args:
            target_pose_l: 左臂目标位姿
            target_pose_r: 右臂目标位姿
            from_frame_l: 左臂参考系
            to_frame_l: 左臂目标系
            from_frame_r: 右臂参考系
            to_frame_r: 右臂目标系
            offset_*: TCP 偏移
            is_dual_arm_syn: 是否同步
        """

        msg = DualArmTargetPose()   # 你的 msg 名字按实际改

        # ===== header =====
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "base_footprint"

        # ===== 左臂 =====
        msg.target_l = target_pose_l
        msg.from_frame_l = from_frame_l
        msg.to_frame_l = to_frame_l
        msg.offset_x_l = offset_x_l
        msg.offset_y_l = offset_y_l
        msg.offset_z_l = offset_z_l

        # ===== 右臂 =====
        msg.target_r = target_pose_r
        msg.from_frame_r = from_frame_r
        msg.to_frame_r = to_frame_r
        msg.offset_x_r = offset_x_r
        msg.offset_y_r = offset_y_r
        msg.offset_z_r = offset_z_r

        # ===== 同步标志 =====
        msg.is_dual_arm_syn = is_dual_arm_syn

        # ===== 发布 =====
        topic_publisher(
            self.node,
            "/endposetarget_Dual",
            DualArmTargetPose,
            msg
        )
    def publish_endposetarget_body(self, target_pose: List[float]):
        """
        发布身体到指定的笛卡尔空间位置到 /endposetarget_body 话题
        消息类型为 std_msgs/msg/Float64MultiArray
        """
        if len(target_pose) != 4:
            self.node.get_logger().error("target_pose长度必须为4，body")
            return None
        msg = Float64MultiArray()
        msg.data = target_pose
        topic_publisher(
            self.node,
            "/endposetarget_Body",
            Float64MultiArray,
            msg
        )


class MoveitCall:
    def __init__(self, node: XARM_manager):
        self.node = node
        self.time_measurement = False
    
        self.joint_action_report_srv = self.node.create_service(
            CommonStreamRos2,
            '/joint_action_report',
            self.joint_action_report_callback
        )

        # INSERT_YOUR_CODE
        self.waypoints_action_report_srv = self.node.create_service(
            CommonStreamRos2,
            '/waypoints_action_report',
            self.waypoints_action_report_callback
        )
        self.node.get_logger().info('/joint_action_report 服务已启动')
        self.joint_space_running_flag = False
        self.report_result = None
        self.report_error_code = None

        self.waypoints_running_flag = False
        self.waypoints_report_result = None
        self.waypoints_report_error_code = None

    def joint_action_report_callback(self, request, response):
        """处理 /joint_action_report 服务请求"""
        # 尝试解析 JSON 数据
        json_obj = []
        try:
            if request.json_data:
                json_obj = json.loads(request.json_data)
                # self.node.get_logger().info(f'  解析的 JSON: {json_obj}')
                
        except json.JSONDecodeError as e:
            self.node.get_logger().warn(f'  JSON 解析失败: {e}')
        
        # INSERT_YOUR_CODE
        # 取其中的result和error code
        if isinstance(json_obj, dict):
            result = json_obj.get("result", None)
            error_code = json_obj.get("error_code", None)
            self.node.get_logger().info(f"joint_action_report: result={result}, error_code={error_code}")
            self.report_result = result
            self.report_error_code = error_code

        # 设置响应
        response.result = True
        self.dual_arm_random_run_flag = False
        response.info = 'joint_action_report 服务调用成功'
        
        # self.get_logger().info(f'  响应: result={response.result}, info="{response.info}"')
        
        # 标记需要关闭节点
        self.joint_space_running_flag = False
        self.node.get_logger().info('MoveitCall: joint space run completed')
        
        return response

    def waypoints_action_report_callback(self, request, response):
        """处理 /waypoints_action_report 服务请求"""
        # 尝试解析 JSON 数据
        json_obj = []
        try:
            if request.json_data:
                json_obj = json.loads(request.json_data)
                # self.node.get_logger().info(f'  解析的 JSON: {json_obj}')
                
        except json.JSONDecodeError as e:
            self.node.get_logger().warn(f'  JSON 解析失败: {e}')
        
        # INSERT_YOUR_CODE
        # 取其中的result和error code
        if isinstance(json_obj, dict):

            result = json_obj.get("result", None)
            error_code = json_obj.get("error_code", None)
            self.node.get_logger().info(f"waypoints_action_report: result={result}, error_code={error_code}")
            self.waypoints_report_result = result
            self.waypoints_report_error_code = error_code

        # 设置响应
        response.result = True
        response.info = 'waypoints_action_report 服务调用成功'
        
        # self.get_logger().info(f'  响应: result={response.result}, info="{response.info}"')
        
        # 标记需要关闭节点
        self.waypoints_running_flag = False
        self.node.get_logger().info('MoveitCall: waypoints run completed')
        
        return response

    def dual_arm_random_run(self):
        if not self.node.xarm_activate_controller(['moveit_left_arm_controller', 'moveit_right_arm_controller']):
            self.node.get_logger().error('激活moveit控制器失败')
            return None

        request = CommonStreamRos2.Request()
        request.json_data = '{"group": "dual_arm","random":true}'
        # INSERT_YOUR_CODE
        self.node.get_logger().info('MoveitCall: 开始双臂随机运动')
        service_caller(
            node=self.node,
            service_name='/joint_space_control',
            service_type=CommonStreamRos2,
            request=request,
        )
        self.joint_space_running_flag = True
        # INSERT_YOUR_CODE
        # 在这里spin等待
        while self.joint_space_running_flag and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.01)
        return self.report_result, self.report_error_code
    
    def dual_arm_random_run_async(self):
        """
        启动双臂随机运动，并返回一个 rclpy.Future，可被 rclpy.spin_until_future_complete 调用等待完成。
        
        返回:
            rclpy.task.Future 对象，可用于 rclpy.spin_until_future_complete()
        
        使用示例:
            future = moveit_call.dual_arm_random_run_async()
            rclpy.spin_until_future_complete(xarm_manager, future)
            if future.result() is not None:
                print('双臂随机运动完成')
        """

        if not self.node.xarm_activate_controller(['moveit_left_arm_controller', 'moveit_right_arm_controller']):
            self.node.get_logger().error('激活moveit控制器失败')
            return None

        request = CommonStreamRos2.Request()
        request.json_data = '{"group": "dual_arm","random":true}'
        self.node.get_logger().info('MoveitCall: 开始双臂随机运动')
        service_caller(
            node=self.node,
            service_name='/joint_space_control',
            service_type=CommonStreamRos2,
            request=request,
        )
        self.joint_space_running_flag = True

        # 使用 rclpy.Future 用于 rclpy.spin_until_future_complete
        done_future = RclpyFuture()
        monitor_timer = None

        def monitor_done():
            nonlocal monitor_timer
            if not self.joint_space_running_flag or not rclpy.ok():
                # 运动完成，设置结果并销毁定时器
                if not done_future.done():
                    done_future.set_result((self.report_result, self.report_error_code))
                if monitor_timer is not None:
                    self.node.destroy_timer(monitor_timer)
                    monitor_timer = None
            # 如果还在运行，定时器会继续触发

        monitor_timer = self.node.create_timer(0.01, monitor_done)
        
        return done_future

    def left_arm_joint_angles(self, angles: list, vel_scale: float = 0.1, acc_scale: float = 0.1):
        """
        控制左臂到指定的关节角度。

        Args:
            angles (list): 长度为7的列表，依次为:
                [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_pitch, elbow_yaw, wrist_pitch, wrist_roll]
            vel_scale (float): 速度缩放系数，默认0.1
            acc_scale (float): 加速度缩放系数，默认0.1

        Returns:
            None
        """
        if not isinstance(angles, list) or len(angles) != 7:
            self.node.get_logger().error("angles 必须是长度为7的列表")
            return None

        if not self.node.xarm_activate_controller(['moveit_left_arm_controller']):
            self.node.get_logger().error('激活moveit控制器失败')
            return None

        # 构造json_data
        json_data = {
            "group": "left_arm",
            "vel_scale": vel_scale,
            "acc_scale": acc_scale,
            "shoulder_pitch_l_joint": angles[0],
            "shoulder_roll_l_joint": angles[1],
            "shoulder_yaw_l_joint": angles[2],
            "elbow_pitch_l_joint": angles[3],
            "elbow_yaw_l_joint": angles[4],
            "wrist_pitch_l_joint": angles[5],
            "wrist_roll_l_joint": angles[6],
            # 右臂字段填0
            "shoulder_pitch_r_joint": 0.0,
            "shoulder_roll_r_joint": 0.0,
            "shoulder_yaw_r_joint": 0.0,
            "elbow_pitch_r_joint": 0.0,
            "elbow_yaw_r_joint": 0.0,
            "wrist_pitch_r_joint": 0.0,
            "wrist_roll_r_joint": 0.0,
        }

        request = CommonStreamRos2.Request()
        request.msg_id = ''
        import json
        request.json_data = json.dumps(json_data)
        request.data = []

        self.joint_space_running_flag = True
        self.node.get_logger().info(f'MoveitCall json: {request.json_data}')

        service_caller(
                    node=self.node,
                    service_name='/joint_space_control',
                    service_type=CommonStreamRos2,
                    request=request,
                )
        self.joint_space_running_flag = True
        while self.joint_space_running_flag and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.01)
        
        return self.report_result, self.report_error_code

    def left_arm_joint_angles_async(self, angles: list, vel_scale: float = 0.1, acc_scale: float = 0.1):
        """
        控制左臂到指定的关节角度。

        Args:
            angles (list): 长度为7的列表，依次为:
                [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_pitch, elbow_yaw, wrist_pitch, wrist_roll]
            vel_scale (float): 速度缩放系数，默认0.1
            acc_scale (float): 加速度缩放系数，默认0.1

        Returns:
            None
        """
        if not isinstance(angles, list) or len(angles) != 7:
            self.node.get_logger().error("angles 必须是长度为7的列表")
            return None

        if not self.node.xarm_activate_controller(['moveit_left_arm_controller']):
            self.node.get_logger().error('激活moveit控制器失败')
            return None

        # 构造json_data
        json_data = {
            "group": "left_arm",
            "vel_scale": vel_scale,
            "acc_scale": acc_scale,
            "shoulder_pitch_l_joint": angles[0],
            "shoulder_roll_l_joint": angles[1],
            "shoulder_yaw_l_joint": angles[2],
            "elbow_pitch_l_joint": angles[3],
            "elbow_yaw_l_joint": angles[4],
            "wrist_pitch_l_joint": angles[5],
            "wrist_roll_l_joint": angles[6],
            # 右臂字段填0
            "shoulder_pitch_r_joint": 0.0,
            "shoulder_roll_r_joint": 0.0,
            "shoulder_yaw_r_joint": 0.0,
            "elbow_pitch_r_joint": 0.0,
            "elbow_yaw_r_joint": 0.0,
            "wrist_pitch_r_joint": 0.0,
            "wrist_roll_r_joint": 0.0,
        }

        request = CommonStreamRos2.Request()
        request.msg_id = ''
        import json
        request.json_data = json.dumps(json_data)
        request.data = []

        self.joint_space_running_flag = True
        self.node.get_logger().info(f'MoveitCall json: {request.json_data}')

        service_caller(
                    node=self.node,
                    service_name='/joint_space_control',
                    service_type=CommonStreamRos2,
                    request=request,
                )
        self.joint_space_running_flag = True
    
        # 使用 rclpy.Future 用于 rclpy.spin_until_future_complete
        done_future = RclpyFuture()
        monitor_timer = None

        def monitor_done():
            nonlocal monitor_timer
            if not self.joint_space_running_flag or not rclpy.ok():
                # 运动完成，设置结果并销毁定时器
                if not done_future.done():
                    done_future.set_result((self.report_result, self.report_error_code))
                if monitor_timer is not None:
                    self.node.destroy_timer(monitor_timer)
                    monitor_timer = None
            # 如果还在运行，定时器会继续触发

        monitor_timer = self.node.create_timer(0.01, monitor_done)
        
        return done_future

    def right_arm_joint_angles(self, angles: list, vel_scale: float = 0.1, acc_scale: float = 0.1):
        """
        控制右臂到指定的关节角度。

        Args:
            angles (list): 长度为7的列表，依次为:
                [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_pitch, elbow_yaw, wrist_pitch, wrist_roll]
            vel_scale (float): 速度缩放系数，默认0.1
            acc_scale (float): 加速度缩放系数，默认0.1

        Returns:
            None
        """
        if not isinstance(angles, list) or len(angles) != 7:
            self.node.get_logger().error("angles 必须是长度为7的列表")
            return None

        if not self.node.xarm_activate_controller(['moveit_right_arm_controller']):
            self.node.get_logger().error('激活moveit控制器失败')
            return None

        # 构造json_data
        json_data = {
            "group": "right_arm",
            "vel_scale": vel_scale,
            "acc_scale": acc_scale,
            # 左臂字段填0
            "shoulder_pitch_l_joint": 0.0,
            "shoulder_roll_l_joint": 0.0,
            "shoulder_yaw_l_joint": 0.0,
            "elbow_pitch_l_joint": 0.0,
            "elbow_yaw_l_joint": 0.0,
            "wrist_pitch_l_joint": 0.0,
            "wrist_roll_l_joint": 0.0,
            # 右臂字段
            "shoulder_pitch_r_joint": angles[0],
            "shoulder_roll_r_joint": angles[1],
            "shoulder_yaw_r_joint": angles[2],
            "elbow_pitch_r_joint": angles[3],
            "elbow_yaw_r_joint": angles[4],
            "wrist_pitch_r_joint": angles[5],
            "wrist_roll_r_joint": angles[6],
        }

        request = CommonStreamRos2.Request()
        request.msg_id = ''
        import json
        request.json_data = json.dumps(json_data)
        request.data = []

        self.joint_space_running_flag = True
        self.node.get_logger().info(f'MoveitCall json: {request.json_data}')

        service_caller(
                    node=self.node,
                    service_name='/joint_space_control',
                    service_type=CommonStreamRos2,
                    request=request,
                )
        self.joint_space_running_flag = True
        while self.joint_space_running_flag and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.01)
        return self.report_result, self.report_error_code

    def right_arm_joint_angles_async(self, angles: list, vel_scale: float = 0.1, acc_scale: float = 0.1):
        """
        控制右臂到指定的关节角度（异步版本）。

        Args:
            angles (list): 长度为7的列表，依次为:
                [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_pitch, elbow_yaw, wrist_pitch, wrist_roll]
            vel_scale (float): 速度缩放系数，默认0.1
            acc_scale (float): 加速度缩放系数，默认0.1

        Returns:
            rclpy.task.Future 对象，可用于 rclpy.spin_until_future_complete()
        """
        if not isinstance(angles, list) or len(angles) != 7:
            self.node.get_logger().error("angles 必须是长度为7的列表")
            return None

        if not self.node.xarm_activate_controller(['moveit_right_arm_controller']):
            self.node.get_logger().error('激活moveit控制器失败')
            return None

        # 构造json_data
        json_data = {
            "group": "right_arm",
            "vel_scale": vel_scale,
            "acc_scale": acc_scale,
            # 左臂字段填0
            "shoulder_pitch_l_joint": 0.0,
            "shoulder_roll_l_joint": 0.0,
            "shoulder_yaw_l_joint": 0.0,
            "elbow_pitch_l_joint": 0.0,
            "elbow_yaw_l_joint": 0.0,
            "wrist_pitch_l_joint": 0.0,
            "wrist_roll_l_joint": 0.0,
            # 右臂字段
            "shoulder_pitch_r_joint": angles[0],
            "shoulder_roll_r_joint": angles[1],
            "shoulder_yaw_r_joint": angles[2],
            "elbow_pitch_r_joint": angles[3],
            "elbow_yaw_r_joint": angles[4],
            "wrist_pitch_r_joint": angles[5],
            "wrist_roll_r_joint": angles[6],
        }

        request = CommonStreamRos2.Request()
        request.msg_id = ''
        import json
        request.json_data = json.dumps(json_data)
        request.data = []

        self.joint_space_running_flag = True
        self.node.get_logger().info(f'MoveitCall json: {request.json_data}')

        service_caller(
                    node=self.node,
                    service_name='/joint_space_control',
                    service_type=CommonStreamRos2,
                    request=request,
                )
        self.joint_space_running_flag = True
    
        # 使用 rclpy.Future 用于 rclpy.spin_until_future_complete
        done_future = RclpyFuture()
        monitor_timer = None

        def monitor_done():
            nonlocal monitor_timer
            if not self.joint_space_running_flag or not rclpy.ok():
                # 运动完成，设置结果并销毁定时器
                if not done_future.done():
                    done_future.set_result((self.report_result, self.report_error_code))
                if monitor_timer is not None:
                    self.node.destroy_timer(monitor_timer)
                    monitor_timer = None
            # 如果还在运行，定时器会继续触发

        monitor_timer = self.node.create_timer(0.01, monitor_done)
        
        return done_future

    def dual_arm_joint_angles(self, left_angles: list, right_angles: list, vel_scale: float = 0.1, acc_scale: float = 0.1):
        """
        控制左右臂到指定的关节角度。

        Args:
            left_angles (list): 长度为7的列表，依次为:
                [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_pitch, elbow_yaw, wrist_pitch, wrist_roll]
            right_angles (list): 长度为7的列表，依次为:
                [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_pitch, elbow_yaw, wrist_pitch, wrist_roll]
            vel_scale (float): 速度缩放系数，默认0.1
            acc_scale (float): 加速度缩放系数，默认0.1

        Returns:
            None
        """
        if not isinstance(left_angles, list) or len(left_angles) != 7:
            self.node.get_logger().error("left_angles 必须是长度为7的列表")
            return None

        if not isinstance(right_angles, list) or len(right_angles) != 7:
            self.node.get_logger().error("right_angles 必须是长度为7的列表")
            return None

        if not self.node.xarm_activate_controller(['moveit_left_arm_controller', 'moveit_right_arm_controller']):
            self.node.get_logger().error('激活moveit控制器失败')
            return None

        # 构造json_data
        json_data = {
            "group": "dual_arm",
            "vel_scale": vel_scale,
            "acc_scale": acc_scale,
            "shoulder_pitch_l_joint": left_angles[0],
            "shoulder_roll_l_joint": left_angles[1],
            "shoulder_yaw_l_joint": left_angles[2],
            "elbow_pitch_l_joint": left_angles[3],
            "elbow_yaw_l_joint": left_angles[4],
            "wrist_pitch_l_joint": left_angles[5],
            "wrist_roll_l_joint": left_angles[6],
            "shoulder_pitch_r_joint": right_angles[0],
            "shoulder_roll_r_joint": right_angles[1],
            "shoulder_yaw_r_joint": right_angles[2],
            "elbow_pitch_r_joint": right_angles[3],
            "elbow_yaw_r_joint": right_angles[4],
            "wrist_pitch_r_joint": right_angles[5],
            "wrist_roll_r_joint": right_angles[6],
        }

        request = CommonStreamRos2.Request()
        request.msg_id = ''
        import json
        request.json_data = json.dumps(json_data)
        request.data = []

        self.joint_space_running_flag = True
        self.node.get_logger().info(f'MoveitCall json: {request.json_data}')

        service_caller(
                    node=self.node,
                    service_name='/joint_space_control',
                    service_type=CommonStreamRos2,
                    request=request,
                )
        self.joint_space_running_flag = True
        while self.joint_space_running_flag and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.01)
        
        return self.report_result, self.report_error_code

    def dual_arm_joint_angles_async(self, left_angles: list, right_angles: list, vel_scale: float = 0.1, acc_scale: float = 0.1):
        """
        控制左右臂到指定的关节角度。

        Args:
            left_angles (list): 长度为7的列表，依次为:
                [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_pitch, elbow_yaw, wrist_pitch, wrist_roll]
            right_angles (list): 长度为7的列表，依次为:
                [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_pitch, elbow_yaw, wrist_pitch, wrist_roll]
            vel_scale (float): 速度缩放系数，默认0.1
            acc_scale (float): 加速度缩放系数，默认0.1

        Returns:
            None
        """
        if not isinstance(left_angles, list) or len(left_angles) != 7:
            self.node.get_logger().error("left_angles 必须是长度为7的列表")
            return None

        if not isinstance(right_angles, list) or len(right_angles) != 7:
            self.node.get_logger().error("right_angles 必须是长度为7的列表")
            return None

        if not self.node.xarm_activate_controller(['moveit_left_arm_controller', 'moveit_right_arm_controller']):
            self.node.get_logger().error('激活moveit控制器失败')
            return None

        # 构造json_data
        json_data = {
            "group": "dual_arm",
            "vel_scale": vel_scale,
            "acc_scale": acc_scale,
            "shoulder_pitch_l_joint": left_angles[0],
            "shoulder_roll_l_joint": left_angles[1],
            "shoulder_yaw_l_joint": left_angles[2],
            "elbow_pitch_l_joint": left_angles[3],
            "elbow_yaw_l_joint": left_angles[4],
            "wrist_pitch_l_joint": left_angles[5],
            "wrist_roll_l_joint": left_angles[6],
            "shoulder_pitch_r_joint": right_angles[0],
            "shoulder_roll_r_joint": right_angles[1],
            "shoulder_yaw_r_joint": right_angles[2],
            "elbow_pitch_r_joint": right_angles[3],
            "elbow_yaw_r_joint": right_angles[4],
            "wrist_pitch_r_joint": right_angles[5],
            "wrist_roll_r_joint": right_angles[6],
        }

        request = CommonStreamRos2.Request()
        request.msg_id = ''
        import json
        request.json_data = json.dumps(json_data)
        request.data = []

        self.joint_space_running_flag = True
        self.node.get_logger().info(f'MoveitCall json: {request.json_data}')

        service_caller(
                    node=self.node,
                    service_name='/joint_space_control',
                    service_type=CommonStreamRos2,
                    request=request,
                )
        self.joint_space_running_flag = True
        # 使用 rclpy.Future 用于 rclpy.spin_until_future_complete
        done_future = RclpyFuture()
        monitor_timer = None

        def monitor_done():
            nonlocal monitor_timer
            if not self.joint_space_running_flag or not rclpy.ok():
                # 运动完成，设置结果并销毁定时器
                if not done_future.done():
                    done_future.set_result((self.report_result, self.report_error_code))
                if monitor_timer is not None:
                    self.node.destroy_timer(monitor_timer)
                    monitor_timer = None
            # 如果还在运行，定时器会继续触发

        monitor_timer = self.node.create_timer(0.01, monitor_done)
        
        return done_future


    def build_left_arm_waypoints_json(self, left_waypoints, vel_scale=0.1, acc_scale=0.1, left_motion_id=0, frame="waist_yaw_link", mode="plan_and_execute", use_cartesian_path=False):
        return {
            "group": "left_arm",
            "frame": frame,
            "mode": mode,
            "use_cartesian_path": use_cartesian_path,
            "left_motion_id": left_motion_id,
            "vel_scale": vel_scale,
            "acc_scale": acc_scale,
            "left_waypoints": left_waypoints,
        }
    def build_right_arm_waypoints_json(self, right_waypoints, vel_scale=0.1, acc_scale=0.1, right_motion_id=0, frame="waist_yaw_link", mode="plan_and_execute", use_cartesian_path=False):
        return {
            "group": "right_arm",
            "frame": frame,
            "mode": mode,
            "use_cartesian_path": use_cartesian_path,
            "right_motion_id": right_motion_id,
            "vel_scale": vel_scale,
            "acc_scale": acc_scale,
            "right_waypoints": right_waypoints,
        }
    
    def build_dual_arm_waypoints_json(self, left_waypoints, right_waypoints, vel_scale=0.1, acc_scale=0.1, left_motion_id=0, right_motion_id=0, frame="waist_yaw_link", mode="plan_and_execute", use_cartesian_path=False):
        return {
            "group": "dual_arm",
            "frame": frame,
            "mode": mode,
            "use_cartesian_path": use_cartesian_path,
            "left_motion_id": left_motion_id,
            "right_motion_id": right_motion_id,
            "vel_scale": vel_scale,
            "acc_scale": acc_scale,
            "left_waypoints": left_waypoints,
            "right_waypoints": right_waypoints,
        }

    def arm_waypoints(self, json_data):

        # 判断json_data中的"group"字段
        group = None
        if isinstance(json_data, dict):
            group = json_data.get('group', None)
        elif isinstance(json_data, str):
            try:
                json_obj = json.loads(json_data)
                group = json_obj.get('group', None)
            except Exception:
                group = None

        if group == "left_arm":
            if not self.node.xarm_activate_controller(['moveit_left_arm_controller']):
                self.node.get_logger().error('激活moveit控制器失败')
                return None
        elif group == "right_arm":
            if not self.node.xarm_activate_controller(['moveit_right_arm_controller']):
                self.node.get_logger().error('激活moveit控制器失败')
                return None
        elif group == "dual_arm":
            if not self.node.xarm_activate_controller(['moveit_left_arm_controller', 'moveit_right_arm_controller']):
                self.node.get_logger().error('激活moveit控制器失败')
                return None
        else:
            self.node.get_logger().error('group字段错误')
            return None

        request = CommonStreamRos2.Request()
        request.msg_id = ''
        request.json_data = json.dumps(json_data)
        request.data = []

        self.node.get_logger().info(f"arm_waypoints json: {request.json_data}")

        service_caller(
            node=self.node,
            service_name='/dual_arm_waypoints',
            service_type=CommonStreamRos2,
            request=request,
        )
        self.waypoints_running_flag = True

        # 在这里spin等待
        while self.waypoints_running_flag and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.01)
        return self.waypoints_report_result, self.waypoints_report_error_code

    def arm_waypoints_async(self, json_data):

        # 判断json_data中的"group"字段
        group = None
        if isinstance(json_data, dict):
            group = json_data.get('group', None)
        elif isinstance(json_data, str):
            try:
                json_obj = json.loads(json_data)
                group = json_obj.get('group', None)
            except Exception:
                group = None

        if group == "left_arm":
            if not self.node.xarm_activate_controller(['moveit_left_arm_controller']):
                self.node.get_logger().error('激活moveit控制器失败')
                return None
        elif group == "right_arm":
            if not self.node.xarm_activate_controller(['moveit_right_arm_controller']):
                self.node.get_logger().error('激活moveit控制器失败')
                return None
        elif group == "dual_arm":
            if not self.node.xarm_activate_controller(['moveit_left_arm_controller', 'moveit_right_arm_controller']):
                self.node.get_logger().error('激活moveit控制器失败')
                return None
        else:
            self.node.get_logger().error('group字段错误')
            return None

        request = CommonStreamRos2.Request()
        request.msg_id = ''
        request.json_data = json.dumps(json_data)
        request.data = []

        self.node.get_logger().info(f"arm_waypoints json: {request.json_data}")

        service_caller(
            node=self.node,
            service_name='/dual_arm_waypoints',
            service_type=CommonStreamRos2,
            request=request,
        )
        self.waypoints_running_flag = True
        # 使用 rclpy.Future 用于 rclpy.spin_until_future_complete
        done_future = RclpyFuture()
        monitor_timer = None

        def monitor_done():
            nonlocal monitor_timer
            if not self.waypoints_running_flag or not rclpy.ok():
                # 运动完成，设置结果并销毁定时器
                if not done_future.done():
                    done_future.set_result((self.waypoints_report_result, self.waypoints_report_error_code))
                if monitor_timer is not None:
                    self.node.destroy_timer(monitor_timer)
                    monitor_timer = None
            # 如果还在运行，定时器会继续触发

        monitor_timer = self.node.create_timer(0.01, monitor_done)
        
        return done_future


class ParamConfiger:
    def __init__(self, node: XARM_manager):
        self.node = node
        self.time_measurement = False

    def get_node_parameter(self, node_name: str, parameter_name: str):
        return get_node_parameter(self.node, node_name, parameter_name)

    def set_node_parameter(self, node_name: str, parameter_name: str, value: any):
        return set_node_parameter(self.node, node_name, parameter_name, value)

    def list_node_parameters(self, node_name: str):
        return list_node_parameters(self.node, node_name)