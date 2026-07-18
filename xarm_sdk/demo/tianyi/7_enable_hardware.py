"""
硬件使能和模式配置示例

本示例演示如何控制 XARM 机器人的硬件使能状态和运行模式。

功能演示:
1. 硬件使能/去使能（双臂、头部、腿部、腰部）
2. 硬件运行模式配置（位置环、力位混合等）
3. 硬件调试信息获取
4. 在真实模式下进行硬件控制
5. 运动控制后读取关节状态

硬件使能说明:
- 硬件使能是控制机器人电机上电/断电的操作
- 只有在真实模式（run_type == "real"）下才能进行硬件操作
- 在仿真模式（run_type == "sim"）下，这些操作会被忽略

运行模式说明:
- mode 0: 力位混合（默认模式）
- mode 1: 电机速度环
- mode 2: 重力补偿
- mode 3: 位置环（常用模式，用于位置控制）
- mode 4: 左手柔顺右手位置
- mode 5: 左手位置右手柔顺


前置条件:
- 已安装并启动 XARM 环境
- 已启动 ROS2 和相关控制器

"""

import rclpy
from xarm_sdk import XARM_manager
from xarm_sdk import MoveitCall, ActionCall, TopicPublisher


def main():
    """
    主函数：演示硬件使能和模式配置
    """
    # ========== 初始化 ==========
    print("=" * 60)
    print("硬件使能和模式配置示例")
    print("=" * 60)
    
    # 初始化 ROS2
    rclpy.init()
    
    # 创建 XARM 管理节点
    xarm_manager = XARM_manager()
    
    # 创建控制接口（XARM 三件套）
    action_call = ActionCall(xarm_manager)
    topic_publisher = TopicPublisher(xarm_manager)
    moveit_call = MoveitCall(xarm_manager)
    
    print(f"\n当前运行模式: {xarm_manager.run_type}")
    print("提示: 硬件操作仅在真实模式（real）下有效")
    
    # ========== 第一部分：硬件使能和模式配置 ==========
    print("\n第一部分：硬件使能和模式配置")
    print("-" * 60)
    
    # 检查是否为真实模式
    if xarm_manager.run_type == "real":
        print("\n【步骤 1】使能各个硬件模块")
        print("说明: 使能硬件后，电机将上电，机器人可以运动")
        
        # 分别使能各个硬件模块
        print("  - 使能双臂...")
        xarm_manager.hardware_arm_enable(True)
        
        print("  - 使能头部...")
        xarm_manager.hardware_head_enable(True)
        
        print("  - 使能腿部...")
        xarm_manager.hardware_leg_enable(True)
        
        print("  - 使能腰部...")
        xarm_manager.hardware_waist_enable(True)
        
        # 或者使用统一函数使能所有硬件（注释掉的代码）
        # print("  - 使能所有硬件（统一函数）...")
        # xarm_manager.hardware_all_enable(True)
        
        print("✓ 所有硬件模块已使能")
        
        print("\n【步骤 2】配置双臂运行模式")
        print("说明: 不同的运行模式适用于不同的控制场景")
        print("\n可用模式:")
        print("  - mode 0: 力位混合（默认，安全模式）")
        print("  - mode 1: 电机速度环")
        print("  - mode 2: 重力补偿")
        print("  - mode 3: 位置环（推荐用于位置控制）")
        print("  - mode 4: 左手柔顺右手位置")
        print("  - mode 5: 左手位置右手柔顺")
        
        # 设置双臂模式为位置环（mode 3）
        print("\n设置双臂模式为 mode 3（位置环）...")
        xarm_manager.hardware_arm_mode(3)
        print("✓ 模式设置完成")
        
        print("\n【步骤 3】获取硬件调试信息")
        print("说明: 硬件调试信息包含硬件状态、错误码等")
        info = xarm_manager.hardware_debug()
        print(f"硬件调试信息:\n{info}")
    else:
        print("\n当前为仿真模式，跳过硬件使能操作")
        print("提示: 在真实模式下运行此脚本才能进行硬件操作")
    
    # ========== 第二部分：运动控制 ==========
    print("\n\n第二部分：运动控制")
    print("-" * 60)
    print("说明: 在硬件使能后，可以进行运动控制")
    
    # 控制左臂到初始工作位置
    print("\n控制左臂到初始工作位置...")
    left_target = [0.0, 1.18, 0.0, -1.3, 0.0, -0.13, 0.18]
    print(f"目标关节角度: {left_target}")
    action_call.jointspace_arm_L_controller(left_target)
    print("✓ 左臂运动完成")
    
    # 控制右臂到初始工作位置
    print("\n控制右臂到初始工作位置...")
    right_target = [0.0, -1.18, 0.0, -1.3, 0.0, -0.13, 0.18]
    print(f"目标关节角度: {right_target}")
    action_call.jointspace_arm_R_controller(right_target)
    print("✓ 右臂运动完成")
    
    # ========== 第三部分：读取关节状态 ==========
    print("\n\n第三部分：读取关节状态")
    print("-" * 60)
    print("说明: 运动完成后，读取实际关节角度验证位置")
    
    # 更新关节状态（阻塞式，等待新消息）
    print("\n更新关节状态...")
    xarm_manager.joint_state_update()
    print("✓ 关节状态已更新")
    
    # 读取左右臂关节角度
    left_arm_joint_angles = xarm_manager.xarm_left_arm_joint_angles()
    right_arm_joint_angles = xarm_manager.xarm_right_arm_joint_angles()
    
    print(f"\n左臂当前关节角度: {left_arm_joint_angles}")
    print(f"右臂当前关节角度: {right_arm_joint_angles}")
    
    # ========== 第四部分：去使能硬件（安全关闭）==========
    print("\n\n第四部分：去使能硬件（安全关闭）")
    print("-" * 60)
    print("说明: 使用完毕后，应去使能硬件以确保安全")
    
    if xarm_manager.run_type == "real":
        print("\n【步骤 1】去使能各个硬件模块")
        print("说明: 去使能硬件后，电机将断电，机器人进入安全状态")
        
        # 分别去使能各个硬件模块
        print("  - 去使能双臂...")
        xarm_manager.hardware_arm_enable(False)
        
        print("  - 去使能头部...")
        xarm_manager.hardware_head_enable(False)
        
        print("  - 去使能腿部...")
        xarm_manager.hardware_leg_enable(False)
        
        print("  - 去使能腰部...")
        xarm_manager.hardware_waist_enable(False)
        
        # 或者使用统一函数去使能所有硬件（注释掉的代码）
        # print("  - 去使能所有硬件（统一函数）...")
        # xarm_manager.hardware_all_enable(False)
        
        print("✓ 所有硬件模块已去使能")
        
        print("\n【步骤 2】恢复默认运行模式")
        print("说明: 恢复为默认的力位混合模式（mode 0）")
        xarm_manager.hardware_arm_mode(0)
        print("✓ 模式已恢复为默认值")
        
        print("\n【步骤 3】获取最终硬件调试信息")
        info = xarm_manager.hardware_debug()
        print(f"最终硬件调试信息:\n{info}")
        
        print("\n✓ 硬件已安全关闭")
    else:
        print("\n当前为仿真模式，跳过硬件去使能操作")
    
    # ========== 总结 ==========
    print("\n" + "=" * 60)
    print("示例运行完成！")
    print("=" * 60)
    print("\n参考文档: README.md")
    
    # 关闭 ROS2
    rclpy.shutdown()


if __name__ == "__main__":
    main()