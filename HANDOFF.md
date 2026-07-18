# HANDOFF

## 任务背景

我们在调试本机 `/home/cheng/cheng` 里的双臂抓取流程，重点节点是：

- `compliant_grasp_execute`
- 入口：`compliant_grasp_execute/main.py`
- 配置：`compliant_grasp_execute/config.yaml`

目标是让机器人从固定初始姿态出发，使用头部 RGB-D 相机识别目标，自动选择左/右手，完成 approach、柔顺下探、夹爪闭合、抬起、撤回、回 Home 的完整抓取流程。

远端板卡信息：

- 主机：`ubuntu@10.11.1.23`
- 密码：`123`
- 远端目标目录：`/home/ubuntu/cheng`
- 本机工作目录：`/home/cheng/cheng`
- 本机远端备份目录：`/home/cheng/文档/niu_7_13`
- 本机 cheng 备份：`/home/cheng/cheng_backup_7_13`

注意：除非用户明确要求，远端 `/home/ubuntu/niu` 不要改，它是源参考之一。

## 已完成的事情

1. 从远端 `/home/ubuntu/niu` 同步了一份到本机：

   ```text
   /home/cheng/文档/niu_7_13
   ```

2. 将 `~/文档/niu_7_13` 的最新版更新进本机：

   ```text
   /home/cheng/cheng
   ```

3. 将更新后的本机 `/home/cheng/cheng` 同步到远端：

   ```text
   ubuntu@10.11.1.23:/home/ubuntu/cheng
   ```

4. 同步时排除了这些目录：

   ```text
   .git/
   .agents/
   .codex/
   ```

5. 最新一次远端同步后做过 checksum dry-run 校验，结果是本机和远端 `/home/ubuntu/cheng` 一致。

6. 已把 `niu_7_13` 里的力传感器标定 JSON 同步到本机 `cheng`：

   ```text
   compliant_grasp_execute/ft_calibration/ft_calibration_left.json
   compliant_grasp_execute/ft_calibration/ft_calibration_right.json
   ```

7. 已新增“识别/抓取前准备姿态”功能：

   配置在 `compliant_grasp_execute/config.yaml`：

   ```yaml
   pre_cycle_move: false

   pre_cycle_left_joints:
   - 0.844
   - 0.026
   - -0.006
   - -2.216
   - -1.529
   - 0.300
   - 0.683

   pre_cycle_right_joints:
   - 0.844
   - -0.026
   - 0.006
   - -2.216
   - 1.529
   - 0.300
   - -0.683
   ```

   代码在 `compliant_grasp_execute/main.py`：

   - 参数：`--pre-cycle-move`
   - 参数：`--pre-cycle-left-joints`
   - 参数：`--pre-cycle-right-joints`
   - 函数：`_move_pre_cycle_joints(...)`
   - 调用位置：`_startup_home_and_activate(...)` 之后，`_detect_pose(...)` 之前。

   当前默认 `pre_cycle_move: false`，所以不会自动多走一步。用户需要时改成 `true`。

## 当前抓取流程

当前整体流程是：

```text
读取 config.yaml
-> 初始化 ROS / xarm / ActionCall / MoveIt / TopicPublisher
-> start_home: 双臂到 home_left_joints / home_right_joints
-> 可选 pre_cycle_move: 双臂到 pre_cycle_left_joints / pre_cycle_right_joints
-> 头部 RGB-D 相机识别
-> 自动选择左手或右手
-> 生成抓取姿态
-> 低肘或高肘分支
-> QP approach
-> 抓取前打开夹爪
-> compliant insert 柔顺下探
-> 闭合夹爪
-> 停止柔顺保持
-> lift 抬起
-> retract 撤回
-> return_home 回 Home
-> 默认不释放物体
-> 写 result json / handoff json
```

关键配置：

```yaml
start_home: true
pre_cycle_move: false
arm: auto
motion_strategy: qp_all
compliant_grasp: true
open_gripper_before_grasp: true
close_gripper: true
return_home: true
release_on_finish: false
```

## 重要配置含义

### Home 和准备姿态

`start_home: true` 时会先让左右手到：

```yaml
home_left_joints
home_right_joints
```

这两个也会用于最后 `return_home: true` 的回家动作。

如果只想在识别前额外摆一个姿态，不要改 `home_*_joints`，应该使用：

```yaml
pre_cycle_move: true
pre_cycle_left_joints
pre_cycle_right_joints
```

### 高肘姿态

高肘最终种子姿态是：

```yaml
elbow_high_ready_left_joints
elbow_high_ready_right_joints
```

进入高肘时流程：

```text
home -> elbow_high_stage -> transition -> elbow_high_ready
```

日志中看到：

```text
ELBOW-HIGH: ... -> ready
```

说明已经到达高肘最终姿态。

### 低姿态抓取角度

低肘/普通抓取不是固定关节角，是实时算 TCP 姿态。

核心参数：

```yaml
grasp_tilt_y_deg: 45.0
continuous_grasp_orientation: true
continuous_grasp_max_yaw_deg: 15.0
diagonal_schedule: true
diagonal_schedule_tilt_scale_end: 0.5
diagonal_schedule_yaw_clamp_end_deg: 45.0
reach_tilt_reduce: true
reach_tilt_scale_end: 0.4
```

如果低姿态太“扎下去”，优先把 `grasp_tilt_y_deg` 从 `45.0` 调到 `30.0`。

### 速度相关

当前慢速调试参数主要是：

```yaml
vel_scale: 0.2
acc_scale: 0.2
qp_stream_rate_hz: 30.0
qp_transit_p_step: 0.001
qp_transit_r_step: 0.008
qp_otg_p_step: 0.0005
qp_otg_r_step: 0.0003
compliant_insert_speed_mps: 0.01
compliant_max_vel: 0.05
gripper_speed_pct: 30.0
```

特别注意：用户曾说把 `qp_speed_scale` 改成了 `0.2`，但当前文件里实际仍是：

```yaml
qp_speed_scale: 0.9
```

如果觉得速度没有明显变慢，要先确认这个值。

### 夹爪力度

夹爪参数：

```yaml
gripper_close_pct: 100.0
gripper_open_pct: 0.0
gripper_speed_pct: 30.0
gripper_force_pct: 50.0
gripper_close_delay_sec: 0.15
```

夹取力度主要调：

```yaml
gripper_force_pct
```

夹取速度主要调：

```yaml
gripper_speed_pct
```

夹到多深主要调：

```yaml
gripper_close_pct
```

## 检测逻辑

检测入口：

```text
compliant_grasp_execute/main.py::_detect_pose(...)
```

头部相机 topic：

```yaml
rgb_topic: /ob_camera_head/color/image_raw
depth_topic: /ob_camera_head/depth/image_raw
```

检测流程：

```text
读取 RGB + depth
-> 用 prompt 做 RGB 分割
-> 得到目标 2D mask
-> 只取 mask 内的 depth 像素
-> 反投影成 3D 点云
-> 过滤无效深度、图像边缘、离群点
-> 计算目标中心、3D bbox、主轴方向
-> TF 转到 waist_yaw_link
```

当前没有显式“排除机械臂点云”的逻辑。如果分割 mask 把机械臂也包进去了，头部深度相机可能把机械臂深度误当成目标深度。

可调：

```yaml
segment_confidence: 0.3
prompt:
- remote-control
```

建议调试时打开：

```yaml
save_dir: /tmp/grasp_debug
save_prefix: grasp_exec
```

这样会保存识别 overlay 和 result json，便于看目标框是否框到了机械臂。

## 可视化现状

当前没有专门的 RViz Marker 可视化功能。

能做的：

1. 在 RViz 里看机器人本体：

   ```text
   RobotModel
   TF
   ```

   前提是系统发布了 `/joint_states`、`/tf`、`/robot_description`。

2. 看 QP 目标 topic：

   ```text
   /endposetarget_L
   /endposetarget_R
   ```

   但它们是自定义消息，RViz 默认不会画。

3. 看检测 overlay：

   开 `save_dir` 后查看 `/tmp/grasp_debug/*_overlay.png`。

建议下一步新增：

```text
/compliant_grasp_execute/markers
```

发布 `visualization_msgs/MarkerArray`，显示：

- 目标中心
- 3D bbox
- approach pose 坐标轴
- grasp pose 坐标轴
- lift pose 坐标轴
- approach -> grasp -> lift 路径线
- 当前选择的 left/right arm 文本

## 当前卡在哪里

主要卡点不是代码跑不起来，而是调试可观测性不够：

1. RViz 只能看机器人本体，不能直接看抓取目标点和路径。
2. 检测结果是否把机械臂深度混入目标，需要靠 overlay/json 和日志判断。
3. 当前 `qp_speed_scale` 仍是 `0.9`，如果用户以为已经是 `0.2`，会导致“调慢不明显”的误判。
4. 本机普通 Python 环境缺 `numpy`，直接运行：

   ```bash
   python3 -m compliant_grasp_execute.main --help
   ```

   会报 `ModuleNotFoundError: No module named 'numpy'`。这不代表机器人运行环境一定坏，可能实际运行要 source ROS/项目环境。

## 下一步计划

建议优先顺序：

1. 如果用户要可视化，给 `compliant_grasp_execute` 增加 RViz MarkerArray 发布：

   - 新增配置：

     ```yaml
     publish_debug_markers: true
     debug_marker_topic: /compliant_grasp_execute/markers
     debug_marker_lifetime_sec: 0.0
     ```

   - 在生成 `det`、`approach_pose7`、`grasp_pose7`、`lift_pose7` 后发布 marker。

2. 打开 `save_dir`，做几次识别，确认 RGB mask 是否会框到机械臂。

3. 如果要真正慢速调试，确认是否要把：

   ```yaml
   qp_speed_scale: 0.2
   ```

   同步到配置。

4. 如果启用识别前准备姿态：

   ```yaml
   pre_cycle_move: true
   ```

   先只用安全、离桌面远的关节角，低速观察。

5. 每次本机确认后，再同步到远端 `/home/ubuntu/cheng`。

## 远端同步命令

之前用过 askpass 脚本：

```text
/tmp/ssh-askpass-niu.sh
```

内容是输出密码 `123`。

推送本机到远端的命令模板：

```bash
env SSH_ASKPASS=/tmp/ssh-askpass-niu.sh \
    SSH_ASKPASS_REQUIRE=force \
    DISPLAY=codex \
    rsync -a --delete \
    --exclude='.git/' \
    --exclude='.agents/' \
    --exclude='.codex/' \
    -e 'ssh -T -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/tmp/known_hosts_niu -o NumberOfPasswordPrompts=1' \
    /home/cheng/cheng/ \
    ubuntu@10.11.1.23:/home/ubuntu/cheng/
```

校验命令模板：

```bash
env SSH_ASKPASS=/tmp/ssh-askpass-niu.sh \
    SSH_ASKPASS_REQUIRE=force \
    DISPLAY=codex \
    rsync -a --checksum --dry-run --itemize-changes \
    --exclude='.git/' \
    --exclude='.agents/' \
    --exclude='.codex/' \
    -e 'ssh -T -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/tmp/known_hosts_niu -o NumberOfPasswordPrompts=1' \
    /home/cheng/cheng/ \
    ubuntu@10.11.1.23:/home/ubuntu/cheng/
```

注意：这里需要网络和 SSH，Codex 工具里通常要申请 escalated 权限。

## 千万不要再踩的坑

1. 不要把远端 `/home/ubuntu/niu` 当目标目录改。它是源参考，不是同步目标。

2. 不要把本机 `/home/cheng/cheng/.git` 同步到远端。之前 `.git` 同步会出问题，固定排除：

   ```text
   .git/
   .agents/
   .codex/
   ```

3. 不要看到 `simple_topdown_grasp.py` 在 IDE tab 里就默认它还在当前流程中。当前主流程是 `compliant_grasp_execute/main.py`。

4. 不要把 `home_left_joints` / `home_right_joints` 当成只影响启动。它们同时影响 `start_home` 和 `return_home`。

5. 如果只是想识别前摆姿态，不要改 `home_*_joints`，用：

   ```yaml
   pre_cycle_move
   pre_cycle_left_joints
   pre_cycle_right_joints
   ```

6. 不要以为 RViz 已经能显示抓取点。当前没有 MarkerArray，只能看 RobotModel/TF 和保存的 debug overlay。

7. 不要以为头部深度一定不会误判机械臂。当前没有机器人 arm mask 排除，mask 包进机械臂就可能污染点云。

8. 不要只改 `vel_scale` / `acc_scale` 就期待所有阶段变慢。`motion_strategy: qp_all` 时，很多阶段主要由 QP 参数控制。

9. 不要忘记 `qp_stream_rate_hz` 在代码里会被限制到 `1..50Hz`，设置 100 不会真的 100Hz。

10. 不要直接运行系统 Python 判断节点一定坏。本机 shell 缺 `numpy`，实际机器人环境可能需要 source ROS/conda/工作区环境。

11. 不要随便删除用户已有改动。当前 git worktree 很脏，里面有很多不是本次会话产生的改动和日志。

12. 不要用 destructive git 命令，比如 `git reset --hard` 或 `git checkout --`，除非用户明确要求。

## 本次会话新增/改动检查

已做检查：

```bash
python3 -m py_compile compliant_grasp_execute/main.py
```

通过。

配置 YAML 检查：

```bash
python3 -c "import yaml; yaml.safe_load(open('compliant_grasp_execute/config.yaml')); print('config yaml ok')"
```

通过。

直接运行模块 `--help` 失败，原因是当前 shell 缺 `numpy`，不是本次代码改动导致。
