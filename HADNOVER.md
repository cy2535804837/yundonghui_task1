# HADNOVER

## 我们在做什么

我们正在给现有抓取/放置流程增加一个“世界坐标放置”桥接节点。现有节点的边界是：

- `compliant_grasp_execute`：识别并抓取物体，抓完保持夹住，写 `/tmp/grasp_handoff.json`
- `grasp_pose_place_execute`：读取 handoff，接收 `waist_yaw_link` 下的 `place_x/place_y/place_z`，执行放置

新需求是：外部系统给一个目标放置点的世界/SLAM map 坐标 `x,y,z`，机器人读取自身 SLAM 位姿 `x,y,yaw`，再读取 ROS TF 里的 `base_footprint -> waist_yaw_link`，把目标点转换成机器人腰部坐标 `waist_yaw_link` 下的放置坐标，然后调用现有放置节点。

## 已经完成了什么

新增包：

```text
world_place_execute/
  __init__.py
  robot_pose.py
  frame_transforms.py
  main.py
  config.yaml
  README.md
```

入口：

```bash
python3 -m world_place_execute.main
```

目标点 HTTP 接口约定：

```text
GET <target_url>
```

返回 JSON：

```json
{
  "x": 1.23,
  "y": 0.45,
  "z": 0.75
}
```

其中 `x/y/z` 单位是米。目标点接口本身可以只返回这三个字段；坐标系由 `world_place_execute/config.yaml` 里的 `target_frame` 表示，默认是 `map`。需要和上游接口确认：它返回的 `x/y/z` 是否和机器人 SLAM pose 坐标系一致。

代码仍兼容可选字段：如果目标点接口额外返回 `frame_id` 或 `frame`，会优先使用接口里的坐标系；如果额外返回 `yaw`，目前只记录不参与放置姿态。

机器人 SLAM pose HTTP 接口默认：

```text
http://192.168.41.6:1448/api/core/slam/v1/localization/pose
```

期望返回：

```json
{"x": 1.0, "y": 2.0, "yaw": 0.5}
```

最终转换链路：

```text
world/map --T_world_base--> base_footprint --T_base_waist--> waist_yaw_link
```

其中：

```text
T_world_base 来自 SLAM HTTP pose: {"x", "y", "yaw"}
T_waist_base 来自 ROS TF: lookup_tf_once("waist_yaw_link", "base_footprint")
T_obj_waist = T_waist_base @ inv(T_world_base) @ T_obj_world
```

离线调试时可以用 `--no-use-live-waist-tf`，回到旧近似：`waist_yaw_link` 和移动底盘中心在 X/Y/Yaw 上重合。

转换后调用现有放置节点：

```bash
python3 -m grasp_pose_place_execute.main \
  --handoff-in /tmp/grasp_handoff.json \
  --arm auto \
  --place-x <place_x> \
  --place-y <place_y> \
  --place-z <place_z>
```

配置文件：

```text
world_place_execute/config.yaml
```

关键配置：

```yaml
target_url: ''
robot_pose_url: http://192.168.41.6:1448/api/core/slam/v1/localization/pose
robot_pose_frame: map
target_frame: map
use_live_waist_tf: true
waist_frame: waist_yaw_link
base_frame: base_footprint
tf_timeout_sec: 3.0
waist_z_in_map: 0.0
place_z_offset: 0.0
handoff_in: /tmp/grasp_handoff.json
arm: auto
json_out: /tmp/world_place_exec.json
```

离线/调试运行：

```bash
python3 -m world_place_execute.main \
  --target-x 1.20 --target-y 0.30 --target-z 0.75 \
  --robot-x 1.00 --robot-y 0.00 --robot-yaw 0.00 \
  --no-use-live-waist-tf \
  --no-execute-place
```

真实运行：

```bash
python3 -m world_place_execute.main \
  --target-url http://<host>/api/place_target
```

输出：

```text
/tmp/world_place_exec.json
/tmp/world_place_exec.place.json
```

`world_place_exec.json` 里会记录：

- 原始 target
- 原始 robot_pose
- 转换后的 `waist_target`
- 调用下游 place 节点的命令参数
- 下游 place 结果

## 下一步计划

1. 确认外部目标放置点接口真实 URL 和返回字段。
2. 确认目标点 `x/y/z` 的坐标系，并把 `target_frame` 配成和 `localization/pose` 返回的机器人 `x/y/yaw` 同一个 `map/world` 坐标系。
3. 确认目标 `z` 的含义：
   - 如果是地图绝对高度，需要设置 `waist_z_in_map`
   - 如果已经是 waist-relative Z，可以保持 `waist_z_in_map: 0.0`
4. 在机器人旁边用 `--no-execute-place` 验证转换结果是否合理：
   - `place_x` 应该是目标在机器人前方的距离
   - `place_y` 左正右负
   - `place_z` 是放置节点可接受的高度
5. 真实联调：
   - 先运行 `compliant_grasp_execute` 抓取并保持物体
   - 再运行 `world_place_execute`
   - 检查 `/tmp/world_place_exec.json` 和 `/tmp/world_place_exec.place.json`
6. 如果 live TF 模式失败，优先检查 TF 树是否存在：
   - `base_footprint`
   - `waist_yaw_link`
   - `lookup_tf_once(target_frame="waist_yaw_link", source_frame="base_footprint")`
7. 如果放置点仍有系统性偏差，再标定/确认 SLAM pose 的 base frame 是否真的是 `base_footprint` 中心；必要时增加额外静态偏移参数。
