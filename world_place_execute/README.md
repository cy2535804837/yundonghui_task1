# world_place_execute

这是一个“世界坐标放置”桥接节点：把外部系统给出的 SLAM map/world 坐标系下的目标放置点，转换成现有放置节点需要的 `waist_yaw_link` 坐标，然后调用 `grasp_pose_place_execute` 执行放置。

它不替代 `grasp_pose_place_execute`，只负责中间坐标转换和调用：

```text
目标 map/world x,y,z + 当前机器人 map/world x,y,yaw
-> 读取 live TF: base_footprint -> waist_yaw_link
-> 得到 waist_yaw_link 下的 place_x,place_y,place_z
-> python3 -m grasp_pose_place_execute.main --place-x ... --place-y ... --place-z ...
```

## 目标点 HTTP 接口

通过 `config.yaml` 配置 `target_url`，或者运行时传 `--target-url`。目标点接口需要支持：

```text
GET <target_url>
```

期望返回 JSON：

```json
{
  "x": 1.23,
  "y": 0.45,
  "z": 0.75，
}
```

字段含义：

- `x/y/z`：目标放置点坐标，单位米

目标点接口本身可以只返回 `x/y/z`。坐标系由 `config.yaml` 里的 `target_frame` 表示，默认是 `map`。你需要和上游接口确认：它返回的 `x/y/z` 是否和机器人 SLAM pose 接口处在同一个 map/world 坐标系。

代码仍然兼容可选字段：如果目标点接口额外返回 `frame_id` 或 `frame`，会优先使用接口里的坐标系；如果额外返回 `yaw`，目前只记录，不参与放置姿态控制。

## 机器人 Pose HTTP 接口

默认机器人定位接口：

```text
http://192.168.41.6:1448/api/core/slam/v1/localization/pose
```

期望返回 JSON：

```json
{"x": 1.0, "y": 2.0, "yaw": 0.5}
```

字段含义：

- `x/y`：机器人底盘在 map/world 坐标系下的位置，单位米
- `yaw`：机器人底盘朝向，单位弧度

这个接口返回的坐标系必须和目标点的 `target_frame` 是同一个 map/world 坐标系。

## 坐标转换

默认模式使用实时 ROS TF，转换链路是：

```text
T_obj_waist = T_waist_base @ inv(T_world_base) @ T_obj_world
```

其中：

- `T_world_base` 来自机器人 SLAM HTTP pose，也就是 `x/y/yaw`
- `T_waist_base` 来自 ROS TF：

```text
lookup_tf_once(target_frame="waist_yaw_link", source_frame="base_footprint")
```

离线调试时可以传 `--no-use-live-waist-tf`，这会使用近似模式：假设 `waist_yaw_link` 和移动底盘中心在 X/Y/Yaw 上重合。

```text
dx = target_x - robot_x
dy = target_y - robot_y
place_x =  cos(robot_yaw) * dx + sin(robot_yaw) * dy
place_y = -sin(robot_yaw) * dx + cos(robot_yaw) * dy
place_z = target_z - waist_z_in_map + place_z_offset
```

因为 SLAM pose 没有高度信息，如果使用近似模式，并且目标 `z` 是 map/world 下的绝对高度，需要配置 `waist_z_in_map`。在默认 live TF 模式下，Z 由 TF 链处理。

## 运行

离线调试，不请求目标点 HTTP，也不启动真实放置节点：

```bash
python3 -m world_place_execute.main \
  --target-x 1.20 --target-y 0.30 --target-z 0.75 \
  --robot-x 1.00 --robot-y 0.00 --robot-yaw 0.00 \
  --no-use-live-waist-tf \
  --no-execute-place
```

真实运行，通过 HTTP 获取目标点并调用现有放置节点：

```bash
python3 -m world_place_execute.main \
  --target-url http://<host>/api/place_target
```

输出文件：

```text
/tmp/world_place_exec.json
/tmp/world_place_exec.place.json
```

其中 `world_place_exec.json` 记录目标点、机器人 pose、转换后的 waist 坐标和调用下游放置节点的参数；`world_place_exec.place.json` 是下游 `grasp_pose_place_execute` 的执行结果。

## MCP 服务

世界坐标入口已经集成到现有放置 MCP：

```bash
python3 -m grasp_pose_place_execute.mcp_server
```

同一个 `place_object` 工具支持两组互斥坐标：

- world/map 坐标：`x/y/z`，内部调用本节点完成转换后放置
- 已转换的腰部坐标：`place_x/place_y/place_z`，直接执行原放置流程

world/map 调用示例：

```json
{
  "x": 2.064436218,
  "y": 0.342832004,
  "z": 0.621329767
}
```

不要在同一次调用中同时传入两组坐标。
