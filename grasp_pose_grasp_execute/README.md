# grasp_pose_grasp_execute

Detect object grasp pose (from `grasp_pose_generation` internals) and execute
single-arm grasp motion via selectable controllers.

## What it does

1. Detect object pose from prompt (`current` or `accelerated` segmentation backend)
2. Convert to TCP grasp pose in `waist_yaw_link`
3. Move selected arm:
   - optional approach waypoint
   - grasp waypoint
   - optional lift waypoint
   - selectable motion strategy:
     - `moveit`: all phases via MoveIt
     - `qpik`: all phases via QPIK action
     - `qp_stream`: all phases via QP topic streaming
     - `auto_hybrid` (default): MoveIt approach + QPIK insertion/lift
4. Pure Robotiq close at grasp pose (default on), optional open-before-approach

## Quick run

```bash
cd /home/nvidia/niu

./grasp_pose_grasp_execute/run.sh \
  --prompt bottle \
  --arm right \
  --pipeline-version accelerated \
  --motion-strategy auto_hybrid \
  --orientation-policy grasp \
  --open-gripper-before-grasp \
  --approach-dx -0.10 \
  --lift-z 0.05

# Need to start the following commands first in a separate terminal to activate the Robotiq gripper
python3 robotiq_node_zmq.py --hand_dev_ip /dev/ttyUSB0 --hand_slave_id 9   --server_set_port 4244 --server_get_port 4245

# best paramters currently
python3 -m grasp_pose_grasp_execute.main   --arm right --prompt sponge --pipeline-version accelerated   --tcp-to-tip-z -0.25 --motion-strategy qp_all   --qp-otg-p-step 0.0015   --use-fixed-grasp-quat --no-use-pour-place-offsets   --grasp-reach-tol-m 0.012 --grasp-z-offset -0.01 --grasp-y-offset 0.025 --grasp-x-offset 0.01   --grasp-tilt-y-deg 15 --lift-tilt-y-deg -30 --approach-dist 0.12  --return-home --retract-to-approach --release-on-finish --gripper-backend zmq   --json-out /tmp/grasp_exec.json

```

Gripper close is **on by default**. Use `--no-close-gripper` for motion-only tests.

Default gripper settings (USB Modbus, no extra server):

```bash
./grasp_pose_grasp_execute/run.sh \
  --prompt bottle \
  --arm right \
  --gripper-serial-port auto \
  --gripper-slave-id 9
```

## Use previously detected pose JSON

```bash
./grasp_pose_grasp_execute/run.sh \
  --detected-pose-json /tmp/grasp_pose.json \
  --arm right
```

## Dry run

```bash
./grasp_pose_grasp_execute/run.sh \
  --prompt bottle \
  --arm right \
  --dry-run
```

## Notes

- Default motion style is `auto_hybrid`, aligned with `bottle_cup_pour_place`:
  - Fixed calibrated grasp quaternion (`--use-fixed-grasp-quat`, on by default)
  - Fixed grasp quaternion only for live detection (offsets off by default; see below)
  - MoveIt approach with orientation retry if planning fails
  - QPIK insertion/lift with live TCP orientation held constant (slow `otg_p_step=0.0008`)
- **`--use-pour-place-offsets`**: only when mimicking `detect_pour_place` fixed cup centroids
  (e.g. `--use-pour-place-offsets` adds right X −0.25 m). Do **not** use with live detection —
  detection already outputs TCP XYZ; adding centroid offsets twice pushes targets out of reach.
- Gripper: no activation at connect by default; lazy `ensure_activated()` on first move only.
  Close runs **after** arm reaches grasp XYZ (waits for TCP), not when QPIK action returns.
  Use `--gripper-force-activate` only to force full re-activate.
- QP/QPIK tuning knobs are exposed in CLI:
  `--qp-otg-p-step`, `--qp-otg-r-step`, `--qp-stream-duration`, `--qp-stream-rate-hz`.
- Gripper opens before approach (default on), closes at grasp pose, then lift runs.
- **Gripper control (default):** direct USB Modbus via `RobotiqUSB/robotiq_driver.py` — same as `bottle_cup_pour_place` (`RightGripperTactile` → `GraspingManager`), **no separate ZMQ server**.
- Optional ZMQ (`--gripper-backend zmq`) only if you already run `robotiq_node_zmq.py`.
- Output JSON includes detection result, planned waypoints, and motion success flags.

## MCP server (agent integration)

`mcp_server.py` exposes the grasp phase to an agent via MCP, built on
[`fastmcp`](https://github.com/jlowin/fastmcp) and following the same pattern
as `bottle_cup_pour_place/mcp_server.py`. Each tool call runs
`python3 -m grasp_pose_grasp_execute.main` as a **subprocess** (clean ROS2
lifecycle), serialised by a run-lock so two calls never fight over the
hardware, and returns `{"state": "succeed"|"failed", "msg", "result", ...}`.

Requires `pip install fastmcp`.

Run it (from the repo root):

```bash
# Streamable-HTTP on 0.0.0.0:8003/  (default)
python3 -m grasp_pose_grasp_execute.mcp_server

# Or stdio (Cursor / Claude Desktop MCP config)
MCP_TRANSPORT=stdio python3 -m grasp_pose_grasp_execute.mcp_server
```

Env overrides: `MCP_TRANSPORT`, `MCP_HOST`, `MCP_PORT` (default 8003), `MCP_PATH`.

Tools exposed:

- `grasp_object(prompt, arm, motion_strategy, release_on_finish, dry_run, …)` —
  detect by prompt and grasp. Set `release_on_finish=false` to keep holding for
  a subsequent place. Writes the handoff file. Pass advanced flags through
  `extra_args`.
- `read_handoff(path)` — read `/tmp/grasp_handoff.json`.

Register with an HTTP MCP client:

```json
{ "mcpServers": { "grasp": { "url": "http://<robot-host>:8003/" } } }
```

Or stdio:

```json
{
  "mcpServers": {
    "grasp": {
      "command": "python3",
      "args": ["-m", "grasp_pose_grasp_execute.mcp_server"],
      "cwd": "/home/ubuntu/niu",
      "env": { "MCP_TRANSPORT": "stdio" }
    }
  }
}
```

Prerequisites match the CLI: the ROS2 stack, segmentation service, and (for
real grasps) the gripper must be running. Use `dry_run: true` to test
detection/planning without hardware. Per-run logs land in
`logs/mcp/grasp_execute/`.

