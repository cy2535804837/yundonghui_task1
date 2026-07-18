# waist_height_adjust

Adjust the robot's **waist (torso) height** to a commanded value.

## How the height is controlled

On the tianyi2 the "body" is a 4-DOF chain

```
[first_leg_pitch_joint, second_leg_pitch_joint, waist_pitch_joint, waist_yaw_joint]
```

The SDK exposes it in **Cartesian** space via
`ActionCall.endpose_body_controller(target_pose)`, where `target_pose` is a
4-element list ordered:

```
[x, z, pitch, yaw]
```

- `x`   — torso forward offset (m)
- `z`   — **waist height (m)**  ← this is what this tool sets
- `pitch` — torso pitch (rad)
- `yaw`   — torso yaw (rad)

The factory **zero pose** is `[0.05, 0.68, 0.0, 0.0]`, i.e. the nominal standing
height is `z = 0.68 m`. Raising/lowering `z` drives the two leg-pitch joints
(a squat/parallelogram linkage) so the torso goes up/down while staying upright.

**Reachable range (observed):** the nominal/tall height is ~`0.68 m`; the
mechanical **floor is ~`0.56 m`**. Asking for a height below the floor makes a
leg-pitch joint exceed its limit, so the body controller **rejects** the goal
(`code=600101 指令超上限位`, "command over upper limit") and the robot does not
move. When that happens the tool prints a clear diagnostic and the result JSON
carries `ok=false`, `moved=false`, the `error_codes`, and a `hint`.

## Usage

Run as a module from the repo root (`/home/ubuntu/niu`):

```bash
# lower the waist to 0.60 m
python3 -m waist_height_adjust.main --height 0.60

# preview the resolved target without moving / touching hardware
python3 -m waist_height_adjust.main --height 0.60 --dry-run

# or via the wrapper
./waist_height_adjust/run.sh --height 0.72
```

The tool:

1. loads `config.yaml`, resolves and **clamps** the target height into the safe
   range `[--min-height, --max-height]`;
2. deactivates all controllers and enables the **leg + waist** hardware
   (`--no-enable-hardware` to skip on sim);
3. sends **one blocking** `endpose_body_controller([x, z, pitch, yaw])` command;
4. waits `--settle-sec` and reads back the body joint angles, printing a JSON
   result (also written to `--json-out`, default `/tmp/waist_height.json`).

## Key options

- `--height` — target waist height in metres (default `0.68`, the zero pose).
- `--min-height` / `--max-height` — **safety clamp** (default `0.45` / `0.90`).
  These are conservative placeholders: confirm the real mechanical travel on
  hardware and widen them in `config.yaml`.
- `--body-x`, `--body-pitch`, `--body-yaw` — the other body-endpose DOFs,
  defaulted to the factory zero pose. Only change if you also need to shift/lean
  the torso.
- `--enable-hardware` / `--no-enable-hardware` — toggle the leg + waist enable.
- `--settle-sec` — settle time before the read-back.
- `--dry-run` — print the target and exit without moving.

All parameters live in `config.yaml`; precedence is
`CLI flag > config.yaml > built-in default`. Regenerate the file from the
effective values with `python3 -m waist_height_adjust.main --write-config`.

## MCP server

The same functionality is exposed as an MCP tool so an agent can adjust the
waist height directly. It mirrors the other servers in this repo
(`grasp_pose_grasp_execute` = 8003, `grasp_pose_place_execute` = 8004,
`compliant_grasp_execute` = 8005) and defaults to port **8006**.

```bash
# Streamable-HTTP on 0.0.0.0:8006/  (default)
python3 -m waist_height_adjust.mcp_server

# stdio transport (for Cursor / Claude Desktop MCP configs)
MCP_TRANSPORT=stdio python3 -m waist_height_adjust.mcp_server
```

It exposes one tool:

- **`set_waist_height(height, body_x=None, body_pitch=None, body_yaw=None,
  enable_hardware=None, settle_sec=None, dry_run=False, timeout_sec=120.0,
  extra_args=None)`** — set the waist height (m) and block until the move
  finishes. Returns `{"state": "succeed"|"failed", "msg": ..., "result": {...},
  "log_path": ...}`. On an out-of-range height the failure `msg` carries the
  tool's `hint` (e.g. "height below the mechanical floor").

Each call runs `waist_height_adjust.main` as an isolated subprocess (so
`rclpy.init()`/`shutdown()` happen once per process), a run lock serialises
calls on the shared body hardware, and `MCP_TRANSPORT` / `MCP_HOST` /
`MCP_PORT` / `MCP_PATH` env vars override the transport defaults.

## Safety notes

- The `[--min-height, --max-height]` clamp guards against typos (e.g. `6.8`
  instead of `0.68`); it does **not** know the true joint limits — verify them
  before widening.
- Make sure the arms/head are in a posture that is safe to raise or lower before
  changing the torso height (moving the whole upper body changes the reachable
  workspace and can bring the grippers toward the table or a fixture).
