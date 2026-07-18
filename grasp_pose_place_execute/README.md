# grasp_pose_place_execute

Placement phase for the pick-and-place pipeline. Runs **after**
`grasp_pose_grasp_execute` has grasped an object and returned home still holding
it. This module moves the same arm that holds the object to a target placement
pose and releases it.

## How it talks to the grasp phase (handoff)

Both phases share a small JSON handoff file (default `/tmp/grasp_handoff.json`):

- The grasp phase writes it (`--handoff-out`) with which arm holds the object:

  ```json
  { "arm": "left", "holding": true, "object": ["banana"], "grasp_pose7": [ ... ] }
  ```

- The placement phase reads it (`--handoff-in`) and uses that arm. After a
  successful release it rewrites the file with `"holding": false, "placed": true`.

`holding` is `true` only when the grasp succeeded, the gripper closed, **and** the
object was not released at finish — so run the grasp with `--no-release-on-finish`.

You can bypass the handoff with `--arm left|right`.

## Motion

Mirrors the grasp path but with a **fixed** orientation (object orientation is
irrelevant for placing):

1. (optional) move the holding arm to its home posture — known start
2. middle/approach waypoint: backed off `--approach-dist` along the tool +Z axis
3. place pose: the target, raised by `--place-z-clearance` so the gripper releases
   slightly above the table and avoids collision
4. open the gripper (release)
5. lift straight up (`--lift-z`, with `--lift-tilt-y-deg`) to clear the table
6. retract back to the middle waypoint (from the lifted height, so it never dips)
7. return to the home joint posture

It reuses the tested motion + gripper primitives from `grasp_pose_grasp_execute`.

### Compliant set-down (F/T admittance) — default on

To stop the set-down from occasionally driving the object into the table (a
fixed-clearance target plus tracking/orientation overshoot can dip low), the
final descent is **compliant**, reusing `compliant_grasp_execute`'s admittance
stack and F/T calibration. Instead of position-driving to the place pose, the
arm:

- QP-streams the bulk of the standoff→place traverse, then hands over for the
  final `--compliant-final-descent-m` (default 6 cm) near the table;
- descends **soft along the place axis**, wrist held rigid, and **stops the
  instant the object/gripper touches the table** (F/T contact or motion stall);
- holds compliant at the touchdown pose while the gripper releases, then
  retracts.

The target is deepened `--compliant-place-extra-drop-m` (default 5 cm) past the
nominal place pose so the object is actually set down on the surface and contact
stops it there — the compliance keeps that contact gentle. Turn it off with
`--no-compliant-place` to use the plain QP descent.

Prerequisite: a valid F/T calibration for the placing arm at
`compliant_grasp_execute/ft_calibration/ft_calibration_{left,right}.json`
(generate with `python3 -m compliant_grasp_execute.ft_calibration.calibrate_ft
--arm <arm>`). Without it the admittance uses an unsafe identity fallback and
logs a loud warning.

## Quick run

End-to-end (grasp then place):

```bash
cd /home/ubuntu/niu

# 1) grasp, keep holding (do NOT release at home), write the handoff
python3 -m grasp_pose_grasp_execute.main --prompt banana \
  --pipeline-version accelerated --tcp-to-tip-z -0.26 --motion-strategy qp_all \
  --qp-otg-p-step 0.0015 --use-fixed-grasp-quat --no-use-pour-place-offsets \
  --grasp-tilt-y-deg 45 --lift-tilt-y-deg -15 --approach-dist 0.12 \
  --return-home --retract-to-approach --no-release-on-finish \
  --continuous-grasp-orientation --continuous-grasp-max-yaw-deg 30 \
  --handoff-out /tmp/grasp_handoff.json --json-out /tmp/grasp_exec.json

# 2a) place back where it was grasped (no target needed; uses handoff grasp_pose7)
python3 -m grasp_pose_place_execute.main \
  --place-z-clearance 0.05 --motion-strategy qp_all --approach-dist 0.12 \
  --handoff-in /tmp/grasp_handoff.json --json-out /tmp/place_exec.json

# 2b) or place at an explicit target TCP pose in waist_yaw_link
python3 -m grasp_pose_place_execute.main \
  --place-x 0.55 --place-y -0.20 --place-z 0.05 --place-z-clearance 0.05 \
  --motion-strategy qp_all --approach-dist 0.12 --place-tilt-y-deg 45 \
  --handoff-in /tmp/grasp_handoff.json --json-out /tmp/place_exec.json
```

## Configuration

All parameters live in `config.yaml` (next to `main.py`) and are auto-loaded, so
you normally just run `python3 -m grasp_pose_place_execute.main`. Edit that file
to change defaults; any CLI flag overrides it for a one-off. Precedence:
CLI flag > `config.yaml` > built-in default. Refresh the file with
`--write-config`, or ignore it with `--config ''`. The grasp module
(`grasp_pose_grasp_execute/config.yaml`) works the same way.

## Key options

- `--place-x/--place-y/--place-z`: target **TCP** pose in `waist_yaw_link` (m).
  **Optional** — if omitted, defaults to the original detected grasp position from
  the handoff (`grasp_pose7`), i.e. places the object back where it was picked.
  Reuses the original grasp orientation unless `--place-fixed-quat` is set.
- `--place-z-clearance`: release this far above the target Z (default 5 cm).
  With `--compliant-place` on, the compliant contact governs the true touchdown
  height, so this mainly sets where the descent hands over.
- `--compliant-place` (default on) / `--no-compliant-place`: compliant F/T
  set-down that stops on table contact. Tunables: `--compliant-final-descent-m`
  (compliant span near the table), `--compliant-place-extra-drop-m` (how far
  past the nominal pose to drive so the object is set down), `--compliant-contact-force-n`,
  `--ft-topic-{left,right}`, `--ft-calib-{left,right}`, and the rest of the
  `--compliant-*` gains (mirrors `compliant_grasp_execute`)
- `--lift-after-release` / `--lift-z` / `--lift-tilt-y-deg`: after releasing, lift
  straight up (default 12 cm, −15° tilt) to clear the table before retracting
- `--place-quat` / `--place-tilt-y-deg`: fixed orientation (defaults to the arm's
  calibrated grasp quat with a 45° nose-down tilt) when an explicit target is given
- `--place-fixed-quat`: when placing back at the grasp pose, use the fixed grasp
  quat instead of reusing the original grasp orientation
- `--arm left|right|auto`: `auto` reads the arm from the handoff
- `--no-require-holding`: place even if the handoff says nothing is held
- `--approach-dist`, `--retract-to-approach`, `--start-home`, `--return-home`
- motion/QP and gripper flags mirror `grasp_pose_grasp_execute`

## MCP server (agent integration)

`mcp_server.py` exposes the place phase to an agent via MCP, built on
[`fastmcp`](https://github.com/jlowin/fastmcp) and following the same pattern
as `bottle_cup_pour_place/mcp_server.py`. Each tool call runs either
`grasp_pose_place_execute.main` or `world_place_execute.main` as a
**subprocess** (clean ROS2 lifecycle), serialised by a run-lock, and returns
`{"state": "succeed"|"failed", "msg", "result", ...}`.

Requires `pip install fastmcp`.

Run it (from the repo root):

```bash
# Streamable-HTTP on 0.0.0.0:8004/  (default)
python3 -m grasp_pose_place_execute.mcp_server

# Or stdio (Cursor / Claude Desktop MCP config)
MCP_TRANSPORT=stdio python3 -m grasp_pose_place_execute.mcp_server
```

Env overrides: `MCP_TRANSPORT`, `MCP_HOST`, `MCP_PORT` (default 8004), `MCP_PATH`.

Tools exposed:

- `place_object(x, y, z, place_x, place_y, place_z, …)` — place the
  held object. Pass world/map `x/y/z` to perform the world-to-waist conversion,
  or pass already-converted waist `place_x/place_y/place_z`. The two coordinate
  groups are mutually exclusive. Omit both groups to place back where grasped.
  Pass advanced flags through `extra_args`.
- `read_handoff(path)` — read `/tmp/grasp_handoff.json`.

Register with an HTTP MCP client:

```json
{ "mcpServers": { "place": { "url": "http://<robot-host>:8004/" } } }
```

Or stdio:

```json
{
  "mcpServers": {
    "place": {
      "command": "python3",
      "args": ["-m", "grasp_pose_place_execute.mcp_server"],
      "cwd": "/home/ubuntu/niu",
      "env": { "MCP_TRANSPORT": "stdio" }
    }
  }
}
```

### Full pick-and-place from an agent

Register both servers, then the agent runs:

1. `grasp.grasp_object` with `prompt: "<object>"` and `release_on_finish: false`
   (keeps holding, writes `/tmp/grasp_handoff.json`).
2. `place.place_object` (no target → place back where picked; pass world
   `x/y/z`, or waist `place_x/place_y/place_z`).

Both phases share `/tmp/grasp_handoff.json` automatically. Run them one at a
time — each call spins up its own ROS2 node, so avoid overlapping calls.
Per-run logs land in `logs/mcp/place_execute/`.
