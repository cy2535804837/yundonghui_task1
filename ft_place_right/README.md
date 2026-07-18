# ft_place_right

Self-contained **wrist force-torque-driven placement** for the RIGHT arm,
extracted from `bottle_cup_pour_place` / `adaptive_place_right`.

This is the **tactile-free** build for a machine that has only wrist
force-torque sensors (no fingertip Tac3D sensors):

- All placement decisions come from the RIGHT wrist FT sensor
  (`/arm_6dof_right`).
- **No** fingertip tactile feedback and **no** grip-decay loop (the
  FT→contact-coefficient decay PD loop was removed because it was found
  unstable).
- The gripper is an **injected hook** — this folder never talks to a
  gripper driver directly. Wire your own gripper; a logging `NoopGripper`
  stub is provided for dry-runs.
- `xarm_sdk` is **not** bundled — it is taken from the system install.

## What it does (phases)

1. **A — characterise object weight (optional, FT only)**
   Sample wrist-Z force with the gripper open vs. closed to get
   `G_obj = |F_empty − F_loaded|` (or pass `--g-obj-mode manual
   --object-weight-N <N>`).
2. **B — descend until contact**
   Fast direct-QP descent, then slow admittance until the wrist-Z force
   delta trips (table contact).
3. **C — load transfer**
   Press the object onto the table and wait until
   `gamma = clip(F_support / G_obj, 0, 1)` exceeds
   `gamma_release_threshold` (the table is bearing the load). No grip
   modulation.
4. **D — release + lift**
   Open the gripper once via the hook, then lift via admittance.

## Layout

```
ft_place_right/
  adaptive_place_right/        # placement package (FT-only)
    main.py                    # CLI entry point
    config.py                  # RightAdaptivePlaceConfig
    adaptive_placer_right.py   # RightAdaptivePlacer orchestrator
    gripper_hook.py            # GripperHook protocol + NoopGripper stub  (NEW)
    spin_thread.py             # background rclpy.spin_once helper          (NEW)
  handover/                    # minimal subset (no tactile / no pipeline)
    config.py
    admittance_arm.py          # AdmittanceArm
  admittance_control/
    AdmittanceController_v3.py
    AdmittanceController_v4_2_fixed.py
    ft_calibration_right.py    # FT sensor calibration tool (RIGHT arm)
    ft_calibration_right.json  # FT calibration data (consumed at runtime)
```

## Run

From **this folder** (it must contain the sibling dirs `handover/` and
`admittance_control/`; `xarm_sdk` comes from the system install):

```bash
cd /home/nvidia/niu/ft_place_right

# Bring up the xarm drivers, the wrist FT sensor, and MoveIt first.

# Dry-run with the NoopGripper stub (manual weight; no real gripping):
python3 -m adaptive_place_right.main --g-obj-mode manual --object-weight-N 1.2

# Descend from the current pose (skip the pre-place jointspace move):
python3 -m adaptive_place_right.main --no-pre-place-move --g-obj-mode manual --object-weight-N 1.2

# Record a CSV trace (t, phase, Fz, F_support, gamma, ..., z_eq):
python3 -m adaptive_place_right.main --log-csv /tmp/place_run.csv --g-obj-mode manual --object-weight-N 1.2
```

## FT sensor calibration

Calibrate the RIGHT wrist FT sensor; the result is written next to the
tool and is what the placement loads at runtime
(`admittance_control/ft_calibration_right.json`, referenced via
`handover.config.DEFAULT_RIGHT_FT_CALIB`):

```bash
cd /home/nvidia/niu/ft_place_right
python3 admittance_control/ft_calibration_right.py            # interactive full calibration
python3 admittance_control/ft_calibration_right.py --drift-only   # refresh bias only
```

If the JSON is missing the placement falls back to identity
`R_sensor_tcp` (expect a Z bias), so calibrate before relying on `gamma`.

## Wiring your own gripper

Implement `gripper_hook.GripperHook` (any object with these methods works
— it is a structural `Protocol`) and pass it into `RightAdaptivePlacer`,
or set it in `main.py` where `NoopGripper()` is currently constructed:

```python
class MyRobotiqGripper:
    def close_to_hold(self) -> None: ...  # grip the object before descent
    def open(self) -> None:         ...   # full release
    def shutdown(self) -> None:     ...   # release + drop connection
```

- `close_to_hold()` is called once before descent when `--grasp-first`.
- `open()` is called for the F_empty weight sample (auto `G_obj`) and as
  the final release before the lift.
- With the default `NoopGripper`, `--g-obj-mode auto` cannot measure a
  real weight (open/close are no-ops), so use `--g-obj-mode manual`.

## Notes / inert flags

Several CLI flags carried over from the original tactile/decay build are
**accepted but no longer have any effect** in this FT-only build:
`--k`, `--target-cf`, `--hold-mode`, `--hold-force-pct`,
`--contact-timeout`, `--decay-release-step`. The active knobs are the
descent (`--descent-speed`, `--fast-descent-speed`, `--force-threshold`,
…), transfer/press (`--transfer-press-depth`, `--transfer-kz`,
`--gamma-release`, `--gamma-debounce`), and lift (`--lift-speed`,
`--lift-height`) parameters.

The original files in `adaptive_place_right/`, `handover/`,
`admittance_control/`, etc. at the repo root were **not modified** — this
folder is an independent copy.
