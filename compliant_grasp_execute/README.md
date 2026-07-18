# compliant_grasp_execute

A **self-contained** fork of `grasp_pose_grasp_execute` whose final grasp insert
is performed under **wrist F/T admittance control** to stop the recurring
collisions with the table (and the joint faults they caused).

Everything needed is copied in here — the grasp pipeline, the proven admittance
stack (from `ft_place_right`), and a both-arms F/T calibrator — so this project
does **not** modify or depend at runtime on `grasp_pose_grasp_execute` or
`ft_place_right`. It still imports the perception package
`grasp_pose_generation` for detection (same as the original executor).

## What is different from the original grasp executor

The whole detect → approach → grasp → lift → home pipeline is identical, **except
the final descent to the grasp pose**:

1. **Approach** (unchanged): move to the standoff pose above the object.
2. **Compliant insert** (new): instead of position-driving straight to the
   planned grasp pose, an admittance controller slews the equilibrium target
   down the insertion axis at a slow, controlled speed. The arm is **soft along
   the insertion axis**, **stiff laterally** (holds the planned X/Y), and the
   **wrist orientation is held rigid**. The descent **stops the instant the
   F/T sensor reports a contact force** along the insertion axis (table or
   object) — so it cannot slam into the table.
3. **Close** (unchanged action, new context): the gripper closes **while the arm
   is still held compliant** at the contact pose, so closing does not fight the
   arm.
4. **Rotate away from table** (new): the wrist rotates about the waist Y axis
   (`--table-clear-rotate-deg`, default −20°) to tilt the gripper away from the
   table.
5. **Lift / retract / home** (unchanged): the usual sequence continues.

**v1 scope (intentionally minimal):** no object-weight re-zero and no compliant
lift. The compliant phase is *descend-to-contact + hold-while-closing* only.

## Layout

```
compliant_grasp_execute/
  main.py                    forked pipeline; --compliant-grasp branch + rotate-away
  joint_limits.py            (verbatim copy) recovery-aware joint-limit guard
  config_io.py               (verbatim copy) YAML config loader
  config.yaml                all CLI params (incl. the compliant_* knobs)
  compliant_insert.py        NEW: tool-axis compliant descend-to-contact
  admittance/                self-contained admittance stack
    AdmittanceController_v3.py        (copy) FT pipeline + TF + quat helpers
    AdmittanceController_v4_2_fixed.py (copy) per-axis M/B/K admittance
    admittance_arm.py        (copy) threaded one-arm admittance runner
    gains.py                 AdmittanceGains / ForceProcessing + calib paths
    spin_thread.py           (copy) background rclpy.spin_once helper
  ft_calibration/
    calibrate_ft.py          BOTH-arms F/T calibrator (--arm left/right)
    ft_calibration_left.json   <- you generate (NOT shipped)
    ft_calibration_right.json  <- you generate (NOT shipped)
```

## Prerequisites (do these first on this machine)

This is a **new machine with a different hardware configuration**, so the old
`ft_place_right` calibration is **invalid**. You must calibrate **both** arms
here before trusting the compliant grasp.

1. **F/T driver running** and publishing both wrench topics:
   ```bash
   ros2 topic hz /arm_6dof_left
   ros2 topic hz /arm_6dof_right
   ```
2. **TF** publishing `waist_yaw_link -> {left,right}_tcp_link`.
3. **Remove every external load** except the permanently-attached gripper/tool.

### Calibrate the F/T sensors (both arms)

```bash
# RIGHT arm
python3 -m compliant_grasp_execute.ft_calibration.calibrate_ft --arm right

# LEFT arm
python3 -m compliant_grasp_execute.ft_calibration.calibrate_ft --arm left
```

The tool drives the arm through several wrist orientations; press **Enter** at
each once it has settled. It auto-detects `R_sensor_tcp` (24 candidates) and
least-squares fits mass / CoM / 6-axis bias, writing
`ft_calibration/ft_calibration_<arm>.json`. Aim for per-pose force residuals
well under ~1 N and condition numbers < 50 (add wrist pitch/roll variety if not).

Quick re-zero of just the bias later (no full re-fit):
```bash
python3 -m compliant_grasp_execute.ft_calibration.calibrate_ft --arm right --drift-only
```

## Run

```bash
# auto-select arm from object position, compliant grasp on by default
python3 -m compliant_grasp_execute.main --prompt banana
```

Everything is configured in `config.yaml`; CLI flags override per-run.
`--no-compliant-grasp` falls back to the original position-controlled insert.

### Key compliant knobs (in `config.yaml` or as flags)

| Param | Default | Meaning |
|-------|---------|---------|
| `compliant_grasp` | `true` | Enable the F/T compliant insert |
| `compliant_contact_force_n` | `1.5` | Contact force (N) along insertion axis that stops the descent |
| `compliant_contact_debounce` | `3` | Consecutive samples over threshold before tripping |
| `compliant_stall_window_s` | `0.7` | Window (s) for STALL contact detection. Must be long enough that a healthy slow descent (~1.3 cm/s × window) clears `stall_eps_m`, or it false-trips early (grasp air) |
| `compliant_stall_eps_m` | `0.004` | Min TCP progress (m) in the stall window; below → treated as contact |
| `compliant_min_insert_m` | `0.008` | Min ACTUAL TCP travel before a contact trip counts (rejects wrist reaction) |
| `compliant_overshoot_m` | `0.02` | How far past the planned grasp depth it may keep seeking contact |
| `compliant_max_insert_m` | `0.20` | Safety ceiling on travel; must be ≥ approach standoff + overshoot, else grasps air |
| `compliant_final_descent_m` | `0.04` | **HYBRID insert.** Only the FINAL this-many metres of the approach→grasp traverse run under F/T admittance; the earlier (longer) portion is driven by the position-controlled QP stream. Set `0` to make the whole insert compliant (old behaviour). See "Hybrid insert" below |
| `compliant_insert_speed_mps` | `0.020` | Descent slew speed (m/s) |
| `compliant_max_lag_m` | `0.025` | Lag throttle: hold the slew while the TCP lags the commanded target by more than this |
| `compliant_lateral_stiffness` | `40.0` | K (N/m) on axes ORTHOGONAL to the insertion direction (hold the planned line) |
| `compliant_insertion_stiffness` | `20.0` | K (N/m) on axes ALONG the insertion direction (compliance) |
| `compliant_soften_threshold` | `0.30` | A waist axis is made soft when its `|dir component|` exceeds this (so tilted inserts don't leave a major axis stiff) |
| `compliant_damping_ratio` | `1.4` | B is set per-axis to `ratio × 2√(K·M)` → near-critical damping on every axis (≥1 avoids shaking/bounce) |
| `compliant_damping` | `3.0` | Absolute floor on the per-axis damping B |
| `compliant_hold_stiffness` | `150.0` | Isotropic K (N/m) for the post-contact HOLD while the gripper closes (stiff → no spring-back) |
| `compliant_mass` | `0.1` | M (kg) on all translation axes |
| `compliant_filter_alpha` | `0.35` | F/T EMA filter (`prev = a·new + (1−a)·prev`). LOWER = more smoothing. Same for both arms; lower it if one side shakes (noisier sensor) |
| `compliant_force_deadzone` | `0.8` | Force (N) ignored as noise; must stay `<` `compliant_contact_force_n` |
| `compliant_torque_deadzone` | `0.08` | Torque (Nm) ignored as noise (wrist held rigid) |
| `compliant_max_vel` | `0.05` | HARD cap (m/s) on admittance output velocity. The descent slews at `compliant_insert_speed_mps` (~0.02); without this cap a residual force on a soft axis drives a runaway (TCP races ahead and slams the table). Keep a small multiple of the slew speed |
| `compliant_max_omega` | `0.5` | HARD cap (rad/s) on admittance angular velocity (wrist held rigid) |
| `compliant_control_rate_hz` | `100.0` | Supervisory descent loop rate (Hz) |
| `compliant_otg_p_step` | `0.008` | QP position OTG step (m) during the insert — too small stalls the descent |
| `compliant_otg_r_step` | `0.005` | QP rotation OTG step (rad) during the insert |
| `compliant_loop_period` | `0.004` | Admittance integration period (s) |
| `compliant_trans_lead_time` | `0.08` | Published carrot = `v_cmd × this`. Must be ≫ loop period or the arm won't move; too large amplifies jitter (see notes) |
| `table_clear_rotate_deg` | `-20.0` | Post-grasp wrist rotation about waist Y to clear the table (0 disables) |
| `grasp_offset_frame` | `waist` | Frame the horizontal grasp offset (`grasp_x_offset`/`grasp_y_offset`) is applied in. **`waist`** (default) applies the offset directly in `waist_yaw_link` X/Y. `object` applies `grasp_x_offset` **along the detected object long axis** and `grasp_y_offset` **across it** (short axis / jaw-gap), so a *fixed* offset grasps the same physical spot at any object heading — but the offset direction then rotates with the object. `grasp_z_offset` is always vertical |
| `grasp_x_offset` | `0.02` | Grasp-target offset along the first horizontal axis (m): waist +X in `waist` frame, or along the object long axis in `object` frame |
| `grasp_y_offset` | `0.0` | Grasp-target offset along the second horizontal axis (m): waist +Y in `waist` frame, or across the object (short axis) in `object` frame |
| `approach_clamp_lateral` | `true` | Clamp the approach standoff's `|Y|` to the grasp `|Y|`. Backing out along `tool_z` usually adds a small outboard Y component (from the arm's fixed grasp orientation), pushing the approach start further from the body center than the grasp. At far-outboard reaches that extra lateral push folds the elbow into its stop (left arm `elbow_pitch` → -2.56 at approach `Y=+0.276` vs grasp `Y=+0.254`) — the approach *start* becomes unreachable even though the grasp is fine, so the standard path aborts and falls back to elbow-high for a perpendicular object. The standoff only needs to be above/behind, so this backs out in X-Z at the grasp's lateral position. No-op for inboard approaches |
| `elbow_low_reorient` | `false` | **Reach-aware reorientation of the standard grasp** (experimental, **disabled — disproven on hardware**). The idea: rotate the grasp about waist X to flatten `tool_z_Y` so the wrist sits over a far-side object, hoping the elbow-low basin could then reach it with a steeper approach. Testing showed it does **not** work — `elbow_pitch` still saturates at its `-2.61` stop regardless of orientation, because the limit is bound by the wrist **position** (extended to `|Y|≳0.24` while the approach pulls it toward the body), not the tool orientation. It also lowered the computed grasp TCP `|Y|` below the proactive threshold, which *defeated* the proactive elbow-high shortcut and wasted two doomed standard attempts before the fallback. Left in as an opt-in flag but off by default; far-side objects are handled by the proactive/fallback elbow-high path, which is the correct strategy for this arm |
| `elbow_low_reorient_trigger_abs_y_m` | `0.24` | (only if `elbow_low_reorient` is on) standard grasp TCP `|Y|` that triggers the reorientation |
| `elbow_low_reorient_max_deg` | `20.0` | (only if `elbow_low_reorient` is on) cap on the waist-X reorientation |
| `diagonal_schedule` | `true` | For diagonal objects (long-axis angle ~20–60° from waist +X), scale the grasp tilt DOWN and the jaw-yaw clamp UP continuously with the angle, instead of the fixed 45° tilt + ±15° yaw. Frees `wrist_pitch` (which the ±15° yaw was pushing past +1.05 at far reaches) and actually aligns the jaws to the long axis (the ±15° clamp left ~20° misaligned). Off below `diagonal_schedule_angle_start_deg` so the proven perpendicular behaviour is unchanged; the elbow-high path takes over at/above `diagonal_schedule_angle_end_deg` |
| `diagonal_schedule_angle_start_deg` | `15.0` | Angle (deg) above which the schedule begins. At/below this the standard 45° tilt + ±15° yaw is used |
| `diagonal_schedule_angle_end_deg` | `60.0` | Angle (deg) at which the schedule reaches full effect. Defaults to 60 to hand off to the elbow-high path |
| `diagonal_schedule_tilt_scale_end` | `0.5` | Tilt scale at the end angle (1.0 at start). 0.5 turns 45° into 22.5° for a near-parallel object. Lower = less nose-down (easier on `wrist_pitch`, shallower approach) |
| `diagonal_schedule_yaw_clamp_end_deg` | `45.0` | Jaw-yaw clamp (deg) at the end angle (`continuous_grasp_max_yaw_deg` at start). Higher = better jaw alignment, more wrist demand; balanced against the reduced tilt |
| `elbow_high_enable_fallback` | `true` | Reconfigure to an elbow-HIGH posture and retry when the elbow-low basin can't hold the grasp orientation (see "Elbow-high" below) |
| `elbow_high_proactive` | `false` | Pick elbow-high up front when the object long axis is ~parallel to the body (angle-from-X ≥ `elbow_high_proactive_angle_min_deg`) — the **parallel half** of the two-strategy axis split (see "Elbow-high" below) — instead of waiting for the elbow-low attempt to fail |
| `elbow_high_proactive_angle_min_deg` | `60.0` | The axis split point (deg, angle-from-waist-X). At/above → elbow-HIGH (parallel bucket); below → elbow-LOW standard (perpendicular/diagonal bucket). Defaults to **60 to match `diagonal_schedule_angle_end_deg`**: diagonal objects (15–60°) stay on elbow-low where the diagonal schedule handles them; only strongly-parallel objects (≥60°) go elbow-high, where the jaw-align yaw is small so `wrist_roll` stays in range (a diagonal object needs a large align yaw that saturates `wrist_roll`'s cramped +1.3 stop). Used by **both** the proactive and the fallback elbow-high triggers |
| `reach_tilt_reduce` | `true` | On the STANDARD (elbow-low) path, back the side-tilt off for a far-outboard object (large `|Y|`) so the elbow can **extend** to reach it instead of folding `elbow_pitch` into its `-2.61` stop. This is the elbow-low way to reach far perpendicular objects (the elbow-high path is reserved for the parallel bucket). Keyed on **lateral** reach only — a forward term was tried and reverted (it over-flattened the tool into the table without fixing the wrist_pitch upper stop it targeted) |
| `reach_tilt_start_abs_y_m` | `0.18` | Object tip `|Y|` (m, waist frame) below which the full 45° side-tilt is kept |
| `reach_tilt_end_abs_y_m` | `0.28` | Object tip `|Y|` (m) at which the reach tilt reduction reaches `reach_tilt_scale_end` |
| `reach_tilt_scale_end` | `0.4` | Side-tilt scale at `reach_tilt_end_abs_y_m` (1.0 at the start). 0.4 turns 45° into ~18° for the farthest-outboard object |
| `elbow_high_clamp_margin_rad` | `0.05` | Clamp the elbow-high seed joints to ≥ this margin inside their hard limits before commanding them (a hand-dragged seed can sit at/past a stop). Smaller = the ready posture stays closer to what you dragged, but the joint starts nearer its hard stop (less QP travel, guard grace covers a joint parked in the margin at phase start). 0.15 is conservative; 0.05 is used so a top-down seed taught near the shoulder-yaw limit isn't rotated away from top-down by the clamp |
| `elbow_high_ready_left_joints` / `_right` | (taught) | Elbow-high seed posture per arm (right = sagittal mirror) |
| `elbow_high_arms` | `left` | Comma-separated arms allowed to use the elbow-high path (proactive AND fallback). Default `left` because only the left seed was hand-taught; the right mirror seed can drive the right `wrist_roll` past its tighter -1.3 lower stop at far-right reaches (object waist Y ~ -0.19m). An arm not in this list uses the standard elbow-low path. Set `left,right` only once a right seed is taught/validated for far-right objects |
| `elbow_high_stage_left_joints` / `_right` | `[0, ±1.18, 0, -1.3, ±1.4, -0.13, ±0.18]` | **Staging** posture visited from home BEFORE the elbow-high reconfigure (and passed back through on return). The arm-up/elbow-bent posture that used to be the home; it's a clean base for the transition sweep into the elbow-high basin. The elbow-**low** path skips it (approaches straight from home) |
| `elbow_high_transition_left_joints` / `_right` | `[]` (auto) | Intermediate jointspace waypoint; empty = midpoint of **stage** & ready |
| `elbow_high_orientation` | `seed` | Elbow-high grasp orientation: `seed` (anchor on the orientation the arm is already in at the seed — only translates, wrist stays in range), `topdown` (force vertical), or `sidetilt` (legacy) |
| `elbow_high_always` | `false` | Route **every** grasp through the top-down elbow-high posture (for arms in `elbow_high_arms`), regardless of object angle/reach. For testing elbow-high as the single strategy. Costs extra motion time and leans on `wrist_pitch` margin for near objects — keep `false` for production, where cheap perpendicular objects use the standard path |
| `elbow_high_align_jaws` | `true` | On the elbow-high (top-down) path, align the jaws to the object's long axis about waist Z so a **parallel object is grasped on its short axis**. Because the tool points ~straight down, this yaw maps onto `wrist_roll` (which has range), and the parallel-jaw 180° symmetry is used to pick the roll-feasible equivalent yaw (the left arm rolls ~−90°, the right ~+90°, matching their asymmetric `wrist_roll` limits). Set `false` to keep the old fixed small ±yaw |
| `elbow_high_align_wrist_margin_rad` | `0.08` | Safety margin (rad) kept from the `wrist_roll` limits when the jaw-symmetry search picks the elbow-high align yaw |
| `elbow_high_jaw_flip_retry` | `true` | If the short-axis alignment saturates `wrist_roll` on the arm's **cramped** stop (left upper `+1.3` / right lower `-1.3`), retry the **same** grip from the 180°-flipped jaw line, which rolls the wrist onto its **roomy** side (left down to `-1.65` / right up to `+1.65`). Keeps *full* alignment — unlike the reduced-yaw retry, which just misaligns the jaws and doesn't free `wrist_roll`. Fires before the reduced-yaw retry, only when the abort was a `wrist_roll` saturation on the elbow-high aligned path |
| `elbow_high_seed_yaw_max_deg` | `90.0` | Search window (deg, about waist Z) for the elbow-high jaw-align yaw. `90` lets the jaws align to **any** object orientation (the yaw maps onto `wrist_roll`, which has room). With `elbow_high_align_jaws=false` this is just a hard clamp on the small fixed yaw; backed off on a reach retry |
| `elbow_high_seed_tilt_up_deg` | `12.0` | Tilt the seed-anchored grasp UP (less nose-down) by this many deg. The hand-taught seed points ~64° nose-down, which drives `wrist_pitch` onto its lower `-0.785` stop mid-descent at a forward object (grasp aborts ~2 cm short). 12° backs the tool to ~52° and also pulls the TCP back (less forward reach), freeing `wrist_pitch`. 0 = use the seed as-is; lower for a more top-down grasp |
| `elbow_high_seed_retry_tilt_up_deg` | `25.0` | Larger seed tilt-up used on the reach retry once `wrist_pitch` saturates. Bigger than the first-build value so the retry actually changes the wrist demand (the yaw-only retry does **not** touch `wrist_pitch`) |
| `elbow_high_qp_lock` | `true` | Tighten the QP solver's joint limits into a window around the elbow-high seed during the approach/grasp so the QP can't drop the elbow into the wrist's hard stop (restored afterwards) |
| `elbow_high_qp_lock_margin_rad` | `0.4` | Half-width (rad) of the per-joint seed window on the **flex** side (the joint may move further into the elbow-high posture). Larger = more reach |
| `elbow_high_qp_lock_drop_margin_rad` | `0.15` | Half-width (rad) on the **drop** side — the direction that LOSES the elbow-high posture (elbow extending back down). Keep SMALL so the elbow is actually held high during the grasp. The drop direction per joint is inferred from the seed sign (negative seed drops by increasing toward 0). `0` = symmetric (use the flex margin, the old behaviour) |
| `elbow_high_qp_lock_joints` | `shoulder_pitch,shoulder_roll,shoulder_yaw,elbow_pitch,elbow_yaw` | Joints pinned into the seed window (default keeps the elbow high, wrists free). Add `wrist_pitch` to also hold the wrist off its lower stop |
| `elbow_high_guard_margin_rad` | `0.01` | Tighter guard margin applied **only to the wrists, only on the elbow-high path**, so the nose-down descent can use the `wrist_pitch` travel it physically has. The wrist stop is -0.785; with margin 0.01 the wrist may travel to -0.775 (the QP's own -0.785 lower limit is the hard backstop, so this is safe). Other joints keep the full margin. `0` disables. If the grasp still aborts at ~-0.775, the descent physically needs the full stop and this config can't complete it — re-teach the seed or use the other arm |
| `elbow_high_topdown` | `true` | (`topdown` mode) grasp **top-down** (gripper noses straight down) instead of the side/tilt orientation |
| `elbow_high_topdown_tilt_deg` | `90.0` | Absolute approach-axis tilt for the top-down grasp (90 = pure vertical) |
| `elbow_high_topdown_max_yaw_deg` | `90.0` | Yaw clamp about the vertical tool axis (wider than side-grasp; aligns jaws across a parallel object) |
| `elbow_high_topdown_retry_tilt_deg` | `55.0` | Shallower tilt used on the reach retry when a pure-vertical top-down saturates `wrist_pitch` at its lower stop |
| `ft_topic_left` / `ft_topic_right` | `/arm_6dof_left` / `_right` | Wrench topics |
| `ft_calib_left` / `ft_calib_right` | (auto) | Override calibration JSON paths |

## Tuning notes

- **Compliance is applied to every waist axis carrying the insertion direction.**
  The approach→grasp direction is decomposed onto the waist axes; any axis whose
  `|component|` exceeds `compliant_soften_threshold` gets the soft
  `insertion_stiffness`, the rest get `lateral_stiffness`. This matters for tilted
  grasps (e.g. 45°), where a single-soft-axis scheme would leave the large Z
  component stiff — that stiff axis fights the descent and its spring force reads
  as a false contact, stopping the arm short (grasp air).
- **Damping is set per axis** to `compliant_damping_ratio × 2√(K·M)` (with a
  `compliant_damping` floor), so each axis is near-critically damped regardless of
  its stiffness. A single scalar B leaves the stiffer axes underdamped, which
  shows up as **shaking in free space** and **bouncing on table contact**. If you
  still see shaking/bounce, raise `compliant_damping_ratio` (e.g. 1.5–1.8); if the
  descent feels sluggish, lower it toward 1.0.
- **One arm "runs away" / slams the table while the other is gentle?** Look at the
  `insert s=.. tcp_drop=.. lag=.. resist=..` telemetry. If `lag` goes **negative**
  (the actual TCP is *ahead* of the commanded slew) and `resist` is **negative**
  (a net force *along* the insertion, i.e. pushing the tool down), that arm's wrench
  has a residual downward bias (under-compensated gravity / a biased force-offset
  zero). On a soft axis (`K≈20 N/m`) even ~2 N of bias commands a ~10 cm equilibrium
  offset, so the admittance drives the tool down fast. Three guards make this safe,
  all applied to **both** arms (a healthy arm has positive `lag` and never trips
  them, so the good side is unaffected):
  - the **runaway guard** (primary): if `lag` goes more negative than
    `compliant_max_lag_m` (the tool is racing *ahead* of the command), stop and
    re-anchor immediately (result `reason: force_runaway_stop`).
  - the **overrun guard**: stop the instant the *actual* `tcp_drop` reaches
    `planned_depth + compliant_overshoot_m`, even if the slew command is still
    behind — the tool is never driven past the intended depth (`max_depth_overrun`).
  - `compliant_max_vel` is a HARD velocity backstop (default 0.20 m/s). Note it must
    stay **well above** the slew speed: the carrot published to the QP is
    `v_cmd × trans_lead_time` and the QP tracks only a fraction of it, so a healthy
    descent runs `v_cmd ≈ 0.1 m/s`. Capping it near the slew (e.g. 0.05) starves the
    carrot, the arm creeps, and the stall detector trips early → **grasp air**.

  These bound the *symptom*. To remove the *cause*, re-zero / re-run
  `calibrate_ft.py --arm <side>` for the biased arm (its descent `resist` should
  start and stay near 0 N until real contact, like the good arm).
- **One arm shakes but the other is smooth?** The gains, mass, damping and
  stiffness are **identical for both arms** — they are *not* set per side. The
  only things that differ left↔right are the F/T calibration file
  (`ft_calibration_<arm>.json`) and the physical sensor. So a side that shakes
  while the other is calm is almost always a **noisier wrench on that arm's
  sensor** leaking through into the admittance command (and getting amplified by
  the velocity-based carrot). The fix is more F/T conditioning, applied to both
  arms: lower `compliant_filter_alpha` (e.g. `0.35 → 0.25`, more smoothing) and/or
  raise `compliant_force_deadzone` (keep it `<` `compliant_contact_force_n`).
  First sanity-check the noisy side's calibration: the residuals in its
  `ft_calibration_<arm>.json` should be comparable to the other arm's (a few N at
  most). If they're much larger, re-run `calibrate_ft.py --arm <side>`.
- **Contact is detected by force OR stall.** A compliant arm yields on contact,
  so against a light object or a tilted graze the force along the insertion axis
  may never reach `compliant_contact_force_n` — but the TCP stops making progress
  (the throttle pins and `tcp_drop` plateaus). The stall detector trips when the
  TCP advances less than `compliant_stall_eps_m` within `compliant_stall_window_s`,
  which is what stops the long "struggle/hesitate" grind to full overshoot depth.
  If it stops *too early* (mid-descent), raise `compliant_stall_eps_m` or
  `compliant_stall_window_s`; if it grinds at the surface, lower them.
- **Speed:** `compliant_insert_speed_mps` (slew speed) and `compliant_otg_p_step`
  (how fast the QP chases the carrot) set the descent rate, bounded by
  `compliant_max_lag_m`. To go faster raise all three together; if you raise the
  speed past what the arm can track, the leash (below) just paces it back (no harm,
  but no speed-up).
- **Smooth leash (no shaking, strong carrot):** the commanded equilibrium is kept a
  CONSTANT lead (`0.9 × max_lag` ≈ 2 cm) ahead of the **actual** tool — this lead is
  the carrot that pulls the arm along the insertion axis. It reproduces the proven
  bang-bang behaviour (which *held* `lag` at ~`max_lag`, a strong carrot ~1.5 cm/s
  descent) but smoothly, so `lag` stays ~constant instead of bouncing off the
  `max_lag` ceiling (advance/hold/advance/hold = shaking). **Do not rate-limit the
  lead to `insert_speed`:** if the arm keeps pace, `s` and the tool advance together
  and the lead never builds, so the carrot collapses, the descent creeps, and the
  stall detector false-trips in the first second (grasp air). The lead is tied
  directly to the actual tool position; `max_vel` + the runaway/overrun guards bound
  any transient. A startup grace (`2 × stall_window`) ignores the initial admittance
  transient so the settle that follows it is not misread as a stall.
- **Hybrid insert (the proper fix for mostly-lateral inserts / shaking).** A tilted
  grasp makes the insert diagonal — part vertical (gravity-assisted, easy) and part
  horizontal (no assist, driven only by the soft spring). When the wrist cannot hold
  the full tilt (common on the **right** arm — it triggers the 22.5° reduced-tilt
  retry), the insertion axis becomes *nearly horizontal* (e.g. `dir=[+0.89,+0.11,−0.44]`)
  and the soft admittance spring has to **drag the whole arm sideways ~10 cm** — it does
  this in a jerky, stick-slip way (the "shake" you see), even though the supervisory
  `lag`/`throttle` telemetry looks healthy. This is **not** a per-side parameter problem
  (the gains are symmetric and correct); it's the wrong controller for a long lateral
  traverse. The fix is the **hybrid insert** (`compliant_final_descent_m`, default
  `0.04 m`): the long approach→grasp traverse is driven by the **position-controlled QP
  stream** (smooth, accurate, drives lateral motion firmly — exactly what the place
  program uses), and the F/T admittance takes over only for the **final few cm of
  descend-to-contact**, where the table-collision risk actually is. To restore the old
  full-span compliant behaviour set `compliant_final_descent_m: 0`. If contact is being
  missed because it happens earlier than the final span, raise `compliant_final_descent_m`;
  if the pre-descent itself risks hitting the table, lower it.
- **Post-contact bounce** is handled by re-anchoring: the instant contact is
  detected (or planned depth is reached), the hold target is snapped to the pose
  actually reached, the integrator velocity is zeroed, and the spring is stiffened
  to `compliant_hold_stiffness`. Without this the descent's commanded equilibrium
  sits ~`max_lag` *below the surface*, so the soft spring keeps driving the arm
  into the table and rebounds while the gripper closes — lifting the gripper off
  the object. If the arm still kicks on contact, lower `compliant_insert_speed_mps`
  and/or `compliant_contact_debounce` (so it trips at a lower force) and raise
  `compliant_hold_stiffness`.
- **Shaking** is also reduced by keeping `compliant_trans_lead_time` modest
  (≈0.08): it must be ≫ the loop period for the arm to move at all, but a large
  value amplifies velocity jitter into target jitter.
- If the arm **stops too early** (false contact): raise `compliant_contact_force_n`,
  or increase `compliant_min_insert_m`. If it **pushes too hard** before
  stopping: lower the threshold, lower `compliant_insert_speed_mps`, and/or lower
  `compliant_insertion_stiffness`.
- If the gripper **barely moves / stops short and grasps air**: the dominant
  cause is the published position *carrot* being too small. The admittance
  commands `v_cmd`, but the QP controller is driven by a target pose; if that
  target is only `v_cmd × loop_period` (≈ sub-millimetre for slow spring-driven
  motion) ahead of the live TCP it falls below the QP OTG resolution and the arm
  does not move — even though the admittance keeps reporting a healthy `v_cmd`.
  The fix is `compliant_trans_lead_time` (default `0.12 s`), which projects the
  carrot `v_cmd × lead_time` ahead (the same trick `rot_lead_time` already uses
  for rotation). If the arm still creeps, raise it toward `0.15-0.2`. Secondary
  causes: keep `compliant_damping` low (≈3), keep the lag throttle on
  (`compliant_max_lag_m`), keep `compliant_otg_p_step` ≈ 0.005 (not the fine QPIK
  step), and confirm `compliant_max_insert_m` ≥ `approach_dist` +
  `compliant_overshoot_m`. Watch the `insert s=.. tcp_drop=.. lag=..` log line and
  the result JSON's `traveled_m` (ACTUAL TCP drop) vs `planned_depth_m` — they
  should end up close, with `throttle_events` modest (not tens of thousands).
- If the F/T reads drift / push with no contact, the calibration is stale —
  re-run the calibrator (or `--drift-only`).

## Elbow-high reconfiguration (awkward object poses)

The arms are 7-DOF (redundant): a given TCP pose has a whole *family* of joint
solutions ("elbow circle"). The QP controller resolves that redundancy
**locally** — it tracks the **nearest** solution to the **current** joint state
each cycle, and there is **no nullspace / posture-bias parameter** exposed. So
starting from the home posture the arm stays in the **elbow-LOW** basin for the
whole approach + insert. For some object poses (classically: the object long
axis lying **parallel to the body**) the wrist orientation needed for the grasp
is simply **not reachable within joint limits inside the elbow-low basin** — the
wrist saturates, the orientation gate fails, and the grasp is skipped or grabs
air.

### Two strategies, selected by the object axis

Elbow-LOW and elbow-HIGH are **complementary** — each one grasps the short axis
for exactly the object orientation the other cannot, so the executor picks
between them by the object's long-axis angle from waist **+X**
(`elbow_high_proactive_angle_min_deg`, default **60°**, matching the diagonal
schedule's handoff `diagonal_schedule_angle_end_deg`):

| Object | short axis | jaws close along | strategy |
|---|---|---|---|
| **perpendicular / diagonal** (angle < 60°) | lateral (Y) → rotating | left–right | **elbow-LOW** (standard side-tilt + diagonal schedule closes on the short axis; wrist stays in range) |
| **parallel** (angle ≥ 60°) | fore-aft (X) | toward/away | **elbow-HIGH** (the taught seed's jaws already point ~fore-aft, so only a **small** jaw-align yaw is needed and `wrist_roll` stays near neutral) |

This split is why the elbow-high jaw alignment no longer saturates: elbow-high is
now used **only** for parallel-ish objects (≥60°), where the required alignment
yaw is small and comfortably inside `wrist_roll`. A **diagonal** object (45–60°)
would need a large align yaw that slams `wrist_roll` into its cramped `+1.3` stop,
so those stay on the elbow-low diagonal-schedule path instead. There is **no reach-based** elbow-high
trigger — routing a far-outboard *perpendicular* object into elbow-high was the
old failure mode (its seed jaws are ~90° wrong for it, so `wrist_roll` slammed
into its `+1.3` stop). Both the **proactive** and the **fallback** elbow-high
triggers are gated on the parallel bucket.

**Far-outboard perpendicular objects stay on elbow-low**, which handles their
reach via a **reach-based side-tilt reduction** (`reach_tilt_reduce`): a far
object (large `|Y|`) folds `elbow_pitch` into its `-2.61` stop because the 45°
nose-down tilt lever-arms the wrist even further out; backing the tilt off toward
level lets the elbow **extend** to reach it. The tilt scales from full at
`reach_tilt_start_abs_y_m` (0.18 m) down to `reach_tilt_scale_end` (0.4×) at
`reach_tilt_end_abs_y_m` (0.28 m).

> **Far-FORWARD objects (large tip `X`, ~0.5 m+) are a genuine reach limit**, not
> a tuning knob. Reaching that far forward with a downward grasp pushes the left
> arm to near-full extension: `wrist_pitch` hits its `+1.05` upper stop (and,
> with a diagonal object, the alignment yaw hits the `wrist_roll` `+1.30` stop).
> Backing the tilt off to relieve the joint only flattens the approach into the
> table. A forward-reach tilt reduction was tried and reverted for exactly this
> reason. Place objects within ~0.45 m forward for a reliable elbow-low grasp.

Elbow-low and elbow-high are **different IK branches separated by a
singularity**, so you cannot Cartesian-interpolate between them — the switch
**must** be a joint-space move. This path does exactly that:

1. The arm is **jointspace-moved into an elbow-HIGH seed posture** (`home →
   stage → transition waypoint → ready`), so the redundancy is now resolved in
   the elbow-high basin. The **stage** (`elbow_high_stage_*_joints`, the old
   arm-up home) is visited first so the sweep into the elbow-high basin starts
   from a clean posture regardless of the actual (tucked) home; the transition
   waypoint is the midpoint of stage & ready. The elbow-**low** path skips all of
   this and approaches straight from home. On return, the arm unwinds back
   through `transition → stage` before homing.
2. The grasp orientation is rebuilt for the elbow-high basin. The strategy is
   chosen by `--elbow-high-orientation` (default **`seed`**):
   - **`seed` (default, recommended):** anchor the grasp on the orientation the
     arm is **already in at the seed** — read live off TF right after the
     jointspace move. Because the orientation already matches, the QP approach
     only has to **translate** down onto the object, so the wrist **never leaves
     the range it is already in** (no saturation, no `目标超出跟踪限`). This is the
     "any grasp that works from elbow-high" path: the seed posture you taught is
     by construction reachable, so a grasp at that same orientation is too. The
     grasp angle is therefore **whatever tilt the seed points at** (e.g. a ~45°
     nose-down), not forced vertical.

     **Axis-adaptive jaw alignment** (`--elbow-high-align-jaws`, on by default):
     the jaws are rotated about waist Z to grasp the object's **short axis**.
     Since elbow-high is now selected only for the **parallel bucket** (angle ≥
     45°), the seed's fore-aft jaws are already close to the short axis, so this
     yaw is small and `wrist_roll` stays near neutral. This is an **absolute**
     alignment: the gripper jaw-closing axis is tool_x, and the code
     measures its current horizontal heading off the **hand-taught seed quat**
     then rotates so it lands on the short-axis direction (`long_yaw + 90°`).
     (A *relative* `+long_yaw` nudge — the earlier bug — is wrong because the
     seed's baseline jaw heading is arbitrary, so it left the jaws pointing near
     the **long** axis, i.e. gripping the object end-to-end.) This lives on the
     elbow-high path because there the ~top-down posture makes a waist-Z yaw map
     almost 1:1 onto **`wrist_roll`**, which *does* have range — unlike the
     elbow-low basin where the same rotation saturates the wrist. The gripper is
     symmetric under 180°, so the alignment has equivalents `θ + k·180°`; the
     search (`--elbow-high-seed-yaw-max-deg`, 90°) picks the one that keeps the
     predicted `wrist_roll` inside its **asymmetric** limits (left `[-1.65,
     +1.3]`, right `[-1.3, +1.65]`) with `--elbow-high-align-wrist-margin-rad` to
     spare. In practice the **left** arm rolls further negative and the **right**
     further positive, matching where each arm's roll has room. Backed off on a
     reach retry.

     **Jaw-flip retry** (`--elbow-high-jaw-flip-retry`, on by default): the
     `wrist_roll`→yaw mapping is only clean for a *truly* top-down seed; a seed
     taught a few degrees off vertical can make even the minimal alignment
     saturate `wrist_roll` on the arm's **cramped** stop (the QP then clamps the
     wrist and can't reach the orientation). When that happens the code retries
     the **same** short-axis grip from the 180°-flipped jaw line — an identical
     line reached from the mirrored side, which rolls the wrist onto its **roomy**
     side. This is what makes the alignment robust to an imperfect seed. NOTE:
     the definitive fix is still to re-teach a genuinely top-down seed whose
     natural jaw axis points along waist **Y** (so the common perpendicular
     object needs ~zero roll); the flip retry is a safety net, not a substitute.
   - **`topdown`:** force the gripper **straight down** (`--elbow-high-topdown-tilt-deg`,
     90°) with a wider yaw clamp (`--elbow-high-topdown-max-yaw-deg`). This needs
     `wrist_pitch` **headroom**: its range is **asymmetric** (`[-0.785, 1.05]`),
     so a pure 90° vertical at a far-forward object can bottom it out at the
     **lower** stop. When that trips the guard, the reach retry drops the tilt to
     `--elbow-high-topdown-retry-tilt-deg` (55°). Only use this if your seed
     genuinely points down with room to spare.
   - **`sidetilt`:** reuse the normal side/tilt grasp orientation (legacy).
   In all modes the rebuilt grasp is streamed by QP from the seed; because QP
   re-seeds off the current posture it **stays elbow-high**.
3. After lift, the return home is routed **back through the transition
   waypoint** so the arm doesn't sweep a large arc straight from elbow-high to
   home.

**Keeping the elbow high during the grasp (`--elbow-high-qp-lock`, default on).**
The QP resolves the 7-DOF redundancy locally and will happily let the **elbow
sag back down** as it streams to the grasp — and a low elbow drives `wrist_pitch`
straight into its **lower** hard stop (the guard then aborts and the arm returns
home). To prevent that, when the elbow-high path engages we tighten the QP
controller's own `joint_lower_limits` / `joint_upper_limits` into a **window
around the seed** for the proximal joints (`--elbow-high-qp-lock-joints`, default
shoulder + elbow) with half-width `--elbow-high-qp-lock-margin-rad` (0.6 rad),
leaving the wrists free to track orientation. The QP solver then **cannot leave
the elbow-high basin** while reaching for the object, so the wrist stays in
range. The window is **restored to the true hard limits** after the grasp (and in
the cleanup path), and the safety **guard always watches the real hard limits**.
The QP controller is re-activated after the seed reconfigure in both the
proactive and fallback flows, so it picks up the window on activation. Tune the
margin **down** if the elbow still sags, **up** if the arm can't reach; add
`wrist_pitch` to the lock-joints list to additionally pin the wrist off its stop.

**Why the elbow lock alone isn't always enough — `wrist_pitch` is orientation-
driven (`--elbow-high-guard-margin-rad`).** Pinning the elbow keeps the arm in
the high basin, but the *wrist* angle for a grasp is set mostly by the grasp
**orientation**, not by the elbow. A ~60–67° nose-down grasp at a far-forward
object needs `wrist_pitch ≈ -0.71` no matter where the elbow is — which is still
**inside** the -0.785 operational stop, but the default 0.1 rad guard margin
aborts at -0.685, *before* the arm can even reach the grasp. So on the elbow-high
path we give **only the wrists** a tighter guard margin
(`--elbow-high-guard-margin-rad`, 0.04) — the wrist gets to use the ~0.07 rad of
travel it physically has, while every other joint keeps the full protective
margin. The override is cleared after the grasp. If a far/steep grasp still
aborts on `wrist_pitch`, lower this toward 0.02, or reduce the grasp tilt
(re-teach the seed less nose-down). The motion at the stop is the slow compliant
descend-to-contact, and the orientation is held, so the wrist stays put near
-0.71 rather than running to the stop.

**The deeper lever — tilt the grasp up (`--elbow-high-seed-tilt-up-deg`).** A
nose-down grasp needs the wrist to flex *further* the deeper it descends, so a
steep seed-anchored grasp at a far object can run `wrist_pitch` out of travel
**mid-descent** (it reaches the stop before the fingers reach the object). The
intended fix is to make the grasp **less nose-down**: the seed-anchored orientation
is tilted UP toward level by `--elbow-high-seed-tilt-up-deg`, keeping the
heading, which directly reduces the `wrist_pitch` the descent needs. On a reach
retry (after the wrist saturated) it uses the larger
`--elbow-high-seed-retry-tilt-up-deg`. Both default to **0** (off): in practice
`wrist_pitch` on this robot is driven by the **reach** from the seed (the seed
parks the gripper far to one side with the wrist already near its stop; sweeping
to a centered object eats the rest of the travel), not by the tool tilt, so
tilting the tool does not free the wrist. The reliable fix is to re-teach the
seed with the gripper over the work area and `wrist_pitch` near neutral (see
"Re-teaching the seed" below).

**Holding the elbow up — asymmetric QP window
(`--elbow-high-qp-lock-drop-margin-rad`).** The QP lock pins the proximal joints
into a window around the seed so the solver can't drop the elbow. The window is
**asymmetric**: `--elbow-high-qp-lock-margin-rad` (0.4) bounds the **flex** side
(the arm may move further into the elbow-high posture), while
`--elbow-high-qp-lock-drop-margin-rad` (0.15) bounds the **drop** side (the
direction that loses the posture — the elbow extending back down). The drop
direction per joint is inferred from the seed sign (a negative seed drops by
increasing toward 0). Keep the drop margin small so the elbow is actually held
high; if the arm then can't reach the object it returns a large position error
rather than sagging — the honest signal that this seed can't reach this object
without dropping the elbow, i.e. re-teach the seed. The full per-joint vector is
logged after each phase (`[JOINT-LIMIT] <arm> <phase> per-joint: ...`) so you can
see `elbow_pitch` staying in its window. `0` = symmetric (the old behaviour,
which let the elbow drift ~0.6 rad and was too loose to hold it up).

**Orientation leash (large reorientations).** In `seed` mode the reorientation
is tiny by construction (the grasp keeps the seed's own orientation), so the
approach is essentially pure translation. The leash mainly matters for `topdown`
mode, where switching to vertical from the seed can be a big reorientation (the
seed gripper does not point straight down). The QP pure-pursuit used to pace
orientation by the **position** fraction,
so on a short path it commanded almost the full rotation in one step and blew the
controller's orientation tracking bound (`目标超出跟踪限`, `dis_ori`), stalling the
wrist into its stop. The streamer now **leashes** the orientation carrot to a
bounded lead (`ori_lead_rad`, ~0.6 rad) over the *measured* orientation and ramps
the final hold the same way, so any size reorientation slews smoothly. Normal
small-rotation moves are unchanged (the position fraction is still the binding
term).

**Triggering (both modes shipped):**
- **Fallback (default on):** fires after the normal elbow-low approach *and* the
  reduced-tilt retry both fail — either as an **orientation-only miss** (TCP
  reached, wrist couldn't hold the tilt) **or** as a **joint-limit guard abort**
  (a wrist joint, typically `wrist_pitch`, saturated at its hard stop mid-
  approach — the clearest "elbow-low can't serve this pose" signal, common for
  objects lying parallel to the body). Zero impact on poses that already work;
  if elbow-high also fails it falls through to the existing best-effort grasp.
- **Proactive (`--elbow-high-proactive`, default off):** chooses elbow-high up
  front when the detected object long-axis angle from waist +X is
  `≥ --elbow-high-proactive-angle-min-deg` (≈ parallel to the body).

**Seed safety (auto-clamp).** The hand-taught seed is clamped so every joint
sits ≥ `--elbow-high-clamp-margin-rad` (0.15) inside its limit. The shipped
left seed has `shoulder_yaw = −2.964`, which is *past* the −2.96 operational
limit; the clamp pulls it to −2.81 (and nudges `wrist_pitch`) so the QP seed
never starts at/over a hard stop.

**A seed must keep headroom on EVERY joint.** A seed with a joint parked at a
limit is unusable. The default `seed` orientation mode largely sidesteps this —
it grasps at the seed's own (reachable) orientation, so it never demands a
vertical wrist the seed can't provide. `topdown` mode is stricter: the original
shipped seed has `shoulder_yaw` at its stop *and* `wrist_pitch` at −0.64 (≈0.06
rad from the guard), so a top-down grasp of a far-forward object drives
`wrist_pitch` past its hard −0.785 stop and aborts no matter how tilt/yaw is
tuned. Either way, a roomier seed is better — capture one directly from a posture
you can physically reach:

1. Hand-drag the arm (`xarm_sdk/demo/tianyi/11_gravity_compensation_drag.py`)
   into an elbow-high posture with the gripper pointing **straight down** above
   where the object actually sits (~0.5 m forward), keeping every joint well
   inside its limits.
2. Capture it (no motion is commanded; warns about any near-limit joint):

```bash
python3 -m compliant_grasp_execute.main --capture-elbow-high-seed --arm left
```

   This writes `elbow_high_ready_left_joints` (or `_right`) into `config.yaml`.
3. Re-run normally. Because the seed already points down above the object, the
   QP approach is a small motion and the wrist stays in range.

**Transition safety.** Jointspace interpolation is **per-joint**, so a move
between two in-limit joint configs keeps every intermediate value in-limits (no
joint-limit fault). The intermediate waypoint exists to shrink the **Cartesian
sweep** (table/body clearance), not for joint limits. Note the jointspace action
exposes **no speed knob**, so the reconfiguration runs at the controller's
default speed — **watch the first reconfiguration** on a new seed/waypoint to
confirm clearance, and override `elbow_high_transition_{left,right}_joints` with
a hand-taught bridge if the auto-midpoint sweeps badly.

## MCP server (agent integration)

`mcp_server.py` exposes the compliant grasp phase to an agent via MCP, built on
[`fastmcp`](https://github.com/jlowin/fastmcp) and following the same pattern as
`grasp_pose_grasp_execute/mcp_server.py`. Each tool call runs
`python3 -m compliant_grasp_execute.main` as a **subprocess** (clean ROS2
lifecycle), serialised by a run-lock so two calls never fight over the hardware,
and returns `{"state": "succeed"|"failed", "msg", "result", ...}`.

Requires `pip install fastmcp`.

Run it (from the repo root):

```bash
# Streamable-HTTP on 0.0.0.0:8005/  (default)
python3 -m compliant_grasp_execute.mcp_server

# Or stdio (Cursor / Claude Desktop MCP config)
MCP_TRANSPORT=stdio python3 -m compliant_grasp_execute.mcp_server
```

Env overrides: `MCP_TRANSPORT`, `MCP_HOST`, `MCP_PORT` (default **8005**, so it
can run alongside the grasp server on 8003 and the place server on 8004),
`MCP_PATH`.

Tools exposed:

- `grasp_object(prompt, arm, pipeline_version, motion_strategy,
  release_on_finish, elbow_high, handoff_out, dry_run, timeout_sec, extra_args)`
  — detect the object and run the full compliant pick (approach → F/T
  descend-to-contact → close → lift → home). Pass `release_on_finish=False` to
  keep holding for a following place phase. `elbow_high` is
  `proactive`|`always`|`on`|`off` (top grasp for parallel-to-body objects);
  `None` keeps the config default. Any other flag can be forwarded via
  `extra_args`, e.g. `["--elbow-high-seed-tilt-up-deg", "12"]`.
- `read_handoff(path)` — read `/tmp/grasp_handoff.json`.

Note on `prompt`: the CLI's `--prompt` is *append*, so the server writes a
merged temp config (a copy of `config.yaml` with `prompt` replaced) and passes
it via `--config` — this cleanly **replaces** the configured object instead of
appending to it. Leave `prompt=""` to use the object in `config.yaml`.

Register with an HTTP MCP client:

```json
{ "mcpServers": { "compliant_grasp": { "url": "http://<robot-host>:8005/" } } }
```

Or stdio:

```json
{
  "mcpServers": {
    "compliant_grasp": {
      "command": "python3",
      "args": ["-m", "compliant_grasp_execute.mcp_server"],
      "cwd": "/home/ubuntu/niu",
      "env": { "MCP_TRANSPORT": "stdio" }
    }
  }
}
```

Per-run logs land in `logs/mcp/compliant_grasp_execute/`. Use `dry_run: true` to
test detection/planning without moving the robot.

## Safety

- The compliant insert refuses to trust an **identity fallback** silently: it
  logs a loud warning if the calibration JSON is missing. Do not run a real
  grasp without a valid calibration for the active arm.
- The recovery-aware joint-limit guard still supervises every motion phase,
  including the compliant descent.
- The compliant insert **aborts** if the arm makes no forward progress before
  reaching `compliant_min_insert_m` within the stall window after the startup
  grace (`reason="blocked_no_progress"`). This is the case where a wrist joint
  is saturated against its hard stop (e.g. the right `wrist_roll` at -1.3 from a
  bad elbow-high seed) or a residual force is pushing the tool backward
  (`tcp_drop` goes negative). Previously the insert ground against the stop
  until the deadline (or a manual stop); it now bails out promptly and logs the
  cause so the seed/orientation can be re-taught instead of retried blindly.
