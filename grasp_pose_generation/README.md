# grasp_pose_generation

Standalone project to generate grasp pose(s) from live camera data, separated
from `bottle_cup_pour_place` execution flow.

It runs only:

1. RGB + depth capture from ROS topics
2. perception API call (segment / yolo)
3. 3D pose estimation + TF (`head_roll_link -> waist_yaw_link`)
4. tip->tcp conversion and grasp orientation policy

No arm motion is executed here.

## Make It Portable For Colleagues

You can copy only this folder (`grasp_pose_generation/`) to another machine.

This version is now **self-contained** for project code:
- no imports from sibling folders `tactile_grasp/` or `detection_only/`
- camera YAML is bundled in `assets/poseestimator/`
- both current/accelerated segmentation paths are vendored under `internal/`

Still required from runtime environment:
- ROS 2 Python stack (`rclpy`, messages, tf2)
- `cv_bridge`
- `xarm_sdk`
- reachable segmentation backend URL (`--base-url`)

Then run:

```bash
./grasp_pose_generation/run.sh --doctor
```

`--doctor` prints path checks and exits.

It supports two local segmentation backends:

- `current`: bundled perception pipeline
- `accelerated`: bundled fast segmentation style (from prior `test_call` logic),
  then the same 3D pose + TF + tip->tcp conversion

## Run

From workspace root:

```bash
python3 -m grasp_pose_generation.main \
  --prompt bottle \
  --pipeline-version current \
  --mode segment \
  --orientation-policy grasp \
  --arm right \
  --json-out /tmp/grasp_pose.json
```

or with portable launcher:

```bash
./grasp_pose_generation/run.sh \
  --prompt bottle \
  --pipeline-version accelerated \
  --mode segment \
  --orientation-policy grasp \
  --arm right
```

Multiple prompts:

```bash
python3 -m grasp_pose_generation.main \
  --prompt bottle --prompt cup \
  --pipeline-version accelerated \
  --segment-confidence 0.3
```

Latency benchmark (reuses one `PerceptionTool` instance):

```bash
python3 -m grasp_pose_generation.main \
  --prompt bottle \
  --pipeline-version current \
  --bench-runs 10 \
  --sam3-upload-format jpeg \
  --sam3-jpeg-quality 85
```

Compare both versions in one command:

```bash
python3 -m grasp_pose_generation.main \
  --prompt cup \
  --pipeline-version compare \
  --bench-runs 20 \
  --json-out /tmp/grasp_pose_compare.json
```

Per-run output now includes stage timing breakdown:

- `stage_timing_sec.capture_sec`
- `stage_timing_sec.segmentation_sec`
- `stage_timing_sec.pose3d_sec`
- `stage_timing_sec.tf_sec`

For bench/compare runs, aggregated stage stats are reported in
`stage_stats_sec` (`avg`, `min`, `max` per stage).

## Colleague Hand-off Checklist

1. Copy `grasp_pose_generation/` folder.
2. Ensure runtime has ROS2 + `xarm_sdk` + `cv_bridge`.
3. Validate:
   - `./grasp_pose_generation/run.sh --doctor`
4. Run detection:
   - `./grasp_pose_generation/run.sh --prompt bottle --pipeline-version accelerated`

## Notes

- The CLI uses a vendored local object-pose pipeline (`internal/object_pose_pipeline.py`).
- Upload format is configurable via CLI and forwarded to environment variables:
  - `SAM3_UPLOAD_FORMAT=jpeg|png`
  - `SAM3_JPEG_QUALITY=1..95`
- Result JSON includes:
  - `pose_tip_waist_yaw_link_rpy`
  - `pose_tcp_waist_yaw_link_pose7`
  - `current_tcp_pose7`
  - `total_time_sec`

