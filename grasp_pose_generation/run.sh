#!/usr/bin/env bash
set -euo pipefail

# Portable launcher for grasp_pose_generation.
# Usage:
#   ./run.sh --prompt bottle --pipeline-version accelerated --bench-runs 5

exec python3 -m grasp_pose_generation.main "$@"

