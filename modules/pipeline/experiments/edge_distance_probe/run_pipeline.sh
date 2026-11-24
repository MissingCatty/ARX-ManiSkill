#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/home/zhaoyechong/projects/ARX-ManiSkill/modules/pipeline/experiments/edge_distance_probe
TASK_RUN="$TASK_ROOT/runs/three_objects_20261123_01"
TASK_GPU=GPU-d9e4f8a1-c4f3-232a-4ceb-a9dde26aa45b
export CUDA_VISIBLE_DEVICES="$TASK_GPU" ARX_SIM_GPU="$TASK_GPU" ARX_PLANNER_GPU="$TASK_GPU" ARX_NEURAL_GPU="$TASK_GPU" LILO_GPU="$TASK_GPU"
source /home/zhaoyechong/projects/ARX-ManiSkill/modules/arx_core/scripts/curobo_arx/environment.sh
exec >"$TASK_RUN/runtime.log" 2>&1
/home/zhaoyechong/projects/ARX-ManiSkill/modules/maniskill/.venv/bin/python "$TASK_ROOT/run_probe.py" capture
/home/zhaoyechong/projects/ARX-ManiSkill/modules/curobo/.venv/bin/python "$TASK_ROOT/run_probe.py" plan
