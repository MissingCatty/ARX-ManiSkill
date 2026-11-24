#!/usr/bin/env bash
# Explicit device precheck; any failure stops before Python or CUDA initialization.
set -euo pipefail
ROOT=/home/zhaoyechong/projects/ARX-ManiSkill/modules/curobo_adapter
if [[ "$#" -lt 2 ]]; then
    echo 'Usage: bash run_isolated.sh GPU-UUID script.py [arguments...]' >&2
    exit 64
fi
TASK_GPU_UUID="$1"
shift
source "$ROOT/environment.sh"
TASK_GPU_FREE="$(nvidia-smi -i "$TASK_GPU_UUID" --query-gpu=memory.free --format=csv,noheader,nounits)"
if [[ "$TASK_GPU_FREE" -lt 8192 ]]; then
    echo "Resource precheck failed on $TASK_GPU_UUID: ${TASK_GPU_FREE} MiB free" >&2
    exit 75
fi
export CUDA_VISIBLE_DEVICES="$TASK_GPU_UUID" PYTHONDONTWRITEBYTECODE=1
exec "$ROOT/.venv/bin/python" "$@"
