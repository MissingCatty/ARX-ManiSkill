#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${CUROBO_PYTHON:-$(dirname "$ROOT")/curobo/.venv/bin/python}"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUROBO_GPU:-1}"
"$PYTHON" scripts/curobo_arx/grasp_input.py
if [[ -z "${GRASP_FIXED_SELECTION:-}" ]]; then
  bash scripts/curobo_arx/run_zerograsp.sh
fi
"$PYTHON" scripts/curobo_arx/prepare.py --output runs/arx_grasp --fine-gripper
"$PYTHON" scripts/curobo_arx/grasp_feedback_execute.py
