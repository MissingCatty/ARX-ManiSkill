#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${CUROBO_PYTHON:-$(dirname "$ROOT")/curobo/.venv/bin/python}"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUROBO_GPU:-1}"
"$PYTHON" scripts/curobo_arx/grasp_input.py
"$PYTHON" scripts/curobo_arx/check_replayed_observation.py
# ZeroGrasp inference and both cuRobo warmups finish before the first arm command.
if [[ -z "${GRASP_FIXED_SELECTION:-}" && "${GRASP_REUSE_NETWORK_INPUT:-0}" != "1" ]]; then
  bash scripts/curobo_arx/run_zerograsp.sh
fi
"$PYTHON" scripts/curobo_arx/prepare.py --output runs/arx_grasp --fine-gripper
"$PYTHON" scripts/curobo_arx/grasp_orientation_ablation_execute.py
