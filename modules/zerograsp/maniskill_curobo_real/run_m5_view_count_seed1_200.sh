#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="maniskill_curobo/envs/maniskill_curobo/bin/python"
CANDIDATES="maniskill_curobo_real/runs/m4_full_depth_candidates_seed1_200"
VIEW4_ROOT="maniskill_curobo_real/runs/pickclutter_m5_instance_esdf_table_4view_seed1_200"
VIEW5_ROOT="maniskill_curobo_real/runs/pickclutter_m5_instance_esdf_table_5view_seed1_200"
MIN_FREE_GPU_MIB="${MIN_FREE_GPU_MIB:-6000}"
PID_FILE="${PID_FILE:-maniskill_curobo_real/runs/pickclutter_m5_view_count_seed1_200_launcher/run.pid}"

mkdir -p "$(dirname "$PID_FILE")"
printf '%s\n' "$$" > "$PID_FILE"

wait_for_gpu() {
  while true; do
    local free_mib
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_GPU_MIB )); then
      echo "[gpu] ${free_mib} MiB free; starting benchmark"
      return
    fi
    echo "[gpu] only ${free_mib:-unknown} MiB free; waiting for ${MIN_FREE_GPU_MIB} MiB"
    sleep 60
  done
}

run_view_count() {
  local view_count="$1"
  local output_root="$2"
  "$PYTHON" -m maniskill_curobo_real.run_multiview_instance_benchmark \
    --env-id PickClutterYCB-v1 \
    --seed-start 1 \
    --seed-end 200 \
    --view-count "$view_count" \
    --stage m5-instance-esdf-table \
    --output-root "$output_root" \
    --reuse-candidate-root "$CANDIDATES" \
    --reuse-existing
}

wait_for_gpu
run_view_count 4 "$VIEW4_ROOT"
run_view_count 5 "$VIEW5_ROOT"

"$PYTHON" -m maniskill_curobo_real.analyze_m5_view_count_benchmark \
  --view4-root "$VIEW4_ROOT" \
  --view5-root "$VIEW5_ROOT"
