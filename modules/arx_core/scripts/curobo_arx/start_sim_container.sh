#!/usr/bin/env bash
set -euo pipefail
ARX_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ARX_CONTAINER_ARGS=()
ARX_DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) ARX_DRY_RUN=1; shift ;;
    --detach) ARX_CONTAINER_ARGS+=(--detach); shift ;;
    *) break ;;
  esac
done
ARX_SIM_GPU_UUID="$(python3 "$ARX_REPO_ROOT/scripts/curobo_arx/gpu_policy.py" resolve "${ARX_SIM_DEVICE:-${ARX_SIM_GPU:-3}}")"
ARX_SIM_GPU_ARGS=(--device "nvidia.com/gpu=$ARX_SIM_GPU_UUID" -e NVIDIA_VISIBLE_DEVICES=void)
ARX_EXTRA_RUN_MOUNTS=()
ARX_RUNTIME_CACHE_MOUNTS=()
if [[ -n "${ARX_SIM_KIT_CACHE:-}" ]]; then
  [[ -d "$ARX_SIM_KIT_CACHE" ]] || { echo 'Registered Kit cache is missing' >&2; exit 1; }
  ARX_RUNTIME_CACHE_MOUNTS=(-v "$ARX_SIM_KIT_CACHE:/isaac-sim/kit/cache:rw")
fi
if [[ -n "${ARX_SIM_EXTRA_RUN_ROOT:-}" ]]; then
  case "$ARX_SIM_EXTRA_RUN_ROOT" in
    /coolas/data/*) ;;
    *) echo 'Extra run root must be a dedicated directory under /coolas/data' >&2; exit 1 ;;
  esac
  [[ -d "$ARX_SIM_EXTRA_RUN_ROOT" ]] || { echo 'Create the registered extra run root before launching' >&2; exit 1; }
  ARX_EXTRA_RUN_MOUNTS=(-v "$ARX_SIM_EXTRA_RUN_ROOT:$ARX_SIM_EXTRA_RUN_ROOT:rw")
fi
ARX_CONTAINER_COMMAND=(podman run --rm "${ARX_CONTAINER_ARGS[@]}"
  --name "${ARX_SIM_CONTAINER:-arx-sim-zhaoyechong-gpu${ARX_SIM_GPU:-3}}"
  --user 0 "${ARX_SIM_GPU_ARGS[@]}" --security-opt=label=disable
  --network=host --shm-size=2g
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=N -e OMNI_KIT_ALLOW_ROOT=1
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e "CUDA_VISIBLE_DEVICES=$ARX_SIM_GPU_UUID"
  -e ARX_SIM_GPU=0 -e "ARX_SIM_GPU_UUID=$ARX_SIM_GPU_UUID"
  -e PYTHONUNBUFFERED=1
  -e "ARX_SIM_CAMERA_ROLES=${ARX_SIM_CAMERA_ROLES:-head}"
  -e "ARX_CUROBO_VALIDATION_DIR=${ARX_CUROBO_VALIDATION_DIR:?Set mailbox}"
  -e "ARX_TABLETOP_CATALOG=${ARX_TABLETOP_CATALOG:?Set catalog}"
  -e "ARX_HISTORICAL_SETUP=${ARX_HISTORICAL_SETUP:?Set initial state}"
  -e "ARX_REPRO_REFERENCE=${ARX_REPRO_REFERENCE:?Set reference}"
  -e "ARX_EGAD_EFFECTIVE_CONFIG=${ARX_EGAD_EFFECTIVE_CONFIG:-}"
  -e "ARX_REPRO_OPS_PORT=${ARX_REPRO_OPS_PORT:-}"
  -e "ARX_REPRO_ROS_DOMAIN=${ARX_REPRO_ROS_DOMAIN:-}"
  -e "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-32}"
  -e PYTHONPATH=/workspace/COOLAS-ARX/shared:/workspace/COOLAS-ARX/arx-ops:/workspace/COOLAS-ARX/arx-sim
  -e LD_LIBRARY_PATH=/isaac-sim/exts/isaacsim.ros2.core/jazzy/lib
  -e ISAAC_SIM_ROOT=/isaac-sim -e ARX_SIM_ASSET_ROOT=/workspace/COOLAS-ARX/asset
  -v "${ARX_REPO_ROOT}:/workspace/COOLAS-ARX:rw"
  -v "${ARX_REPO_ROOT}:${ARX_REPO_ROOT}:rw"
  -v "${ARX_REPO_ROOT}/asset:/workspace/asset:ro"
  "${ARX_EXTRA_RUN_MOUNTS[@]}"
  "${ARX_RUNTIME_CACHE_MOUNTS[@]}"
  -w /workspace/COOLAS-ARX --entrypoint /bin/bash
  "${ARX_SIM_IMAGE:-nvcr.io/nvidia/isaac-sim:6.0.1}"
  /isaac-sim/python.sh "${ARX_SIM_ENTRYPOINT:-scripts/curobo_arx/static_pick/start_sim.py}" "$@")
if [[ "$ARX_DRY_RUN" == 1 ]]; then
  printf '%q ' "${ARX_CONTAINER_COMMAND[@]}"
  printf '\n'
else
  exec "${ARX_CONTAINER_COMMAND[@]}"
fi
