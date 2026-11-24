#!/usr/bin/env bash
set -euo pipefail
ARX_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
ARX_EGAD_CAMPAIGN="$(realpath -- "${1:?Usage: start_egad49.sh CAMPAIGN [--detach]}")"
shift
export ARX_SIM_CONTAINER="${ARX_SIM_CONTAINER:-arx-egad49-lilo-gpu3}"
export ARX_SIM_IMAGE="${ARX_SIM_IMAGE:-localhost/arx-isaac-warm:6.0.1}"
# A single physical 4090 is CUDA/Vulkan device 0 inside this owned container.
export ARX_SIM_DEVICE="${ARX_SIM_DEVICE:-3}" ARX_SIM_GPU="${ARX_SIM_GPU:-0}"
export ARX_SIM_ENTRYPOINT=scripts/curobo_arx/lilo_vla/egad_start_sim.py
export ARX_REPRO_REFERENCE="${ARX_EGAD_CAMPAIGN}/runtime/historical_reference"
export ARX_TABLETOP_CATALOG="${ARX_REPRO_REFERENCE}/catalog.json"
export ARX_HISTORICAL_SETUP="${ARX_REPRO_REFERENCE}/initial_setup.json"
export ARX_CUROBO_VALIDATION_DIR="${ARX_EGAD_CAMPAIGN}/runtime/mailbox"
export ARX_EGAD_EFFECTIVE_CONFIG="${ARX_EGAD_CAMPAIGN}/runtime/effective_config.json"
exec bash "${ARX_REPO_ROOT}/scripts/curobo_arx/start_sim_container.sh" "$@" \
  --task grasp --webrtc-port 49210 --run-dir "${ARX_EGAD_CAMPAIGN}/runtime/simulator_runtime"
