#!/usr/bin/env bash
set -euo pipefail
ARX_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
ARX_CLEAR_OUTPUT="$(realpath -- "${1:?Usage: start_clear_bottle.sh OUTPUT [--detach|--dry-run]}")"
shift
export ARX_REPRO_REFERENCE="${ARX_CLEAR_OUTPUT}/historical_reference"
export ARX_CUROBO_VALIDATION_DIR="${ARX_CLEAR_OUTPUT}/mailbox"
export ARX_TABLETOP_CATALOG="${ARX_REPRO_REFERENCE}/catalog.json"
export ARX_HISTORICAL_SETUP="${ARX_REPRO_REFERENCE}/initial_setup.json"
[[ -f "${ARX_REPRO_REFERENCE}/setup.json" ]] || { echo 'Run init_clear_bottle.py first.' >&2; exit 1; }
exec bash "${ARX_REPO_ROOT}/scripts/curobo_arx/start_sim_container.sh" "$@" \
  --task grasp --webrtc-port 49210 --run-dir "${ARX_CLEAR_OUTPUT}/simulator_runtime"
