#!/usr/bin/env bash
# Current fresh-inference bottle workflow; see ../README.md.
set -euo pipefail
ARX_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${ARX_REPO_ROOT}/scripts/curobo_arx/environment.sh"
ARX_CLEAR_OUTPUT="$(realpath -- "${1:-${ARX_CLEAR_REPRO_OUTPUT:?Usage: record_clear_bottle.sh OUTPUT [--mailbox PATH] [--audit]}}")"
if [[ $# -gt 0 ]]; then shift; fi
ARX_CLEAR_TRANSPORT="${ARX_CLEAR_MAILBOX:-${ARX_CLEAR_OUTPUT}/mailbox}"
ARX_CLEAR_AUDIT="${ARX_RESEARCH_AUDIT}"
if [[ -e "${ARX_CLEAR_OUTPUT}/fresh_rgbd.log" || -e "${ARX_CLEAR_OUTPUT}/grasp_report.json" ]]; then
  echo 'This run already contains an attempt. Initialize a new output directory.' >&2
  exit 1
fi
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mailbox) ARX_CLEAR_TRANSPORT="$(realpath -- "${2:?Missing mailbox}")"; shift 2 ;;
    --audit) ARX_CLEAR_AUDIT=1; shift ;;
    --no-audit) ARX_CLEAR_AUDIT=0; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$ARX_CLEAR_AUDIT" in
  0) ARX_CLEAR_AUDIT_ARGS=(--no-audit) ;;
  1) ARX_CLEAR_AUDIT_ARGS=(--audit) ;;
  *) echo 'ARX_RESEARCH_AUDIT must be 0 or 1' >&2; exit 2 ;;
esac
cd "$ARX_REPO_ROOT"
bash scripts/curobo_arx/apply_curobo_performance_patch.sh
"$ARX_NEURAL_PYTHON" scripts/curobo_arx/static_pick/reproduce_setup.py \
  --output "$ARX_CLEAR_OUTPUT" --mailbox "$ARX_CLEAR_TRANSPORT"
exec "$ARX_NEURAL_PYTHON" scripts/curobo_arx/static_pick/reproduce_record.py \
  --output "$ARX_CLEAR_OUTPUT" --reference-run "${ARX_CLEAR_REFERENCE:-${ARX_REPO_ROOT}/scripts/curobo_arx/static_pick/reference}" \
  --curobo-python "$CUROBO_PYTHON" --neural-gpu "$ARX_NEURAL_GPU" --planner-gpu "$ARX_PLANNER_GPU" \
  --trajopt-policy "$GRASP_TRAJOPT_POLICY" --mailbox "$ARX_CLEAR_TRANSPORT" "${ARX_CLEAR_AUDIT_ARGS[@]}"
