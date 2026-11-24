#!/usr/bin/env bash
# Reuse identical SDF queries without changing collision costs or constraints.
set -euo pipefail
ARX_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ARX_CUROBO_SOURCE="${1:-${ARX_REPO_ROOT}/../curobo/source}"
ARX_CUROBO_PIN=8e734f3ced1df898990bcd92de40abce475907db
if [[ "$(git -C "$ARX_CUROBO_SOURCE" rev-parse HEAD)" != "$ARX_CUROBO_PIN" ]]; then
  echo "cuRobo revision differs from the validated revision: $ARX_CUROBO_PIN" >&2
  exit 1
fi
ARX_LATEST_PATCH="${ARX_REPO_ROOT}/scripts/curobo_arx/patches/curobo_swept_mesh_bounds.patch"
if git -C "$ARX_CUROBO_SOURCE" apply --reverse --check "$ARX_LATEST_PATCH" 2>/dev/null; then
  echo 'cuRobo performance patch stack already applied.'
  exit 0
fi
for ARX_PATCH_NAME in curobo_sweep_current_sdf.patch curobo_swept_mesh_bounds.patch; do
  ARX_SWEEP_PATCH="${ARX_REPO_ROOT}/scripts/curobo_arx/patches/${ARX_PATCH_NAME}"
  if git -C "$ARX_CUROBO_SOURCE" apply --reverse --check "$ARX_SWEEP_PATCH" 2>/dev/null; then
    echo "cuRobo patch already applied: $ARX_PATCH_NAME"
  else
    git -C "$ARX_CUROBO_SOURCE" apply --check "$ARX_SWEEP_PATCH"
    git -C "$ARX_CUROBO_SOURCE" apply "$ARX_SWEEP_PATCH"
    echo "Applied cuRobo patch: $ARX_PATCH_NAME"
  fi
done
