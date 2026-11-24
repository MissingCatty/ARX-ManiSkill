#!/usr/bin/env bash
set -euo pipefail
cd /home/zhaoyechong/projects/COOLAS-ARX
source scripts/curobo_arx/static_pick/env.sh
ARX_STATIC_PY=/home/zhaoyechong/projects/ARX-ManiSkill/modules/curobo/.venv/bin/python
export GRASP_VIDEO_DIR=/workspace/COOLAS-ARX/runs/arx_dynamic_pick/static_00/frames
test -f runs/arx_dynamic_pick/static_00/baseline_path_check.json
test -f runs/arx_dynamic_pick/static_00/zerograsp/object_1.grasp.npy
set +e
"$ARX_STATIC_PY" scripts/curobo_arx/static_pick/run_stage.py static_00 grasp_execute.py
ARX_STATIC_RC=$?
set -e
"$ARX_STATIC_PY" scripts/curobo_arx/static_pick/finalize.py static_00
if [[ -f runs/arx_dynamic_pick/static_00/frames/frames.json ]]; then
 "$ARX_STATIC_PY" scripts/curobo_arx/dynamic_pick/encode_walltime.py runs/arx_dynamic_pick/static_00
fi
exit "$ARX_STATIC_RC"
