#!/usr/bin/env bash
set -euo pipefail
cd /home/zhaoyechong/projects/COOLAS-ARX
source scripts/curobo_arx/static_pick/env.sh
ARX_STATIC_PY=/home/zhaoyechong/projects/ARX-ManiSkill/modules/curobo/.venv/bin/python
export GRASP_VIDEO_DIR=/workspace/COOLAS-ARX/runs/arx_dynamic_pick/pole_00/frames
test -f runs/arx_dynamic_pick/pole_00/baseline_path_check.json
test -f runs/arx_dynamic_pick/pole_00/zerograsp/object_1.grasp.npy
set +e
"$ARX_STATIC_PY" scripts/curobo_arx/static_pick/run_stage.py pole_00 grasp_execute.py
ARX_STATIC_RC=$?
set -e
"$ARX_STATIC_PY" scripts/curobo_arx/static_pick/finalize.py pole_00
if [[ -f runs/arx_dynamic_pick/pole_00/frames/frames.json ]]; then
 "$ARX_STATIC_PY" scripts/curobo_arx/dynamic_pick/encode_walltime.py runs/arx_dynamic_pick/pole_00
 "$ARX_STATIC_PY" - <<'PYCODE'
from pathlib import Path
import json
p=Path('runs/arx_dynamic_pick/pole_00')
old=p/'dynamic_obstacle_grasp.mp4';new=p/'static_pole_grasp.mp4'
old.rename(new)
r=json.loads((p/'video_report.json').read_text());r['video']=str(new.resolve());(p/'video_report.json').write_text(json.dumps(r,indent=2))
PYCODE
fi
exit "$ARX_STATIC_RC"
