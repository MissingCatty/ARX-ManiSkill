#!/usr/bin/env bash
set -euo pipefail
cd /home/zhaoyechong/projects/COOLAS-ARX
source scripts/curobo_arx/static_pick/env.sh
ARX_STATIC_PY=/home/zhaoyechong/projects/ARX-ManiSkill/modules/curobo/.venv/bin/python
"$ARX_STATIC_PY" - <<'PY'
from pathlib import Path
import time
p=Path('runs/arx_dynamic_pick/mailbox/ready.json');end=time.monotonic()+240
while not p.exists():
 if time.monotonic()>end:raise TimeoutError('Side sim not ready')
 time.sleep(1)
PY
"$ARX_STATIC_PY" scripts/curobo_arx/static_pick/setup_pole.py pole_00
"$ARX_STATIC_PY" scripts/curobo_arx/static_pick/run_stage.py pole_00 grasp_input.py
"$ARX_STATIC_PY" scripts/curobo_arx/static_pick/run_stage.py pole_00 prepare.py --output runs/arx_dynamic_pick/pole_00 --fine-gripper
"$ARX_STATIC_PY" scripts/curobo_arx/static_pick/verify_pole.py pole_00
