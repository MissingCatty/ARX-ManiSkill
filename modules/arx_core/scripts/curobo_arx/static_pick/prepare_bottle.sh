#!/usr/bin/env bash
set -euo pipefail
cd /home/zhaoyechong/projects/COOLAS-ARX
source scripts/curobo_arx/static_pick/env.sh
export CUDA_VISIBLE_DEVICES=0
ARX_STATIC_PY=/home/zhaoyechong/projects/ARX-ManiSkill/modules/curobo/.venv/bin/python
"$ARX_STATIC_PY" scripts/curobo_arx/static_pick/setup_bottle.py
"$ARX_STATIC_PY" scripts/curobo_arx/static_pick/run_stage.py bottle_00 grasp_input.py
"$ARX_STATIC_PY" scripts/curobo_arx/static_pick/run_stage.py bottle_00 prepare.py --output runs/arx_dynamic_pick/bottle_00 --fine-gripper
"$ARX_STATIC_PY" scripts/curobo_arx/static_pick/verify_bottle.py bottle_00
