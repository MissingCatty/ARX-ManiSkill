#!/usr/bin/env bash
set -euo pipefail
cd /home/zhaoyechong/projects/COOLAS-ARX
source scripts/curobo_arx/static_pick/env.sh
export CUDA_VISIBLE_DEVICES=0
ARX_STATIC_PY=/home/zhaoyechong/projects/ARX-ManiSkill/modules/curobo/.venv/bin/python
"$ARX_STATIC_PY" - <<'PY'
from pathlib import Path
import time
p=Path('runs/arx_dynamic_pick/bottle_00/input_report.json');end=time.monotonic()+420
while not p.exists():
 if time.monotonic()>end:raise TimeoutError('Pole RGBD unavailable')
 time.sleep(1)
PY
podman run -d --name coolas-arx-bottle-zerograsp-20261022 --device nvidia.com/gpu=0 --ipc=host --network=none -v /home/zhaoyechong/projects/zerograsp:/workspace:rw -v /home/zhaoyechong/projects/zerograsp/source:/opt/app:ro -v /coolas/data/zhaoyechong/COOLAS-ARX/dynamic_pick_20261022/experiment/bottle_00:/experiment:rw -v /home/zhaoyechong/projects/ARX-ManiSkill/modules/arx_core/scripts/curobo_arx:/arx:ro -w /opt/app -e PYTHONPATH=/opt/app:/workspace/scripts -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e TORCH_HOME=/workspace/.cache/torch -e HF_HUB_OFFLINE=1 localhost/zerograsp-standalone:152f67c27269 python /arx/zerograsp_worker.py
"$ARX_STATIC_PY" scripts/curobo_arx/static_pick/run_stage.py bottle_00 zerograsp_request.py warmup
"$ARX_STATIC_PY" scripts/curobo_arx/static_pick/run_stage.py bottle_00 zerograsp_request.py infer_raw
