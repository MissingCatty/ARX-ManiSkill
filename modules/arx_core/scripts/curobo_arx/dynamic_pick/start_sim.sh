#!/usr/bin/env bash
set -euo pipefail
cd /workspace/COOLAS-ARX
/isaac-sim/python.sh -m pip install -r arx-sim/requirements.txt
export ARX_CUROBO_VALIDATION_DIR=/workspace/COOLAS-ARX/runs/arx_dynamic_pick/mailbox
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/isaac-sim/exts/isaacsim.ros2.core/jazzy/lib:${LD_LIBRARY_PATH:-}"
exec /isaac-sim/python.sh scripts/curobo_arx/dynamic_pick/sim_runner.py --task grasp --run-dir /workspace/COOLAS-ARX/runs/arx_dynamic_pick/session
