#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ZG="${ZEROGRASP_ROOT:-$(dirname "$ROOT")/zerograsp}"
NAME="coolas-arx-zerograsp-zhaoyechong"
if ! podman container exists "$NAME"; then
podman run -d --name "$NAME" --device "nvidia.com/gpu=${GPU:-1}" --ipc=host --network=none \
  -v "$ZG:/workspace:rw" -v "$ZG/source:/opt/app:ro" -v "$ROOT/runs/arx_grasp:/experiment:rw" -v "$ROOT/scripts/curobo_arx:/arx:ro" \
  -w /opt/app -e PYTHONPATH=/opt/app:/workspace/scripts -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONUNBUFFERED=1 -e TORCH_HOME=/workspace/.cache/torch -e HF_HUB_OFFLINE=1 \
  localhost/zerograsp-standalone:152f67c27269 python /arx/zerograsp_worker.py
elif [[ "$(podman ps --filter "name=^${NAME}$" --format '{{.Names}}')" != "$NAME" ]]; then
  podman start "$NAME"
fi
python "$ROOT/scripts/curobo_arx/zerograsp_request.py" warmup
python "$ROOT/scripts/curobo_arx/zerograsp_request.py" infer_raw
