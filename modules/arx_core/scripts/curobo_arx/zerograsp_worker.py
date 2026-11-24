"""Keep ZeroGrasp loaded in the isolated GPU container between ARX requests."""
import json
import sys
import time
import traceback
from pathlib import Path

import infer

mailbox = Path('/experiment/zerograsp_worker')
mailbox.mkdir(exist_ok=True)
load_model = infer.load_model
models = {}


def cached_model(config, checkpoint):
    key = (str(checkpoint), config.img_height, config.img_width)
    if key not in models:
        models[key] = load_model(config, checkpoint)
    return models[key]


infer.load_model = cached_model
while True:
    request_path = mailbox / 'request.json'
    if not request_path.exists():
        time.sleep(0.05)
        continue
    request = json.loads(request_path.read_text())
    request_path.unlink()
    started = time.monotonic()
    try:
        action = request['action']
        if action not in ('warmup', 'infer', 'infer_raw'):
            raise ValueError('Unknown ZeroGrasp action')
        output = '/experiment/zerograsp_warmup' if action == 'warmup' else '/experiment/zerograsp'
        sys.argv = ['infer.py', '--rgb', '/experiment/camera/color.png',
                    '--depth', '/experiment/camera/depth.png', '--mask', '/experiment/camera/mask.png',
                    '--camera', '/experiment/camera/camera.yml', '--output', output,
                    '--postprocess', 'raw' if action == 'infer_raw' else 'paper']
        infer.main()
        result = {'id': request['id'], 'ok': True, 'action': action,
                  'wall_seconds': time.monotonic() - started, 'models_loaded': len(models)}
        if action == 'warmup':
            (mailbox / 'ready.json').write_text(json.dumps(result))
    except Exception:
        result = {'id': request['id'], 'ok': False, 'error': traceback.format_exc()}
    temporary = mailbox / 'response.tmp'
    temporary.write_text(json.dumps(result))
    temporary.replace(mailbox / 'response.json')
