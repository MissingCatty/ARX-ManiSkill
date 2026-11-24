"""Warm or infer using the resident ARX ZeroGrasp worker."""
import argparse
import json
import time
import uuid
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('action', choices=['warmup', 'infer', 'infer_raw'])
args = parser.parse_args()
mailbox = Path(__file__).resolve().parents[2] / 'runs/arx_grasp/zerograsp_worker'
mailbox.mkdir(exist_ok=True)
if (mailbox / 'request.json').exists():
    raise RuntimeError('ZeroGrasp already has a pending request')
request = {'id': uuid.uuid4().hex, 'action': args.action}
temporary = mailbox / 'request.tmp'
temporary.write_text(json.dumps(request))
temporary.replace(mailbox / 'request.json')
deadline = time.monotonic() + 300
while time.monotonic() < deadline:
    response = mailbox / 'response.json'
    if response.exists():
        result = json.loads(response.read_text())
        if result['id'] == request['id']:
            if not result['ok']:
                raise RuntimeError(result['error'])
            print(json.dumps(result))
            break
    time.sleep(0.1)
else:
    raise TimeoutError('ZeroGrasp request timed out')
