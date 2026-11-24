"""Wait for published readiness and a live health reply, never just a socket."""
import json
from pathlib import Path
import time

from resident_backend import rpc


def wait_ready(path,ready_file,process,kind,timeout=1800):
    path=Path(path);ready_file=Path(ready_file)
    deadline=time.monotonic()+timeout
    while True:
        if process.poll() is not None:
            raise RuntimeError('Service exited before readiness: '+kind)
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('Service readiness timed out: '+kind)
        if ready_file.exists():
            ready=json.loads(ready_file.read_text())
            if ready.get('pid')!=process.pid:raise RuntimeError('Stale service readiness PID: '+kind)
            if ready.get('status')=='ready' and path.exists():
                try:
                    health=rpc(path,{'command':'health'},timeout=min(120,remaining))
                except (FileNotFoundError,ConnectionRefusedError,ConnectionResetError):
                    # Only retry the same owned process while its startup is in
                    # progress. Never restart it or replay a request.
                    time.sleep(.1)
                    continue
                if health.get('pid')!=process.pid or health.get('kind')!=kind:
                    raise RuntimeError('Wrong service answered health: '+kind)
                if health.get('status')=='ready':return health
        time.sleep(.1)
