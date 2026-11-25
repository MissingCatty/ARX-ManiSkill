"""Follow the parallel repeat, verify completed records, publish both sites."""
import argparse
from pathlib import Path
import time
import traceback

from arx_project.artifacts import atomic_json
from arx_project.json_io import read_json
from arx_project.layout import now
from .metrics import summarize
from .publication import publish,sync_public


def watch(root,control_socket):
    last=None
    while True:
        try:
            summary=summarize(root)
            terminal=summary['phase'] in ['complete','stopped']
            exited=(root/'services_stopped.json').exists() and read_json(root/'services_stopped.json')['all_owned_exited']
            stamp=(summary['phase'],tuple(summary['methods'][m]['completed']//5 for m in ['graspgenx','zerograsp']),terminal and exited)
            if stamp!=last:
                if terminal and exited and not (root/'acceptance.json').exists():
                    from .audit import audit
                    audit(root)
                publish(root)
                if control_socket:sync_public(root,control_socket)
                last=stamp
            if terminal and exited:return
        except Exception:
            atomic_json(root/'publication_error.json',{'at':now(),'error':traceback.format_exc()})
            print(traceback.format_exc(),flush=True)
        time.sleep(10)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--ssh-control',type=Path);args=parser.parse_args();watch(args.root,args.ssh_control)
