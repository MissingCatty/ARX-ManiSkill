"""Wait for V4 acceptance, then register and launch V6 once, without retries."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from arx_project.layout import ROOT, ensure_run_path, now, read_json, write_json
from .run import CONFIG, prepare


def baseline_state(source):
    progress = read_json(source/'progress.json')
    if progress['phase'] == 'stopped':
        return 'baseline_stopped'
    if progress['phase'] != 'complete':
        return 'waiting_for_baseline'
    if not (source/'services_stopped.json').exists() or not read_json(
            source/'services_stopped.json')['all_owned_exited']:
        return 'waiting_for_baseline_services_exit'
    if not (source/'acceptance.json').exists():
        return 'waiting_for_baseline_acceptance'
    audit = read_json(source/'acceptance.json')
    if not audit['passed'] or not audit['record_integrity']:
        return 'baseline_audit_failed'
    if not (source/'publication_complete.json').exists():
        return 'waiting_for_baseline_publication'
    return 'ready'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = ensure_run_path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    with (root/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root/'registration.json').exists() or (root/'STOP').exists():
            raise RuntimeError('Queue already registered or stopped; no automatic replay')
        source = ROOT/read_json(CONFIG)['baseline_run']
        while True:
            state = baseline_state(source)
            write_json(root/'queue_status.json', {'at': now(), 'state': state,
                'baseline': str(source.relative_to(ROOT)), 'pid': os.getpid()})
            if state in ['baseline_stopped', 'baseline_audit_failed']:
                raise RuntimeError('Baseline cannot transfer a complete paired cohort: '+state)
            if state == 'ready':
                break
            time.sleep(5)
        prepare(root)
        unit = 'arx-sam3-zerograsp-'+root.name.replace('_', '-')
        python = str(ROOT/'environments/simulation/bin/python')
        command = ['systemd-run', '--user', '--unit='+unit, '--collect',
            '--property=WorkingDirectory='+str(ROOT), '--property=KillMode=control-group',
            '--property=TimeoutStopSec=45', '--property=Environment=PYTHONUNBUFFERED=1',
            '--property=StandardOutput=append:'+str(root/'controller.log'),
            '--property=StandardError=append:'+str(root/'controller.log'),
            python, '-u', '-m', 'modules.pipeline.experiments.sam3_zerograsp_comparison.run',
            'run', '--root', str(root)]
        subprocess.run(command, check=True)
        watch_command = ['systemd-run', '--user', '--unit='+unit+'-watch', '--collect',
            '--property=WorkingDirectory='+str(ROOT), '--property=Environment=PYTHONUNBUFFERED=1',
            '--property=StandardOutput=append:'+str(root/'watch.log'),
            '--property=StandardError=append:'+str(root/'watch.log'),
            python, '-u', '-m', 'modules.pipeline.experiments.sam3_zerograsp_comparison.watch',
            '--root', str(root)]
        subprocess.run(watch_command, check=True)
        write_json(root/'queue_status.json', {'at': now(), 'state': 'submitted', 'unit': unit+'.service'})


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        # No services or physical trials are started by this queue before submission.
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        raise
