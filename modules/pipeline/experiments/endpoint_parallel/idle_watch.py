"""Dispatch one registered comparison when a healthy GPU has no compute users."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import time
from unittest.mock import patch

from arx_project.layout import ROOT, now, read_json, write_json
from modules.pipeline.monitoring.gpu_auto_start import tick, inventory
from modules.pipeline.experiments.graspgenx_native_postprocessing.offline import guard
from .resources import compute_processes, idle_devices


def dispatch(root, gpu):
    reg = guard(root)
    if reg['gpu'] != 'pending' or (root / 'started.json').exists() or list((root / 'claims').glob('*.json')):
        raise RuntimeError('Only a fresh, unassigned timing registration can be dispatched')
    backup = root / 'registration_revisions/device_assignment_01.json'
    backup.parent.mkdir(); shutil.copy2(root / 'registration.json', backup)
    reg.update(gpu=gpu, device_assignment_pending=False, device_assigned_at=now())
    write_json(root / 'registration.json', reg)
    command = ['systemd-run', '--user', '--unit', 'arx-v8-windowed-idle-20261127-01',
        '--property=WorkingDirectory=' + str(ROOT), '--property=StandardOutput=append:' + str(root / 'controller.log'),
        '--property=StandardError=inherit', str(ROOT / 'environments/curobo/bin/python'), '-u', '-m',
        'modules.pipeline.experiments.parallel_planning.run', '--root', str(root)]
    subprocess.run(command, check=True, timeout=30)
    write_json(root / 'monitor/dispatch.json', {'at': now(), 'gpu': gpu, 'command': command, 'automatic_retries': False})


def watch(config):
    root = ROOT / config['run']; output = ROOT / config['status_file']
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state, previous, last_write = {}, None, 0.
        while True:
            observed = {}
            def query():
                cards = inventory(); processes = compute_processes()
                observed.update(inventory=cards, compute_processes=processes)
                return idle_devices(cards, processes, config.get('excluded_gpu_uuids', []))
            result = tick(config, state, query=query)
            if result['state'] != previous or time.monotonic() - last_write >= 5:
                write_json(output, {'at': now(), **result, **observed, 'automatic_retries': False})
                if result['state'] != previous: print(now(), result['state'], flush=True)
                previous, last_write = result['state'], time.monotonic()
            if result['terminal']:
                if result['state'] == 'experiment_complete':
                    from .report import create
                    from modules.pipeline.experiments.parallel_planning import run as controller, audit
                    from modules.pipeline.experiments.graspgenx_native_postprocessing import planning as original_review
                    def archived(path): return guard(path, completed=True)
                    with patch.object(controller, 'guard', archived), patch.object(audit, 'guard', archived), patch.object(original_review, 'guard', archived):
                        create(root)
                    with (root / 'website_refresh.log').open('x') as log:
                        subprocess.run([str(ROOT / 'environments/curobo/bin/python'),
                            str(ROOT / 'website/tools/publish_indexed_planning_plan.py')], cwd=ROOT,
                            check=True, stdout=log, stderr=subprocess.STDOUT, timeout=180)
                        subprocess.run([config['node'], 'build.mjs'], cwd=ROOT / 'website',
                            check=True, stdout=log, stderr=subprocess.STDOUT, timeout=180)
                    write_json(root / 'idle_verification_complete.json', {'at': now(), 'passed': True,
                        'report_created': True, 'local_website_refreshed': True, 'default_activated': False})
                return
            time.sleep(1)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', type=Path)
    p.add_argument('--dispatch-root', type=Path); p.add_argument('--gpu')
    args = p.parse_args()
    if args.dispatch_root: dispatch(args.dispatch_root.resolve(), args.gpu)
    else: watch(read_json(args.config))
