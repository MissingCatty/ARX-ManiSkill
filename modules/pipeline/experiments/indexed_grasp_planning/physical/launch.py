"""One CUDA-admitted controller; no restarting a claimed physical registration."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from arx_project.layout import ROOT, ensure_run_path, now, write_json
from .registration import guard, prepare, raw


def unit(root): return 'arx-v8-physical-'+root.name.replace('_', '-')+'.service'


def monitor_policy(root):
    return {'run': str(root.relative_to(ROOT)), 'controller_unit': unit(root),
        'status_file': str((root/'monitor/status.json').relative_to(ROOT)),
        'readiness_marker': 'preflight/all_ready.json', 'dispatch_marker_files': ['dispatch_claim.json'],
        'poll_interval_seconds': 1, 'admission_plans_mib': [{'execution': 14000}],
        'cuda_probe_before_dispatch': {'python': 'environments/curobo/bin/python',
            'minimum_free_mib': 14000, 'timeout_seconds': 30, 'failure_cooldown_seconds': 600},
        'start_commands': [[str(ROOT/'environments/geometric_ik/bin/python'), '-m',
            'modules.pipeline.experiments.indexed_grasp_planning.physical.launch', 'dispatch',
            '--root', str(root), '--gpu', '{execution_uuid}']],
        'automatic_experiment_retries': False, 'physical_slot_budget': 20}


def watch(root):
    guard(root)
    if any((root/p).exists() for p in ['STOP', 'started.json', 'dispatch_claim.json']):
        raise ValueError('Do not redispatch this physical registration')
    if raw(root/'monitor_config.json') != monitor_policy(root): raise ValueError('Resource policy changed')
    name = 'arx-v8-physical-watch-'+root.name.replace('_', '-')+'.service'
    subprocess.run(['systemd-run', '--user', '--unit='+name, '--property=WorkingDirectory='+str(ROOT),
        '--property=StandardOutput=append:'+str(root/'monitor.log'), '--property=StandardError=inherit',
        str(ROOT/'environments/geometric_ik/bin/python'), '-m', 'modules.pipeline.monitoring.gpu_auto_start',
        '--config', str(root/'monitor_config.json')], check=True)
    write_json(root/'monitor_controller.json', {'at': now(), 'unit': name})


def dispatch(root, gpu):
    reg = guard(root)
    if not reg['device_assignment_pending'] or any((root/n).exists() for n in ['STOP', 'started.json', 'dispatch_claim.json']):
        raise ValueError('Physical controller was already claimed')
    with (root/'dispatch_claim.json').open('x') as stream: json.dump({'at': now(), 'gpu': gpu}, stream)
    reg.update(gpu=gpu, device_assignment_pending=False);write_json(root/'registration.json', reg)
    command = ['systemd-run', '--user', '--unit='+unit(root), '--property=Type=exec',
        '--property=WorkingDirectory='+str(ROOT), '--property=StandardOutput=append:'+str(root/'controller.log'),
        '--property=StandardError=inherit', str(ROOT/'environments/geometric_ik/bin/python'), '-u', '-m',
        'modules.pipeline.experiments.indexed_grasp_planning.physical.controller', '--root', str(root)]
    subprocess.run(command, check=True, timeout=20)
    write_json(root/'controller.json', {'at': now(), 'unit': unit(root), 'gpu': gpu})


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('stage', choices=['prepare', 'watch', 'dispatch'])
    p.add_argument('--root', type=Path, required=True);p.add_argument('--gpu');args = p.parse_args()
    root = ensure_run_path(args.root)
    if args.stage == 'prepare': prepare(root)
    elif args.stage == 'watch': watch(root)
    else: dispatch(root, args.gpu)
