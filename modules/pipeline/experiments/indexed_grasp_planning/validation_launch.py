"""One-shot dispatch through the original CUDA admission watcher."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from arx_project.layout import ROOT, ensure_run_path, now, write_json
from modules.pipeline.experiments.graspgenx_postprocessing.planning import raw, verify


def controller_unit(root): return 'arx-v8-'+root.name.replace('_', '-')+'.service'


def monitor_policy(root):
    return {'run': str(root.relative_to(ROOT)), 'controller_unit': controller_unit(root),
        'status_file': str((root/'monitor/status.json').relative_to(ROOT)), 'readiness_marker': 'startup.json',
        'dispatch_marker_files': ['dispatch_claim.json'], 'poll_interval_seconds': 1,
        'admission_plans_mib': [{'planner': 10000}],
        'cuda_probe_before_dispatch': {'python': 'environments/curobo/bin/python', 'minimum_free_mib': 10000,
            'timeout_seconds': 30, 'failure_cooldown_seconds': 600},
        'start_commands': [[str(ROOT/'environments/geometric_ik/bin/python'), '-m',
            'modules.pipeline.experiments.indexed_grasp_planning.validation_launch', 'dispatch',
            '--root', str(root), '--gpu', '{planner_uuid}']],
        'automatic_experiment_retries': False, 'physical_grasp_budget': 0}


def watch(root):
    if any((root/p).exists() for p in ['STOP', 'started.json', 'dispatch_claim.json']):
        raise ValueError('Never redispatch an already claimed validation registration')
    verify(raw(root/'registration.json'))
    if raw(root/'monitor_config.json') != monitor_policy(root): raise ValueError('Resource policy changed')
    unit = 'arx-v8-watch-'+root.name.replace('_', '-')+'.service'
    subprocess.run(['systemd-run', '--user', '--unit='+unit, '--property=WorkingDirectory='+str(ROOT),
        '--property=StandardOutput=append:'+str(root/'monitor.log'), '--property=StandardError=inherit',
        str(ROOT/'environments/geometric_ik/bin/python'), '-m', 'modules.pipeline.monitoring.gpu_auto_start',
        '--config', str(root/'monitor_config.json')], check=True)
    write_json(root/'monitor_controller.json', {'at': now(), 'unit': unit})


def dispatch(root, gpu):
    reg = raw(root/'registration.json');verify(reg)
    if not reg['device_assignment_pending'] or any((root/p).exists() for p in ['STOP', 'started.json']):
        raise ValueError('The validation registration was already dispatched or stopped')
    with (root/'dispatch_claim.json').open('x') as stream: json.dump({'at': now(), 'gpu': gpu}, stream)
    reg.update(gpu=gpu, device_assignment_pending=False);write_json(root/'registration.json', reg)
    env = {'CUDA_HOME': str(ROOT/'environments/runtime_bundle/cuda'),
        'CUDA_PATH': str(ROOT/'environments/runtime_bundle/cuda'),
        'TORCH_EXTENSIONS_DIR': str(ROOT/'tmp/curobo/torch_extensions_sm89'), 'TORCH_CUDA_ARCH_LIST': '8.9',
        'CUROBO_KERNEL_BACKEND': 'cuda_core', 'GRASP_TRAJOPT_POLICY': 'interactive',
        'PYTHONUNBUFFERED': '1', 'XDG_CACHE_HOME': str(ROOT/'tmp/curobo/arx-curobo/cache'), 'MAX_JOBS': '2',
        'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'PATH': str(ROOT/'environments/runtime_bundle/cuda/bin')+':'+os.environ['PATH']}
    command = ['systemd-run', '--user', '--unit='+controller_unit(root), '--property=Type=exec',
        '--property=WorkingDirectory='+str(ROOT), '--property=StandardOutput=append:'+str(root/'controller.log'),
        '--property=StandardError=inherit']
    command += ['--setenv='+k+'='+v for k, v in env.items()]
    command += [str(ROOT/'environments/geometric_ik/bin/python'), '-m',
        'modules.pipeline.experiments.indexed_grasp_planning.validation_replay', 'run', '--root', str(root)]
    write_json(root/'controller.json', {'at': now(), 'unit': controller_unit(root), 'gpu': gpu, 'cuda_environment': env})
    subprocess.run(command, check=True, timeout=20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('stage', choices=['watch', 'dispatch'])
    parser.add_argument('--root', type=Path, required=True);parser.add_argument('--gpu');args = parser.parse_args()
    root = ensure_run_path(args.root)
    if args.stage == 'watch': watch(root)
    else: dispatch(root, args.gpu)
