"""Poll GPU memory each second and dispatch a configured task at most once."""
import argparse
import fcntl
import itertools
import json
import os
from pathlib import Path
import subprocess
import time

from arx_project.layout import ROOT, now, read_json, write_json


def inventory():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=5)
    return [{'index': int(index), 'uuid': uuid.strip(), 'free_mib': int(free),
             'utilization_percent': int(util) if util.strip().isdigit() else None}
            for index, uuid, free, util in (line.split(',') for line in result.stdout.strip().splitlines())]


def choose(gpus, plans):
    for plan in plans:
        names = list(plan)
        admitted = []
        for devices in itertools.permutations(gpus, len(names)):
            if all(gpu['free_mib'] >= plan[name] for name, gpu in zip(names, devices)):
                margin = min(gpu['free_mib']-plan[name] for name, gpu in zip(names, devices))
                admitted.append((margin, devices))
        if admitted:
            _, devices = max(admitted, key=lambda row: row[0])
            return {name: gpu for name, gpu in zip(names, devices)}
    return None


def unit_active(name):
    result = subprocess.run(['systemctl', '--user', 'show', name, '--property=ActiveState', '--value'],
        capture_output=True, text=True, timeout=5)
    return result.returncode == 0 and result.stdout.strip() in ['active', 'activating', 'reloading']


def cuda_probe(gpu, policy):
    """Check a candidate in an isolated process; do not load experiment models."""
    script = '''import json,torch
try:
 free,total=torch.cuda.mem_get_info()
 x=torch.ones(262144,device="cuda"); y=(x+1).sum(); torch.cuda.synchronize()
 print(json.dumps({"passed":float(y.item())==524288.,"free_mib":free/1024**2}))
except Exception as e:
 print(json.dumps({"passed":False,"error":str(e)})); raise SystemExit(1)
'''
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu['uuid'], CUDA_DEVICE_ORDER='PCI_BUS_ID')
    try:
        result = subprocess.run([str(ROOT/policy['python']), '-c', script], env=environment,
            capture_output=True, text=True, timeout=policy.get('timeout_seconds', 30))
        answer = json.loads(result.stdout.strip().splitlines()[-1])
        answer['passed'] = (result.returncode == 0 and answer.get('passed', False)
            and answer.get('free_mib', 0) >= policy['minimum_free_mib'])
        return answer
    except (subprocess.TimeoutExpired, ValueError, IndexError, OSError) as error:
        return {'passed': False, 'error': str(error)}


def tick(config, state, *, query=inventory, active=unit_active, execute=subprocess.run,
         probe=cuda_probe, clock=time.monotonic):
    root = ROOT/config['run']
    if (root/'STOP').exists():
        return {'state': 'stopped_by_experiment_gate', 'terminal': True}
    progress = read_json(root/'progress.json')
    if progress.get('phase') in ['complete', 'stopped']:
        return {'state': 'experiment_'+progress['phase'], 'terminal': True}
    try:
        gpus = query()
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError, ValueError) as error:
        # A transient NVML query failure is not an experiment failure. With no
        # fresh inventory, dispatch nothing and poll again on the next tick.
        return {'state': 'waiting_for_gpu_query', 'terminal': False, 'error': str(error),
            'task_dispatched': state.get('dispatched', False)}
    if active(config['controller_unit']):
        state['dispatched'] = True
        executing = (root/config.get('readiness_marker', 'preflight/all_ready.json')).exists()
        return {'state': 'execution_active' if executing else 'preflight_active',
                'terminal': False, 'gpus': gpus, 'duplicate_start_prevented': True}
    reg = read_json(root/'registration.json')
    pending = reg.get('continuation', {}).get('pending_case_ids', [r['id'] for r in reg.get('schedule', [])])
    claimed = [case for case in pending if (root/'claims'/(case+'.json')).exists()]
    markers = [root/name for name in config.get('dispatch_marker_files', [])]
    if state.get('dispatched') or (root/'started.json').exists() or claimed or any(p.exists() for p in markers):
        return {'state': 'already_started_not_retried', 'terminal': True, 'claimed_new_slots': claimed}
    candidates = gpus
    checks = []
    policy = config.get('cuda_probe_before_dispatch')
    if policy:
        cooldowns = state.setdefault('cuda_cooldowns', {})
        candidates = [g for g in gpus if cooldowns.get(g['uuid'], {}).get('until', 0) <= clock()]
        while True:
            selected = choose(candidates, config['admission_plans_mib'])
            if selected is None:break
            failures = []
            for gpu in selected.values():
                answer = probe(gpu, policy)
                checks.append({'at': now(), 'gpu': gpu['uuid'], 'index': gpu['index'], **answer})
                if not answer['passed']:
                    failures.append(gpu['uuid'])
                    cooldowns[gpu['uuid']] = {'until': clock()+policy.get('failure_cooldown_seconds', 600),
                        'last_check': checks[-1]}
            if not failures:break
            candidates = [g for g in candidates if g['uuid'] not in failures]
    else:
        selected = choose(candidates, config['admission_plans_mib'])
    if checks:
        write_json(root/'monitor/cuda_checks_latest.json', {'at': now(), 'checks': checks})
    if selected is None:
        return {'state': 'waiting_for_gpu', 'terminal': False, 'gpus': gpus,
            'cuda_checks': checks, 'cuda_cooldowns': state.get('cuda_cooldowns', {})}
    if policy:
        write_json(root/'monitor/cuda_admission.json', {'at': now(), 'selected': selected, 'checks': checks})
    context = {'gpu_uuids': ','.join(gpu['uuid'] for gpu in selected.values()),
               'gpu_indices': ','.join(str(gpu['index']) for gpu in selected.values())}
    for role, gpu in selected.items():
        context[role+'_uuid'], context[role+'_index'] = gpu['uuid'], str(gpu['index'])
    state['dispatched'] = True
    commands = [[argument.format_map(context) for argument in command] for command in config['start_commands']]
    for command in commands:
        execute(command, check=True, timeout=30)
    return {'state': 'task_dispatched', 'terminal': False, 'gpus': gpus,
            'selected': selected, 'commands': commands, 'cuda_checks': checks, 'automatic_retries': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    output = ROOT/config['status_file']
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        state, previous, last_write = {}, None, 0.
        while True:
            try:
                result = tick(config, state)
            except BaseException as error:
                write_json(output, {'at': now(), 'state': 'monitor_error', 'error': str(error),
                    'task_dispatched': state.get('dispatched', False)})
                raise
            if result['state'] != previous or time.monotonic()-last_write >= 5:
                write_json(output, {'at': now(), 'controller_unit': config['controller_unit'],
                    'poll_interval_seconds': config.get('poll_interval_seconds', 1), **result})
                if result['state'] != previous:
                    print(now(), result['state'], flush=True)
                previous, last_write = result['state'], time.monotonic()
            if result['terminal']:
                return
            time.sleep(config.get('poll_interval_seconds', 1))


if __name__ == '__main__':
    main()
