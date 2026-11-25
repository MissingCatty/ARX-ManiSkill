"""Poll GPU memory each second and dispatch a configured task at most once."""
import argparse
import fcntl
import itertools
import json
from pathlib import Path
import subprocess
import time

from arx_project.layout import ROOT, now, read_json, write_json


def inventory():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=5)
    return [{'index': int(index), 'uuid': uuid.strip(), 'free_mib': int(free),
             'utilization_percent': int(util)}
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


def tick(config, state, *, query=inventory, active=unit_active, execute=subprocess.run):
    root = ROOT/config['run']
    if (root/'STOP').exists():
        return {'state': 'stopped_by_experiment_gate', 'terminal': True}
    progress = read_json(root/'progress.json')
    if progress.get('phase') in ['complete', 'stopped']:
        return {'state': 'experiment_'+progress['phase'], 'terminal': True}
    gpus = query()
    if active(config['controller_unit']):
        state['dispatched'] = True
        executing = (root/'preflight/all_ready.json').exists()
        return {'state': 'execution_active' if executing else 'preflight_active',
                'terminal': False, 'gpus': gpus, 'duplicate_start_prevented': True}
    reg = read_json(root/'registration.json')
    pending = reg.get('continuation', {}).get('pending_case_ids', [r['id'] for r in reg['schedule']])
    claimed = [case for case in pending if (root/'claims'/(case+'.json')).exists()]
    if state.get('dispatched') or (root/'started.json').exists() or claimed:
        return {'state': 'already_started_not_retried', 'terminal': True, 'claimed_new_slots': claimed}
    selected = choose(gpus, config['admission_plans_mib'])
    if selected is None:
        return {'state': 'waiting_for_gpu', 'terminal': False, 'gpus': gpus}
    context = {'gpu_uuids': ','.join(gpu['uuid'] for gpu in selected.values()),
               'gpu_indices': ','.join(str(gpu['index']) for gpu in selected.values())}
    for role, gpu in selected.items():
        context[role+'_uuid'], context[role+'_index'] = gpu['uuid'], str(gpu['index'])
    state['dispatched'] = True
    commands = [[argument.format_map(context) for argument in command] for command in config['start_commands']]
    for command in commands:
        execute(command, check=True, timeout=30)
    return {'state': 'task_dispatched', 'terminal': False, 'gpus': gpus,
            'selected': selected, 'commands': commands, 'automatic_retries': False}


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
