"""Run the authorized shared-GPU comparison with asynchronous load records."""
import argparse
import json
from pathlib import Path
import subprocess
from threading import Event, Thread
import traceback

import numpy as np

from arx_project.layout import ROOT, now, read_json, write_json, ensure_run_path
from modules.pipeline.experiments.parallel_planning.run import run
from .resources import compute_processes


def summarize_load(rows):
    good = [r for r in rows if 'GPU_utilization_percent' in r]
    active = [r for r in good if r['phase'] == 'planning']
    result = {'samples': len(rows), 'valid_samples': len(good), 'query_errors': len(rows) - len(good),
        'planning_samples': len(active), 'exclusive_timing': False, 'user_authorized_shared_device': True}
    for label, values in [('all', good), ('planning', active)]:
        if not values: continue
        result[label] = {'minimum_free_mib': min(r['free_mib'] for r in values),
            'median_GPU_utilization_percent': float(np.median([r['GPU_utilization_percent'] for r in values])),
            'maximum_GPU_utilization_percent': max(r['GPU_utilization_percent'] for r in values),
            'maximum_foreign_compute_contexts': max(r['foreign_compute_contexts'] for r in values)}
    return result


def record_load(root, gpu, stop):
    rows = []
    try:
        with (root / 'gpu_load.jsonl').open('x') as output:
            while not stop.is_set():
                progress = read_json(root / 'progress.json')
                row = {'at': now(), 'phase': progress['phase'], 'completed_requests': progress['completed_requests'],
                    'current': progress.get('current'), 'gpu_uuid': gpu}
                try:
                    result = subprocess.run(['nvidia-smi', '-i', gpu,
                        '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'],
                        check=True, capture_output=True, text=True, timeout=5)
                    free, util = result.stdout.strip().split(',')
                    own = read_json(root / 'planner_process.json')['pid'] if (root / 'planner_process.json').exists() else None
                    processes = compute_processes()
                    row.update(free_mib=int(free), GPU_utilization_percent=int(util),
                        foreign_compute_contexts=sum(p['uuid'] == gpu and p['pid'] != own for p in processes))
                except Exception as error:
                    row['query_error'] = str(error)
                rows.append(row); output.write(json.dumps(row, ensure_ascii=False) + '\n'); output.flush()
                write_json(root / 'gpu_load_latest.json', row)
                stop.wait(1.)
    finally:
        write_json(root / 'gpu_load_summary.json', summarize_load(rows))


def main(root):
    registration = read_json(root / 'registration.json')
    if registration['config'].get('require_idle_device') or not registration['config'].get('user_authorized_shared_device'):
        raise RuntimeError('Shared execution requires the explicit registered user authorization')
    write_json(root / 'timing_environment.json', {'at': now(), 'exclusive_timing': False,
        'user_authorized_shared_device': True, 'GPU_index': 2, 'GPU_uuid': registration['gpu'],
        'reason': 'User explicitly authorized GPU2 with concurrent compute processes',
        'load_recording': 'asynchronous_one_second_samples', 'original_memory_and_CUDA_gates_preserved': True})
    stop = Event(); thread = Thread(target=record_load, args=(root, registration['gpu'], stop), name='gpu-load')
    thread.start()
    try:
        run(root)
    except BaseException:
        write_json(root / 'STOP', {'at': now(), 'error': traceback.format_exc(), 'automatic_retry': False})
        raise
    finally:
        stop.set(); thread.join(timeout=15)
    if thread.is_alive(): raise RuntimeError('GPU load recorder did not stop')
    if not read_json(root / 'gpu_load_summary.json')['planning_samples']:
        raise RuntimeError('No planning-phase GPU load observation was recorded')
    from .report import create
    create(root)
    write_json(root / 'shared_verification_complete.json', {'at': now(), 'passed': True,
        'planning_requests': 30, 'default_activated': False, 'GPU_load_recorded': True})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    main(ensure_run_path(p.parse_args().root))
