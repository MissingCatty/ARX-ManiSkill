"""Run once on shared GPU2, then review complete timing and trajectory records."""
import argparse
from pathlib import Path
from threading import Event, Thread
import traceback

from arx_project.layout import ROOT, now, read_json, write_json, ensure_run_path
from modules.pipeline.experiments.parallel_planning.run import run as original_run
from modules.pipeline.experiments.endpoint_parallel.shared_runner import record_load


def run(root):
    reg = read_json(root / 'registration.json')
    cfg = reg['config']
    if not cfg['user_authorized_shared_device'] or reg['gpu'] != cfg['fixed_gpu_uuid']:
        raise RuntimeError('Shared GPU2 authorization differs from the registration')
    write_json(root / 'timing_environment.json', {'at': now(), 'exclusive_timing': False,
        'user_authorized_shared_device': True, 'GPU_index': 2, 'GPU_uuid': reg['gpu'],
        'original_memory_and_CUDA_gates_preserved': True,
        'load_recording': 'asynchronous_one_second_samples'})
    stop = Event()
    thread = Thread(target=record_load, args=(root, reg['gpu'], stop), name='gpu-load')
    thread.start()
    write_json(ROOT / 'tmp/state_updates/active.json', {'at': now(),
        'run': str(root.relative_to(ROOT)), 'phase': 'running', 'complete': False})
    try:
        original_run(root)
    except BaseException:
        write_json(root / 'STOP', {'at': now(), 'error': traceback.format_exc(), 'automatic_retry': False})
        raise
    finally:
        stop.set()
        thread.join(timeout=15)
    if thread.is_alive() or not read_json(root / 'gpu_load_summary.json')['planning_samples']:
        raise RuntimeError('GPU load record incomplete')
    from modules.pipeline.experiments.parallel_planning.audit import audit
    audit(root)
    from .report import create
    create(root)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    root = ensure_run_path(p.parse_args().root)
    try:
        run(root)
    except BaseException:
        if not (root / 'STOP').exists():
            write_json(root / 'STOP', {'at': now(), 'error': traceback.format_exc(), 'automatic_retry': False})
        raise
