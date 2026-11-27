"""Own one warmed planner and one simulator; stop on infrastructure faults."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from threading import Event, Thread
import time
import traceback

from arx_project.layout import ROOT, now, read_json, sha256, write_json, ensure_run_path
from arx_project.client import Client
from modules.pipeline.experiments.indexed_grasp_planning.physical.controller import environment, wait_ready
from .registration import guard


def load_record(root, gpu, stop):
    with (root / 'gpu_load.jsonl').open('x') as stream:
        while not stop.is_set():
            row = {'at': now(), 'gpu': gpu}
            try:
                r = subprocess.run(['nvidia-smi', '-i', gpu, '--query-gpu=memory.free,utilization.gpu',
                    '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=5)
                free, util = r.stdout.strip().split(',')
                row.update(free_mib=int(free), GPU_utilization_percent=int(util),
                    phase=read_json(root / 'progress.json')['phase'])
            except Exception as error: row['error'] = str(error)
            stream.write(json.dumps(row) + '\n'); stream.flush(); stop.wait(1)


def run(root):
    reg = guard(root)
    carried = reg.get('continuation') or {}
    inherited = set(carried.get('terminal_slot_ids', carried.get('completed_slot_ids', [])))
    claims = {p.stem for p in (root / 'claims').glob('*.json')}
    if (root / 'started.json').exists() or claims != inherited:
        raise RuntimeError('No restarting an already-started comparison')
    from modules.pipeline.monitoring.gpu_auto_start import inventory, cuda_probe
    while True:
        guard(root, full=False)
        gpu = next(g for g in inventory() if g['uuid'] == reg['gpu'])
        if gpu['free_mib'] >= reg['config']['combined_startup_free_mib']: break
        status = read_json(root / 'progress.json'); status.update(at=now(), phase='waiting_for_gpu', free_mib=gpu['free_mib'])
        write_json(root / 'progress.json', status); time.sleep(5)
    probe = cuda_probe(gpu, {'python': 'environments/curobo/bin/python', 'minimum_free_mib': 14000, 'timeout_seconds': 30})
    write_json(root / 'preflight/cuda_admission.json', probe)
    if not probe['passed']: raise RuntimeError('GPU2 actual CUDA/resource admission failed')
    write_json(root / 'started.json', {'at': now(), 'pid': os.getpid(), 'gpu': reg['gpu']})
    status = read_json(root / 'progress.json'); status.update(at=now(), phase='warming')
    write_json(root / 'progress.json', status)
    socket = ROOT / 'tmp/rpc' / sha256(root / 'registration.json')[:12] / 'plan.sock'
    socket.parent.mkdir(parents=True)
    children = {}; streams = []; stop = Event()
    thread = Thread(target=load_record, args=(root, reg['gpu'], stop), name='gpu-load'); thread.start()
    def start(name, module, args):
        stream = (root / (name + '.log')).open('x'); streams.append(stream)
        python = 'curobo' if name == 'planner' else 'simulation'
        command = [str(ROOT / 'environments' / python / 'bin/python'), '-u', '-m', module, *args]
        child = subprocess.Popen(command, cwd=ROOT, env=environment(reg['gpu']), stdout=stream, stderr=subprocess.STDOUT)
        children[name] = child
        write_json(root / (name + '_process.json'), {'at': now(), 'pid': child.pid, 'command': command})
        return child
    try:
        planner = start('planner', 'modules.pipeline.experiments.state_update_execution.worker',
            ['--root', str(root / 'resident'), '--socket', str(socket), '--warmup-trial', str(root / 'warmup'),
             '--descriptor', str(root / 'gripper_descriptor.json'), '--campaign', str(root)])
        wait_ready(planner, root / 'resident/curobo_memory_state_ready.json')
        child = start('simulation', 'modules.pipeline.experiments.state_update_execution.simulation',
            ['--root', str(root), '--socket', str(socket)])
        while child.poll() is None:
            if planner.poll() is not None: raise RuntimeError('Resident planner exited during execution')
            time.sleep(1)
        if child.returncode or not (root / 'execution_complete.json').exists():
            raise RuntimeError('Execution stopped; preserve simulator logs and STOP')
        guard(root)
    finally:
        for name, child in reversed(list(children.items())):
            if child.poll() is None:
                if name == 'planner':
                    try: Client(socket, timeout=10).call({'command': 'shutdown'})
                    except Exception: child.terminate()
                else: child.terminate()
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=10)
        stop.set(); thread.join(timeout=15)
        for stream in streams: stream.close()
        write_json(root / 'services_stopped.json', {'at': now(), 'all_owned_exited':
            all(p.poll() is not None for p in children.values()) and not thread.is_alive(),
            'returncodes': {name: p.returncode for name, p in children.items()}})
    from .review import review
    from .report import report
    review(root); report(root)
    status = read_json(root / 'progress.json'); status.update(at=now(), phase='complete')
    write_json(root / 'progress.json', status)
    write_json(ROOT / 'tmp/state_updates/execution_active.json', {'at': now(),
        'run': str(root.relative_to(ROOT)), 'phase': 'complete', 'complete': True})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    root = ensure_run_path(p.parse_args().root)
    try: run(root)
    except BaseException:
        if not (root / 'STOP').exists():
            write_json(root / 'STOP', {'at': now(), 'error': traceback.format_exc(), 'automatic_retry': False})
        status = read_json(root / 'progress.json'); status.update(at=now(), phase='stopped')
        write_json(root / 'progress.json', status)
        raise
