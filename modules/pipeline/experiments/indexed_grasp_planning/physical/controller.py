"""Own one GPU planner, one CPU ordering proxy and one simulator worker."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from arx_project.layout import ROOT, now, sha256, write_json
from arx_project.client import Client
from .registration import guard, raw


def environment(gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, CUDA_DEVICE_ORDER='PCI_BUS_ID',
        CUDA_HOME=str(ROOT/'environments/runtime_bundle/cuda'), CUDA_PATH=str(ROOT/'environments/runtime_bundle/cuda'),
        TORCH_EXTENSIONS_DIR=str(ROOT/'tmp/curobo/torch_extensions_sm89'), TORCH_CUDA_ARCH_LIST='8.9',
        CUROBO_KERNEL_BACKEND='cuda_core', GRASP_TRAJOPT_POLICY='interactive',
        XDG_CACHE_HOME=str(ROOT/'tmp/curobo/arx-curobo/cache'), CUBLAS_WORKSPACE_CONFIG=':4096:8',
        PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', MAX_JOBS='2',
        ARX_PLANNER_GPU=gpu, ARX_SIM_GPU=gpu, LILO_GPU=gpu)
    env['PATH'] = str(ROOT/'environments/runtime_bundle/cuda/bin')+':'+env['PATH']
    env.pop('PYTHONPATH', None)
    return env


def wait_ready(child, marker, timeout=1800):
    deadline = time.monotonic()+timeout
    while not marker.exists():
        if child.poll() is not None: raise RuntimeError('Resident worker exited before readiness: '+str(marker))
        if time.monotonic() > deadline: raise TimeoutError('Worker readiness timed out: '+str(marker))
        time.sleep(1)


def run(root):
    reg = guard(root)
    if reg['device_assignment_pending'] or (root/'started.json').exists() or list((root/'claims').glob('*.json')):
        raise RuntimeError('Physical registration is not fresh and CUDA assigned')
    write_json(root/'started.json', {'at': now(), 'pid': os.getpid(), 'gpu': reg['gpu']})
    write_json(root/'progress.json', {'at': now(), 'phase': 'warming', 'completed_requests': 0,
        'total_requests': 20, 'results': []})
    socket_root = ROOT/'tmp/rpc'/sha256(root/'registration.json')[:12];socket_root.mkdir(parents=True)
    planner_socket = socket_root/'plan.sock';proxy_socket = socket_root/'order.sock'
    children = {};streams = []

    def start(name, command, env):
        log = (root/(name+'.log')).open('x');streams.append(log)
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        children[name] = child
        write_json(root/(name+'_process.json'), {'at': now(), 'pid': child.pid, 'command': command})
        return child

    try:
        args = ['--kind', 'curobo_indexed_planning', '--root', str(root/'resident'),
            '--socket', str(planner_socket), '--warmup-trial', str(root/'warmup'),
            '--descriptor', str(root/'gripper_descriptor.json')]
        child = start('planner', [str(ROOT/'environments/curobo/bin/python'), '-u', '-m',
            'modules.curobo.indexed_planning.execution_service', *args], environment(reg['gpu']))
        wait_ready(child, root/'resident/curobo_indexed_planning_ready.json')
        args = ['--root', str(root/'proxy'), '--socket', str(proxy_socket), '--warmup-trial', str(root/'warmup'),
            '--descriptor', str(root/'gripper_descriptor.json'), '--campaign', str(root),
            '--planner-socket', str(planner_socket)]
        child = start('proxy', [str(ROOT/'environments/geometric_ik/bin/python'), '-u', '-m',
            'modules.pipeline.experiments.indexed_grasp_planning.physical.proxy', *args],
            dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
        wait_ready(child, root/'proxy/curobo_indexed_planning_ready.json', timeout=120)
        child = start('simulation', [str(ROOT/'environments/simulation/bin/python'), '-u', '-m',
            'modules.pipeline.experiments.indexed_grasp_planning.physical.simulation',
            '--root', str(root), '--socket', str(proxy_socket)], environment(reg['gpu']))
        while child.poll() is None:
            if any(p.poll() is not None for k,p in children.items() if k != 'simulation'):
                raise RuntimeError('Resident planning/proxy process exited during execution')
            time.sleep(1)
        if child.returncode != 0 or not (root/'execution_complete.json').exists():
            raise RuntimeError('Physical simulator stopped; see simulation.log and preserved STOP')
        guard(root)
    finally:
        for name, child in reversed(list(children.items())):
            if child.poll() is None:
                socket = planner_socket if name == 'planner' else proxy_socket if name == 'proxy' else None
                if socket:
                    try: Client(socket, timeout=10).call({'command': 'shutdown'})
                    except Exception: child.terminate()
                else: child.terminate()
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired: child.kill();child.wait(timeout=10)
        for stream in streams: stream.close()
        write_json(root/'services_stopped.json', {'at': now(),
            'all_owned_exited': all(p.poll() is not None for p in children.values()),
            'returncodes': {n:p.returncode for n,p in children.items()}})
    from .review import review
    from .report import report, publish
    checked = review(root);report(root)
    status = raw(root/'progress.json');status.update(at=now(), phase='complete');write_json(root/'progress.json', status)
    pointer = ROOT/'tmp/indexed_grasp_planning/funnel_active.json';active = raw(pointer)
    active.update(at=now(), phase='physical_pilot_complete', physical_run=str(root.relative_to(ROOT)),
        next_stage='review_physical_pilot_before_any_full_benchmark')
    write_json(pointer, active)
    try:
        publish(root)
    except Exception:
        write_json(root/'publication_error.json', {'at': now(), 'error': traceback.format_exc(),
            'physical_schedule_complete': True, 'requires_physical_retry': False})
        print(traceback.format_exc(), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--root', type=Path, required=True);args = p.parse_args()
    try: run(args.root)
    except BaseException:
        if not (args.root/'STOP').exists():
            write_json(args.root/'STOP', {'at': now(), 'error': traceback.format_exc(), 'automatic_retry': False})
        raise
