"""Register and replay existing grasp requests; no simulator or robot actions."""
import argparse
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

from arx_project.layout import ROOT, ensure_run_path, now, read_json, sha256, write_json

CASES = ['M0002_S1_open_avocado', 'M0004_S1_open_banana',
         'M0005_S1_open_bar_soap', 'M0011_S1_open_bread']
BASELINE = ROOT/'tmp/mobile_dual_arm/v4_high100_resume_20261125_02'


def prepare(root, staged, gpu):
    if (root/'registration.json').exists():
        raise RuntimeError('Replay registration exists')
    root.mkdir(parents=True, exist_ok=True)
    stage = read_json(staged/'staging.json')
    for relative, entry in stage['files'].items():
        if sha256(ROOT/relative) != entry['before_sha256']:
            raise RuntimeError('Running source differs from staged predecessor: '+relative)
    schedule = []
    for case in CASES:
        original = BASELINE/'trials'/case
        inputs = ['state.json', 'candidates/candidate_pool.json']
        schedule.append({'id': case, 'source': str(original),
            'input_hashes': {name: sha256(original/name) for name in inputs}})
        for mode in ['before', 'after']:
            target = root/mode/'trials'/case
            for name in inputs:
                destination = target/name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original/name, destination)
    warm = root/'warmup'
    warm.mkdir()
    for name in ['state.json', 'closed_state.json']:
        shutil.copy2(BASELINE/'preflight/head_warmup'/name, warm/name)
    shutil.copy2(BASELINE/'gripper_descriptor.json', root/'gripper_descriptor.json')
    write_json(root/'registration.json', {'at': now(), 'gpu': gpu, 'staged': str(staged),
        'stage_sha256': sha256(staged/'staging.json'), 'schedule': schedule,
        'physical_grasp_budget': 0, 'simulator_instances': 0,
        'purpose': 'Same existing native candidates, same native solver and collision rules; I/O implementation only',
        'random_seed': 2026112501, 'policy': 'interactive', 'configuration_hashes': {
            'active_experiment': sha256(ROOT/'configs/experiments/active_experiment.json'),
            'modules': sha256(ROOT/'configs/modules.json')}})


def execute(root, mode):
    reg = read_json(root/'registration.json')
    if (root/mode/'started.json').exists():
        raise RuntimeError('This native replay already started')
    write_json(root/mode/'started.json', {'at': now(), 'pid': os.getpid()})
    if mode == 'after':
        from modules.curobo.candidate_io.loader import load_staged, load_file
        load_staged(Path(reg['staged']), planning=True, client=False)
    from modules.curobo.dual_arm.placement_service import create
    args = SimpleNamespace(root=root/mode/'resident', warmup_trial=root/'warmup',
                           descriptor=root/'gripper_descriptor.json')
    args.root.mkdir()
    planner = create(args)
    import numpy as np
    import torch
    rows = []
    for index, item in enumerate(reg['schedule']):
        directory = root/mode/'trials'/item['id']
        for name, digest in item['input_hashes'].items():
            if sha256(directory/name) != digest:
                raise RuntimeError('Replay input changed')
        seed = reg['random_seed']+index
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        kwargs = {}
        if mode == 'after':
            from arx_project.artifacts import publish_candidate_pool, finish_candidate_transport
            source = directory/'candidates/candidate_pool.json'
            payload = json.loads(source.read_text())
            source.rename(source.with_name('registered_source_pool.json'))
            kwargs['candidate_transport'] = publish_candidate_pool(directory, payload)
        torch.cuda.synchronize(); started = time.perf_counter()
        result = planner.request(directory, **kwargs)
        torch.cuda.synchronize()
        wall = time.perf_counter()-started
        if mode == 'after':
            finish_candidate_transport(directory)
        rows.append({'id': item['id'], 'wall_seconds': wall, 'success': result['success'],
            'selected': result.get('selected'), 'arms': result.get('arm_results'),
            'rank_bounds': result.get('acceleration', {}).get('rank_bounds')})
        print('REPLAY', mode, item['id'], round(wall, 3), result['success'],
            result.get('selected', {}).get('arm'), result.get('selected', {}).get('index'), flush=True)
    write_json(root/mode/'summary.json', {'at': now(), 'mode': mode, 'results': rows,
        'physical_grasps': 0, 'simulator_instances': 0})


def compare(root):
    import numpy as np
    before, after = [read_json(root/mode/'summary.json')['results'] for mode in ['before', 'after']]
    rows = []
    for left, right in zip(before, after):
        if left['id'] != right['id']:
            raise RuntimeError('Replay order changed')
        passed = left['success'] == right['success']
        errors = {}
        if left['success'] and right['success']:
            a, b = left['selected'], right['selected']
            passed &= a['arm'] == b['arm'] and a['index'] == b['index'] and a['score'] == b['score']
            for key in ['contact_pose_world', 'grasp_world', 'pregrasp_world', 'lift_world']:
                difference = float(np.abs(np.asarray(a[key])-b[key]).max())
                errors[key] = difference
                passed &= difference == 0.
            for key in ['q_grasp', 'q_pre', 'q_lift']:
                difference = float(np.abs(np.asarray(a[key])-b[key]).max())
                errors[key] = difference
                passed &= difference <= 1e-4
        rows.append({'id': left['id'], 'passed': bool(passed), 'maximum_differences': errors,
            'before_seconds': left['wall_seconds'], 'after_seconds': right['wall_seconds'],
            'speedup': left['wall_seconds']/right['wall_seconds'],
            'arm': right.get('selected', {}).get('arm'), 'index': right.get('selected', {}).get('index')})
    report = {'at': now(), 'passed': all(row['passed'] for row in rows), 'results': rows,
        'physical_grasps': 0, 'simulator_instances': 0,
        'method': 'Same GPU, input hashes, source model, collision policy and seeds; serialized native replays'}
    write_json(root/'verification.json', report)
    return report


def launch(root, staged, gpu):
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free',
                                  '--format=csv,noheader,nounits'], text=True)
    free = {a.strip(): int(b) for a,b in (line.split(',') for line in text.strip().splitlines())}
    if free.get(gpu, 0) < 10000:
        raise RuntimeError('Native replay requires 10000 MiB free before model loading')
    prepare(root, staged, gpu)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, ARX_PLANNER_GPU=gpu, ARX_SIM_GPU=gpu,
        LILO_GPU=gpu, GRASP_TRAJOPT_POLICY='interactive', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    for mode in ['before', 'after']:
        command = ['bash', '-c', 'source "$1" || exit; shift; exec "$@"', 'candidate-io-replay',
            str(ROOT/'modules/arx_core/scripts/curobo_arx/environment.sh'),
            str(ROOT/'environments/curobo/bin/python'), '-u', '-m',
            'modules.curobo.candidate_io.replay', 'execute', '--root', str(root), '--mode', mode]
        with (root/(mode+'.log')).open('x') as stream:
            subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                           check=True, timeout=1800)
    result = compare(root)
    print('REPLAY_VERIFIED', result['passed'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['launch', 'execute', 'compare'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--staged', type=Path)
    parser.add_argument('--gpu')
    parser.add_argument('--mode', choices=['before', 'after'])
    args = parser.parse_args()
    root = ensure_run_path(args.root)
    if args.stage == 'execute':execute(root, args.mode)
    elif args.stage == 'compare':print(compare(root))
    else:launch(root, args.staged.resolve(), args.gpu)
