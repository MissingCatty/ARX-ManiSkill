"""Compare accelerated planning on frozen observations, with zero robot motion."""
import argparse
from copy import deepcopy
from pathlib import Path
import os
import shutil
import subprocess
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from arx_project.layout import ROOT, ensure_run_path, now, read_json, sha256, write_json
from modules.curobo.dual_arm.adapter import choose_plan


def historical_proof(source, root):
    reports = []
    for trial in sorted((source/'trials').iterdir()):
        if not (trial/'result.json').exists():
            continue
        for directory in [*(trial/'stationing').glob('candidate_*'), trial]:
            if not (directory/'plan_result.json').exists():
                continue
            originals = {side: read_json(directory/'arms'/side/'plan_result.json') for side in ['right', 'left']}
            state = read_json(directory/'state.json')
            expected = choose_plan(originals, state)
            bounded = deepcopy(originals)
            right = originals['right']
            limit = None
            skipped = 0
            if right['success']:
                limit = right['selected']['index']+1
                left = bounded['left']
                events = left.get('candidate_events', [])
                skipped = max(0, len(events)-limit)
                if left['success'] and left['selected']['index'] >= limit:
                    bounded['left'] = {'success': False}
            actual = choose_plan(bounded, state)
            if actual != expected:
                raise RuntimeError('Recorded ranking equivalence failed')
            reports.append({'directory': str(directory.relative_to(source)), 'expected_arm': expected,
                'bounded_arm': actual, 'second_arm_prefix': limit, 'recorded_candidate_checks_skipped': skipped})
    report = {'passed': True, 'recorded_plans': len(reports),
              'recorded_second_arm_checks_omittable': sum(r['recorded_candidate_checks_skipped'] for r in reports),
              'basis': 'Recorded feasibility outcomes and unchanged tie-break; no new IK or physical grasp',
              'results': reports}
    write_json(root/'historical_rank_proof.json', report)
    return report


def prepare(source, root, gpu):
    if root.exists():
        raise ValueError('Use a new comparison registration')
    reg = read_json(source/'registration.json')
    completed = {p.parent.name: read_json(p) for p in (source/'trials').glob('*/result.json')}
    right = next(k for k, v in sorted(completed.items()) if v['success'] and v.get('selected_arm') == 'right')
    left = min((k for k, v in completed.items() if v['success'] and v.get('selected_arm') == 'left'),
               key=lambda k: completed[k]['wall_seconds'])
    slow = 'M0036_S2_edge_spatula'
    cases = list(dict.fromkeys([right, left, slow]))
    if len(cases) != 3 or slow not in completed:
        raise ValueError('Need right, left and full-secondary-scan completed fixtures')
    root.mkdir(parents=True)
    (root/'warmup').mkdir()
    for filename in ['state.json', 'closed_state.json']:
        shutil.copy2(source/'preflight/head_warmup'/filename, root/'warmup'/filename)
    inputs = {}
    for case in cases:
        for name in ['state.json', 'candidates/candidate_pool.json']:
            path = source/'trials'/case/name
            inputs[str(path)] = sha256(path)
    sources = {str(ROOT/name): sha256(ROOT/name) for name in [
        'modules/curobo/dual_arm/accelerated_adapter.py',
        'modules/pipeline/experiments/dual_arm_grasp/benchmark_acceleration.py']}
    sources.update(reg['source_hashes'])
    write_json(root/'registration.json', {'at': now(), 'source_run': str(source), 'cases': cases,
        'gpu': gpu, 'minimum_free_mib': 8000, 'source_hashes': sources, 'input_hashes': inputs,
        'physical_grasp_budget': 0, 'simulation_instances': 0, 'planning_comparison_requests': 6,
        'same_gpu_pairwise_measurements': True, 'selection': 'posthoc engineering fixtures: right/left/long failed secondary search',
        'user_authorization': 'User approved bounding the second-arm candidate search and batching opening checks',
        'baseline_and_accelerated_use_same_collision_models': True})
    historical_proof(source, root)


def reset(pool):
    import torch
    torch.manual_seed(20261125)
    np.random.seed(20261125)
    for context in pool.contexts.values():
        context.planner.reset_seed()
    if pool.pure_solver is not None:
        pool.pure_solver.reset_seed()


def serial_opening(pool, work, canonical):
    import torch
    from modules.curobo.dual_arm.accelerated_adapter import opening_samples
    fractions, target, observed = opening_samples(canonical)
    valid, spheres = [], []
    started = time.perf_counter()
    for fraction in fractions:
        state = deepcopy(canonical)
        for name, value in observed.items():
            state['qpos'][state['joint_names'].index(name)] = float(value+fraction*(target-value))
        passed = True
        try:
            context = pool.bind(work, state, include_target=True)
        except AssertionError as error:
            if str(error) != 'Measured state collides in unchanged ARX model':
                raise
            context, passed = pool.contexts['open'], False
        q = context.measured(state)[None, None]
        spheres.append(context.checker.get_kinematics(q).robot_spheres.detach().cpu().numpy().copy())
        valid.append(passed)
    torch.cuda.synchronize()
    return {'valid': valid, 'spheres': np.concatenate(spheres, axis=0),
            'wall_seconds': time.perf_counter()-started}


def compare_opening(original, accelerated, state, directory, side):
    from modules.curobo.dual_arm import accelerated_adapter as algorithm
    write_json(directory/'candidates/candidate_pool.json', {'candidates': [], 'scope': 'opening_gate_only'})
    work = original.prepare_arm(directory, state, side, 'state.json', planned_open=True)
    canonical = read_json(work/'state.json')
    reference = serial_opening(original.pool, work/'serial_opening', canonical)
    captured = []
    native = algorithm.validate_sphere_batch
    def capture(checker, spheres, q):
        captured.append(spheres.detach().cpu().numpy().copy())
        return native(checker, spheres, q)
    with patch.object(algorithm, 'validate_sphere_batch', capture):
        actual = accelerated.opening_sweep(work, canonical)
    expected_invalid = np.flatnonzero(~np.asarray(reference['valid']))
    expected_first = int(expected_invalid[0]) if len(expected_invalid) else None
    passed = actual.get('first_invalid_sample') == expected_first
    error = None
    if captured:
        error = float(np.abs(reference['spheres']-captured[0]).max())
        passed &= error <= 2e-7 and reference['valid'] == actual['valid_per_sample']
        passed &= np.array_equal(reference['spheres'][..., 3], captured[0][..., 3])
    elif expected_first != 0:
        passed = False
    report = {'passed': bool(passed), 'side': side, 'samples': len(reference['valid']),
        'maximum_sphere_difference_m': error, 'first_invalid_sample': expected_first,
        'serial_seconds': reference['wall_seconds'], 'batched_seconds': actual['wall_seconds'],
        'serial_valid_per_sample': reference['valid'], 'batched': actual}
    write_json(work/'opening_equivalence.json', report)
    if not passed:
        raise RuntimeError('Native opening mask/FK equivalence failed')
    return report, canonical, reference['spheres']


def run(root):
    reg = read_json(root/'registration.json')
    if (root/'started.json').exists():
        raise ValueError('Comparison already started; retain prior evidence')
    for name, digest in {**reg['source_hashes'], **reg['input_hashes']}.items():
        if sha256(name) != digest:
            raise RuntimeError('Registered comparison input/source changed: '+name)
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free',
        '--format=csv,noheader,nounits'], text=True)
    free = dict((a.strip(), int(b)) for a, b in (row.split(',') for row in inventory.strip().splitlines()))
    if free[reg['gpu']] < reg['minimum_free_mib']:
        raise RuntimeError('Insufficient free memory for isolated replay')
    os.environ['CUDA_VISIBLE_DEVICES'] = reg['gpu']
    from modules.curobo.dual_arm.adapter import create
    from modules.curobo.dual_arm.accelerated_adapter import AcceleratedDualArmPlanner
    import torch
    torch.set_num_threads(4)
    probe = torch.zeros(1, device='cuda'); torch.cuda.synchronize()
    write_json(root/'started.json', {'at': now(), 'pid': os.getpid(), 'gpu_name': torch.cuda.get_device_name(),
                                   'gpu_free_before_start_mib': free[reg['gpu']]})
    original = create(SimpleNamespace(root=root/'resident', warmup_trial=root/'warmup'))
    accelerated = AcceleratedDualArmPlanner(original.pool, original.model, original.root)
    source = Path(reg['source_run'])
    state = read_json(source/'trials'/reg['cases'][0]/'state.json')
    openings = []
    for side in ['right', 'left']:
        report, canonical, spheres = compare_opening(original, accelerated, state, root/'opening'/side, side)
        openings.append(report)
        print('OPENING', side, report, flush=True)
        if side == 'left':
            # Place a tiny obstacle in the finger sweep, with both endpoint
            # configurations clear, to detect endpoint-only implementations.
            midpoint = spheres[len(spheres)//2, 0, :, :3]
            ends = spheres[[0, -1], 0]
            margins = []
            for point in midpoint:
                clearance = (np.linalg.norm(ends[..., :3]-point, axis=-1)-ends[..., 3]).min()
                margins.append(clearance)
            index = int(np.argmax(margins))
            if margins[index] <= .001:
                raise RuntimeError('Could not construct an endpoint-clear interior collision fixture')
            body = np.asarray(canonical['links']['body4'])
            point = midpoint[index]@body[:3, :3].T+body[:3, 3]
            obstructed = deepcopy(state)
            obstructed['scene_cuboids']['opening_probe'] = {'center': point.tolist(), 'dims': [.001]*3}
            report, _, _ = compare_opening(original, accelerated, obstructed, root/'opening/interior_obstacle', side)
            if report['first_invalid_sample'] in [None, 0, report['samples']-1]:
                raise RuntimeError('Interior collision fixture did not exercise intermediate widths')
            openings.append(report)
            print('OPENING_INTERIOR_COLLISION', report, flush=True)
    results = []
    for i, case in enumerate(reg['cases']):
        pair = {}
        modes = ['original', 'accelerated'] if i % 2 == 0 else ['accelerated', 'original']
        for mode in modes:
            directory = root/'plans'/case/mode
            (directory/'candidates').mkdir(parents=True)
            for name in ['state.json', 'candidates/candidate_pool.json']:
                shutil.copy2(source/'trials'/case/name, directory/name)
            reset(original.pool)
            started = time.perf_counter()
            print('PLAN_BEGIN', case, mode, flush=True)
            result = (original if mode == 'original' else accelerated).request(directory)
            pair[mode] = {'seconds': time.perf_counter()-started, 'result': result}
            print('PLAN_END', case, mode, pair[mode]['seconds'], result['success'], flush=True)
        a, b = [pair[m]['result'] for m in ['original', 'accelerated']]
        passed = a['success'] == b['success']
        if a['success']:
            passed &= a['selected']['arm'] == b['selected']['arm']
            passed &= a['selected']['index'] == b['selected']['index']
            passed &= np.array_equal(a['selected']['contact_pose_world'], b['selected']['contact_pose_world'])
        report = {'case': case, 'passed': bool(passed),
            'original_seconds': pair['original']['seconds'], 'accelerated_seconds': pair['accelerated']['seconds'],
            'speedup': pair['original']['seconds']/pair['accelerated']['seconds'],
            'original_arm': a.get('selected', {}).get('arm'), 'accelerated_arm': b.get('selected', {}).get('arm'),
            'original_index': a.get('selected', {}).get('index'), 'accelerated_index': b.get('selected', {}).get('index')}
        results.append(report)
        write_json(root/'progress.json', {'phase': 'planning_replay', 'completed_pairs': len(results), 'results': results})
        if not passed:
            raise RuntimeError('Paired planning selected a different grasp or arm')
    write_json(root/'acceptance.json', {'at': now(), 'passed': True, 'physical_grasps': 0,
        'same_gpu': reg['gpu'], 'paired_cases': results, 'opening_checks': openings,
        'historical_rank_proof': read_json(root/'historical_rank_proof.json')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--gpu')
    args = parser.parse_args()
    destination = ensure_run_path(args.root)
    if args.stage == 'prepare':
        prepare(args.source.resolve(), destination, args.gpu)
    else:
        try:
            run(destination)
        except BaseException:
            write_json(destination/'STOP', {'at': now(), 'error': traceback.format_exc(), 'physical_grasps': 0})
            raise
