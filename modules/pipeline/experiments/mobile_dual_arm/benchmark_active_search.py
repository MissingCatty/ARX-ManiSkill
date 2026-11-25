"""Paired cached-observation station search, with no simulation or robot motion."""
import argparse
from pathlib import Path
import os
import shutil
import subprocess
import time
import traceback
from types import SimpleNamespace

import numpy as np

from arx_project.layout import ROOT, ensure_run_path, now, read_json, sha256, write_json
from modules.mobile_manipulation.station_search import StationSearch
from modules.pipeline.experiments.mobile_dual_arm.active_search import make_search, StationBackend


CASES = ['M0001_S1_open_apple', 'M0030_S2_edge_peach', 'M0036_S2_edge_spatula']


def prepare(root, source, gpu):
    if root.exists():
        raise ValueError('Replay registration already exists; use a new directory')
    root.mkdir(parents=True)
    old = read_json(source/'registration.json')
    files = {Path(p) for p in old['source_hashes']}
    for folder in ['modules/mobile_manipulation', 'modules/curobo/dual_arm',
                   'modules/pipeline/experiments/mobile_dual_arm']:
        files.update((ROOT/folder).glob('*.py'))
    files.update(ROOT/p for p in ['modules/reachability/orientation_index.py',
        'configs/modules.json', 'configs/experiments/inverse_reachability_placement.json',
        'configs/experiments/sam3_graspgenx_v4.json', 'tests/test_inverse_reachability.py'])
    source_hashes = {}
    for file in sorted(files):
        if not file.is_relative_to(ROOT):
            raise ValueError('Registered source outside project')
        relative = file.relative_to(ROOT)
        destination = root/'source_registered'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, destination)
        source_hashes[str(relative)] = sha256(file)
    input_hashes = {}
    for case in CASES:
        case_root = source/'trials'/case
        for name in ['stationing/coarse/state.json', 'stationing/coarse/candidates/candidate_pool.json',
                     'stationing/measurement.json']:
            file = case_root/name
            input_hashes[str(file.relative_to(ROOT))] = sha256(file)
    warm = root/'warmup'; warm.mkdir()
    for filename in ['state.json', 'closed_state.json']:
        src = source/'preflight/head_warmup'/filename
        shutil.copy2(src, warm/filename)
        input_hashes[str(src.relative_to(ROOT))] = sha256(src)
        input_hashes[str((warm/filename).relative_to(ROOT))] = sha256(warm/filename)
    config = read_json(ROOT/'configs/experiments/inverse_reachability_placement.json')
    index = ROOT/config['index_directory']
    for name in ['completion.json', 'order.npy', 'offsets.npy']:
        input_hashes[str((index/name).relative_to(ROOT))] = sha256(index/name)
    for name in ['完成.json', '地图元数据.json']:
        p = ROOT/config['map_directory']/name
        input_hashes[str(p.relative_to(ROOT))] = sha256(p)
    write_json(root/'registration.json', {'at': now(), 'source_run': str(source.relative_to(ROOT)),
        'cases': CASES, 'gpu': gpu, 'minimum_free_mib': 8000,
        'source_hashes': source_hashes, 'input_hashes': input_hashes,
        'settings': read_json(ROOT/'configs/experiments/sam3_graspgenx_v4.json')['placement'],
        'options': config, 'physical_grasp_budget': 0, 'simulation_instances': 0,
        'paired_search_requests': 6, 'maximum_full_planning_calls_per_search': 3,
        'baseline': 'Original geometry-grid proposals, same accelerated native dual-arm planner',
        'treatment': 'Indexed unchanged-grasp map placement, native endpoint gate, full-pool grid fallback',
        'coarse_observations': 'Identical cached pre-stationing perception; no simulator target pose for search',
        'fixture_selection': 'Prior engineering fixtures: apple, left-arm peach and long-search spatula',
        'user_authorization': 'User approved integrating the map station generator and offline complete-planning comparison',
        'resource_fault': 'Stop and retain; no automatic retry',
        'old_stopped_batch': 'Unchanged; no retries or remaining-scene execution'})


def verify(reg):
    for name, expected in {**reg['source_hashes'], **reg['input_hashes']}.items():
        if sha256(ROOT/name) != expected:
            raise RuntimeError('Registered source/input changed: '+name)


def run(root):
    reg = read_json(root/'registration.json')
    if (root/'started.json').exists() or (root/'STOP').exists():
        raise ValueError('Replay already started or stopped; keep prior evidence')
    verify(reg)
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free',
        '--format=csv,noheader,nounits'], text=True)
    free = dict((a.strip(), int(b)) for a, b in (line.split(',') for line in inventory.strip().splitlines()))
    if free[reg['gpu']] < reg['minimum_free_mib']:
        raise RuntimeError('Not enough free GPU memory for isolated replay')
    os.environ['CUDA_VISIBLE_DEVICES'] = reg['gpu']
    os.environ['GRASP_TRAJOPT_POLICY'] = reg['options']['trajopt_policy']
    import torch
    torch.set_num_threads(4)
    torch.zeros(1, device='cuda'); torch.cuda.synchronize()
    write_json(root/'started.json', {'at': now(), 'pid': os.getpid(),
        'gpu_uuid': reg['gpu'], 'gpu_free_mib': free[reg['gpu']], 'physical_grasps': 0})
    from modules.curobo.dual_arm.placement_service import create
    from modules.pipeline.experiments.dual_arm_grasp.benchmark_acceleration import reset
    warm_started = time.perf_counter()
    planner = create(SimpleNamespace(root=root/'resident', warmup_trial=root/'warmup'))
    warm_seconds = time.perf_counter()-warm_started
    baseline = StationSearch(planner.model, reg['settings'], StationBackend(planner.request))
    map_search = make_search(planner.model, reg['settings'], planner.request, reg['options'])
    from modules.mobile_manipulation.placement import SweepChecker
    geometry_started = time.perf_counter()
    initial = ROOT/reg['source_run']/'trials'/reg['cases'][0]/'stationing/coarse/state.json'
    SweepChecker(planner.model, read_json(initial), reg['settings']['transit_clearance_m'])
    geometry_seconds = time.perf_counter()-geometry_started
    from modules.curobo.dual_arm.profiling import SolverTrace
    trace = SolverTrace(torch.cuda.synchronize)
    trace.install(planner.pool)
    write_json(root/'readiness.json', {'at': now(), 'passed': True, 'planner_warmup_seconds': warm_seconds,
        'shared_geometry_startup_seconds': geometry_seconds, 'uniform_static_geometry_warmup': True,
        'runtime_environment': {'GRASP_TRAJOPT_POLICY': os.environ['GRASP_TRAJOPT_POLICY'],
            'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'),
            'OPENBLAS_NUM_THREADS': os.environ.get('OPENBLAS_NUM_THREADS')},
        'index_load_seconds': map_search.generator.index.load_seconds, 'same_resident_planner': True,
        'kinematics': planner.model.proof, 'collision_spheres': planner.model.sphere_proof,
        'fingertip_table_policy': {'profile_id': planner.pool.contact_profile.profile_id,
            'maximum_tip_penetration_m': planner.pool.contact_profile.maximum_penetration_m,
            'distal_finger_length_m': planner.pool.contact_profile.distal_length_m}})
    print('READY', warm_seconds, flush=True)
    pairs = []
    for number, case in enumerate(reg['cases']):
        case_root = ROOT/reg['source_run']/'trials'/case
        coarse = case_root/'stationing/coarse'
        state = read_json(coarse/'state.json')
        pool = read_json(coarse/'candidates/candidate_pool.json')
        target = read_json(case_root/'stationing/measurement.json')['world_median']
        pair = {'case': case}
        order = ['grid', 'map'] if number % 2 == 0 else ['map', 'grid']
        for mode in order:
            verify(reg)
            reset(planner.pool)
            trace.reset()
            print('SEARCH_BEGIN', case, mode, flush=True)
            write_json(root/'progress.json', {'phase': 'searching', 'case': case, 'mode': mode,
                'completed_pairs': len(pairs), 'completed_results': pairs, 'at': now()})
            selected, report = (baseline if mode == 'grid' else map_search).select(
                state, pool, target, root/'searches'/case/mode)
            torch.cuda.synchronize()
            core = trace.report()
            write_json(root/'searches'/case/mode/'solver_timing.json', core)
            pair[mode] = {'success': report['success'], 'seconds': report['total_seconds'],
                'solver_components': core['components'],
                'timing_seconds': report['timing_seconds'], 'full_planning_calls': report['full_planning_calls'],
                'endpoint_checks': report['endpoint_checks'], 'fallback_used': report['fallback_used'],
                'selected_arm': selected['plan_result']['selected']['arm'] if selected else None,
                'selected_candidate': selected['plan_result']['selected']['index'] if selected else None,
                'selected_source': selected['proposal']['source'] if selected else None}
            write_json(root/'pairs'/f'{case}.json', pair)
            print('SEARCH_END', case, mode, pair[mode], flush=True)
        pair['observed_regression'] = pair['grid']['success'] and not pair['map']['success']
        pair['speedup'] = pair['grid']['seconds']/pair['map']['seconds']
        pairs.append(pair)
    verify(reg)
    acceptance = {'at': now(), 'passed': True, 'complete': True, 'physical_grasps': 0,
        'same_gpu': reg['gpu'], 'paired_cases': pairs,
        'baseline_planning_successes': sum(p['grid']['success'] for p in pairs),
        'map_planning_successes': sum(p['map']['success'] for p in pairs),
        'observed_regressions': sum(p['observed_regression'] for p in pairs),
        'grid_total_seconds': sum(p['grid']['seconds'] for p in pairs),
        'map_total_seconds': sum(p['map']['seconds'] for p in pairs),
        'source_and_input_checksums_unchanged': True, 'default_activation': False}
    write_json(root/'acceptance.json', acceptance)
    write_json(root/'progress.json', {'phase': 'complete', **acceptance})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--gpu')
    args = parser.parse_args()
    root = ensure_run_path(args.root)
    if args.stage == 'prepare':
        prepare(root, args.source.resolve(), args.gpu)
    else:
        try:
            run(root)
        except BaseException:
            write_json(root/'STOP', {'at': now(), 'error': traceback.format_exc(), 'physical_grasps': 0})
            raise
