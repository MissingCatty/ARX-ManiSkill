"""Compare native batch capacities on frozen V6 inputs without simulation."""
import argparse
from itertools import permutations
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np

from arx_project.layout import ROOT, ensure_run_path, now, read_json, sha256, write_json

VARIANTS = {'control': 0, 'batch16': 16, 'batch32': 32}


def prepare(root, source, gpu, selected=None):
    if root.exists():
        raise ValueError('Use a fresh V8 registration; existing attempts are preserved')
    from modules.pipeline.experiments.current_configuration.configuration import load_current
    current = load_current(verify_dataset=True)
    acceptance = read_json(source/'acceptance.json')
    if not acceptance['passed'] or not acceptance['task_complete']:
        raise ValueError('The source V6 comparison is not complete and accepted')
    cases, excluded, input_hashes = [], [], {}
    trials = source/'methods/graspgenx/trials'
    for directory in sorted(trials.iterdir()):
        if not directory.is_dir():continue
        pool = directory/'candidates/candidate_pool.json'
        if not pool.is_file():
            excluded.append(directory.name)
            continue
        if selected and directory.name not in selected:continue
        cases.append({'id': directory.name, 'source_directory': str(directory)})
        for path in [directory/'state.json', pool, directory/'plan_result.json']:
            input_hashes[str(path)] = sha256(path)
    if selected and {c['id'] for c in cases} != set(selected):
        raise ValueError('Some selected engineering fixtures have no frozen candidate observation')
    if not selected and (len(cases) != 69 or len(excluded) != 5):
        raise ValueError('The full V8 cohort must contain 69 observations and five stationing failures')
    root.mkdir(parents=True)
    warmup = source/'methods/zerograsp/preflight/head_warmup'
    for name in ['state.json', 'closed_state.json']:
        target = root/'warmup'/name; target.parent.mkdir(exist_ok=True)
        shutil.copy2(warmup/name, target)
        input_hashes[str(warmup/name)] = sha256(warmup/name)
    source_files = [ROOT/'configs/modules.json', ROOT/'configs/experiments/active_experiment.json',
        ROOT/'configs/experiments/dual_arm_interleaving.json', ROOT/'configs/policies/fingertip_table.json',
        ROOT/'modules/curobo/arx_adapter.py', ROOT/'modules/maniskill/resident_planner.py',
        ROOT/'modules/maniskill/fast_resident_planner.py', ROOT/'modules/maniskill/arx_pipeline/planner.py',
        ROOT/'modules/maniskill/checked_native_services.py', ROOT/'modules/curobo/scene_cache.py', Path(__file__)]
    for folder in ['modules/curobo/dual_arm', 'modules/curobo/fingertip_contact',
                   'modules/curobo/candidate_io', 'modules/curobo/indexed_planning']:
        source_files.extend((ROOT/folder).glob('*.py'))
    source_files += [Path(name) for name in current['configuration_hashes']]
    source_files += [ROOT/'modules/curobo/source/curobo/_src/solver'/name
        for name in ['solver_ik.py','solver_ik_cfg.py']]
    source_files = sorted(set(source_files))
    for path in source_files:
        target = root/'source_registered'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
    write_json(root/'registration.json', {'at': now(), 'version': 'V8',
        'user_authorization': '2026-11-26：开始吧，执行V8优化方案；先做批量IK独立规划对照。',
        'source_run': str(source), 'cases': cases, 'common_stationing_failures': excluded,
        'case_selection': 'posthoc_engineering_fixtures' if selected else 'all_69_frozen_observations_no_outcome_filtering',
        'gpu': gpu, 'minimum_startup_free_mib': 10000, 'minimum_request_free_mib': 1536,
        'variants': VARIANTS, 'planning_requests': len(cases)*len(VARIANTS),
        'physical_grasp_budget': 0, 'simulation_instances': 0, 'perception_calls': 0,
        'automatic_retries': False, 'random_seed': 20261126,
        'scheduler': {'first': 16, 'subsequent': 64},
        'source_hashes': {str(p): sha256(p) for p in source_files}, 'input_hashes': input_hashes})
    write_json(root/'progress.json', {'at': now(), 'phase': 'registered',
        'completed_requests': 0, 'total_requests': len(cases)*len(VARIANTS)})


def verify_hashes(registration):
    for filename, digest in {**registration['source_hashes'], **registration['input_hashes']}.items():
        if sha256(filename) != digest:
            raise RuntimeError('Registered planning source/input changed: '+filename)


def run(root):
    from types import SimpleNamespace
    reg = read_json(root/'registration.json')
    if (root/'started.json').exists() or (root/'STOP').exists():
        raise ValueError('This V8 registration already started; create a new run')
    verify_hashes(reg)
    os.environ.update(CUDA_VISIBLE_DEVICES=reg['gpu'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
        GRASP_TRAJOPT_POLICY='interactive', CUBLAS_WORKSPACE_CONFIG=':4096:8',
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    import torch
    from arx_project.artifacts import publish_candidate_pool, finish_candidate_transport, close_owned_transports
    from modules.curobo.indexed_planning.service import create
    from modules.curobo.scene_cache import mesh_stores, prune_inactive_meshes
    torch.set_num_threads(4)
    free, total = torch.cuda.mem_get_info()
    if free/1024**2 < reg['minimum_startup_free_mib']:
        raise RuntimeError('Insufficient free CUDA memory before V8 model loading')
    torch.zeros(1, device='cuda'); torch.cuda.synchronize()
    write_json(root/'started.json', {'at': now(), 'pid': os.getpid(), 'gpu': reg['gpu'],
        'device': torch.cuda.get_device_name(), 'free_mib': free/1024**2})
    begin = time.perf_counter()
    planner = create(SimpleNamespace(root=root/'resident', warmup_trial=root/'warmup',
        ik_batch_size=16, warm_batch_sizes=(16, 32)))
    pool = planner.pool
    from modules.curobo.indexed_planning.warmup import warm_solver_branches
    branches = warm_solver_branches(planner,root/'warmup/state.json',root/'solver_prewarm')
    write_json(root/'startup.json', {'at': now(), 'seconds': time.perf_counter()-begin,
        'model_loads': pool.loads, 'all_variants_share_models_and_current_scene': True,
        'batch_capacities_warmed': sorted(pool.batch_engines),
        'negative_optimizer_paths_and_partial_batch_rows_verified':branches['passed'],
        'allocated_mib': torch.cuda.memory_allocated()/1024**2,
        'reserved_mib': torch.cuda.memory_reserved()/1024**2,
        'peak_allocated_mib': torch.cuda.max_memory_allocated()/1024**2})
    completed, rows = 0, []
    orders = list(permutations(VARIANTS))
    try:
        for ordinal, case in enumerate(reg['cases']):
            pair = {}
            for variant in orders[ordinal % len(orders)]:
                free, _ = torch.cuda.mem_get_info()
                if free/1024**2 < reg['minimum_request_free_mib']:
                    raise RuntimeError('Insufficient CUDA reserve before next V8 request')
                directory = root/'plans'/case['id']/variant
                directory.mkdir(parents=True)
                source = Path(case['source_directory'])
                shutil.copy2(source/'state.json', directory/'state.json')
                payload = read_json(source/'candidates/candidate_pool.json')
                descriptor = publish_candidate_pool(directory, payload)
                pool.ik_batch_size = VARIANTS[variant]
                torch.manual_seed(reg['random_seed']); np.random.seed(reg['random_seed'])
                for context in pool.contexts.values():context.planner.reset_seed()
                pool.pure_solver.reset_seed()
                for engine in pool.batch_engines.values():engine.reset_seed()
                write_json(root/'progress.json', {'at': now(), 'phase': 'planning',
                    'completed_requests': completed, 'total_requests': reg['planning_requests'],
                    'current': {'id': case['id'], 'variant': variant}, 'results': rows})
                print('PLAN_BEGIN', case['id'], variant, flush=True)
                torch.cuda.synchronize(); began = time.perf_counter()
                result = planner.request(directory, candidate_transport=descriptor)
                torch.cuda.synchronize(); seconds = time.perf_counter()-began
                finish_candidate_transport(directory)
                io = read_json(directory/'candidate_io.json')
                if io['source'] != 'shared_memory' or io['decode_count'] != 1 or io['full_arm_pool_copies'] != 0:
                    raise RuntimeError('V8 replay changed the shared-memory candidate contract')
                candidate_checks, batches = 0, []
                for side in ['right', 'left']:
                    path = directory/'arms'/side/'plan_result.json'
                    if path.is_file():candidate_checks += len(read_json(path).get('candidate_events', []))
                    for receipt in sorted((directory/'arms'/side/'search_windows').glob('*.json')):
                        batch = read_json(receipt)['result'].get('batch_ik')
                        if batch:batches.append({'arm': side, **batch})
                pair[variant] = {'seconds': seconds, 'success': result['success'],
                    'selected_arm': result.get('selected', {}).get('arm'),
                    'selected_source_index': result.get('selected', {}).get('index'),
                    'candidate_checks': candidate_checks, 'batches': batches,
                    'request_free_mib': free/1024**2,
                    'selected_contact_pose_world': result.get('selected', {}).get('contact_pose_world')}
                write_json(directory/'replay_metrics.json', pair[variant])
                completed += 1
                print('PLAN_END', case['id'], variant, round(seconds, 4),
                    'success', result['success'], 'checks', candidate_checks, flush=True)
                prune_inactive_meshes(mesh_stores(planner), torch.cuda.synchronize)
            control = pair['control']
            comparison = {mode: {'success_retained': not control['success'] or value['success'],
                'same_selection': all(value[k] == control[k] for k in
                    ['success', 'selected_arm', 'selected_source_index', 'selected_contact_pose_world']),
                'speedup': control['seconds']/value['seconds']} for mode, value in pair.items() if mode != 'control'}
            rows.append({'id': case['id'], 'variants': pair, 'comparison': comparison})
            write_json(root/'progress.json', {'at': now(), 'phase': 'comparison',
                'completed_requests': completed, 'total_requests': reg['planning_requests'], 'results': rows})
        verify_hashes(reg)
        summaries = {}
        for mode in VARIANTS:
            times = [r['variants'][mode]['seconds'] for r in rows]
            summaries[mode] = {'requests': len(times), 'successes': sum(r['variants'][mode]['success'] for r in rows),
                'median_seconds': float(np.median(times)), 'p90_seconds': float(np.percentile(times, 90)),
                'total_seconds': sum(times)}
        retained = {m: all(r['comparison'][m]['success_retained'] for r in rows) for m in ['batch16', 'batch32']}
        write_json(root/'acceptance.json', {'at': now(), 'passed': True, 'record_integrity': True,
            'task_complete': completed == reg['planning_requests'], 'planning_requests': completed,
            'summaries': summaries, 'control_successes_retained': retained, 'paired_cases': rows,
            'physical_grasps': 0, 'simulation_instances': 0, 'perception_calls': 0,
            'same_gpu_and_resident_models': True, 'automatic_default_activation': False})
        write_json(root/'progress.json', {'at': now(), 'phase': 'complete',
            'completed_requests': completed, 'total_requests': reg['planning_requests'], 'results': rows})
    finally:
        close_owned_transports()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--gpu')
    parser.add_argument('--cases', nargs='+')
    args = parser.parse_args(); root = ensure_run_path(args.root)
    if args.stage == 'prepare':
        prepare(root, args.source.resolve(), args.gpu, args.cases)
    else:
        try:run(root)
        except BaseException:
            write_json(root/'STOP', {'at': now(), 'error': traceback.format_exc(), 'physical_grasps': 0})
            raise


if __name__=='__main__':main()
