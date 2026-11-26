"""Compare request-scoped arm contexts against scalar, shared-pool planning."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from types import SimpleNamespace

import numpy as np

from arx_project.layout import ROOT, ensure_run_path, now, read_json, sha256, write_json
from .replay import prepare as prepare_inputs, verify_hashes


def prepare(root, gpu, selected=None):
    config = read_json(ROOT/'configs/experiments/indexed_grasp_planning.json')
    previous = ROOT/config['native_batch_implementation']['full_run']
    accepted = read_json(previous/'acceptance.json')
    reviewed = read_json(previous/'independent_review.json')
    if not accepted['task_complete'] or not reviewed['passed']:
        raise RuntimeError('Finish and independently accept the native-batch replay first')
    source = ROOT/config['baseline_run']
    prepare_inputs(root, source, gpu, selected)
    reg = read_json(root/'registration.json')
    reg.update(experiment='binding_reuse', native_batching=False,
        user_authorization='2026-11-26：按V8优化方案继续，批量完成后推进上下文复用。',
        variants=({'control_1': False, 'reuse_1': True, 'reuse_2': True, 'control_2': False}
            if selected else {'control': False, 'reuse': True}),
        preceding_native_batch_review=str(previous/'independent_review.json'),
        comparison='C_alone_versus_A_both_scalar_IK_same_16_64_scheduler',
        cache_lifetime='one_complete_planning_request',
        independent_arm_contexts=True, shared_pure_kinematic_solver=True,
        stale_world_receipt_or_start_gate_fault='stop_no_automatic_retry')
    reg['planning_requests'] = len(reg['cases'])*len(reg['variants'])
    for filename in ['binding_replay.py', 'binding_followup.py']:
        path = Path(__file__).with_name(filename)
        target = root/'source_registered'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        reg['source_hashes'][str(path)] = sha256(path)
    policy = ROOT/'configs/experiments/indexed_grasp_binding_reuse.json'
    reg['source_hashes'][str(policy)] = sha256(policy)
    reg['model_epoch'] = hashlib.sha256(json.dumps(reg['source_hashes'],
        sort_keys=True).encode()).hexdigest()
    write_json(root/'registration.json', reg)
    write_json(root/'progress.json', {'at': now(), 'phase': 'registered',
        'completed_requests': 0, 'total_requests': reg['planning_requests']})


def launch(root, gpu, selected=None):
    config_path = ROOT/'configs/experiments/indexed_grasp_planning.json'
    config = read_json(config_path)
    active = ROOT/config['execution']['active_run']
    if (active/'controller.json').is_file():
        unit = read_json(active/'controller.json')['unit']
        state = subprocess.check_output(['systemctl', '--user', 'show', unit,
            '-p', 'MainPID', '-p', 'ActiveState'], text=True)
        fields = dict(line.split('=', 1) for line in state.strip().splitlines())
        if fields.get('MainPID', '0') != '0' or fields.get('ActiveState') in ['active', 'activating']:
            raise RuntimeError('The preceding registered V8 planner has not exited')
    prepare(root, gpu, selected)
    write_json(root/'configuration_before.json', config)
    config['execution'].update(active_run=str(root.relative_to(ROOT)),
        current_phase='binding_reuse_pilot' if selected else 'binding_reuse_full_replay',
        next_stage='independently_check_binding_reuse_before_geometric_and_index_variants')
    config['implementation_status'] = 'native_batch_accepted_binding_reuse_in_progress'
    record = config.setdefault('binding_reuse_implementation', {})
    record.update(adapter='modules/curobo/indexed_planning/binding_service.py',
        configuration='configs/experiments/indexed_grasp_binding_reuse.json',
        offline_checks=20, automatic_default_activation=False)
    record['pilot_run' if selected else 'full_run'] = str(root.relative_to(ROOT))
    for variant in config['variants']:
        if variant['id'] == 'control': variant['state'] = 'native_batch_full_replay_complete'
        if variant['id'] == 'binding_reuse': variant['state'] = 'pilot_running' if selected else 'full_replay_running'
    write_json(config_path, config)
    native = ROOT/config['native_batch_implementation']['full_run']
    environment = read_json(native/'controller.json')['cuda_environment']
    unit = 'arx-v8-'+root.name.replace('_', '-')+'.service'
    command = ['systemd-run', '--user', '--unit='+unit, '--property=Type=exec',
        '--property=WorkingDirectory='+str(ROOT),
        '--property=StandardOutput=append:'+str(root/'controller.log'), '--property=StandardError=inherit']
    command += ['--setenv='+name+'='+value for name, value in environment.items()]
    command += [str(ROOT/'environments/curobo/bin/python'), '-m',
        'modules.pipeline.experiments.indexed_grasp_planning.binding_replay', 'run', '--root', str(root)]
    subprocess.run(command, check=True)
    write_json(root/'controller.json', {'at': now(), 'unit': unit,
        'run': str(root.relative_to(ROOT)), 'physical_grasps': 0, 'cuda_environment': environment})
    print(json.dumps({'unit': unit, 'run': str(root.relative_to(ROOT)),
        'planning_requests': read_json(root/'registration.json')['planning_requests']}, ensure_ascii=False))


def warm_branches(planner, root):
    """Warm failed-target code paths on both independent native contexts."""
    import torch
    warmup = read_json(root/'warmup/state.json')
    reports = []
    for side, worker in planner.pool.workers.items():
        state = planner.model.bind(warmup, side, root/'prewarm'/side/'state')
        context = planner.pool.bind(root/'prewarm'/side/'open', state)
        planner.pool.contact_view(context, state)
        start = context.measured(state)
        far = np.eye(4); far[:3, 3] = [10., 10., 10.]
        if context.ik(far, start) is not None or len(worker.unconstrained_ik(context, far, start)):
            raise RuntimeError('Unreachable-goal warmup unexpectedly returned an admitted solution')
        reports.append({'arm': side, 'negative_native_and_pure_paths_warmed': True})
    torch.cuda.synchronize()
    planner.pool.reset_seeds()
    return reports


def run(root):
    reg = read_json(root/'registration.json')
    if (root/'started.json').exists() or (root/'STOP').exists():
        raise ValueError('This registration already started; keep its records')
    verify_hashes(reg)
    os.environ.update(CUDA_VISIBLE_DEVICES=reg['gpu'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
        GRASP_TRAJOPT_POLICY='interactive', CUBLAS_WORKSPACE_CONFIG=':4096:8',
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    import torch
    from arx_project.artifacts import publish_candidate_pool, finish_candidate_transport, close_owned_transports
    from modules.curobo.indexed_planning.binding_service import create
    from modules.curobo.scene_cache import mesh_stores, prune_inactive_meshes
    torch.set_num_threads(4)
    free, _ = torch.cuda.mem_get_info()
    if free/1024**2 < reg['minimum_startup_free_mib']:
        raise RuntimeError('Insufficient CUDA memory before loading arm contexts')
    write_json(root/'started.json', {'at': now(), 'pid': os.getpid(), 'gpu': reg['gpu'],
        'device': torch.cuda.get_device_name(), 'free_mib': free/1024**2})
    began = time.perf_counter()
    planner = create(SimpleNamespace(root=root/'resident', warmup_trial=root/'warmup'))
    pool = planner.pool
    warmed = warm_branches(planner, root)
    write_json(root/'startup.json', {'at': now(), 'seconds': time.perf_counter()-began,
        'model_loads': pool.loads, 'worker_memory': pool.memory_loads,
        'negative_branches': warmed, 'shared_pure_solver': pool.workers['left'].pure_solver
            is pool.workers['right'].pure_solver,
        'allocated_mib': torch.cuda.memory_allocated()/1024**2,
        'reserved_mib': torch.cuda.memory_reserved()/1024**2,
        'native_batching': False, 'physical_grasps': 0})
    completed, rows = 0, []
    try:
        for ordinal, case in enumerate(reg['cases']):
            pair = {}
            order = list(reg['variants'])
            if ordinal % 2:
                order = ([order[1], order[0], order[3], order[2]] if len(order) == 4 else order[::-1])
            for mode in order:
                free, _ = torch.cuda.mem_get_info()
                if free/1024**2 < reg['minimum_request_free_mib']:
                    raise RuntimeError('Insufficient CUDA reserve before the next binding comparison')
                directory = root/'plans'/case['id']/mode
                directory.mkdir(parents=True)
                source = Path(case['source_directory'])
                shutil.copy2(source/'state.json', directory/'state.json')
                descriptor = publish_candidate_pool(directory, read_json(source/'candidates/candidate_pool.json'))
                torch.manual_seed(reg['random_seed']); np.random.seed(reg['random_seed'])
                pool.reset_seeds()
                write_json(root/'progress.json', {'at': now(), 'phase': 'planning',
                    'completed_requests': completed, 'total_requests': reg['planning_requests'],
                    'current': {'id': case['id'], 'variant': mode}, 'results': rows})
                print('PLAN_BEGIN', case['id'], mode, flush=True)
                torch.cuda.synchronize(); began = time.perf_counter()
                with pool.request_scope(case['id']+'/'+mode, enabled=reg['variants'][mode],
                                        model_epoch=reg['model_epoch']):
                    result = planner.request(directory, candidate_transport=descriptor)
                    bindings = pool.binding_report()
                torch.cuda.synchronize(); seconds = time.perf_counter()-began
                finish_candidate_transport(directory)
                io = read_json(directory/'candidate_io.json')
                if io['source'] != 'shared_memory' or io['decode_count'] != 1 or io['full_arm_pool_copies'] != 0:
                    raise RuntimeError('Binding comparison changed candidate transport')
                checks = sum(len(read_json(path).get('candidate_events', [])) for path in
                    [directory/'arms'/side/'plan_result.json' for side in ['right', 'left']] if path.is_file())
                pair[mode] = {'seconds': seconds, 'success': result['success'],
                    'selected_arm': result.get('selected', {}).get('arm'),
                    'selected_source_index': result.get('selected', {}).get('index'),
                    'selected_contact_pose_world': result.get('selected', {}).get('contact_pose_world'),
                    'candidate_checks': checks, 'binding_seconds': bindings['seconds'],
                    'binding_calls': bindings['calls'], 'binding_cache_hits': bindings['cache_hits'],
                    'full_world_rebinds': bindings['full_world_rebinds'], 'request_free_mib': free/1024**2}
                write_json(directory/'binding_reuse.json', bindings)
                write_json(directory/'replay_metrics.json', pair[mode])
                completed += 1
                print('PLAN_END', case['id'], mode, round(seconds, 4),
                    'success', result['success'], 'bindings', bindings['full_world_rebinds'],
                    'hits', bindings['cache_hits'], flush=True)
                prune_inactive_meshes(mesh_stores(planner), torch.cuda.synchronize)
            comparison = {}
            for mode, value in pair.items():
                if not reg['variants'][mode]: continue
                control = pair[mode.replace('reuse', 'control')]
                comparison[mode] = {'success_retained': not control['success'] or value['success'],
                    'same_selection': all(value[key] == control[key] for key in
                        ['success', 'selected_arm', 'selected_source_index', 'selected_contact_pose_world']),
                    'same_candidate_checks': value['candidate_checks'] == control['candidate_checks'],
                    'speedup': control['seconds']/value['seconds']}
            rows.append({'id': case['id'], 'variants': pair, 'comparison': comparison})
            write_json(root/'progress.json', {'at': now(), 'phase': 'comparison',
                'completed_requests': completed, 'total_requests': reg['planning_requests'], 'results': rows})
        verify_hashes(reg)
        summaries = {}
        for group, enabled in [('control', False), ('reuse', True)]:
            entries = [value for row in rows for mode, value in row['variants'].items()
                       if reg['variants'][mode] == enabled]
            times = [entry['seconds'] for entry in entries]
            summaries[group] = {'requests': len(entries), 'successes': sum(e['success'] for e in entries),
                'median_seconds': float(np.median(times)), 'p90_seconds': float(np.percentile(times, 90)),
                'total_seconds': sum(times), 'binding_seconds': sum(e['binding_seconds'] for e in entries),
                'full_world_rebinds': sum(e['full_world_rebinds'] for e in entries),
                'binding_cache_hits': sum(e['binding_cache_hits'] for e in entries)}
        retained = {mode: all(row['comparison'][mode]['success_retained'] for row in rows)
            for mode, enabled in reg['variants'].items() if enabled}
        write_json(root/'acceptance.json', {'at': now(), 'passed': True, 'record_integrity': True,
            'task_complete': completed == reg['planning_requests'], 'planning_requests': completed,
            'summaries': summaries, 'paired_cases': rows, 'control_successes_retained': retained,
            'all_original_selections_retained': all(c['same_selection'] for row in rows for c in row['comparison'].values()),
            'physical_grasps': 0, 'simulation_instances': 0, 'perception_calls': 0,
            'automatic_default_activation': False, 'native_batching': False})
        write_json(root/'progress.json', {'at': now(), 'phase': 'complete',
            'completed_requests': completed, 'total_requests': reg['planning_requests'], 'results': rows})
    finally:
        close_owned_transports()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['launch', 'run'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gpu')
    parser.add_argument('--cases', nargs='+')
    args = parser.parse_args(); root = ensure_run_path(args.root)
    if args.stage == 'launch':
        launch(root, args.gpu, args.cases)
    else:
        try: run(root)
        except BaseException:
            write_json(root/'STOP', {'at': now(), 'error': traceback.format_exc(), 'physical_grasps': 0})
            raise


if __name__ == '__main__':
    main()
