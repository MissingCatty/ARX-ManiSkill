"""Registered V4: 100 scenes, deterministic stationing, fresh dual-arm grasps."""
import argparse
from collections import Counter
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

from arx_project.layout import ROOT, ensure_run_path, now, read_json, sha256, write_json
from arx_project.registry import module_records
from modules.pipeline.experiments.dual_arm_grasp.run import (
    Services, allocation_probes, choose_gpus, current, gpu_inventory, imports)

HERE = Path(__file__).resolve().parent
CONFIG = ROOT/'configs/experiments/sam3_graspgenx_v4.json'
COMPONENTS = ['sam3', 'rayst3r', 'graspgenx', 'curobo_dual_arm']


def verify_dataset(dataset):
    frozen = read_json(dataset/'frozen_dataset.json')
    for name, digest in frozen['files'].items():
        if sha256(dataset/name) != digest:
            raise RuntimeError('Frozen V4 scene changed: '+name)
    schedule = read_json(dataset/'manifest.json')['schedule']
    if len(schedule) != 100 or len({r['id'] for r in schedule}) != 100:
        raise ValueError('V4 requires 100 distinct registered scenes')
    if sorted(Counter(r['layout'] for r in schedule).values()) != [20]*5:
        raise ValueError('V4 layout quota mismatch')
    return schedule


def prepare(root):
    if root.exists():
        raise ValueError('Use a new run; existing STOP and claims remain')
    config = read_json(CONFIG)
    base = current()
    dataset = ROOT/config['dataset']
    schedule = verify_dataset(dataset)
    roles, floors, probes = choose_gpus()
    root.mkdir(parents=True)
    for name in ['preflight', 'resident', 'claims', 'trials', 'source_registered']:
        (root/name).mkdir()
    write_json(root/'preflight/device_allocation.json', probes)
    imports()
    from repair_contract import descriptor
    write_json(root/'gripper_descriptor.json', descriptor())
    shutil.copy2(dataset/'object_catalog.json', root/'object_catalog.json')
    write_json(root/'common_initial_conditions.json', {'base_pose': schedule[0]['base_pose'],
        'initial_q': schedule[0]['robot_initial_q']})
    sources = {str(CONFIG): sha256(CONFIG), **base['configuration_hashes']}
    folders = ['arx_project', 'modules/mobile_manipulation', 'modules/curobo/dual_arm',
        'modules/curobo/fingertip_contact', 'modules/maniskill/arx_pipeline',
        'modules/maniskill/planning_common', 'modules/pipeline/experiments/mobile_dual_arm',
        'modules/pipeline/experiments/dual_arm_grasp',
        'modules/pipeline/experiments/repaired_grasp_benchmark',
        'modules/pipeline/experiments/diverse_grasp_benchmark',
        'modules/pipeline/experiments/diverse_object_scenes',
        'modules/pipeline/experiments/pointcloud_benchmark']
    for folder in folders:
        for path in (ROOT/folder).glob('*.py'):
            sources[str(path)] = sha256(path)
    for name in ['configs/modules.json', 'configs/experiments/active_experiment.json',
        'modules/arx_core/arms.py', 'modules/curobo/arx_adapter.py',
        'modules/maniskill/resident_planner.py', 'modules/maniskill/fast_resident_planner.py',
        'modules/maniskill/short_lift_protocol.py', 'modules/maniskill/short_lift_planner.py',
        'modules/maniskill/checked_native_services.py',
        'modules/maniskill/compat_tools/physical_table_pair_20261118_01.py']:
        sources[str(ROOT/name)] = sha256(ROOT/name)
    for name in sources:
        path = Path(name)
        destination = root/'source_registered'/path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    weights = {}
    for kind in COMPONENTS:
        for name in module_records()[kind].get('weights', []):
            location = ROOT/name
            for file in ([location] if location.is_file() else sorted(p for p in location.rglob('*') if p.is_file())):
                weights[str(file)] = {'size': file.stat().st_size, 'sha256': sha256(file)}
    write_json(root/'registration.json', {'at': now(), 'version': 'V4', 'schedule': schedule,
        'physical_slot_budget': 100, 'automatic_retries': False, 'configuration_id': base['profile_id'],
        'source_hashes': sources, 'weights': weights, 'dataset': str(dataset),
        'dataset_freeze_sha256': sha256(dataset/'frozen_dataset.json'),
        'base_dataset_freeze_sha256': base['dataset_freeze_sha256'],
        'preserved_v3_dataset_freeze_sha256': sha256(ROOT/'simulation/scenes/latest200/frozen_dataset.json'),
        'gpu_roles': roles, 'minimum_free_mib': floors, 'required_components': COMPONENTS,
        'placement': config['placement'], 'user_authorization': config['user_request'],
        'success': {'planned_lift_m': .03, 'minimum_hold_lift_m': .01, 'hold_seconds': 3.,
                    'bilateral_contact': True, 'joint_tolerance_rad': .005, 'forbidden_contacts': 0},
        'development_warmup_entry': read_json(ROOT/'simulation/scenes/latest200/manifest.json')['schedule'][0]})
    summarize(root, 'registered')
    print('REGISTERED_V4', root, roles, flush=True)


def guard(root):
    reg = read_json(root/'registration.json')
    if (root/'STOP').exists():
        raise RuntimeError('STOP exists')
    for name, digest in reg['source_hashes'].items():
        if sha256(name) != digest:
            raise RuntimeError('Registered source changed: '+name)
    if sha256(Path(reg['dataset'])/'frozen_dataset.json') != reg['dataset_freeze_sha256']:
        raise RuntimeError('V4 scene registration changed')
    base = current()
    if base['profile_id'] != reg['configuration_id'] or base['dataset_freeze_sha256'] != reg['base_dataset_freeze_sha256']:
        raise RuntimeError('Base configuration or preserved V3 scenes changed')
    if sha256(ROOT/'simulation/scenes/latest200/frozen_dataset.json') != reg['preserved_v3_dataset_freeze_sha256']:
        raise RuntimeError('Preserved V3 scenes changed')
    return reg


def summarize(root, phase):
    reg = read_json(root/'registration.json')
    rows = [read_json(p) for p in sorted((root/'trials').glob('*/result.json'))]
    summary = {'at': now(), 'phase': phase, 'version': 'V4', 'registered': 100, 'completed': len(rows),
        'not_completed': 100-len(rows),
        'station_ready': sum(r.get('stationing', {}).get('success', False) for r in rows),
        'planning_passes': sum(r.get('planning_success', False) for r in rows),
        'physical_attempts': sum(r.get('physical_attempt', False) for r in rows),
        'successes': sum(r.get('success', False) for r in rows),
        'selected_arms': dict(Counter(r['selected_arm'] for r in rows if 'selected_arm' in r)),
        'failure_reasons': dict(Counter(r.get('reason', 'unknown') for r in rows if not r['success']))}
    summary['by_layout'] = {layout: {'completed': len(group),
        'planning_passes': sum(r.get('planning_success', False) for r in group),
        'successes': sum(r['success'] for r in group)}
        for layout in dict.fromkeys(r['layout'] for r in reg['schedule'])
        for group in [[r for r in rows if r['layout'] == layout]]}
    write_json(root/'progress.json', summary)
    write_json(root/'summary.json', {**summary, 'results': rows})
    return summary


def run(root):
    reg = guard(root)
    if (root/'started.json').exists() or list((root/'claims').glob('*.json')):
        raise RuntimeError('This registration has started; do not replay claimed slots')
    verify_dataset(Path(reg['dataset']))
    free = gpu_inventory()
    if any(free.get(g, 0) < v for g, v in reg['minimum_free_mib'].items()):
        raise RuntimeError('Registered GPU resources unavailable')
    probes = allocation_probes(set(reg['gpu_roles'].values()))
    write_json(root/'preflight/device_allocation_before_run.json', probes)
    if not all(r['passed'] for r in probes.values()):
        raise RuntimeError('Device allocation failed')
    for name, entry in reg['weights'].items():
        if Path(name).stat().st_size != entry['size'] or sha256(name) != entry['sha256']:
            raise RuntimeError('Model changed: '+name)
    os.environ['CUDA_VISIBLE_DEVICES'] = reg['gpu_roles']['simulation']
    imports()
    from modules.pipeline.experiments.mobile_dual_arm.runtime import Stationing, capture
    from modules.curobo.dual_arm.model import ArmModel
    from repair_contract import make_candidates
    spec = importlib.util.spec_from_file_location('v4_repaired_simulation', HERE.parent/'repaired_grasp_benchmark/simulation.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    executor, evaluator, _ = module.install(root)
    model = ArmModel()
    write_json(root/'started.json', {'at': now(), 'pid': os.getpid()})
    services = None
    try:
        warm_entry = reg['development_warmup_entry']
        env = executor.make(warm_entry['model_id'], scene_spec=warm_entry['scene'])
        try:
            import torch
            env.reset(seed=warm_entry['seed'])
            for _ in range(80):
                env._step_action(torch.tensor(env.initial_q[None], device=env.device))
            warm = root/'preflight/head_warmup'
            capture(env, warm, executor, warm_entry)
            write_json(warm/'closed_state.json', read_json(warm/'state.json'))
        finally:
            env.close()
        services = Services(root, reg)
        services.start()
        make_candidates(warm)
        response = services.request('curobo_dual_arm', warm)
        write_json(root/'preflight/full_chain.json', response)
        if not response['result']['success']:
            raise RuntimeError('Development full-chain planning failed')
        write_json(root/'preflight/all_ready.json', {'passed': True, 'at': now(), 'components': COMPONENTS})
        normal = {'success', 'sam3_no_target_mask', 'rayst3r_empty_pointcloud',
            'coarse_sam3_no_target_mask', 'coarse_rayst3r_empty_pointcloud', 'coarse_invalid_depth',
            'no_safe_reachable_station', 'no_arm_has_executable_candidate',
            'no_bilateral_target_contact', 'insufficient_hold_height', 'forbidden_robot_contact',
            'measured_closure_lift_unreachable', 'measured_pregrasp_approach_unreachable',
            'gripper_open_not_converged', 'open_not_converged', 'Target has fewer than 50 visible pixels'}
        normal.update(stage+'_not_converged' for stage in ['pregrasp', 'approach', 'close', 'lift', 'hold'])
        for number, entry in enumerate(reg['schedule']):
            guard(root)
            if shutil.disk_usage(root).free < 10*1024**3:
                raise RuntimeError('Insufficient disk space')
            trial = root/'trials'/entry['id']
            claim = root/'claims'/(entry['id']+'.json')
            if trial.exists() or claim.exists():
                raise RuntimeError('Scene already claimed')
            write_json(claim, {'state': 'claimed', 'at': now(), 'scene': entry['id'], 'attempt': 1})
            write_json(root/'current_trial.json', {'number': number+1, 'scene': entry['id'], 'at': now()})

            def worker(script, directory, gpu, log, **flags):
                directory = Path(directory)
                try:
                    if script == 'neural.py':
                        write_json(directory/'entry.json', entry)
                        sam = services.request('sam3', directory)
                        if not sam['result'].get('target_found', True):
                            write_json(directory/'perception_failure.json', {'reason': 'sam3_no_target_mask'})
                            return 2
                        ray = services.request('rayst3r', directory)
                        if ray['result'].get('point_count', 0) < 20:
                            write_json(directory/'perception_failure.json', {'reason': 'rayst3r_empty_pointcloud'})
                            return 2
                        response = services.request('graspgenx', directory, seed=2026112500+number)
                        make_candidates(directory)
                    else:
                        response = services.request('curobo_dual_arm', directory, **flags)
                    write_json(log, response)
                    return response['returncode']
                except BaseException:
                    write_json(directory/'infrastructure_error.json', {'at': now(), 'error': traceback.format_exc()})
                    raise
            executor.worker = worker
            station = Stationing(services, executor, model, reg['placement'], 2026112500+number)
            print('BEGIN_V4', number+1, entry['id'], flush=True)
            result = executor.trial(entry, trial, reg['gpu_roles']['rayst3r'],
                reg['gpu_roles']['curobo_dual_arm'], 'head', prepare_observation=station.prepare)
            if (trial/'perception_failure.json').exists():
                result['reason'] = read_json(trial/'perception_failure.json')['reason']
                write_json(trial/'result.json', result)
            evaluator.verify_trial(trial)
            if result['reason'] not in normal or list(trial.rglob('infrastructure_error.json')):
                raise RuntimeError('Execution/configuration/recording fault: '+result['reason'])
            if result.get('stationing', {}).get('motion_started'):
                for name in ['command.json', 'execution.jsonl', 'receipt.json', 'settled_state.json', 'motion.mp4', 'head.mp4']:
                    path = trial/'stationing'/name
                    if not path.is_file() or not path.stat().st_size:
                        raise RuntimeError('Missing station evidence: '+name)
            if result.get('physical_attempt'):
                for name in ['execution.jsonl', 'execution.mp4', 'head.mp4', 'stages.json',
                             'contact_entities.json', 'independent_evaluation.json']:
                    path = trial/name
                    if not path.is_file() or not path.stat().st_size:
                        raise RuntimeError('Missing execution evidence: '+name)
            write_json(claim, {'state': 'complete', 'at': now(), 'scene': entry['id'],
                              'result_sha256': sha256(trial/'result.json')})
            summarize(root, 'running')
            print('RESULT_V4', number+1, entry['id'], result.get('selected_arm'), result['reason'], flush=True)
        guard(root)
        verify_dataset(Path(reg['dataset']))
        summarize(root, 'complete')
    finally:
        if services is not None:
            services.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run'])
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = ensure_run_path(args.root)
    if args.stage == 'prepare':
        prepare(root)
    else:
        with (root/'controller.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:
                run(root)
            except BaseException:
                write_json(root/'STOP', {'at': now(), 'error': traceback.format_exc()})
                summarize(root, 'stopped')
                raise


if __name__ == '__main__':
    main()
