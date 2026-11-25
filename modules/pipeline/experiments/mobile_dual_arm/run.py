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
import time
import traceback

from arx_project.layout import ROOT, ensure_run_path, now, read_json, sha256, write_json
from arx_project.registry import module_records
from modules.pipeline.experiments.mobile_dual_arm.campaign import (
    load_variant, map_resources, preserved_predecessor, verify_preserved_and_resources)
from modules.pipeline.experiments.mobile_dual_arm.continuation import execution_entries, verify_inheritance
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


def prepare(root, *, variant_path=None, development_only=False):
    if root.exists():
        raise ValueError('Use a new run; existing STOP and claims remain')
    config = read_json(CONFIG)
    variant = load_variant(variant_path, CONFIG)
    base = current()
    dataset = ROOT/config['dataset']
    schedule = verify_dataset(dataset)
    candidates = None
    priority = variant and variant['station_search_options'].get('placement_method') == 'high_workspace_priority'
    if variant and not priority:
        previous_probes = read_json(ROOT/variant['predecessor']/'preflight/device_allocation.json')
        candidates = [gpu for gpu, result in previous_probes.items() if result['passed']]
    if priority:
        from modules.pipeline.experiments.mobile_dual_arm.workspace_priority.resources import select_resources
        ray_record = module_records()['rayst3r']
        roles, floors, probes = select_resources(ray_minimum_mib=ray_record.get('startup_free_mib', 18000))
    else:
        roles, floors, probes = choose_gpus(candidate_gpus=candidates)
    root.mkdir(parents=True)
    for name in ['preflight', 'resident', 'claims', 'trials', 'source_registered']:
        (root/name).mkdir()
    write_json(root/'preflight/device_allocation.json', probes)
    if candidates is not None:
        write_json(root/'preflight/device_candidates.json', {'freshly_probed': candidates,
            'selection': 'Previously working devices rechecked; previously failed devices deferred',
            'previous_probe_file': variant['predecessor']+'/preflight/device_allocation.json'})
    imports()
    from repair_contract import descriptor
    write_json(root/'gripper_descriptor.json', descriptor())
    shutil.copy2(dataset/'object_catalog.json', root/'object_catalog.json')
    write_json(root/'common_initial_conditions.json', {'base_pose': schedule[0]['base_pose'],
        'initial_q': schedule[0]['robot_initial_q']})
    sources = {str(CONFIG): sha256(CONFIG), **base['configuration_hashes']}
    if variant:
        for name in [variant['configuration_file'], variant['station_search_configuration'],
                     'modules/reachability/orientation_index.py']:
            sources[str(ROOT/name)] = sha256(ROOT/name)
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
    if variant and variant['station_search_options'].get('placement_method') == 'high_workspace_priority':
        for folder in ['modules/mobile_manipulation/workspace_priority',
                       'modules/pipeline/experiments/mobile_dual_arm/workspace_priority']:
            for path in (ROOT/folder).rglob('*.py'):
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
    bindings = variant['component_modules'] if variant else {}
    for kind in COMPONENTS:
        record = module_records()[bindings.get(kind, kind)]
        for name in record.get('source_files', []):
            path = ROOT/name
            sources[str(path)] = sha256(path)
            destination = root/'source_registered'/path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        source_directories = list(record.get('source_directories', []))
        if '/optimized/' in record.get('adapter', ''):
            source_directories.append('modules/rayst3r/optimized')
        for folder in source_directories:
            for path in (ROOT/folder).rglob('*.py'):
                sources[str(path)] = sha256(path)
                destination = root/'source_registered'/path.relative_to(ROOT)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
        for field in ['adapter', 'reconstruction']:
            if field in record:
                name = ROOT/record[field]
                sources[str(name)] = sha256(name)
                destination = root/'source_registered'/name.relative_to(ROOT)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(name, destination)
        for name in record.get('weights', []):
            location = ROOT/name
            for file in ([location] if location.is_file() else sorted(p for p in location.rglob('*') if p.is_file())):
                weights[str(file)] = {'size': file.stat().st_size, 'sha256': sha256(file)}
    extra = {}
    if variant:
        prior_weights = read_json(ROOT/variant['predecessor']/'registration.json')['weights']
        if weights != prior_weights:
            raise ValueError('Comparison model weights differ from the original batch')
        extra = {'variant': variant['variant'], 'component_modules': bindings,
                 'station_search_options': variant['station_search_options'],
                 'variant_configuration': variant['configuration_file'],
                 'publication_prefix': variant['publication_prefix'], 'report_document': variant['report_document'],
                 'predecessor': preserved_predecessor(variant, sha256(dataset/'frozen_dataset.json')),
                 'map_resources': map_resources(variant['station_search_options'])}
    write_json(root/'registration.json', {'at': now(), 'version': 'V4', 'schedule': schedule,
        'physical_slot_budget': 0 if development_only else 100,
        'development_only': development_only,
        'automatic_retries': False, 'configuration_id': base['profile_id'],
        'source_hashes': sources, 'weights': weights, 'dataset': str(dataset),
        'dataset_freeze_sha256': sha256(dataset/'frozen_dataset.json'),
        'base_dataset_freeze_sha256': base['dataset_freeze_sha256'],
        'preserved_v3_dataset_freeze_sha256': sha256(ROOT/'simulation/scenes/latest200/frozen_dataset.json'),
        'gpu_roles': roles, 'minimum_free_mib': floors, 'required_components': COMPONENTS,
        'placement': config['placement'],
        'user_authorization': variant['user_authorization'] if variant else config['user_request'],
        'success': {'planned_lift_m': .03, 'minimum_hold_lift_m': .01, 'hold_seconds': 3.,
                    'bilateral_contact': True, 'joint_tolerance_rad': .005, 'forbidden_contacts': 0},
        'development_warmup_entry': read_json(ROOT/'simulation/scenes/latest200/manifest.json')['schedule'][0], **extra})
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
    verify_preserved_and_resources(reg)
    return reg


def summarize(root, phase):
    reg = read_json(root/'registration.json')
    records = [read_json(p) for p in sorted((root/'trials').glob('*/result.json'))]
    claims = {p.stem: read_json(p) for p in (root/'claims').glob('*.json')}
    complete = {name for name, claim in claims.items() if claim['state'] == 'complete'}
    unfinished = len(claims)-len(complete)
    inherited_interruptions = len(reg.get('continuation', {}).get('interrupted_case_ids', []))
    rows = [row for row in records if row['id'] in complete]
    summary = {'at': now(), 'phase': phase, 'version': 'V4', 'registered': len(reg['schedule']),
        'completed': len(rows), 'result_records': len(records),
        'not_completed': len(reg['schedule'])-len(rows),
        'not_started': len(reg['schedule'])-len(claims),
        'interrupted': unfinished if phase == 'stopped' else inherited_interruptions,
        'in_progress': unfinished-inherited_interruptions if phase == 'running' else 0,
        'recorded_physical_attempts': sum(r.get('physical_attempt', False) for r in records),
        'station_ready': sum(r.get('stationing', {}).get('high_workspace_reached',
                            r.get('stationing', {}).get('success', False)) for r in rows),
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
    if (root/'started.json').exists():
        raise RuntimeError('This registration has started; do not replay claimed slots')
    if reg.get('continuation'):
        verify_inheritance(root, reg, full=True)
    elif list((root/'claims').glob('*.json')):
        raise RuntimeError('This registration has started; do not replay claimed slots')
    verify_dataset(Path(reg['dataset']))
    verify_preserved_and_resources(reg, full_resources=True)
    free = gpu_inventory()
    while any(free.get(g, 0) < v for g, v in reg['minimum_free_mib'].items()):
        if not reg.get('continuation'):
            raise RuntimeError('Registered GPU resources unavailable')
        write_json(root/'preflight/resource_admission_wait.json', {'at': now(),
            'state': 'waiting_before_any_model_or_new_slot', 'free_mib': free,
            'required_mib': reg['minimum_free_mib']})
        time.sleep(10)
        guard(root)
        free = gpu_inventory()
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
    priority = reg.get('station_search_options', {}).get('placement_method') == 'high_workspace_priority'
    if priority:
        from modules.pipeline.experiments.mobile_dual_arm.workspace_priority.runtime import HighWorkspaceStationing as Stationing
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
        search = None
        if reg.get('station_search_options'):
            from modules.pipeline.experiments.mobile_dual_arm.active_search import service_search
            from modules.mobile_manipulation.placement import backproject
            from PIL import Image
            import numpy as np
            search = service_search(model, reg['placement'], services, reg['station_search_options'])
            warm_state = read_json(warm/'state.json')
            measured = backproject(np.asarray(Image.open(warm/'camera/mask.png')) > 0,
                np.load(warm/'camera/depth_m.npy').squeeze(), warm_state['intrinsics'], warm_state['camera_world_optical'])
            selected, check = search.select(warm_state, read_json(warm/'candidates/candidate_pool.json'),
                                            np.median(measured, axis=0), root/'preflight/map_stationing')
            write_json(root/'preflight/map_stationing.json', {'passed': selected is not None,
                       'physical_grasps': 0, 'body_motions': 0, 'search_summary': {k:v for k,v in check.items() if k!='events'}})
            if selected is None:
                raise RuntimeError('Development map stationing preflight failed')
        if priority:
            from modules.mobile_manipulation.workspace_priority.handoff import validate_prepared
            env = executor.make(warm_entry['model_id'], scene_spec=warm_entry['scene'])
            interface = root/'preflight/priority_pipeline'
            interface.mkdir()
            try:
                env.reset(seed=warm_entry['seed'])
                for _ in range(80):
                    env._step_action(torch.tensor(env.initial_q[None], device=env.device))
                prepared = Stationing(services, executor, model, reg['placement'],
                    2026112500, station_search=search).prepare(env, interface, warm_entry)
                write_json(root/'preflight/priority_pipeline.json', {'passed': prepared['success'],
                    'physical_grasps': 0, 'base_and_lift_motion': prepared.get('motion_started', False),
                    'workspace_scope': prepared.get('workspace_scope'), 'receipt': prepared})
                if not prepared['success']:
                    raise RuntimeError('High-workspace development pipeline failed: '+str(prepared))
                validate_prepared(interface, prepared, executor.snapshot(env, env.get_obs()))
            finally:
                env.close()
            from modules.pipeline.experiments.mobile_dual_arm.workspace_priority.resources import verify_ready
            verify_ready(root, reg)
        write_json(root/'preflight/all_ready.json', {'passed': True, 'at': now(), 'components': COMPONENTS})
        if reg.get('development_only'):
            write_json(root/'preflight/complete.json', {'passed': True, 'at': now(),
                'formal_slots_claimed': 0, 'physical_grasps': 0,
                'base_and_lift_pipeline_verified': priority})
            summarize(root, 'preflight_complete')
            return
        normal = {'success', 'sam3_no_target_mask', 'rayst3r_empty_pointcloud',
            'coarse_sam3_no_target_mask', 'coarse_rayst3r_empty_pointcloud', 'coarse_invalid_depth',
            'no_safe_reachable_station', 'no_arm_has_executable_candidate',
            'no_bilateral_target_contact', 'insufficient_hold_height', 'forbidden_robot_contact',
            'measured_closure_lift_unreachable', 'measured_pregrasp_approach_unreachable',
            'gripper_open_not_converged', 'open_not_converged', 'Target has fewer than 50 visible pixels'}
        normal.update(stage+'_not_converged' for stage in ['pregrasp', 'approach', 'close', 'lift', 'hold'])
        normal.add('unable_to_plan')
        for number, entry in execution_entries(reg):
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
            station = Stationing(services, executor, model, reg['placement'], 2026112500+number,
                                 station_search=search)
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
                if priority:
                    from modules.pipeline.experiments.mobile_dual_arm.workspace_priority.runtime import audit_segment
                    for relative in result['stationing']['stationing_segments']:
                        segment = (trial/relative).resolve()
                        if not segment.is_relative_to((trial/'stationing/segments').resolve()):
                            raise RuntimeError('Invalid station segment path')
                        if audit_segment(segment) != read_json(segment/'motion_verification.json'):
                            raise RuntimeError('Station segment evidence changed')
            if priority and result.get('planning_success'):
                proof = read_json(trial/'prepared_plan_execution_gate.json')
                if not proof['passed'] or proof['selected_arm'] != result['selected_arm']:
                    raise RuntimeError('Prepared grasp execution gate is missing or inconsistent')
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
        verify_preserved_and_resources(reg, full_resources=True)
        summarize(root, 'complete')
    finally:
        if services is not None:
            services.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--variant', type=Path)
    parser.add_argument('--development-only', action='store_true',
                        help='Register only the development preflight; claim no formal scenes')
    args = parser.parse_args()
    root = ensure_run_path(args.root)
    if args.stage == 'prepare':
        prepare(root, variant_path=args.variant, development_only=args.development_only)
    else:
        if args.variant is not None or args.development_only:
            parser.error('Variant settings are frozen during prepare')
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
