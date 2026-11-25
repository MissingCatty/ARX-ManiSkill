"""Activate the validated factory after the original V4 batch is fully accepted."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time

from arx_project.layout import ROOT, ensure_run_path, now, read_json, sha256, write_json

ADAPTER = 'modules/curobo/dual_arm/accelerated_service.py'


def readiness(comparison, main):
    if (comparison/'STOP').exists():
        return 'comparison_failed'
    if not (comparison/'acceptance.json').is_file():
        return 'waiting_for_comparison'
    accepted = read_json(comparison/'acceptance.json')
    if not accepted['passed'] or accepted['physical_grasps'] != 0:
        return 'comparison_failed'
    if len(accepted.get('paired_cases', [])) != 3 or not all(p['passed'] for p in accepted['paired_cases']):
        return 'comparison_failed'
    if len(accepted.get('opening_checks', [])) < 3 or not all(p['passed'] for p in accepted['opening_checks']):
        return 'comparison_failed'
    if (main/'STOP').exists():
        return 'main_batch_stopped'
    for name in ['publication_complete.json', 'acceptance.json', 'services_stopped.json']:
        if not (main/name).is_file():
            return 'waiting_for_main_acceptance'
    publication = read_json(main/'publication_complete.json')
    acceptance = read_json(main/'acceptance.json')
    stopped = read_json(main/'services_stopped.json')
    if not acceptance['passed'] or not acceptance['task_complete'] or acceptance['completed'] != 100:
        return 'main_acceptance_failed'
    if not publication['task_complete'] or not publication['audit_passed'] or not stopped['all_owned_exited']:
        return 'waiting_for_main_shutdown'
    # The completion marker is emitted after model shutdown; wait until the
    # controller itself has exited too, before changing its registered files.
    pid = read_json(main/'started.json')['pid']
    if (Path('/proc')/str(pid)).exists():
        return 'waiting_for_main_shutdown'
    return 'ready'


def activate(comparison, main):
    folder = comparison/'activation'
    receipt = folder/'activation.json'
    if receipt.is_file():
        return read_json(receipt)
    state = readiness(comparison, main)
    if state != 'ready':
        raise RuntimeError('Activation is not ready: '+state)
    for registration in [read_json(main/'registration.json'), read_json(comparison/'registration.json')]:
        for name, digest in registration['source_hashes'].items():
            if sha256(name) != digest:
                raise RuntimeError('Registered source changed before activation: '+name)
    from arx_project.registry import load_adapter, module_records
    from unittest.mock import patch
    records = module_records()
    candidate = {**records['curobo_dual_arm'], 'adapter': ADAPTER}
    with patch('arx_project.registry.module_records', return_value={**records, 'curobo_dual_arm': candidate}):
        if not callable(load_adapter('curobo_dual_arm').create):
            raise RuntimeError('Accelerated factory cannot be loaded by the registry')
    folder.mkdir(parents=True)
    variant_path = ROOT/'configs/experiments/dual_arm_acceleration.json'
    paths = [ROOT/'configs/modules.json', ROOT/'configs/experiments/active_experiment.json',
             ROOT/'configs/experiments/scene_dataset.json', variant_path]
    previous = {}
    for path in paths:
        destination = folder/'before'/path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        previous[str(path)] = sha256(path)
    config = read_json(paths[1])
    history = ROOT/'configs/experiments/history'/('sam3_graspgenx_v4_before_acceleration_'+str(int(time.time()))+'.json')
    history.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(paths[1], history)
    record = {'at': now(), 'adapter': ADAPTER, 'variant': 'rank_bound_batch_opening',
        'parent_configuration_sha256': previous[str(paths[1])],
        'parent_configuration_snapshot': str(history.relative_to(ROOT)),
        'validation_report': str((comparison/'acceptance.json').relative_to(ROOT)),
        'validation_sha256': sha256(comparison/'acceptance.json'),
        'completed_main_run': str(main.relative_to(ROOT)),
        'main_acceptance_sha256': sha256(main/'acceptance.json'),
        'physical_robot_camera_and_success_unchanged': True,
        'additional_physical_grasps': 0}
    try:
        variant = read_json(variant_path)
        variant.update(activation='active_for_new_run_registrations',
                       validation_report=record['validation_report'])
        write_json(variant_path, variant)
        modules = read_json(paths[0])
        modules['modules']['curobo_dual_arm'].update(adapter=ADAPTER, variant=record['variant'])
        policies = modules['modules']['curobo_dual_arm'].setdefault('policies', [])
        if 'configs/experiments/dual_arm_acceleration.json' not in policies:
            policies.append('configs/experiments/dual_arm_acceleration.json')
        write_json(paths[0], modules)
        pointer = read_json(paths[2])
        pointer.update(grasp_tests_authorized=False, grasp_test_budget=0, last_completed_run=str(main.relative_to(ROOT)))
        write_json(paths[2], pointer)
        config.update(profile_id=config['profile_id']+'_rankbatch01', at=now(),
            state='v4_100_complete_acceleration_ready', planner_variant=record['variant'],
            grasp_tests_authorized=False, physical_grasp_budget=0, acceleration_revision=record)
        for name in list(config['configuration_hashes']):
            config['configuration_hashes'][name] = sha256(name)
        for name in [ADAPTER, 'modules/curobo/dual_arm/accelerated_adapter.py',
                     'configs/experiments/dual_arm_acceleration.json']:
            config['configuration_hashes'][str(ROOT/name)] = sha256(ROOT/name)
        write_json(paths[1], config)
        from modules.pipeline.experiments.current_configuration.configuration import load_current
        load_current(verify_dataset=True)
        if not callable(load_adapter('curobo_dual_arm').create):
            raise RuntimeError('Updated registry failed its factory loading check')
    except BaseException:
        for path in paths:
            shutil.copy2(folder/'before'/path.relative_to(ROOT), path)
        raise
    record['previous_file_hashes'] = previous
    record['new_file_hashes'] = {str(p): sha256(p) for p in paths}
    write_json(receipt, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--main-run', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    comparison, main_run = ensure_run_path(args.comparison), ensure_run_path(args.main_run)
    while True:
        state = readiness(comparison, main_run)
        write_json(comparison/'activation_status.json', {'at': now(), 'state': state})
        if state == 'ready':
            activate(comparison, main_run)
            try:
                from modules.pipeline.experiments.mobile_dual_arm.finish import publish
                publish(main_run)
            except (OSError, subprocess.SubprocessError, RuntimeError) as error:
                write_json(comparison/'activation_status.json', {'at': now(),
                    'state': 'activated_website_pending', 'error': str(error)})
                if not args.watch:
                    raise
                time.sleep(30)
                continue
            write_json(comparison/'activation_status.json', {'at': now(), 'state': 'activated_and_published'})
            return
        if state.endswith('failed') or state == 'main_batch_stopped':
            return
        if not args.watch:
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
