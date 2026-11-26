"""Promote a completed planning-only comparison for future map-planner runs."""
import argparse
from pathlib import Path
import shutil
import subprocess

from arx_project.layout import ROOT,now,read_json,sha256,write_json


def activate(comparison):
    comparison = Path(comparison).resolve()
    accepted = read_json(comparison/'acceptance.json')
    registration = read_json(comparison/'registration.json')
    if (not accepted['passed'] or not accepted['same_selected_arm_index_and_pose']
            or len(accepted['paired_cases']) != len(registration['cases'])
            or read_json(comparison/'progress.json')['completed_requests'] != registration['planning_requests']):
        raise RuntimeError('Interleaved planning comparison has not passed')
    controller = read_json(comparison/'controller.json')['unit']
    status = subprocess.check_output(['systemctl','--user','show',controller,
        '-p','ActiveState','-p','MainPID','-p','ExecMainStatus'],text=True)
    fields = dict(line.split('=',1) for line in status.strip().splitlines())
    if fields.get('ActiveState') != 'inactive' or fields.get('MainPID') != '0' or fields.get('ExecMainStatus') != '0':
        raise RuntimeError('Comparison process must exit successfully before activation')
    for name,digest in registration['source_hashes'].items():
        if sha256(name) != digest:
            raise RuntimeError('Validated comparison source changed: '+name)
    final = Path(registration['source_run'])
    if not read_json(final/'acceptance.json')['task_complete']:
        raise RuntimeError('Source V6 task must be complete')
    folder = comparison/'activation'
    if folder.exists():
        raise ValueError('Keep the existing activation record')
    folder.mkdir()
    paths = [ROOT/'configs/modules.json',ROOT/'configs/experiments/active_experiment.json',
        ROOT/'configs/experiments/dual_arm_interleaving.json']
    for path in paths:
        previous = folder/'before'/path.relative_to(ROOT)
        previous.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,previous)
    current = read_json(paths[1])
    # Only the two explicitly reviewed shared hooks differ from the former
    # configuration. Their pre-change copies must match that registration.
    hooks = {'modules/curobo/dual_arm/adapter.py','modules/curobo/candidate_io/store.py'}
    for name,digest in current['configuration_hashes'].items():
        if sha256(name) == digest:
            continue
        relative = Path(name).relative_to(ROOT).as_posix()
        if relative not in hooks or sha256(comparison/'source_before'/relative) != digest:
            raise RuntimeError('Unexpected configuration change before activation: '+relative)
    adapter = 'modules/curobo/dual_arm/interleaved_map_service.py'
    record = {'at':now(),'variant':'alternating_candidate_windows','candidate_batch_size':16,
        'subsequent_candidate_batch_size':64,
        'adapter':adapter,'validation_report':str((comparison/'acceptance.json').relative_to(ROOT)),
        'validation_sha256':sha256(comparison/'acceptance.json'),
        'source_complete_run':str(final.relative_to(ROOT)),
        'source_acceptance_sha256':sha256(final/'acceptance.json'),
        'physical_robot_camera_scene_and_success_unchanged':True,'additional_physical_grasps':0}
    history = ROOT/'configs/experiments/history/dual_arm_interleaving_activation_20261126.json'
    if history.exists():
        raise ValueError('Activation history already exists')
    try:
        registry = read_json(paths[0])
        module = registry['modules']['curobo_dual_arm_map']
        module.update(adapter=adapter,variant=record['variant'])
        files = module.setdefault('source_files',[])
        for name in [adapter,'modules/curobo/dual_arm/interleaved_adapter.py',
                     'modules/curobo/dual_arm/adapter.py']:
            if name not in files:
                files.append(name)
        policies = module.setdefault('policies',[])
        if 'configs/experiments/dual_arm_interleaving.json' not in policies:
            policies.append('configs/experiments/dual_arm_interleaving.json')
        write_json(paths[0],registry)
        policy = read_json(paths[2])
        policy.update(activation='active_for_new_run_registrations',validation_report=record['validation_report'])
        write_json(paths[2],policy)
        current.update(at=now(),state='interleaved_search_ready_for_new_run',
            planner_variant=record['variant'],candidate_search_revision=record,
            grasp_tests_authorized=False,physical_grasp_budget=0)
        for name in current['configuration_hashes']:
            current['configuration_hashes'][name] = sha256(name)
        for name in [adapter,'modules/curobo/dual_arm/interleaved_adapter.py',
                     'configs/experiments/dual_arm_interleaving.json']:
            current['configuration_hashes'][str(ROOT/name)] = sha256(ROOT/name)
        write_json(paths[1],current)
        from arx_project.registry import load_adapter
        from modules.pipeline.experiments.current_configuration.configuration import load_current
        assert callable(load_adapter('curobo_dual_arm_map').create)
        load_current(verify_dataset=True)
    except BaseException:
        for path in paths:
            shutil.copy2(folder/'before'/path.relative_to(ROOT),path)
        raise
    record['new_file_hashes'] = {str(path.relative_to(ROOT)):sha256(path) for path in paths}
    write_json(folder/'activation.json',record)
    write_json(history,record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    print(activate(parser.parse_args().root))
