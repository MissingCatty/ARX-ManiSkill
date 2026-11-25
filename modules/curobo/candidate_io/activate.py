"""Apply verified candidate I/O changes after the running V6 releases its sources."""
import argparse
from copy import deepcopy
import fcntl
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from arx_project.layout import ROOT, ensure_run_path, now, read_json, sha256, write_json

RUNTIME = ['arx_project/json_io.py', 'arx_project/artifacts.py', 'arx_project/candidate_rpc.py',
           'modules/curobo/candidate_io/store.py']
MODULES = ROOT/'configs/modules.json'
ACTIVE = ROOT/'configs/experiments/active_experiment.json'


def active(unit):
    result = subprocess.run(['systemctl', '--user', 'is-active', unit],
                            capture_output=True, text=True)
    return result.stdout.strip() in ['active', 'activating', 'reloading', 'deactivating']


def ready(run):
    prefix = 'arx-sam3-zerograsp-'+run.name.replace('_', '-')
    if active(prefix+'.service') or active(prefix+'-watch.service'):
        return False, 'waiting_for_current_controller_and_publication'
    if not (run/'services_stopped.json').exists() or not read_json(run/'services_stopped.json')['all_owned_exited']:
        return False, 'waiting_for_current_model_services_exit'
    if not (run/'acceptance.json').exists():
        return False, 'waiting_for_current_independent_acceptance'
    acceptance = read_json(run/'acceptance.json')
    if not acceptance['passed'] or not acceptance['record_integrity']:
        raise RuntimeError('Current run has unresolved record damage; keep its sources frozen')
    return True, 'ready'


def apply(root, staged, native, saved, run, offline):
    stage = read_json(staged/'staging.json')
    for name in [native/'verification.json', saved/'verification.json', offline]:
        if not read_json(name)['passed']:
            raise RuntimeError('Validation failed: '+str(name))
    validation = read_json(offline)
    for relative, digest in validation['runtime_sha256'].items():
        if sha256(ROOT/relative) != digest:
            raise RuntimeError('Runtime helper changed after validation: '+relative)
    for relative, digest in validation['staged_file_hashes'].items():
        if sha256(staged/'staged_source'/relative) != digest:
            raise RuntimeError('Staged source changed after offline validation: '+relative)
    for relative, record in stage['files'].items():
        if sha256(ROOT/relative) != record['before_sha256']:
            raise RuntimeError('Working source changed since validation: '+relative)
        if sha256(staged/'staged_source'/relative) != record['after_sha256']:
            raise RuntimeError('Staged source changed since validation: '+relative)
    modules, config = read_json(MODULES), read_json(ACTIVE)
    for filename, digest in config['configuration_hashes'].items():
        if sha256(filename) != digest:
            raise RuntimeError('Current source is not its registered predecessor: '+filename)
    previous_physics = {k:v for k,v in config.items() if k not in ['at','configuration_hashes','candidate_io_software']}
    proof = {'at': now(), 'predecessor_run': str(run.relative_to(ROOT)),
        'predecessor_registration_sha256': sha256(run/'registration.json'),
        'predecessor_acceptance_sha256': sha256(run/'acceptance.json'),
        'stage_sha256': sha256(staged/'staging.json'),
        'native_replay_verification': str((native/'verification.json').relative_to(ROOT)),
        'native_replay_verification_sha256': sha256(native/'verification.json'),
        'saved_pool_verification_sha256': sha256(saved/'verification.json'),
        'offline_verification_sha256': sha256(offline), 'files': stage['files'],
        'runtime_dependencies': {relative: sha256(ROOT/relative) for relative in RUNTIME},
        'physical_grasps_added': 0, 'physical_configuration_changed': False}
    before = root/'source_before'
    for relative in [*stage['files'], 'configs/modules.json', 'configs/experiments/active_experiment.json']:
        target = before/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/relative, target)
    write_json(root/'activation_registration.json', proof)
    history = ROOT/'configs/experiments/history/candidate_io_activation_20261125_01'
    history.mkdir(parents=True, exist_ok=False)
    shutil.copy2(MODULES, history/'modules_before.json')
    shutil.copy2(ACTIVE, history/'active_experiment_before.json')
    try:
        for relative in stage['files']:
            destination = ROOT/relative
            pending = destination.with_suffix(destination.suffix+'.candidate-io-pending')
            shutil.copy2(staged/'staged_source'/relative, pending)
            pending.replace(destination)
        for key in ['curobo_dual_arm', 'curobo_dual_arm_map']:
            record = modules['modules'][key]
            record['source_files'] = sorted(set(record.get('source_files', [])) | set(RUNTIME))
            record['candidate_data_delivery'] = 'one request decode; shared arm views; optional shared-memory handoff'
            record['candidate_transport_format'] = 'arx_candidate_json_shm_v1'
        write_json(MODULES, modules)
        for filename in list(config['configuration_hashes']):
            config['configuration_hashes'][filename] = sha256(filename)
        for relative in [*stage['files'], *RUNTIME]:
            config['configuration_hashes'][str(ROOT/relative)] = sha256(ROOT/relative)
        config['candidate_io_software'] = {'activated_at': now(),
            'verification': str((root/'activation_registration.json').relative_to(ROOT)),
            'candidate_filter_order_and_pose_changes': False,
            'background_recording': True, 'shared_memory_handoff': True}
        config['at'] = now()
        if {k:v for k,v in config.items() if k not in ['at','configuration_hashes','candidate_io_software']} != previous_physics:
            raise RuntimeError('I/O activation unexpectedly changed a physical configuration field')
        write_json(ACTIVE, config)
        from modules.pipeline.experiments.current_configuration.configuration import load_current
        load_current(verify_dataset=True)
        write_json(history/'activation.json', proof)
        write_json(root/'activation_complete.json', {'at':now(), 'passed':True,
            'files_changed':len(stage['files']), 'runtime_dependencies':RUNTIME,
            'physical_configuration_unchanged':True, 'old_results_and_stop_markers_preserved':True})
    except BaseException:
        for relative in [*stage['files'], 'configs/modules.json', 'configs/experiments/active_experiment.json']:
            shutil.copy2(before/relative, ROOT/relative)
        write_json(root/'activation_failed.json', {'at':now(), 'rolled_back':True, 'error':traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'staged', 'native', 'saved', 'run', 'offline']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    root = ensure_run_path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    with (root/'activation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root/'activation_complete.json').exists() or (root/'activation_failed.json').exists():
            raise RuntimeError('Activation already has a terminal receipt')
        while True:
            passed, state = ready(args.run.resolve())
            validations = [args.native/'verification.json',args.saved/'verification.json',args.offline]
            available = all(p.exists() for p in validations)
            if available and not all(read_json(p)['passed'] for p in validations):
                raise RuntimeError('Validation did not pass; do not activate')
            write_json(root/'status.json', {'at':now(), 'state':state if not passed else
                ('validations_ready' if available else 'waiting_for_validation'),
                'current_run':str(args.run.resolve().relative_to(ROOT))})
            if passed and available:
                break
            time.sleep(5)
        apply(root,args.staged.resolve(),args.native.resolve(),args.saved.resolve(),args.run.resolve(),args.offline.resolve())
        write_json(root/'status.json', {'at':now(), 'state':'activated',
            'current_run':str(args.run.resolve().relative_to(ROOT))})
        print('CANDIDATE_IO_ACTIVATED',root,flush=True)


if __name__ == '__main__':
    main()
