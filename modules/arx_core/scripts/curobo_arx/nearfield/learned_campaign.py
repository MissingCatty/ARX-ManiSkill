"""Bounded advisor21 continuation: admit final models, then exactly four slots."""
import argparse
import copy
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import traceback
from contextlib import contextmanager
from scripts.curobo_arx.data_factory.common import ROOT, read, write, sha, digest, verify_files, space
from scripts.curobo_arx.data_factory.ledger import Ledger
from scripts.curobo_arx.data_factory.watchdog import exclusive_stop, process_info
from .prepare_v2 import predecessors
from .physical_startup import startup_expired
from . import campaign as launcher

PYTHON = ROOT.parent / 'tmp/arx-show-harness-env/bin/python'


@contextmanager
def registered_launcher_project(reg):
    """Host mounts use the registered project; executable code stays frozen."""
    original = launcher.ROOT
    project = Path(reg['project_root']).resolve()
    if not project.is_dir():raise RuntimeError('Registered host project is unavailable')
    launcher.ROOT = project
    try:yield
    finally:launcher.ROOT = original


def require_execution_authority(result, authorization_file=None):
    if not result['record_integrity_pass']:
        raise RuntimeError('Training records are not verified')
    if authorization_file is not None:
        authorization_file = Path(authorization_file).resolve()
        expected = ROOT / 'research_plan/chat/gemini回复/gemini回复22.md'
        if authorization_file != expected or not authorization_file.is_file():
            raise ValueError('Expected the user-invoked advisor22 four-slot instruction')
        return {'performance_admission_required': False, 'authorization': str(authorization_file),
            'authorization_sha256': sha(authorization_file),
            'scope': 'Exactly the original four diagnostic slots, unmodified checkpoint975 outputs',
            'old_admission_result_preserved': result['learned_closed_loop_admitted']}
    if not result['learned_closed_loop_admitted']:
        raise RuntimeError('The newly registered learned-policy admission did not pass')
    return {'performance_admission_required': True, 'authorization': None}


def prepare(training, out, checks, policy_uuid, compute_uuid, sim_uuid, *, authorization_file=None):
    training, out, checks = (Path(p).resolve() for p in (training, out, checks))
    if out.exists():raise FileExistsError('Preserve prior registrations')
    result = read(training / 'final_verification.json'); tr = read(training / 'registration.json')
    authority = require_execution_authority(result, authorization_file)
    verify_files(result['artifacts'])
    for name in ('source_files', 'data_files', 'model_files', 'image_files'):verify_files(tr[name])
    check = read(checks)
    if not check['pass']:raise RuntimeError('Offline runtime checks did not pass')
    verify_files(check['sources']);verify_files(check['evidence'])
    if len({policy_uuid, compute_uuid, sim_uuid}) != 3:raise ValueError('Use three explicitly separate free GPUs')
    plan = read(training / 'closed_loop_cases.json')
    if len(plan['slots']) != 4 or plan['max_physical_trials'] != 4:raise ValueError('Only four fixed slots')
    parent = Path(plan['slots'][0]['source_physical_campaign']); old = read(parent / 'registration.json')
    if any(Path(t['source_physical_campaign']) != parent for t in plan['slots']):raise ValueError('Unexpected source campaign')
    idle = predecessors([parent])
    if not read(parent / 'postrun_verification.json')['record_integrity_pass']:raise ValueError('Historical records not verified')
    verify_files({**old['execution_source_files'], **old['dependencies']})
    out.mkdir(parents=True);space(out)
    source = out / 'source_snapshot/COOLAS-ARX';source.parent.mkdir()
    previous = Path(old['source_snapshot'])
    shutil.copytree(previous, source, symlinks=True, ignore=shutil.ignore_patterns('__pycache__'))
    (source.parent / 'curobo').symlink_to(ROOT.parent / 'curobo', target_is_directory=True)
    (source.parent / 'tmp').symlink_to(ROOT.parent / 'tmp', target_is_directory=True)
    cache = ROOT / 'runs/data_factory_runtime_cache'
    if not all((cache / name).is_dir() for name in ('ov', 'nvidia')):
        raise RuntimeError('Original shared Isaac runtime cache mount is unavailable')
    (source / 'runs').mkdir(exist_ok=True)
    (source / 'runs/data_factory_runtime_cache').symlink_to(cache, target_is_directory=True)
    execution = {str(source / Path(p).relative_to(previous)) if Path(p).is_relative_to(previous) else p: h
        for p, h in old['execution_source_files'].items()}
    frozen_training = Path(tr['source_snapshot']) / 'training'
    for f in frozen_training.rglob('*'):
        if f.is_file() and f.suffix in ('.py', '.cpp'):
            dest = source / 'training' / f.relative_to(frozen_training);dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest);execution[str(dest)] = sha(dest)
    names = ('training/nearfield_arx/learned_policy.py', 'scripts/curobo_arx/nearfield/runtime_learned.py',
        'scripts/curobo_arx/nearfield/learned_campaign.py', 'scripts/curobo_arx/nearfield/learned_report.py',
        'scripts/curobo_arx/nearfield/test_learned.py')
    for name in names:
        dest = source / name;shutil.copy2(ROOT / name, dest);execution[str(dest)] = sha(dest)
    # Historical snapshots contained the simulator/worker entry points only;
    # the newly frozen coordinator and reporter also need their support files.
    # Add missing files without replacing any inherited physical executor.
    for package in ('scripts/curobo_arx/data_factory', 'scripts/curobo_arx/nearfield',
                    'scripts/curobo_arx/seed_augmentation'):
        for file in (ROOT / package).glob('*.py'):
            dest = source / file.relative_to(ROOT)
            if not dest.exists():
                shutil.copy2(file, dest);execution[str(dest)] = sha(dest)
    reg = copy.deepcopy(old)
    for key in ('entry_wait_continuation_of', 'startup_continuation_of', 'serialization_recovery_receipt',
                'physical_data_contract', 'preserved_completed_results', 'consumed_physical_budget'):
        reg.pop(key, None)
    deps = {**old['dependencies'], **tr['model_files'], **tr['data_files'], **tr['image_files']}
    for file in (training / 'registration.json', training / 'final_verification.json', training / 'closed_loop_cases.json',
                 parent / 'postrun_verification.json', parent / 'final_idle_verification.json', checks):
        deps[str(file)] = sha(file)
    deps.update(result['artifacts'])
    if authority['authorization']:
        shutil.copy2(authority['authorization'], out / 'advisor22.md')
        deps[str(out / 'advisor22.md')] = sha(out / 'advisor22.md')
        deps[str(training / 'four_slot_continuation.json')] = sha(training / 'four_slot_continuation.json')
        write(out / 'execution_authority.json', authority)
        deps[str(out / 'execution_authority.json')] = sha(out / 'execution_authority.json')
    for renderer, origin in old['reference_files'].items():
        dest = out / 'references' / (renderer + '.json');dest.parent.mkdir(exist_ok=True);shutil.copy2(origin, dest)
        reg['reference_files'][renderer] = str(dest);deps[str(dest)] = sha(dest)
    for origin, name in ((parent / 'storage_policy.json', 'storage_policy.json'),
                         (training / 'advisor21.md', 'advisor21.md'), (checks, 'offline_checks.json')):
        shutil.copy2(origin, out / name);deps[str(out / name)] = sha(out / name)
    trials = []
    for slot in plan['slots']:
        t = copy.deepcopy(slot['source_trial']);t.update(trial_id=slot['slot_id'], method=slot['method'],
            policy='learned_' + slot['method'], source_teacher_trial=slot['source_trial']['trial_id'],
            truth_replay_only_no_policy_training=False)
        trials.append(t)
        catalog = copy.deepcopy(old['catalog'])
        for asset in catalog['objects']:
            asset['collection_enabled'] = asset['object_id'] == t['object_id']
            if asset['collection_enabled']:asset['collection_placements'] = [t['placement']]
        path = out / 'trial_ledgers' / (t['trial_id'] + '.sqlite3');path.parent.mkdir(exist_ok=True)
        ledger = Ledger(path);ledger.initialize(catalog, {**old['quotas'], 'first_object': t['object_id'],
            'max_physical_attempts': 1, 'max_total_dispatches': 1, 'physical_per_bin': 1,
            'planning_rejections_per_bin': 1, 'goals_per_object': 1, 'successes_per_bin': 1});ledger.close()
    checkpoints = {};warmups = {}
    for method in ('C', 'D'):
        folder = Path(result['methods'][method]['checkpoint']);checkpoints[method] = str(folder)
        cert = read(folder / 'checkpoint_receipt.json')
        for name, h in cert['files'].items():deps[str(folder / name)] = h
        deps[str(folder / 'checkpoint_receipt.json')] = sha(folder / 'checkpoint_receipt.json')
        import json
        rows = {r['sample_id']: r for r in (json.loads(l) for l in (training / 'model_rows' / method / 'train.jsonl').read_text().splitlines())}
        warmups[method] = [rows[i] for i in tr['warmup_sample_ids']]
    from training.nearfield_arx.publish import ROBOT
    policy = {'checkpoints': checkpoints, 'base_model': tr['model'], 'training': str(training),
        'training_registration_sha256': sha(training / 'registration.json'), 'gpu_uuid': policy_uuid,
        'robot_usd': str(ROBOT), 'visual_contract': read(Path(tr['dataset']) / 'input_contract.json'),
        'max_new_tokens': tr['max_new_tokens'], 'warmups': warmups, 'request_timeout_seconds': 120,
        'C_max_motion_requests': plan['C_max_motion_requests'], 'D_max_refines': 2,
        'max_policy_calls': plan['max_policy_calls'], 'fixed_G0': True,
        'decode_axes': 'Exact measured EEF of the model-input image; check drift before dispatch',
        'no_teacher_goal_or_forced_GRASP': True, 'no_retry': True}
    env = os.environ.copy();env.update(PYTHONPATH=str(source), CUDA_VISIBLE_DEVICES='')
    probe = subprocess.run([str(PYTHON), '-c',
        'import scripts.curobo_arx.nearfield.learned_campaign; '
        'import scripts.curobo_arx.nearfield.runtime_learned; '
        'import scripts.curobo_arx.nearfield.learned_report; '
        'import training.nearfield_arx.learned_policy; '
        'import scripts.curobo_arx.nearfield.resources'], cwd=source, env=env,
        capture_output=True, text=True, timeout=60)
    write(out / 'snapshot_import_preflight.json', {'pass': probe.returncode == 0,
        'exit_code': probe.returncode, 'stdout': probe.stdout, 'stderr': probe.stderr,
        'gpu_and_simulation_started': False})
    if probe.returncode:raise RuntimeError('Frozen snapshot imports incomplete: ' + probe.stderr)
    deps[str(out / 'snapshot_import_preflight.json')] = sha(out / 'snapshot_import_preflight.json')
    reg.update(stage='arx_learned_cd_four_slot_v1', created_unix=time.time(), campaign_started_unix=time.time(),
        deadline_unix=time.time() + 3 * 3600, wall_hours=3, production=False, seed_trials=trials,
        seeds={t['seed_id']: old['seeds'][t['seed_id']] for t in trials}, source_snapshot=str(source),
        execution_source_files=execution, source_identity=digest(execution), dependencies=deps,
        runtime_module='scripts.curobo_arx.nearfield.runtime_learned',
        coordinator_module='scripts.curobo_arx.nearfield.learned_campaign',
        max_physical_trials=4, global_physical_budget=4, recovery_slots_total=0,
        carried_results=[], prior_results=[], prerequisite_results=[],
        requested_devices={'compute_uuid': compute_uuid, 'sim_uuid': sim_uuid, 'policy_uuid': policy_uuid},
        startup_timeout_seconds=1800, automatic_startup_retries=0, automatic_attempt_retries=0,
        learned_policy=policy, execution_authority=authority, formal_48_trials_authorized_here=False)
    write(out / 'predecessor_exit_receipt.json', idle)
    write(out / 'source_change_receipt.json', {'parent': str(parent), 'training': str(training),
        'added_sources': names, 'physical_planner_robot_scene_and_success_unchanged': True,
        'source_Gstar_use': 'Historical entry setup and offline evaluation only; never generates learned local actions',
        'four_new_slots': True, 'old_trials_replayed': False})
    write(out / 'registration.json', reg);write(out / 'catalog.json', reg['catalog'])
    verify_files({**execution, **deps})
    write(out / 'state.json', {'status': 'prepared', 'physical_trials': 0, 'scheduled_slots': 4})
    return out


def run(campaign):
    campaign = Path(campaign).resolve();reg = read(campaign / 'registration.json');directory = campaign / 'workers/policy'
    if (campaign / 'STOP').exists() or directory.exists():raise RuntimeError('No service or experiment retries')
    from scripts.curobo_arx.data_factory.launch import devices, environment, allocate
    from .resources import probe_cuda
    devices_now = devices();allocate(devices_now, 1, reg['requested_devices']['compute_uuid'], [reg['requested_devices']['sim_uuid']])
    device = next(d for d in devices_now if d['uuid'] == reg['learned_policy']['gpu_uuid'])
    if device['free_mb'] < 18000:raise RuntimeError('Policy GPU needs 18000 MiB free before loading both models')
    proof = probe_cuda(device, environment(device));write(campaign / 'policy_cuda_preflight.json', proof)
    if not proof['pass']:raise RuntimeError('Policy CUDA initialization failed')
    directory.mkdir(parents=True);(directory / 'queue').mkdir();(directory / 'done').mkdir()
    env = environment(device);env['PYTHONPATH'] = reg['source_snapshot']
    command = [str(PYTHON), '-m', reg.get('policy_module', 'training.nearfield_arx.learned_policy'), str(campaign)]
    process = None;finished = threading.Event()
    try:
        with (directory / 'process.log').open('x') as log:
            process = subprocess.Popen(command, cwd=reg['source_snapshot'], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(directory / 'process.json', {'pid': process.pid, 'command': command, 'owned': True, **(process_info(process.pid) or {})})
        deadline = time.monotonic() + 600
        while not (directory / 'ready.json').exists():
            if process.poll() is not None or time.monotonic() >= deadline or (campaign / 'STOP').exists():
                raise RuntimeError('Pinned learned policy service failed to become ready')
            time.sleep(.25)
        def monitor():
            while not finished.wait(.5):
                if process.poll() is not None:
                    exclusive_stop(campaign / 'STOP', {'reason': 'owned_learned_policy_service_exited'});return
        watcher = threading.Thread(target=monitor, daemon=True);watcher.start()
        previous = launcher.startup_expired
        launcher.startup_expired = lambda group, elapsed: startup_expired(group, elapsed, 1800)
        try:
            with registered_launcher_project(reg):
                launcher.run(campaign, reg['requested_devices']['compute_uuid'], reg['requested_devices']['sim_uuid'])
        finally:launcher.startup_expired = previous
    except BaseException:
        error = traceback.format_exc();exclusive_stop(campaign / 'STOP', {'reason': 'learned_coordinator_fault', 'error': error})
        write(campaign / 'coordinator_fault.json', {'error': error})
        if not (campaign / 'launch.json').exists():
            write(campaign / 'state.json', {'status': 'fault', 'physical_trials': 0, 'unconfirmed_world': False, 'error': error})
        raise
    finally:
        finished.set()
        if process is not None:
            write(directory / 'STOP', {'reason': 'bounded learned four-slot campaign finished'})
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=30)
            write(directory / 'process_exit.json', {'pid': process.pid, 'exit_code': process.returncode, 'wall': time.time()})
    from .learned_report import report
    return report(campaign)


def wait_and_run(training, out, checks, devices):
    """One continuation only; a failed gate never launches a physical World."""
    training = Path(training).resolve();statefile = training / 'four_slot_continuation.json'
    lock = (training / 'four_slot_continuation.lock').open('a+');fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if statefile.exists():raise FileExistsError('One registered continuation, no automatic repeat')
    write(statefile, {'status': 'waiting_for_registered_training', 'physical_trials': 0,
        'checks': str(Path(checks).resolve()), 'planned_campaign': str(Path(out).resolve()), 'devices': devices})
    try:
        while not (training / 'final_verification.json').exists():
            if (training / 'STOP').exists() or (training / 'FAULT.json').exists():
                raise RuntimeError('Training stopped; do not start policy trials')
            if time.time() > read(training / 'registration.json')['created_unix'] + 5 * 3600:
                raise TimeoutError('Training completion wait expired; no physical dispatch')
            time.sleep(5)
        result = read(training / 'final_verification.json')
        if not result['learned_closed_loop_admitted']:
            write(statefile, {'status': 'not_admitted', 'physical_trials': 0,
                'reasons': {m: r['learned_closed_loop_admission'] for m, r in result['methods'].items()}});return
        campaign = prepare(training, out, checks, *devices)
        write(statefile, {'status': 'running_registered_four_slots', 'campaign': str(campaign)})
        reg = read(campaign / 'registration.json')
        env = os.environ.copy();env['PYTHONPATH'] = reg['source_snapshot']
        command = [str(PYTHON), '-m', 'scripts.curobo_arx.nearfield.learned_campaign', 'run', str(campaign)]
        with (campaign / 'coordinator.log').open('x') as log:
            process = subprocess.Popen(command, cwd=reg['source_snapshot'], env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(campaign / 'execution_coordinator.json', {'pid': process.pid, 'command': command,
            'owned': True, **(process_info(process.pid) or {})})
        code = process.wait()
        write(campaign / 'execution_coordinator_exit.json', {'pid': process.pid, 'exit_code': code, 'wall': time.time()})
        if code:raise RuntimeError('Frozen four-slot coordinator failed; no automatic retry')
        write(statefile, {'status': 'finished', 'campaign': str(campaign), 'result': read(campaign / 'learned_verification.json')})
    except BaseException:
        write(statefile, {'status': 'fault', 'error': traceback.format_exc(), 'automatic_retry': False});raise


if __name__ == '__main__':
    p = argparse.ArgumentParser();sub = p.add_subparsers(dest='action', required=True)
    q = sub.add_parser('wait');q.add_argument('--training', required=True);q.add_argument('--out', required=True)
    q.add_argument('--checks', required=True);q.add_argument('--policy-uuid', required=True)
    q.add_argument('--compute-uuid', required=True);q.add_argument('--sim-uuid', required=True)
    q = sub.add_parser('run');q.add_argument('campaign');a = p.parse_args()
    if a.action == 'run':run(a.campaign)
    else:wait_and_run(a.training, a.out, a.checks, (a.policy_uuid, a.compute_uuid, a.sim_uuid))
