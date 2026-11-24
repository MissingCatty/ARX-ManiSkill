"""Nine conditional pure-D ARX slots. No retries and no changed robot/physics."""
import argparse
import copy
import fcntl
import itertools
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

from scripts.curobo_arx.data_factory.common import ROOT, read, write, sha, digest, verify_files, space
from scripts.curobo_arx.data_factory.launch import devices, environment
from scripts.curobo_arx.data_factory.watchdog import exclusive_stop, process_info
from scripts.curobo_arx.seed_augmentation.campaign import available_memory
from training.nearfield_arx.lateral_campaign import ledgers
from training.nearfield_arx.train import json_lines
from training.nearfield_arx.advisor40_metrics import offline_gate
from .advisor34_launch import owned_process, wait_ready, healthy, container_command
from .advisor31_campaign import exited
from .physical_startup import startup_expired
from .resources import probe_cuda
from .visual_grounding_spec import expected_reset

PARENT = ROOT / 'runs/nearfield_advisor34_20261111_01'
OBJECTS = ('household_hope_milk_carton', 'raisin_box', 'cheez_it')


def make_cases(inventory, old):
    source = {r['object_id']: r for r in inventory}
    trials, seeds = [], {}
    for obj in OBJECTS:
        src = source[obj]; seed = src['seed']
        # The old coarse/source grasp defines H only; the new canonical label
        # does not retarget the entrance or any post-H inference action.
        prior = np.asarray(seed['goal']['grasp_world']); handoff = prior.copy()
        handoff[:3, 3] -= .08 * handoff[:3, 2]
        history = [t for t in old['seed_trials'] if t['object_id'] == obj and t['method'] == 'D'
                   and t.get('vision_arm') == 'fresh' and t.get('fixed_entry_backoff_m', .08) == .08]
        if history:
            # Milk reuses the already executed historical H verbatim.
            handoff = np.asarray(history[0]['entry_world']); prior = np.asarray(history[0]['G0_world'])
        for shift, condition in ((0., 'normal'), (-.015, 'minus15'), (.015, 'plus15')):
            placement = copy.deepcopy(src['placement'])
            if shift:
                placement['placement']['position'][0] += shift
                placement['offset_xy_m'][0] += shift
                placement['minimum_support_border_m'] -= abs(shift)
                if placement['minimum_support_border_m'] <= 0:
                    raise ValueError('Declared target shift exceeds original tabletop support')
                placement['placement_id'] = digest(['advisor40', src['placement']['placement_id'], shift])[:20]
                placement['registered_counterfactual_delta_world_m'] = [shift, 0., 0.]
            ident = digest(['advisor40', src['seed_id'], placement['placement_id']])[:32]
            new_seed = copy.deepcopy(seed)
            new_seed.update(seed_id=ident, placement=placement, source_seed_id=src['seed_id'])
            seeds[ident] = new_seed
            row = {k: copy.deepcopy(src[k]) for k in ('object_id', 'source_family', 'source_split', 'geometry_group', 'renderer')}
            diagnostic = np.array(src['source_Gstar_world']); diagnostic[0, 3] += shift
            row.update(trial_id=f'a40_{len(trials):02d}_{obj}_{condition}', seed_id=ident,
                source_seed_id=src['seed_id'], placement=placement, method='D', policy='advisor40_pure_full6d',
                phase='pure_visual_validation', condition=condition, vision_arm='fresh', repeat=1,
                G0_world=prior.tolist(), source_Gstar_world=diagnostic.tolist(),
                entry_world=handoff.tolist(), offset_goal_world=handoff.tolist(),
                fixed_entry_backoff_m=.08, target_delta_world_m=[shift, 0., 0.],
                perturbation={'type': 'registered_target_world_X', 'delta_m': shift},
                no_retry=True, G0_role='Historical H generation only; never a post-H guide',
                historical_H_reused=bool(history))
            trials.append(row)
    validate_cases(trials)
    return trials, seeds


def validate_cases(trials):
    if len(trials) != 9 or len({t['trial_id'] for t in trials}) != 9:
        raise ValueError('Exactly nine unique conditional slots required')
    for obj in OBJECTS:
        rows = [t for t in trials if t['object_id'] == obj]
        if len(rows) != 3 or {t['condition'] for t in rows} != {'normal', 'minus15', 'plus15'}:
            raise ValueError('Each object needs normal and both target shifts')
        if any(t['entry_world'] != rows[0]['entry_world'] or t['G0_world'] != rows[0]['G0_world'] for t in rows):
            raise ValueError('H or prior followed the displaced object')
        if any(t['method'] != 'D' or not t['no_retry'] for t in rows):
            raise ValueError('Unexpected method/retry')


def prepare_spec(root):
    root = Path(root).resolve()
    if (root / 'physical_ready.json').exists():
        raise FileExistsError('Do not overwrite conditional physical specification')
    old = read(PARENT / 'registration.json')
    trials, seeds = make_cases(read(root / 'predecessor_snapshot/advisor39/source_inventory.json'), old)
    spec = {'trials': trials, 'seeds': seeds, 'old_registration': str(PARENT / 'registration.json'),
            'old_registration_sha256': sha(PARENT / 'registration.json'),
            'maximum_physical_slots': 9, 'maximum_model_calls_per_slot': 3,
            'maximum_refines_per_slot': 2, 'automatic_closure': False, 'automatic_retry': False,
            'world_X_target_shifts_m': [0., -.015, .015], 'H_fixed_for_each_object': True,
            'canonical_goals_not_used_for_H_or_post_H_control': True,
            'expected_resets': {t['trial_id']: expected_reset(seeds[t['seed_id']], t) for t in trials},
            'success': 'Original bilateral grasp and lift >60mm at EVERY sample throughout final >=3-second hold',
            'resources_wait_seconds': 12 * 3600, 'physical_wall_seconds': 12 * 3600,
            'stopping': 'Resource/configuration/record/idle uncertainty stops remaining slots; ordinary grasp failures continue'}
    write(root / 'physical_spec.json', spec)
    sources = [*sorted((ROOT / 'scripts/curobo_arx/nearfield').glob('advisor40_*.py')),
               ROOT / 'scripts/curobo_arx/nearfield/runtime_advisor40.py']
    # New code is pinned before model results; the frozen parent supplies all
    # old robot, environment, planner, renderer and success-evaluator settings.
    files = {str(p): sha(p) for p in sources + [root / 'physical_spec.json', PARENT / 'registration.json']}
    write(root / 'physical_ready.json', {'command': [sys.executable, '-u', '-m',
        'scripts.curobo_arx.nearfield.advisor40_online', 'run', str(root)], 'files': files,
        'CPU_preparation_only': True, 'simulator_started': False, 'physical_trials': 0})
    return {'conditional_slots': 9, 'simulator_started': False}


def choose_roles(inventory):
    choices = []
    for sim, policy, compute in itertools.product(inventory, repeat=3):
        if sim['uuid'] in (policy['uuid'], compute['uuid']) or sim['free_mb'] < 9000:
            continue
        shared = policy['uuid'] == compute['uuid']
        if shared and policy['free_mb'] < 14000:
            continue
        if not shared and (policy['free_mb'] < 9500 or compute['free_mb'] < 6500):
            continue
        score = min(sim['free_mb'] - 9000, policy['free_mb'] - (14000 if shared else 9500),
                    compute['free_mb'] - (14000 if shared else 6500))
        choices.append((score, {'sim_uuid': sim['uuid'], 'policy_uuid': policy['uuid'], 'compute_uuid': compute['uuid']}))
    return max(choices, key=lambda r: r[0])[1] if choices else None


def prepare_runtime(root, roles):
    spec = read(root / 'physical_spec.json'); old = read(spec['old_registration'])
    out = root / 'physical'; out.mkdir(exist_ok=False); space(out)
    verify_files({**old['execution_source_files'], **old['dependencies']})
    prior = Path(old['source_snapshot']); source = out / 'source_snapshot/COOLAS-ARX'
    shutil.copytree(prior, source, symlinks=True, ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('tmp', 'curobo'):
        (source.parent / name).symlink_to(ROOT.parent / name, target_is_directory=True)
    execution = {str(source / Path(p).relative_to(prior)) if Path(p).is_relative_to(prior) else p: h
                 for p, h in old['execution_source_files'].items()}
    added = [*sorted((ROOT / 'scripts/curobo_arx/nearfield').glob('advisor40_*.py')),
             ROOT / 'scripts/curobo_arx/nearfield/runtime_advisor40.py',
             ROOT / 'scripts/curobo_arx/nearfield/advisor36_contract.py',
             *sorted((ROOT / 'training/nearfield_arx').glob('*.py'))]
    for p in added:
        dest = source / p.relative_to(ROOT); shutil.copy2(p, dest); execution[str(dest)] = sha(dest)
    reg = copy.deepcopy(old); now = time.time()
    for key in ('carried_results', 'prior_results', 'prerequisite_results', 'carried_terminal_ids', 'ancestor_campaigns'):
        reg[key] = []
    for key in ('advisor34_policy', 'advisor33_policy', 'advisor31_policy', 'closedloop30_policy'):
        reg.pop(key, None)
    reg.update(stage='advisor40_pure_visual_9', project_root=str(ROOT), source_snapshot=str(source),
        execution_source_files=execution, source_identity=digest(execution), seeds=spec['seeds'], seed_trials=spec['trials'],
        runtime_module='scripts.curobo_arx.nearfield.runtime_advisor40',
        policy_module='scripts.curobo_arx.nearfield.advisor40_policy',
        coordinator_module='scripts.curobo_arx.nearfield.advisor40_online',
        requested_devices=roles, created_unix=now, campaign_started_unix=now,
        deadline_unix=now + spec['physical_wall_seconds'], wall_hours=spec['physical_wall_seconds'] / 3600,
        global_physical_budget=9, max_physical_trials=9, envs_per_gpu=1, object_block_attempts=1,
        training_authorized_here=False, no_automatic_retry=True, automatic_attempt_retries=0,
        automatic_startup_retries=0, supplemental_interruptions={},
        continuation_of=str(PARENT), remaining_trial_ids=[t['trial_id'] for t in spec['trials']],
        pure_visual_policy={'G0_after_H': False, 'canonical_after_H': False,
            'model_decisions_maximum': 3, 'refines_maximum': 2, 'full_six_values_preserved': True,
            'explicit_GRASP_only': True, 'budget_in_model_prompt': False,
            'geometry_closure_controller': False, 'numeric_anomaly_bounds_m_deg': [.120, 45.],
            'original_IK_collision_joint_limits_nominal_closed_checks': True},
        execution_authority={'file': str(root / 'user_authority.json'), 'sha256': sha(root / 'user_authority.json')})
    training = read(root / 'registration.json'); checkpoint = root / 'D/checkpoint-2865'
    reg['learned_policy'].update(base_model=training['model'], gpu_uuid=roles['policy_uuid'],
        checkpoints={'D': str(checkpoint)}, max_policy_calls={'D': 3}, max_neural_calls={'D': 3},
        D_max_refines=2, no_teacher_goal_or_forced_GRASP=True, fixed_G0=False,
        warmups={'D': [json_lines(root / 'dataset/train.jsonl')[i] for i in (0, 1)]},
        training_bindings={'D': {'registration': str(root / 'registration.json'), 'sha256': sha(root / 'registration.json')}})
    dependencies = dict(old['dependencies'])
    dependencies.update(read(root / 'model_pins.json'))
    for seed in spec['seeds'].values():
        dependencies.update(seed['files'])
    for p in checkpoint.iterdir():
        if p.is_file():
            dependencies[str(p)] = sha(p)
    for p in (root / 'physical_spec.json', root / 'offline_gate.json', root / 'registration.json', root / 'user_authority.json'):
        dependencies[str(p)] = sha(p)
    for item in reg['learned_policy']['warmups']['D']:
        dependencies.update({p: sha(p) for p in item['images']})
    reg['dependencies'] = dependencies
    shutil.copy2(PARENT / 'storage_policy.json', out / 'storage_policy.json')
    write(out / 'source_change_receipt.json', {'new_sources': {str(p): sha(p) for p in added},
        'original_parent_registration': spec['old_registration'],
        'robot_physics_initial_conditions_success_unchanged_except_declared_target_X': True,
        'new_policy': reg['pure_visual_policy']})
    write(out / 'expected_resets.json', spec['expected_resets'])
    ledgers(out, reg); write(out / 'catalog.json', reg['catalog']); write(out / 'registration.json', reg)
    # Verify the exact runtime imports under the pinned Isaac Python without
    # creating a World or loading a GPU. Never download a container image.
    name = 'arx-a40-import-' + digest(str(out))[:12]
    if subprocess.run(['podman', 'container', 'exists', name], capture_output=True, timeout=30).returncode != 1:
        raise RuntimeError('Previous owned import probe is not confirmed absent')
    command = ['podman', 'run', '--pull=never', '--name', name, '--user', '0', '--security-opt=label=disable',
        '-v', str(ROOT.parent) + ':' + str(ROOT.parent) + ':rw', '-w', str(source),
        '--entrypoint', '/isaac-sim/python.sh', 'localhost/arx-s02-isaac:6.0.1', '-c',
        'import scripts.curobo_arx.nearfield.runtime_advisor40; print("advisor40_runtime_import_ok")']
    try:
        probe = subprocess.run(command, capture_output=True, text=True, timeout=600)
        proof = {'pass': probe.returncode == 0 and 'advisor40_runtime_import_ok' in probe.stdout,
                 'command': command, 'stdout': probe.stdout[-4000:], 'stderr': probe.stderr[-4000:],
                 'simulator_World_started': False, 'image_download': False}
        write(out / 'container_import_preflight.json', proof)
        if not proof['pass']:
            raise RuntimeError('Pinned Isaac runtime import failed')
    finally:
        subprocess.run(['podman', 'rm', '--force', name], capture_output=True, timeout=60)
        absent = subprocess.run(['podman', 'container', 'exists', name], capture_output=True, timeout=30).returncode == 1
        write(out / 'import_container_exit.json', {'absent': absent, 'name': name})
        if not absent:
            raise RuntimeError('Owned import container exit unconfirmed')
    verify_files({**execution, **dependencies})
    write(out / 'full_preflight.json', {'pass': True, 'files_checked': len({**execution, **dependencies}),
                                       'physical_trials': 0})
    return out, reg


def summarize(out, reg):
    rows = [r for p in sorted(out.glob('groups/*/results.json')) for r in read(p)['trials']]
    if len({r['trial_id'] for r in rows}) != len(rows):
        raise RuntimeError('Repeated physical slot')
    ids = {r['trial_id'] for r in rows}
    pending = [t['trial_id'] for t in reg['seed_trials'] if t['trial_id'] not in ids]
    interrupted = [ident for ident in pending if list(out.glob('groups/*/trials/' + ident))]
    unexecuted = [ident for ident in pending if ident not in interrupted]
    result = {'scheduled': 9, 'completed': len(rows), 'successes': sum(bool(r['success']) for r in rows),
        'ordinary_failed_completed': sum(not r['success'] for r in rows),
        'unexecuted': unexecuted, 'interrupted_unknown': interrupted,
        'unexecuted_or_interrupted': pending,
        'trials': rows, 'by_object': {obj: {t['condition']: {'success': t['success'], 'outcome': t['outcome'],
            'physical_dispatches': t['physical_dispatches']} for t in rows if t['object_id'] == obj} for obj in OBJECTS}}
    write(out / 'summary.json', result)
    return result


def run(root):
    root = Path(root).resolve(); lock = (root / 'online.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'STOP').exists() or (root / 'ONLINE_STARTED.json').exists():
        raise RuntimeError('No physical replay or STOP bypass')
    ready = read(root / 'physical_ready.json'); verify_files(ready['files'])
    gate = read(root / 'offline_gate.json'); score = read(root / 'offline/new_D_test/metrics.json')
    if not (gate['pass'] and offline_gate(score)['pass'] and gate['offline_record_integrity_pass']
            and gate['required_offline_analyses_complete']):
        raise RuntimeError('Offline primary thresholds/evidence do not authorize physical dispatch')
    if gate['training_registration_sha256'] != sha(root / 'registration.json'):
        raise RuntimeError('Training/gate binding mismatch')
    if sha(root / 'offline/new_D_test/predictions.jsonl') != score['predictions_sha256']:
        raise RuntimeError('Sealed test predictions changed')
    write(root / 'ONLINE_STARTED.json', {'wall': time.time(), 'pid': os.getpid(), 'gate_sha256': sha(root / 'offline_gate.json')})
    spec = read(root / 'physical_spec.json'); waiting_deadline = time.monotonic() + spec['resources_wait_seconds']
    processes = {}; active = None; name = None; out = None; error = None; unresolved = False
    try:
        stable = 0; previous = None
        while stable < 3:
            if (root / 'STOP').exists() or time.monotonic() > waiting_deadline:
                raise RuntimeError('Stopped or physical resource wait deadline')
            inventory = devices(); roles = choose_roles(inventory)
            stable = stable + 1 if roles and roles == previous else int(bool(roles))
            previous = roles
            write(root / 'state.json', {'status': 'offline_pass_waiting_for_physical_resources',
                'inventory': inventory, 'roles': roles, 'stable_samples': stable, 'physical_trials_started': 0, 'wall': time.time()})
            if stable < 3:
                time.sleep(5)
        out, reg = prepare_runtime(root, roles)
        exited(PARENT)
        by = {d['uuid']: d for d in devices()}
        for uuid in sorted(set(roles.values())):
            proof = probe_cuda(by[uuid], environment(by[uuid]))
            write(out / ('cuda_' + str(by[uuid]['index']) + '.json'), proof)
            if not proof['pass']:
                raise RuntimeError('CUDA preflight failed')
        for role, python, module, args in (
            ('policy', ROOT.parent / 'tmp/arx-show-harness-env/bin/python', reg['policy_module'], [str(out)]),
            ('planner', ROOT.parent / 'curobo/.venv/bin/python', reg['worker_module'],
             ['--directory', str(out / 'workers/planner'), '--registration', str(out / 'registration.json')])):
            directory = out / 'workers' / role
            env = environment(by[roles['policy_uuid' if role == 'policy' else 'compute_uuid']])
            env.update(PYTHONPATH=reg['source_snapshot'], PYTHONUNBUFFERED='1')
            processes[role] = owned_process([str(python), '-m', module, *args], reg['source_snapshot'], env, directory)
            if role == 'policy':
                (directory / 'queue').mkdir(); (directory / 'done').mkdir()
            wait_ready(processes[role], directory, out)
        groups = {}
        for t in reg['seed_trials']:
            groups.setdefault(t['renderer'], []).append(t['trial_id'])
        for index, (renderer, ids) in enumerate(groups.items()):
            healthy(out, processes); space(out)
            if (root / 'STOP').exists() or (out / 'STOP').exists() or time.time() > reg['deadline_unix']:
                raise RuntimeError('Stopped before next physical group')
            if next(d['free_mb'] for d in devices() if d['uuid'] == roles['sim_uuid']) < 9000:
                raise RuntimeError('Simulator memory below original floor')
            if available_memory() < reg['minimum_available_host_memory_bytes']:
                raise RuntimeError('Host memory below original floor')
            ident = f'advisor40_{index:02d}_{renderer.lower()}'
            name = 'arx-' + root.name + '-' + ident
            if subprocess.run(['podman', 'container', 'exists', name], capture_output=True, timeout=30).returncode != 1:
                raise RuntimeError('Old World absence unconfirmed')
            path = out / 'group_registrations' / (ident + '.json'); group = out / 'groups' / ident
            write(path, {'group_id': ident, 'renderer': renderer, 'gpu_uuid': roles['sim_uuid'], 'env_count': 1,
                'planner_directory': str(out / 'workers/planner'), 'candidate_directory': str(out / 'no_neural_requests'),
                'trial_ids': ids, 'stop_unix': reg['deadline_unix'], 'registration_sha256': sha(out / 'registration.json')})
            command = container_command(out, reg, roles['sim_uuid'], path, name)
            with path.with_suffix('.log').open('x') as log:
                active = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write(path.with_suffix('.process.json'), {'pid': active.pid, 'container': name, 'command': command,
                'owned': True, **(process_info(active.pid) or {})})
            started = time.monotonic()
            while active.poll() is None:
                healthy(out, processes)
                if (root / 'STOP').exists() or (out / 'STOP').exists() or time.time() > reg['deadline_unix']:
                    raise RuntimeError('Stopped during physical execution')
                if startup_expired(group, time.monotonic() - started, 1800):
                    raise RuntimeError('Original simulator startup budget exhausted')
                summary = summarize(out, reg)
                write(root / 'state.json', {'status': 'running_conditional_physical', 'completed': summary['completed'],
                    'successes': summary['successes'], 'scheduled': 9, 'wall': time.time()})
                time.sleep(2)
            code = active.returncode; active = None
            if subprocess.run(['podman', 'container', 'exists', name], capture_output=True, timeout=30).returncode != 1:
                raise RuntimeError('World removal unconfirmed')
            if code or (group / 'FAULT.json').exists():
                raise RuntimeError('Simulator/configuration/record/idle failure; subsequent slots stopped')
            if not read(group / 'shutdown_receipt.json')['all_trajectories_idle']:
                raise RuntimeError('Missing final idle receipt')
            if len(read(group / 'results.json')['trials']) != len(ids):
                raise RuntimeError('Unaccounted physical slots')
        verify_files({**reg['execution_source_files'], **reg['dependencies']})
    except BaseException:
        error = traceback.format_exc()
        write(root / 'physical_FAULT.json', {'error': error, 'wall': time.time()})
        exclusive_stop(root / 'STOP', {'reason': 'physical_resource_configuration_record_or_idle_failure', 'error': error})
    finally:
        if active is not None and active.poll() is None:
            subprocess.run(['podman', 'kill', '--signal', 'TERM', name], capture_output=True, timeout=30)
            try:
                active.wait(timeout=30)
            except subprocess.TimeoutExpired:
                unresolved = True
            if subprocess.run(['podman', 'container', 'exists', name], capture_output=True, timeout=30).returncode != 1:
                unresolved = True
        for role, process in processes.items():
            write(out / 'workers' / role / 'STOP', {'reason': 'Bounded advisor40 run ended'})
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    unresolved = True
            write(out / 'workers' / role / 'process_exit.json', {'pid': process.pid, 'exit_code': process.poll()})
        summary = summarize(out, reg) if out else {'completed': 0, 'successes': 0, 'unexecuted_or_interrupted': 9}
        state = {'status': 'fault' if error or unresolved else 'complete', 'summary': summary,
                 'unconfirmed_world_or_process': unresolved, 'wall': time.time()}
        if out:
            write(out / 'state.json', state)
        write(root / 'state.json', state)
    if error or unresolved:
        raise RuntimeError(error or 'Owned execution exit unconfirmed')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=('prepare', 'run')); p.add_argument('root')
    a = p.parse_args()
    print(prepare_spec(a.root)) if a.action == 'prepare' else run(a.root)
