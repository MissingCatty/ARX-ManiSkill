"""Three explicitly authorized observational ARX trials after a failed offline gate.

The failed training gate remains unchanged. No retraining, automatic retry,
post-H goal guide, synthetic Z, or automatic closure is introduced.
"""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

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
from .advisor40_online import PARENT, OBJECTS, choose_roles
from .visual_grounding_spec import expected_reset

TRAINING = ROOT / 'runs/nearfield_advisor40_20261112_01'
MODULE = 'scripts.curobo_arx.nearfield.advisor40_preview'
USER_REQUEST = '你先用这个训出来的真正抓一下试试，出几个视频看看'


def validate_preview(spec):
    trials = spec['trials']
    if len(trials) != 3 or len({t['trial_id'] for t in trials}) != 3:
        raise ValueError('Exactly three observational slots are authorized')
    if {t['object_id'] for t in trials} != set(OBJECTS):
        raise ValueError('One normal trial for each of the three training objects')
    if spec['maximum_physical_slots'] != 3 or spec['automatic_retry'] or spec['automatic_closure']:
        raise ValueError('Unexpected preview budget or retry/closure change')
    original = read(TRAINING / 'physical_spec.json')
    by = {t['object_id']: t for t in original['trials'] if t['condition'] == 'normal'}
    for trial in trials:
        if trial['condition'] != 'normal' or trial['method'] != 'D' or not trial['no_retry']:
            raise ValueError('Only normal-entry D without retries is authorized')
        for key in ('entry_world', 'G0_world', 'placement', 'renderer', 'seed_id', 'target_delta_world_m'):
            if trial[key] != by[trial['object_id']][key]:
                raise ValueError('Frozen normal case changed: ' + key)
        seed = spec['seeds'][trial['seed_id']]
        if spec['expected_resets'][trial['trial_id']] != expected_reset(seed, trial):
            raise ValueError('Frozen reset binding changed')


def check_authority(root):
    authority = read(root / 'user_authority.json')
    if authority.get('user_request') != USER_REQUEST or authority.get('kind') != 'explicit_observational_preview':
        raise RuntimeError('Missing explicit user preview authorization')
    if authority.get('maximum_slots') != 3 or not authority.get('allow_execution_despite_offline_performance_failure'):
        raise RuntimeError('No authorization for these three below-threshold observations')
    if not authority.get('robot_physics_success_frozen') or not authority.get('original_results_preserved'):
        raise RuntimeError('Original constraints and results must remain')
    validate_preview(read(root / 'physical_spec.json'))
    verify_files(read(root / 'preview_registration.json')['files'])
    receipt = read(TRAINING / 'D/training_receipt.json')
    if not receipt['pass'] or receipt['optimizer_steps'] != 2865 or not receipt['ten_epochs_completed']:
        raise RuntimeError('The requested epoch10 model is not complete')
    metrics = read(TRAINING / 'offline/new_D_test/metrics.json')
    if sha(TRAINING / 'offline/new_D_test/predictions.jsonl') != metrics['predictions_sha256']:
        raise RuntimeError('Sealed-test evidence changed')
    measured_gate = offline_gate(metrics)
    if measured_gate != read(root / 'offline_performance_observed.json')['gate']:
        raise RuntimeError('Original offline result changed')
    return {'explicit_user_authorization': True, 'offline_performance_pass': measured_gate['pass'],
            'activation': 'User-requested observational three-trial preview; not gate-qualified validation',
            'original_performance_gate_preserved': True}


def prepare(root):
    root = Path(root).resolve()
    if root.exists():
        raise FileExistsError('Preserve existing preview registration')
    original = read(TRAINING / 'physical_spec.json')
    trials = []
    for item in original['trials']:
        if item['condition'] == 'normal':
            trial = copy.deepcopy(item)
            trial.update(trial_id='preview_' + item['trial_id'], phase='user_observational_preview',
                         parent_unexecuted_trial_id=item['trial_id'])
            trials.append(trial)
    seeds = {t['seed_id']: original['seeds'][t['seed_id']] for t in trials}
    spec = {**original, 'trials': trials, 'seeds': seeds, 'maximum_physical_slots': 3,
            'world_X_target_shifts_m': [0.], 'resources_wait_seconds': 2 * 3600,
            'physical_wall_seconds': 4 * 3600,
            'expected_resets': {t['trial_id']: expected_reset(seeds[t['seed_id']], t) for t in trials}}
    validate_preview(spec)
    verify_files(read(TRAINING / 'source_pins.json'))
    verify_files(read(TRAINING / 'physical_ready.json')['files'])
    checkpoint = TRAINING / 'D/checkpoint-2865'
    cert = read(checkpoint / 'checkpoint_receipt.json')
    verify_files({str(checkpoint / p): h for p, h in cert['files'].items()})
    root.mkdir(parents=True); space(root)
    write(root / 'user_authority.json', {'user_request': USER_REQUEST, 'kind': 'explicit_observational_preview',
        'maximum_slots': 3, 'allow_execution_despite_offline_performance_failure': True,
        'robot_physics_success_frozen': True, 'original_results_preserved': True,
        'training_authorized_here': False, 'automatic_retry': False, 'created_unix': time.time()})
    write(root / 'physical_spec.json', spec)
    score = read(TRAINING / 'offline/new_D_test/metrics.json')
    write(root / 'offline_performance_observed.json', {'gate': offline_gate(score), 'metrics': score['overall'],
        'metrics_source': str(TRAINING / 'offline/new_D_test/metrics.json'),
        'original_offline_analysis_complete': (TRAINING / 'offline_gate.json').exists(),
        'failed_gate_unchanged_and_not_relabelled_as_pass': True})
    paths = [Path(__file__).resolve(), ROOT / 'scripts/curobo_arx/nearfield/test_advisor40_preview.py',
        TRAINING / 'registration.json', TRAINING / 'D/training_receipt.json',
        TRAINING / 'offline/new_D_test/metrics.json', TRAINING / 'offline/new_D_test/predictions.jsonl',
        checkpoint / 'checkpoint_receipt.json', root / 'user_authority.json', root / 'physical_spec.json',
        root / 'offline_performance_observed.json']
    files = {str(p): sha(p) for p in paths}
    files.update(read(TRAINING / 'physical_ready.json')['files'])
    write(root / 'preview_registration.json', {'training_root': str(TRAINING), 'files': files,
        'observational_slots': 3, 'original_conditional_nine_not_dispatched': True,
        'maximum_refines': 2, 'explicit_GRASP_only': True, 'automatic_retry': False,
        'scope': 'Actual ARX simulation, same normal H and resets, original physical success criterion'})
    write(root / 'state.json', {'status': 'registered_preview', 'physical_trials_started': 0, 'wall': time.time()})
    (root / 'research.md').write_text('# 第10轮模型三物体观察性试抓\n\n用户在知悉离线未达标后明确要求实际试抓并录视频。本轮仅牛奶盒、葡萄干盒、Cheez-It各一次正常入口。保持原模型、机器人、物理、初态、成功判据和纯六维执行逻辑。此结果是观察性试抓，不将失败的离线门槛改判通过。无自动重试。\n')
    write(root / 'sources.json', {'training': str(TRAINING), 'execution': str(ROOT / 'scripts/curobo_arx/nearfield/runtime_advisor40.py'),
        'authorization': str(root / 'user_authority.json'), 'original_offline_metrics': str(TRAINING / 'offline/new_D_test/metrics.json')})
    (root / 'research_log.md').write_text('# 运行记录\n\n三次正常入口已登记，尚未开始运动。旧训练与离线结果保留。\n')
    return check_authority(root)


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
    reg.update(stage='advisor40_user_preview_three', project_root=str(ROOT), source_snapshot=str(source),
        execution_source_files=execution, source_identity=digest(execution), seeds=spec['seeds'], seed_trials=spec['trials'],
        runtime_module='scripts.curobo_arx.nearfield.runtime_advisor40',
        policy_module='scripts.curobo_arx.nearfield.advisor40_policy',
        coordinator_module=MODULE,
        requested_devices=roles, created_unix=now, campaign_started_unix=now,
        deadline_unix=now + spec['physical_wall_seconds'], wall_hours=spec['physical_wall_seconds'] / 3600,
        global_physical_budget=3, max_physical_trials=3, envs_per_gpu=1, object_block_attempts=1,
        training_authorized_here=False, no_automatic_retry=True, automatic_attempt_retries=0,
        automatic_startup_retries=0, supplemental_interruptions={},
        continuation_of=str(PARENT), remaining_trial_ids=[t['trial_id'] for t in spec['trials']],
        pure_visual_policy={'G0_after_H': False, 'canonical_after_H': False,
            'model_decisions_maximum': 3, 'refines_maximum': 2, 'full_six_values_preserved': True,
            'explicit_GRASP_only': True, 'budget_in_model_prompt': False,
            'geometry_closure_controller': False, 'numeric_anomaly_bounds_m_deg': [.120, 45.],
            'original_IK_collision_joint_limits_nominal_closed_checks': True},
        execution_authority={'file': str(root / 'user_authority.json'), 'sha256': sha(root / 'user_authority.json')})
    training = read(TRAINING / 'registration.json'); checkpoint = TRAINING / 'D/checkpoint-2865'
    reg['learned_policy'].update(base_model=training['model'], gpu_uuid=roles['policy_uuid'],
        checkpoints={'D': str(checkpoint)}, max_policy_calls={'D': 3}, max_neural_calls={'D': 3},
        D_max_refines=2, no_teacher_goal_or_forced_GRASP=True, fixed_G0=False,
        warmups={'D': [json_lines(TRAINING / 'dataset/train.jsonl')[i] for i in (0, 1)]},
        training_bindings={'D': {'registration': str(TRAINING / 'registration.json'), 'sha256': sha(TRAINING / 'registration.json')}})
    dependencies = dict(old['dependencies'])
    dependencies.update(read(TRAINING / 'model_pins.json'))
    for seed in spec['seeds'].values():
        dependencies.update(seed['files'])
    for p in checkpoint.iterdir():
        if p.is_file():
            dependencies[str(p)] = sha(p)
    for p in (root / 'physical_spec.json', root / 'offline_performance_observed.json', root / 'preview_registration.json', root / 'user_authority.json', TRAINING / 'registration.json', TRAINING / 'D/training_receipt.json'):
        dependencies[str(p)] = sha(p)
    for item in reg['learned_policy']['warmups']['D']:
        dependencies.update({p: sha(p) for p in item['images']})
    reg['dependencies'] = dependencies
    reg['observational_preview'] = check_authority(root)
    reg['learned_policy']['training'] = str(TRAINING)
    reg['observational_preview']['original_conditional_nine_not_executed'] = True
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
    result = {'scheduled': len(reg['seed_trials']), 'completed': len(rows), 'successes': sum(bool(r['success']) for r in rows),
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
    authorization = check_authority(root)
    write(root / 'ONLINE_STARTED.json', {'wall': time.time(), 'pid': os.getpid(),
        'authorization_sha256': sha(root / 'user_authority.json'), 'authority': authorization})
    def terminated(signum, _):
        raise InterruptedError('Preview signal ' + str(signum))
    signal.signal(signal.SIGTERM, terminated)
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
            write(root / 'state.json', {'status': 'authorized_preview_waiting_for_resources',
                'inventory': inventory, 'roles': roles, 'stable_samples': stable, 'physical_trials_started': 0, 'wall': time.time()})
            if stable < 3:
                time.sleep(5)
        write(root / 'resource_admission.json', {'roles': roles, 'inventory': inventory, 'stable_samples': stable, 'wall': time.time()})
        out, reg = prepare_runtime(root, roles)
        write(out / 'predecessor_exit.json', exited(PARENT))
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
            ident = f'preview40_{index:02d}_{renderer.lower()}'
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
                write(root / 'state.json', {'status': 'running_observational_physical', 'completed': summary['completed'],
                    'successes': summary['successes'], 'scheduled': 3, 'wall': time.time()})
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
        summary = summarize(out, reg) if out else {'completed': 0, 'successes': 0, 'unexecuted_or_interrupted': 3}
        state = {'status': 'fault' if error or unresolved else 'complete', 'summary': summary,
                 'unconfirmed_world_or_process': unresolved, 'wall': time.time()}
        if out:
            write(out / 'state.json', state)
        write(root / 'state.json', state)
    if out:
        media(root)
    if error or unresolved:
        raise RuntimeError(error or 'Owned execution exit unconfirmed')


def media(root):
    from .report import video
    import imageio_ffmpeg
    root = Path(root).resolve(); out = root / 'physical'
    if not (out / 'registration.json').exists():
        return {'videos': 0}
    reg = read(out / 'registration.json'); summary = summarize(out, reg)
    videos = root / 'videos'; videos.mkdir(exist_ok=True)
    results = {t['trial_id']: t for t in summary['trials']}; details = []
    for trial in reg['seed_trials']:
        ident = trial['trial_id']; result = results.get(ident)
        episodes = sorted(out.glob('groups/*/trials/' + ident + '/**/samples.jsonl'))
        if len(episodes) > 1:
            raise RuntimeError('Unexpected repeated trial episode')
        record = {'trial_id': ident, 'object_id': trial['object_id'], 'result': result,
                  'status': 'completed' if result else 'interrupted_unknown' if episodes else 'unexecuted',
                  'motion_video': None, 'full_wrist_video': None}
        if episodes:
            ep = episodes[0].parent
            rows = [json.loads(s) for s in episodes[0].read_text().splitlines() if s.strip()]
            if rows:
                full = videos / (ident + '_full.mp4')
                if not full.with_suffix('.json').exists():
                    if full.exists():
                        raise RuntimeError('Unverified existing video; preserve it')
                    playlist = full.with_suffix('.ffconcat')
                    if any("'" in row['rgb_path'] for row in rows):
                        raise ValueError('Unsafe video input path')
                    playlist.write_text('ffconcat version 1.0\n' + ''.join(
                        "file '" + row['rgb_path'] + "'\nduration 0.2\n" for row in rows)
                        + "file '" + rows[-1]['rgb_path'] + "'\n")
                    ff = imageio_ffmpeg.get_ffmpeg_exe()
                    subprocess.run([ff, '-hide_banner', '-loglevel', 'error', '-nostdin', '-n',
                        '-f', 'concat', '-safe', '0', '-i', str(playlist), '-vf', 'fps=5', '-an',
                        '-c:v', 'libx264', '-threads', '2', '-pix_fmt', 'yuv420p', str(full)], check=True)
                    subprocess.run([ff, '-hide_banner', '-loglevel', 'error', '-nostdin',
                        '-i', str(full), '-f', 'null', '-'], capture_output=True, check=True)
                    write(full.with_suffix('.json'), {'path': str(full), 'sha256': sha(full),
                        'actual_frames': len(rows), 'decode_pass': True, 'source_episode': str(ep),
                        'scope': 'All actual unannotated near-field wrist frames; includes inference/planning waits; pre-H is outside capture'})
                record['full_wrist_video'] = read(full.with_suffix('.json'))
                motion = videos / (ident + '_motion.mp4')
                if motion.with_suffix('.json').exists():
                    record['motion_video'] = read(motion.with_suffix('.json'))
                elif (ep / 'requests.json').exists():
                    record['motion_video'] = video(ep, motion)
                record['episode'] = str(ep)
                if result and (ep / 'physical_receipt.json').exists():
                    from .advisor40_audit import physical
                    proof = physical(ep, persist=False)
                    if proof['pass'] != result['success'] or digest(proof) != digest(read(ep / 'physical_receipt.json')):
                        raise RuntimeError('Independent physical outcome mismatch')
                    record['independent_physical_verified'] = True
        details.append(record)
    write(root / 'video_results.json', {'summary': summary, 'details': details,
        'checkpoint': str(TRAINING / 'D/checkpoint-2865'), 'offline_gate_unchanged': True})
    names = {'household_hope_milk_carton': '牛奶盒', 'raisin_box': '葡萄干盒', 'cheez_it': 'Cheez-It'}
    lines = ['# 第10轮模型三物体观察性试抓', '',
        '用户明确授权在离线未达标后做实际ARX仿真试抓。每个物体一次正常入口，模型原样执行完整六维动作，只在显式GRASP时合爪。机器人、物理、碰撞和成功标准不变。', '',
        '| 物体 | 结果 | 动作视频 | 完整腕部录像 |', '|---|---|---|---|']
    for row in details:
        result = row['result']; outcome = ('成功' if result['success'] else '失败：' + result['outcome']) if result else row['status']
        link = lambda x: '[打开视频](' + x['path'] + ')' if x else '无已录制动作'
        lines.append('| ' + ' | '.join([names[row['object_id']], outcome, link(row['motion_video']), link(row['full_wrist_video'])]) + ' |')
    lines += ['', f"完整结果 {summary['completed']}/3；成功 {summary['successes']}；未执行 {len(summary['unexecuted'])}；中断或未知 {len(summary['interrupted_unknown'])}。",
        '', '录像来自本次真实仿真采集的原始腕部帧。动作视频剪去推理和规划等待；完整录像保留全部近场采集帧。进入H之前的远场运动不在腕部录像范围内。',
        '', '这三次观察不修改原训练的离线失败结论，也不计为原9次达标验证。']
    (root / 'PREVIEW_RESULTS.md').write_text('\n'.join(lines) + '\n')
    return {'completed': summary['completed'], 'videos': sum(bool(t['full_wrist_video']) for t in details)}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=('prepare', 'run', 'media')); p.add_argument('root')
    a = p.parse_args()
    print({'prepare': prepare, 'run': run, 'media': media}[a.action](Path(a.root)))
