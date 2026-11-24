"""Immutable, bounded seed-augmentation registrations and owned ARX execution."""
from arx_project.layout import canonical_path
import argparse
import collections
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time
import traceback
import numpy as np
from scripts.curobo_arx.data_factory.common import ROOT, read, write, sha, digest, verify_files, space
from scripts.curobo_arx.data_factory.ledger import Ledger
from .contract import source_seed, perturbed_entry, POLICY, ANCHOR_WAIT

CASES = ['household_hope_milk_carton', 'raisin_box', 'mustard_bottle', 'household_ycb_brick']


def prepare(parent, out, hours=4., predecessor=None):
    parent = Path(parent).resolve(); out = Path(out).resolve()
    if out.exists():
        raise FileExistsError('Never overwrite a seed-augmentation registration')
    old = read(parent/'registration.json')
    policy_source = ROOT/'runs/research_contract_20261019/storage_policy.json'
    missing_policy_origin = not policy_source.is_file()
    if missing_policy_origin:
        policy_source = parent/'storage_policy.json'
    if not policy_source.is_file():
        raise FileNotFoundError('Original or inherited registered storage policy is required')
    verify_files({**old['execution_source_files'], **old['dependencies']}); space(ROOT/'runs')
    previous = Path(predecessor).resolve() if predecessor else None
    previous_reg = read(previous/'registration.json') if previous else None
    spent = 0
    if previous:
        state = read(previous/'state.json')
        if state['status'] not in ('stopped', 'complete') or state.get('unconfirmed_world') or state.get('error'):
            raise ValueError('Previous bounded pilot must be terminal with confirmed idle')
        if read(previous/'finalization.json')['status'] != 'complete':
            raise ValueError('Finish predecessor accounting first')
        spent = previous_reg.get('prior_physical_attempts', 0)+read(previous/'summary.json')['physical_dispatches']
        for path in (previous/'group_registrations').glob('*.process.json'):
            if subprocess.run(['podman', 'container', 'exists', read(path)['container']], capture_output=True, timeout=30).returncode != 1:
                raise ValueError('Previous ARX World absence is not confirmed')
    db = sqlite3.connect('file:'+str(parent/'ledger.sqlite3')+'?mode=ro', uri=True); db.row_factory = sqlite3.Row
    seeds = copy.deepcopy(previous_reg['seeds']) if previous else {}; exclusions = []
    for name in ([] if previous else CASES):
        candidates = [dict(row) for row in db.execute(
            "SELECT * FROM attempts WHERE state='committed' AND object_id=? ORDER BY started", (name,))]
        for row in candidates:
            try:
                seed = source_seed(row)
            except ValueError as exc:
                exclusions.append({'episode_id': row['id'], 'reason': str(exc)}); continue
            seeds[seed['seed_id']] = seed; break
        else:
            raise ValueError('No admissible stationary source seed for '+name)
    db.close(); out.mkdir(parents=True)
    source = out/'source_snapshot/COOLAS-ARX'; source.parent.mkdir()
    prior_source = Path(old['source_snapshot'])
    shutil.copytree(prior_source, source, symlinks=True, ignore=shutil.ignore_patterns('__pycache__'))
    (source.parent/'curobo').symlink_to(ROOT.parent/'curobo', target_is_directory=True)
    package = 'scripts/curobo_arx/seed_augmentation'
    shutil.copytree(ROOT/package, source/package, ignore=shutil.ignore_patterns('__pycache__'))
    reg = copy.deepcopy(old); execution = {}
    for path, value in old['execution_source_files'].items():
        original = Path(path); target = source/original.relative_to(prior_source) if original.is_relative_to(prior_source) else original
        execution[str(target)] = value
    changes = {}
    for file in sorted((source/package).glob('*.py')):
        relative = file.relative_to(source); value = sha(file)
        execution[str(file)] = value; reg['source_files'][str(ROOT/relative)] = value; changes[str(relative)] = value
    reg['execution_source_files'] = execution
    for key in ['resume', 'lineage', 'bulk_storage', 'controlled_resource_pause']:
        reg.pop(key, None)
    for key, value in list(reg['reference_files'].items()):
        dest = out/'references'/(key+'.json'); dest.parent.mkdir(exist_ok=True); shutil.copy2(value, dest)
        reg['reference_files'][key] = str(dest); reg['dependencies'][str(dest)] = sha(dest)
    for seed in seeds.values():
        reg['dependencies'].update(seed['files'])
    trials = []
    offsets = [([.01, 0, 0], [0, 0, 5.]), ([0, -.02, 0], [5., 0, 0])]
    for seed_id, seed in sorted(seeds.items(), key=lambda pair: CASES.index(pair[1]['object_id'])):
        cases = [('control', i, [0., 0., 0.], [0., 0., 0.]) for i in range(2)]
        for index, (delta, omega) in enumerate(offsets):
            for policy in (['stitch', 'direct'] if index == 0 else ['direct', 'stitch']):
                cases.append((policy, index, delta, omega))
        for policy, repeat, delta, omega in cases:
            trials.append({'trial_id': f'{len(trials):02d}__{seed["object_id"]}__{policy}__{repeat}',
                'seed_id': seed_id, 'object_id': seed['object_id'], 'renderer': seed['renderer'],
                'policy': policy, 'repeat': repeat, 'placement': seed['placement'],
                'entry_world': perturbed_entry(seed, delta, omega),
                'perturbation': {'source_eef_translation_m': delta, 'source_eef_rotation_vector_degrees': omega}})
    if spent:
        while len(trials) > 24-spent:
            last = next(t for t in reversed(trials) if t['policy'] != 'control')
            trials = [t for t in trials if not (t['seed_id'] == last['seed_id'] and t['repeat'] == last['repeat'] and t['policy'] != 'control')]
    if len(trials) < 8:
        raise ValueError('Insufficient remaining bounded budget for seed controls')
    quotas = {**reg['quotas'], 'max_physical_attempts': 1, 'max_total_dispatches': 1,
              'physical_per_bin': 1, 'planning_rejections_per_bin': 1, 'goals_per_object': 1, 'successes_per_bin': 1}
    for trial in trials:
        scoped = copy.deepcopy(reg['catalog'])
        for asset in scoped['objects']:
            asset['collection_enabled'] = asset['object_id'] == trial['object_id']
            if asset['collection_enabled']:
                asset['collection_placements'] = [trial['placement']]
        path = out/'trial_ledgers'/(trial['trial_id']+'.sqlite3'); path.parent.mkdir(exist_ok=True)
        ledger = Ledger(path); ledger.initialize(scoped, {**quotas, 'first_object': trial['object_id']}); ledger.close()
    now = time.time()
    reg.update(created_unix=now, deadline_unix=now+hours*3600, wall_hours=hours,
        source_snapshot=str(source), source_identity=digest(reg['source_files']),
        stage='success_seed_augmentation_pilot', production=False, envs_per_gpu=1, object_block_attempts=1,
        candidate_backend='success_seed_replay', seeds=seeds, seed_trials=trials, join_policy=POLICY,
        max_physical_trials=len(trials), max_seed_model_replay_checks=8, no_automatic_retry=True, max_hypotheses=1,
        seed_control_activation='Both two registered control replays must be admitted before four perturbation comparisons for that seed',
        production_adoption=False, existing_production_remains_running=True,
        minimum_available_host_memory_bytes=40*1024**3,
        cumulative_parent_budget_not_reset=True, no_production_samples_credited_from_pilot=True,
        anchor_wait_policy=ANCHOR_WAIT, prior_physical_attempts=spent, combined_physical_cap=24)
    if previous:
        reg['deadline_unix'] = min(reg['deadline_unix'], previous_reg['deadline_unix'])
        reg['predecessor'] = str(previous)
        reg['dependencies'].update({str(previous/'registration.json'): sha(previous/'registration.json'),
            str(previous/'summary.json'): sha(previous/'summary.json'), str(previous/'STOP'): sha(previous/'STOP')})
        import ast
        previous_source = Path(previous_reg['source_snapshot'])
        teacher_relative = package+'/teacher.py'; contract_relative = package+'/contract.py'
        if sha(source/teacher_relative) != sha(previous_source/teacher_relative):
            raise ValueError('Model-check reuse requires identical teacher code')
        def helper_nodes(path):
            chosen = {}
            for node in ast.parse(path.read_text()).body:
                if isinstance(node, ast.FunctionDef) and node.name in ('joint_values', 'replay_command'):
                    chosen[node.name] = ast.dump(node, include_attributes=False)
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'POLICY' for t in node.targets):
                    chosen['POLICY'] = ast.dump(node.value, include_attributes=False)
            return chosen
        if helper_nodes(source/contract_relative) != helper_nodes(previous_source/contract_relative):
            raise ValueError('Source-replay model helpers changed')
        if digest(seeds) != digest(previous_reg['seeds']):
            raise ValueError('Model-check source seed data changed')
        for key in ('model_reference', 'planning_mesh_capacity', 'visual_geometry_policy', 'render_configs'):
            if reg[key] != previous_reg[key]:raise ValueError('Model-check configuration changed')
        prior_model = read(previous/'model_screen.json')
        if not prior_model.get('pass') or len(prior_model['rows']) != 8:
            raise ValueError('All eight prior source-model checks must pass')
        reg['reference_model_screen'] = {'path': str(previous/'model_screen.json'),
            'sha256': sha(previous/'model_screen.json'), 'teacher_sha256': sha(source/teacher_relative),
            'same_teacher_helpers_source_seeds_and_model_configuration': True,
            'runtime_change': 'Hold the accepted anchor target for 1-2 simulation seconds before the unchanged point-state gate; no source command/model/success changes'}
        reg['dependencies'][str(previous/'model_screen.json')] = sha(previous/'model_screen.json')
    reg['storage_policy_provenance'] = {'path': str(policy_source), 'sha256': sha(policy_source),
        'original_contract_path_missing': missing_policy_origin, 'same_inherited_user_storage_policy': True}
    reg['dependencies'][str(policy_source)] = sha(policy_source)
    reg['preparation_complete'] = True
    reg['source_parent'] = str(parent)
    write(out/'registration.json', reg); write(out/'catalog.json', reg['catalog'])
    write(out/'seed_selection.json', {'seeds': seeds, 'excluded_stationarity_or_identity': exclusions})
    write(out/'source_change_receipt.json', {'new_package_only': changes, 'original_parent_execution_files_unchanged': True})
    shutil.copy2(policy_source, out/'storage_policy.json')
    write(out/'sources.json', {'primary': [{'url': 'https://mimicgen.github.io/docs/modules/datagen.html',
        'use': 'Connecting and reusing successful subtask trajectories; ARX implementation inspired, not exact MimicGen reproduction'}],
        'source_data': str(parent), 'new_neural_model_or_weights': False})
    (out/'research.md').write_text('''# Success-seed connection pilot

Goal: measure whether new H entries connected to the actual original stationary S can produce more accepted local demonstrations than direct H-to-G replanning. Keep the original G, source approach/close/lift/hold command payloads and source object placement. Re-execute every new trial and record fresh wrist RGB-D; never concatenate old observations or carry over success.

One hypothesis,24 slots,4h,no automatic retries,pause_every=never. Four previously successful ARX seeds; two control replays each, then matched1cm/5deg and2cm/5deg entries with stitch/direct order reversed. Activate variants for a seed only after both controls pass admission. Failed control leaves subsequent slots NOT_ACTIVATED, not failed grasps. No trained VLA or global production promotion; current collection/watchdog keep running. This pilot tests new entry recovery on successful objects, not broad recovery of historical no-G placements.

Freeze all robot/scene/camera configuration, historical seed initial placement/rendering and original task/record/image criteria. Check source hashes, current frozen runtime properties, original IK/PRM/trajopt or original optimized source commands with current dense collision/limit/derivative/camera validation. Connector uses a joint goal matching original S, avoids the actual target, and must pass original convergence plus registered joint/finger/velocity/object eligibility before tail dispatch. All extra commands are explicitly recorded and audited. Stop on resource/config/record/idle uncertainty. Local storage first,1GiB floor; owned new worker and World only, explicit GPU UUIDs; no reset/commit/push.

Metrics: controls and stitched/direct admitted per registered activated attempt, actual dispatches, unexecuted/not-activated/fault separate, elapsed seconds per admitted trajectory, distinct entry versus fixed object-placement coverage. All descendants stay in source object's dataset split. Require repeatable controls and no observed loss before any wider validation; this small selected pilot cannot establish a population success rate.
''')
    (out/'research_log.md').write_text('Registered before physical execution; parent collection/STOPs/results remain untouched.\n')
    (out/'autoresearch-results.tsv').write_text('iteration\thypothesis\tcontrols\tstitch\tdirect\tphysical_trials\tstatus\n')
    write(out/'state.json', {'status': 'prepared', 'scheduled_slots': len(trials), 'physical_trials': 0, 'prior_physical_attempts': spent})
    if previous:
        write(out/'corrected_continuation.json', {'predecessor': str(previous), 'old_STOP_retained': True,
            'prior_physical_attempts': spent, 'new_slots': len(trials), 'combined_cap': 24,
            'deadline_not_extended': reg['deadline_unix'], 'reference_model_screen': reg['reference_model_screen'],
            'original_join_and_task_thresholds_unchanged': True})
    verify_files({**execution, **reg['dependencies']})
    return out


def available_memory():
    values = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(values['MemAvailable'].strip().split()[0])*1024


def run(campaign, compute_uuid, sim_uuid):
    from scripts.curobo_arx.data_factory.launch import devices, allocate, environment, seed_kit_cache
    from scripts.curobo_arx.data_factory.watchdog import process_info,exclusive_stop
    campaign = Path(campaign).resolve(); reg = read(campaign/'registration.json')
    lock=(campaign/'coordinator.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if reg.get('preparation_complete') is not True or not (campaign/'storage_policy.json').is_file():
        raise ValueError('Seed campaign preparation is incomplete; no services may start')
    if (campaign/'launch.json').exists():
        raise FileExistsError('Never relaunch an existing seed experiment')
    verify_files({**reg['execution_source_files'], **reg['dependencies']}); space(campaign)
    inventory = devices(); compute, simulations = allocate(inventory, 1, compute_uuid, [sim_uuid])
    memory = available_memory()
    if memory < reg['minimum_available_host_memory_bytes']:
        write(campaign/'resource_preflight.json', {'pass': False, 'available_host_bytes': memory, 'inventory': inventory})
        raise RuntimeError('Wait for at least the registered 40GiB available host memory; do not stop production')
    source = Path(reg['source_snapshot']); worker_dir = campaign/'workers/planner'; worker_dir.mkdir(parents=True)
    command = [str(ROOT.parent/'curobo/.venv/bin/python'), '-m', 'scripts.curobo_arx.seed_augmentation.worker',
               '--directory', str(worker_dir), '--registration', str(campaign/'registration.json')]
    worker = None; active = None; container = None; error = None; model_rejected = False
    write(campaign/'launch.json', {'wall': time.time(), 'registration_sha256': sha(campaign/'registration.json'),
        'compute': compute, 'simulator': simulations[0], 'available_host_bytes': memory,
        'deadline_unix': reg['deadline_unix'], 'existing_production_touched': False})
    try:
        with (worker_dir/'process.log').open('x') as log:
            worker = subprocess.Popen(command, cwd=source, env=environment(compute), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(worker_dir/'process.json', {'pid': worker.pid, 'gpu_uuid': compute_uuid, 'command': command,
              'owned': True, **(process_info(worker.pid) or {})})
        until = time.monotonic()+float(reg.get('worker_startup_timeout_seconds',180))
        while not (worker_dir/'ready.json').exists():
            if worker.poll() is not None or time.monotonic() > until:
                raise RuntimeError('Isolated seed planner failed to start')
            time.sleep(.25)
        from scripts.curobo_arx.data_factory.workers import WorkerClient
        client = WorkerClient(worker_dir); reference = reg.get('reference_model_screen')
        if reference and sha(reference['path']) != reference['sha256']:
            raise RuntimeError('Sealed reference model checks changed')
        screen = read(reference['path'])['rows'] if reference else []
        for seed_id, seed in ({} if reference else reg['seeds']).items():
            entry = read(Path(seed['episode'])/'actual_entry.json')
            renderer = reg['render_configs'][seed['renderer']]
            context = {'asset': next({key: asset[key] for key in ('usd_path', 'asset_sha256')}
                        for asset in reg['catalog']['objects'] if asset['object_id'] == seed['object_id']),
                'eef_from_camera': (np.linalg.inv(entry['snapshot']['link_world']['right_eef_link'])@
                                    np.asarray(entry['camera_world_optical'])).tolist(),
                'intrinsics': entry['intrinsics'], 'shape': [renderer['camera']['height'], renderer['camera']['width']],
                'clipping_range_m': entry['camera_clipping_range_m']}
            for phase in ('approach', 'lift'):
                if len(screen) >= reg['max_seed_model_replay_checks']:
                    raise RuntimeError('Seed model activation budget exhausted')
                folder = campaign/'model_screen'/seed_id/phase; folder.mkdir(parents=True)
                snapshot = entry['snapshot'] if phase == 'approach' else read(Path(seed['episode'])/'close_execution.json')['final']
                goal = copy.deepcopy(seed['goal']['grasp_world'])
                if phase == 'lift':goal[2][3] += .12
                payload = {'seed_id': seed_id, 'phase': phase, 'label': phase,
                    'snapshot': snapshot, 'scene': str(Path(seed['episode'])/(phase+'_scene.json')),
                    'goal_world': goal, 'grasp_world': seed['goal']['grasp_world'],
                    'visual_context': context, 'output': str(folder)}
                write(folder/'model_request.json', {'action': 'seed_replay', **payload, 'physical_commands': 0})
                job = client.submit('seed_replay', **payload); until = time.monotonic()+900
                while True:
                    response = client.poll(job)
                    if response is not None:break
                    if worker.poll() is not None or time.monotonic() > until:
                        raise RuntimeError('Seed model activation worker failed or exceeded time budget')
                    time.sleep(.1)
                write(folder/'model_response.json', response)
                if not response['ok']:raise RuntimeError(response['error'])
                screen.append({'seed_id': seed_id, 'object_id': seed['object_id'], 'phase': phase,
                    'pass': not bool(response['result'].get('rejected')), 'rejected': response['result'].get('rejected'),
                    'service_seconds': response['seconds'], 'physical_commands': 0})
                write(campaign/'model_screen.json', {'status': 'running', 'rows': screen, 'physical_commands': 0})
                print('SEED_MODEL_CHECK', json.dumps(screen[-1]), flush=True)
        model_rejected = not all(row['pass'] for row in screen)
        write(campaign/'model_screen.json', {'status': 'complete', 'pass': not model_rejected,
              'rows': screen, 'physical_commands': 0, 'new_model_queries': 0 if reference else len(screen), 'reference': reference})
        if model_rejected:return
        grouped = collections.defaultdict(list)
        for trial in reg['seed_trials']:
            grouped[trial['renderer']].append(trial['trial_id'])
        for renderer, trial_ids in grouped.items():
            if (campaign/'STOP').exists() or time.time() >= reg['deadline_unix']:
                break
            verify_files({**reg['execution_source_files'], **reg['dependencies']}); space(campaign)
            if available_memory() < reg['minimum_available_host_memory_bytes']:
                raise RuntimeError('Host memory below registered seed-World startup floor')
            free = next(d['free_mb'] for d in devices() if d['uuid'] == sim_uuid)
            if free < 7500:
                raise RuntimeError('Isolated simulator no longer has its startup memory floor')
            ident = 'seed_'+renderer.lower(); container = 'arx-seed-augmentation-'+campaign.name+'-'+renderer.lower()
            if subprocess.run(['podman', 'container', 'exists', container], capture_output=True, timeout=30).returncode != 1:
                raise RuntimeError('Seed World name already exists or absence cannot be confirmed')
            launch = {'group_id': ident, 'env_count': 1, 'gpu_uuid': sim_uuid, 'renderer': renderer,
                      'registration_sha256': sha(campaign/'registration.json'),
                      'trial_ids': trial_ids, 'planner_directory': str(worker_dir),
                      'candidate_directory': str(campaign/'no_neural_requests'), 'stop_unix': reg['deadline_unix']}
            path = campaign/'group_registrations'/(ident+'.json'); write(path, launch)
            cache = ROOT/'runs/data_factory_runtime_cache'; kit = seed_kit_cache(cache/sim_uuid)
            from scripts.curobo_arx.data_factory.storage import container_storage_mounts
            cmd = ['podman', 'run', '--rm', '--name', container, '--user', '0', '--device', 'nvidia.com/gpu='+sim_uuid,
                   '--security-opt=label=disable', '--network=host', '--shm-size=2g',
                   '-e', 'ACCEPT_EULA=Y', '-e', 'PRIVACY_CONSENT=N', '-e', 'OMNI_KIT_ALLOW_ROOT=1',
                   '-e', 'CUDA_VISIBLE_DEVICES='+sim_uuid, '-e', 'CUDA_DEVICE_ORDER=PCI_BUS_ID', '-e', 'NVIDIA_VISIBLE_DEVICES=void',
                   '-e', 'PYTHONUNBUFFERED=1', '-e', 'PYTHONDONTWRITEBYTECODE=1',
                   '-v', str(cache/'ov')+':/root/.cache/ov:rw', '-v', str(cache/'nvidia')+':/root/.cache/nvidia:rw',
                   '-v', str(kit)+':/isaac-sim/kit/cache:rw', '-v', str(ROOT.parent)+':'+str(ROOT.parent)+':rw',
                   '-v', '/coolas/data/arx_teacher_zhaoyechong:/coolas/data/arx_teacher_zhaoyechong:ro',
                   *container_storage_mounts(campaign,reg),
                   '-v', str(canonical_path(ROOT/'asset'))+':/workspace/asset:ro', '-v', str(ROOT)+':/workspace/COOLAS-ARX:ro',
                   '-w', str(source), '--entrypoint', '/isaac-sim/python.sh', 'localhost/arx-s02-isaac:6.0.1',
                   '-m', 'scripts.curobo_arx.seed_augmentation.runtime', '--campaign', str(campaign), '--launch', str(path)]
            with path.with_suffix('.log').open('x') as log:
                active = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write(path.with_suffix('.process.json'), {'pid': active.pid, 'container': container, 'command': cmd,
                  'owned':True,**(process_info(active.pid) or {})})
            while active.poll() is None:
                write(campaign/'state.json', {'status': 'running', 'renderer': renderer, 'wall': time.time()})
                if worker.poll() is not None:
                    exclusive_stop(campaign/'STOP', {'wall': time.time(), 'reason': 'Owned seed planner exited'})
                if (campaign/'STOP').exists() or time.time() >= reg['deadline_unix']:
                    subprocess.run(['podman', 'kill', '--signal', 'TERM', container], capture_output=True, timeout=30)
                time.sleep(2)
            if active.returncode != 0:
                raise RuntimeError('Seed ARX World exited with a failure')
            shutdown = read(campaign/'groups'/ident/'shutdown_receipt.json')
            if not shutdown['all_trajectories_idle'] or subprocess.run(['podman', 'container', 'exists', container], capture_output=True, timeout=30).returncode != 1:
                raise RuntimeError('Seed robot idle or World removal not confirmed')
            active = None
        verify_files({**reg['execution_source_files'], **reg['dependencies']})
    except BaseException:
        error = traceback.format_exc(); write(campaign/'FAULT.json', {'wall': time.time(), 'error': error})
        raise
    finally:
        unresolved = False
        if active is not None and active.poll() is None:
            subprocess.run(['podman', 'kill', '--signal', 'TERM', container], capture_output=True, timeout=30)
            try:active.wait(timeout=reg['attempt_seconds']+60)
            except subprocess.TimeoutExpired:unresolved = True
        if not unresolved and worker is not None:
            write(worker_dir/'STOP', {'wall': time.time(), 'reason': 'Bounded seed pilot ended'})
            try:worker.wait(timeout=30)
            except subprocess.TimeoutExpired:worker.terminate()
        if unresolved:
            write(campaign/'cleanup_unknown.json', {'container': container, 'automatic_reuse': False})
        summary = summarize(campaign)
        write(campaign/'state.json', {'status': 'fault' if error or unresolved else ('complete' if summary['accounted_slots'] == len(reg['seed_trials']) else 'stopped'),
            'error': error, 'unconfirmed_world': unresolved, 'model_activation_rejected': model_rejected,
            'finished_unix': time.time(), **summary})


def summarize(campaign):
    campaign = Path(campaign); reg = read(campaign/'registration.json'); completed = {}
    for path in (campaign/'groups').glob('*/seed_results.json'):
        for row in read(path)['trials']:
            if row['trial_id'] in completed:
                raise RuntimeError('Duplicate seed trial result')
            completed[row['trial_id']] = row
    unknown = []; unexecuted = []; not_activated = []; by_policy = {}
    for trial in reg['seed_trials']:
        ident = trial['trial_id']; row = completed.get(ident)
        if row is None:
            began = any((campaign/'groups').glob('*/trials/'+ident))
            (unknown if began else unexecuted).append(ident)
        elif not row['completed']:
            not_activated.append(ident)
    for policy in ('control', 'stitch', 'direct'):
        rows = [r for r in completed.values() if r['policy'] == policy and r['completed']]
        by_policy[policy] = {'completed': len(rows), 'success': sum(r['success'] for r in rows),
                            'admitted': sum(r['admitted'] for r in rows),
                            'physical_dispatches': sum(r['physical_dispatches'] for r in rows),
                            'failure_reasons': dict(collections.Counter(r['outcome'] for r in rows if not r['success']))}
    result = {'scheduled_slots': len(reg['seed_trials']), 'accounted_slots': len(completed),
              'physical_dispatches': sum(r.get('physical_dispatches', 0) for r in completed.values()),
              'policies': by_policy, 'not_activated': not_activated, 'unexecuted': unexecuted, 'unknown': unknown,
              'production_promoted': False}
    write(campaign/'summary.json', result)
    lines = ['# 成功种子入口扩充验证', '',
             '| 方法 | 完成试验 | 原物理成功 | 合格示教 | 实际运动 |', '|---|---:|---:|---:|---:|']
    for policy, row in by_policy.items():
        lines.append(f'| {policy} | {row["completed"]} | {row["success"]} | {row["admitted"]} | {row["physical_dispatches"]} |')
    lines += ['', f'未激活 {len(not_activated)}；未执行 {len(unexecuted)}；故障/未知 {len(unknown)}。',
              '', 'control先验证原成功后段可重复；stitch从扰动H连接到原S后复用原后段；direct从相同H直接重规划到相同G。',
              '每条新增轨迹必须实际执行并重新记录右腕图像，通过原收敛/接触/连续抬升/图像和记录门槛。额外连接和原后段指令单列审查，不复用旧图像或旧成功结论。',
              '本轮物体摆放保持各种子原条件，只增加入口覆盖；不证明新物体/新摆放泛化，不替换原生产算法。']
    (campaign/'ACCEPTANCE.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--parent', required=True); p.add_argument('--out', required=True); p.add_argument('--hours', type=float, default=4.); p.add_argument('--predecessor')
    p = sub.add_parser('run'); p.add_argument('campaign'); p.add_argument('--compute-uuid', required=True); p.add_argument('--sim-uuid', required=True)
    p = sub.add_parser('summarize'); p.add_argument('campaign')
    args = parser.parse_args()
    if args.action == 'prepare':print(prepare(args.parent, args.out, args.hours, args.predecessor))
    elif args.action == 'run':run(args.campaign, args.compute_uuid, args.sim_uuid)
    else:print(json.dumps(summarize(args.campaign)))
