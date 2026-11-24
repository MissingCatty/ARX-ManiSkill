"""One immutable26-slot advisor31 campaign; no retries or parent STOP removal."""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from scripts.curobo_arx.data_factory.common import ROOT, read, write, sha, digest, verify_files, space
from scripts.curobo_arx.data_factory.launch import devices
from scripts.curobo_arx.data_factory.watchdog import process_info, exclusive_stop
from training.nearfield_arx.lateral_campaign import ledgers
from .advisor31_spec import make_trials, validate_trials
from .advisor31_rules import VERSION, POLICY
from .closedloop30_policy import validate_checkpoints
from .visual_grounding_spec import OBJECTS, expected_reset

PARENT = ROOT/'runs/nearfield_g0_attribution_20261110_01'
DONOR = ROOT/'runs/nearfield_closedloop30_20261110_01'
OLD24 = ROOT/'runs/nearfield_visual_grounding_physical_20261110_01'
OFFLINE = ROOT/'runs/nearfield_visual_grounding_20261110_01'
PLAN = ROOT/'research_plan/docs/nearfield_advisor31_20261110.md'


def exited(root):
    """Prove old processes/Worlds absent, without inventing an idle receipt."""
    root = Path(root); rows = []
    for path in [*root.glob('workers/*/process.json'), *root.glob('group_registrations/*.process.json')]:
        record = read(path); live = process_info(record['pid'])
        if live and (not record.get('start_ticks') or live.get('start_ticks') == record['start_ticks']):
            raise RuntimeError('Registered predecessor process remains alive: '+str(path))
        absent = None
        if record.get('container'):
            result = subprocess.run(['podman', 'container', 'exists', record['container']], capture_output=True, timeout=30)
            if result.returncode != 1: raise RuntimeError('Prior World absence unconfirmed')
            absent = True
        rows.append({'file': str(path), 'sha256': sha(path), 'old_process_absent': True, 'container_absent': absent})
    return {'root': str(root), 'processes': rows,
            'old_shutdown_receipts': {str(p): read(p) for p in root.glob('groups/*/shutdown_receipt.json')},
            'scope': 'Process/World absence only; interrupted historical action remains unknown'}


def cancellation_gate(root):
    root = Path(root); cancel = read(root/'cancellation_receipt.json')
    project = Path(cancel['authority']).parents[3]
    parent, old24, offline = (project/'runs'/p.name for p in (PARENT, OLD24, OFFLINE))
    verify_files(cancel['evidence'])
    if cancel != read(parent/'CANCEL_REMAINING_ADVISOR31.json'):
        raise ValueError('Cancellation identity changed')
    if len(cancel['cancelled_unexecuted_slots']) != 48 or cancel['completed_retained'] != 23:
        raise ValueError('Old72 accounting changed')
    if not (parent/'STOP').exists() or not (old24/'STOP').exists(): raise ValueError('Historical STOP removed')
    if not read(offline/'verification.json')['record_integrity_pass']:
        raise ValueError('Offline visual diagnosis integrity failed')
    return {'parents': [exited(parent), exited(old24)], 'cancelled': 48,
            'historical_completed': 23, 'historical_unknown': 1, 'all_old_STOPs_retained': True}


def runtime_dependencies(old, seeds):
    """Retain runtime inputs; keep the full old manifest as provenance only."""
    chosen = {p: h for p, h in old['dependencies'].items() if not Path(p).is_relative_to(ROOT/'runs')}
    for seed in seeds.values(): chosen.update(seed['files'])
    def paths(value):
        if isinstance(value, str) and value.startswith('/') and len(value) < 4096:
            p = Path(value)
            if p.is_file():
                chosen[str(p)] = old['dependencies'].get(str(p)) or sha(p)
        elif isinstance(value, dict):
            for k, v in value.items():
                if k not in {'dependencies', 'source_files', 'execution_source_files', 'seeds'}: paths(v)
        elif isinstance(value, list):
            for v in value: paths(v)
    paths(old)
    for folder in [old['learned_policy']['base_model'], *old['learned_policy']['checkpoints'].values()]:
        for p in Path(folder).rglob('*'):
            if p.is_file() and not any(n.startswith('.') for n in p.relative_to(folder).parts):
                chosen[str(p)] = old['dependencies'].get(str(p)) or sha(p)
    return chosen


def prepare(out, policy_index, compute_index, sim_index):
    out = Path(out).resolve()
    if (out/'registration.json').exists() or (out/'source_snapshot').exists():
        raise FileExistsError('Preserve existing source snapshot and registration')
    if not (out/'authorization.json').is_file(): raise ValueError('Missing advisor31 execution authority')
    space(out); predecessors = cancellation_gate(out)
    old = read(PARENT/'registration.json'); donor = read(DONOR/'registration.json')
    trials, seeds = make_trials(donor)
    command = [sys.executable, '-m', 'unittest', 'scripts.curobo_arx.nearfield.test_advisor31',
               'scripts.curobo_arx.nearfield.test_visual_grounding', 'scripts.curobo_arx.nearfield.test_closedloop30',
               'training.nearfield_arx.test_closure_records', '-v']
    check = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=240)
    (out/'offline_checks.log').write_text(check.stdout+check.stderr)
    write(out/'offline_checks.json', {'pass': check.returncode == 0, 'command': command,
                                    'physical_trials': 0, 'log_sha256': sha(out/'offline_checks.log')})
    if check.returncode: raise RuntimeError('Offline checks failed')
    prior = Path(old['source_snapshot']); source = out/'source_snapshot/COOLAS-ARX'
    shutil.copytree(prior, source, symlinks=True, ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('tmp', 'curobo'): (source.parent/name).symlink_to(ROOT.parent/name, target_is_directory=True)
    execution = {str(source/Path(p).relative_to(prior)) if Path(p).is_relative_to(prior) else p: h
                 for p, h in old['execution_source_files'].items()}
    added = sorted((ROOT/'scripts/curobo_arx/nearfield').glob('advisor31_*.py'))
    added += [ROOT/'scripts/curobo_arx/nearfield'/n for n in ('runtime_advisor31.py', 'test_advisor31.py',
               'visual_grounding_spec.py', 'visual_grounding_report.py')]
    for p in added:
        dest = source/p.relative_to(ROOT); shutil.copy2(p, dest); execution[str(dest)] = sha(dest)
    reg = copy.deepcopy(old); donors = {}
    deps = runtime_dependencies(old, seeds)
    for obj in OBJECTS:
        t = next(t for t in donor['seed_trials'] if t['method'] == 'D' and t['condition'] == 'normal' and t['object_id'] == obj)
        result = read(next(DONOR.glob('groups/*/trials/'+t['trial_id']+'/trial_result.json')))
        ep = Path(result['attempts'][0]['episode']); request = Path(read(ep/'policy_requests.json')[0]['request'])
        req = read(request); prompt = request.with_name(req['request_id']+'_input.json'); images = read(prompt)['images']
        files = {str(p): sha(p) for p in [request, prompt, Path(req['observation']), *map(Path, images)]}
        donors[obj] = {'source_trial_id': t['trial_id'], 'source_episode': str(ep), 'request': str(request),
                       'images': images, 'files': files, 'G0_world': t['G0_world'], 'entry_world': t['entry_world']}
        deps.update(files)
    inventory = devices(); by_index = {d['index']: d for d in inventory}
    if len({policy_index, compute_index, sim_index}) != 3: raise ValueError('Use three distinct devices')
    roles = {role+'_uuid': by_index[i]['uuid'] for role, i in
             (('policy', policy_index), ('compute', compute_index), ('sim', sim_index))}
    for key in ('carried_results', 'prior_results', 'prerequisite_results'): reg[key] = []
    for key in ('wait_for_existing_campaign',): reg.pop(key, None)
    reg.update(stage=VERSION, advisor31_policy=POLICY, source_snapshot=str(source), execution_source_files=execution,
        source_identity=digest(execution), seeds=seeds, seed_trials=trials, visual_donors=donors,
        created_unix=time.time(), campaign_started_unix=time.time(), deadline_unix=time.time()+36*3600, wall_hours=36,
        max_physical_trials=26, global_physical_budget=26, max_replay_trials=0, replay_slots_total=0, recovery_slots_total=0,
        runtime_module='scripts.curobo_arx.nearfield.runtime_advisor31',
        policy_module='scripts.curobo_arx.nearfield.advisor31_policy',
        coordinator_module='scripts.curobo_arx.nearfield.advisor31_campaign',
        requested_devices=roles, policy_memory_floor_mib=9500, training_authorized_here=False,
        no_automatic_retry=True, automatic_attempt_retries=0, automatic_startup_retries=0,
        policy_GPU_residency='One immutable C or D model on CUDA at a time, both stored in host RAM',
        policy_cuda_memory_fraction=.35, execution_authority=read(out/'authorization.json'),
        control_pose_matching={'translation_m': .005, 'rotation_degrees': 2., 'finger_joint_m': .0005,
                               'apply_equally_to_all_arms': True})
    reg['learned_policy']['gpu_uuid'] = roles['policy_uuid']
    validate_checkpoints(reg); validate_trials(trials)
    shutil.copy2(PLAN, out/'PLAN.md'); shutil.copy2(PLAN, out/'research.md')
    shutil.copy2(PARENT/'storage_policy.json', out/'storage_policy.json')
    write(out/'case_manifest.json', {'slots': trials, 'physical_budget': 26, 'regressions': 2, 'target_shift': 24,
                                    'no_performance_selection': True, 'automatic_retries': 0})
    write(out/'expected_resets.json', {t['case_id']: expected_reset(seeds[t['seed_id']], t) for t in trials})
    write(out/'predecessor_exit_receipt.json', predecessors)
    write(out/'runtime_dependency_selection.json', {'old_manifest': str(PARENT/'registration.json'),
        'old_manifest_sha256': sha(PARENT/'registration.json'), 'old_entries': len(old['dependencies']),
        'selected_entries_before_new_evidence': len(deps), 'selected': deps,
        'excluded': {p: h for p, h in old['dependencies'].items() if p not in deps},
        'scope': 'Unused historical training/experiment payload remains untouched; actual models, seed records, assets, source and configs fully checked before launch'})
    write(out/'source_change_receipt.json', {'added': [str(p.relative_to(ROOT)) for p in added],
        'new_rules': POLICY, 'robot_and_scene_physics_unchanged': True, 'historical_evaluator_bytes_unchanged': True,
        'new_evaluator': 'Preserve strict and advisor30 scores; only registered contact-admitted close tracking waiver',
        'registered_new_target_X_offsets_m': [-.015, .015], 'historical_results_and_STOPs_preserved': True,
        'periodic_success_status_dedup_only': True, 'capture_safety_checks_skipped': 0})
    write(out/'resource_plan.json', {'inventory': inventory, 'roles': roles,
        'free_memory_mib': {'policy': 9500, 'compute': 9000, 'sim': 9000}, 'stable_samples': 3,
        'other_users_jobs_untouched': True})
    write(out/'sources.json', {'primary_sources': ['https://arxiv.org/abs/2004.07780', 'https://arxiv.org/abs/2608.06154'],
        'methodology_reference_only': True, 'paper_reproduction': False, 'advisor': reg['execution_authority'],
        'frozen_checkpoints': reg['learned_policy']['checkpoints'], 'no_training': True})
    for p in (PARENT/'registration.json', PARENT/'STOP', OLD24/'registration.json', OLD24/'STOP',
              DONOR/'registration.json', DONOR/'closedloop30_verification.json', OFFLINE/'verification.json'):
        deps[str(p)] = sha(p)
    command = ['podman', 'run', '--pull=never', '--rm', '-i', '--user', '0', '--security-opt=label=disable',
        '-v', str(ROOT.parent)+':'+str(ROOT.parent)+':rw',
        '-v', '/coolas/data/arx_teacher_zhaoyechong:/coolas/data/arx_teacher_zhaoyechong:ro',
        '-w', str(source), '--entrypoint', '/isaac-sim/python.sh', 'localhost/arx-s02-isaac:6.0.1', '-c',
        'import json,sys,pathlib; import scripts.curobo_arx.nearfield.runtime_advisor31; '
        'paths=json.load(sys.stdin); missing=[p for p in paths if not pathlib.Path(p).is_file()]; '
        'print(json.dumps({"missing":missing,"checked":len(paths)})); sys.exit(bool(missing))']
    probe = subprocess.run(command, input=json.dumps(list({*deps, *execution})), capture_output=True, text=True, timeout=120)
    write(out/'container_preflight.json', {'pass': probe.returncode == 0, 'command': command,
        'stdout': probe.stdout, 'stderr': probe.stderr, 'GPU_or_World_started': False, 'no_image_download': True})
    if probe.returncode: raise RuntimeError('Container source/path preflight failed')
    for p in out.iterdir():
        if p.is_file() and p.name not in {'state.json', 'research_log.md'}: deps[str(p)] = sha(p)
    reg['dependencies'] = deps; ledgers(out, reg); write(out/'catalog.json', reg['catalog'])
    write(out/'registration.json', reg)
    started = time.time(); verify_files({**execution, **deps})
    write(out/'full_preflight.json', {'pass': True, 'checked': len({**execution, **deps}),
        'registered_manifest_entries': len(execution)+len(deps),
        'seconds': time.time()-started, 'physical_trials': 0, 'registration_sha256': sha(out/'registration.json')})
    (out/'research_log.md').write_text('26 new slots registered; old48 cancelled; original unknown retained. No physical attempt yet.\n')
    write(out/'state.json', {'status': 'prepared', 'scheduled_slots': 26, 'physical_trials': 0})
    return out


def run(campaign):
    from . import closure_physical, learned_report
    from .advisor31_report import report
    root = Path(campaign).resolve(); reg = read(root/'registration.json')
    lock = (root/'run.lock').open('a+'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'STOP').exists() or (root/'STARTED.json').exists(): raise RuntimeError('No retry or STOP bypass')
    write(root/'STARTED.json', {'wall': time.time(), 'registration_sha256': sha(root/'registration.json')})
    try:
        verify_files({**reg['execution_source_files'], **reg['dependencies']})
        write(root/'fresh_predecessor_exit.json', cancellation_gate(root))
        plan = read(root/'resource_plan.json'); stable = 0
        while stable < plan['stable_samples']:
            if (root/'STOP').exists() or time.time() >= reg['deadline_unix']: raise RuntimeError('Stopped or resource deadline')
            inventory = {d['uuid']: d['free_mb'] for d in devices()}
            stable = stable+1 if all(inventory.get(reg['requested_devices'][k+'_uuid'], 0) >= v
                                   for k, v in plan['free_memory_mib'].items()) else 0
            write(root/'state.json', {'status': 'waiting_for_resources', 'stable_samples': stable,
                                    'free_memory_mib': inventory, 'physical_trials': 0, 'wall': time.time()})
            if stable < plan['stable_samples']: time.sleep(10)
        validate_trials(reg['seed_trials']); learned_report.report = report
        result = closure_physical.run(root)
        write(root/'delivery_receipt.json', {'complete': result['complete'], 'completed': result['completed'],
                                            'sha256': sha(root/'advisor31_verification.json')})
        with (root/'research_log.md').open('a') as f:
            f.write(f'Completed {result["completed"]}/26; all outcomes retained; no automatic promotion.\n')
    except BaseException:
        error = traceback.format_exc(); exclusive_stop(root/'STOP', {'reason': 'advisor31_coordinator_fault', 'error': error})
        write(root/'coordinator_fault.json', {'error': error})
        write(root/'state.json', {'status': 'fault', 'wall': time.time(), 'error': error})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=('prepare', 'run')); p.add_argument('root')
    p.add_argument('--policy-index', type=int); p.add_argument('--compute-index', type=int); p.add_argument('--sim-index', type=int)
    a = p.parse_args()
    if a.action == 'prepare': print(prepare(a.root, a.policy_index, a.compute_index, a.sim_index))
    else: run(a.root)
