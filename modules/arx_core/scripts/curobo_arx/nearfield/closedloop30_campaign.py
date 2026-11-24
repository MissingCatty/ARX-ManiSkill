"""One bounded advisor30 C/D replication, with a new versioned task metric."""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from scripts.curobo_arx.data_factory.common import ROOT, read, write, sha, digest, verify_files, space
from scripts.curobo_arx.data_factory.launch import devices
from training.nearfield_arx.lateral_campaign import ledgers
from .prepare_v2 import predecessors
from .closedloop30_policy import validate_checkpoints
from .closedloop30_rules import VERSION, POLICY

PREVIOUS = ROOT/'runs/nearfield_main48_20261110_07'
AUTH = ROOT/'research_plan/chat/gemini回复/gemini回复30.md'
PLAN = ROOT/'research_plan/docs/nearfield_closedloop30_20261110.md'


def make_trials(original):
    # Priority is fixed before any new observation, including all3 old unknowns.
    priority = [('01_normal', 'D'), ('02_normal', 'D'), ('03_normal', 'D'),
                ('02_joint_prior_entry_offset', 'D'), ('02_entry_offset', 'D'),
                ('03_entry_offset', 'D'), ('03_joint_prior_entry_offset', 'D'),
                ('00_normal', 'D'), ('00_entry_offset', 'D'),
                ('01_joint_prior_entry_offset', 'C')]
    selected = [t for t in original if t['method'] in ('C', 'D')]
    ordered = [next(t for t in selected if (t['case_id'], t['method']) == pair) for pair in priority]
    ordered += [t for t in selected if (t['case_id'], t['method']) not in priority]
    result = []
    for i, old in enumerate(ordered):
        t = copy.deepcopy(old)
        t.update(trial_id=f'r30_{i:02d}_{old["case_id"]}_{old["method"]}',
                 historical_trial_id=old['trial_id'], policy='advisor30_'+old['method'])
        result.append(t)
    if len(result) != 24 or len({t['historical_trial_id'] for t in result}) != 24:
        raise RuntimeError('Exactly one new C/D attempt per original case is required')
    return result


def prepare(out, policy_index=1, compute_index=2, sim_index=3):
    out = Path(out).resolve()
    if out.exists(): raise FileExistsError('Preserve every earlier registration')
    old = read(PREVIOUS/'registration.json'); completed = read(PREVIOUS/'completion_verification.json')
    if not completed['pass'] or read(PREVIOUS/'state.json')['status'] != 'complete':
        raise RuntimeError('Historical original48 has not concluded with verified exit')
    idle = predecessors([PREVIOUS])
    out.mkdir(parents=True); space(out)
    write(out/'state.json', {'status': 'preparing', 'physical_trials': 0})
    added = sorted((ROOT/'scripts/curobo_arx/nearfield').glob('closedloop30_*.py'))
    added += [ROOT/'scripts/curobo_arx/nearfield/runtime_closedloop30.py', ROOT/'scripts/curobo_arx/nearfield/test_closedloop30.py']
    command = [sys.executable, '-m', 'unittest', 'scripts.curobo_arx.nearfield.test_closedloop30',
        'training.nearfield_arx.test_stage_recovery', 'training.nearfield_arx.test_closure_recovery',
        'training.nearfield_arx.test_closure_records', 'scripts.curobo_arx.nearfield.test_learned', '-v']
    check = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=240)
    (out/'offline_checks.log').write_text(check.stdout+check.stderr)
    write(out/'offline_checks.json', {'pass': check.returncode == 0, 'command': command,
        'log_sha256': sha(out/'offline_checks.log'), 'sources': {str(p): sha(p) for p in added},
        'no_GPU_service_or_World_started': True})
    if check.returncode: raise RuntimeError('Offline checks failed')
    verify_files({**old['execution_source_files'], **old['dependencies']})
    verify_files(read(PREVIOUS/'main48_verification.json')['evidence'])
    source = out/'source_snapshot/COOLAS-ARX'; prior = Path(old['source_snapshot'])
    shutil.copytree(prior, source, symlinks=True, ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('tmp', 'curobo'): (source.parent/name).symlink_to(ROOT.parent/name, target_is_directory=True)
    execution = {str(source/Path(p).relative_to(prior)) if Path(p).is_relative_to(prior) else p: h
                 for p, h in old['execution_source_files'].items()}
    for p in added:
        dest = source/p.relative_to(ROOT); dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest); execution[str(dest)] = sha(dest)
    for p, h in old['execution_source_files'].items():
        dest = source/Path(p).relative_to(prior) if Path(p).is_relative_to(prior) else Path(p)
        if sha(dest) != h: raise RuntimeError('Inherited source modified: '+str(dest))
    inventory = devices(); roles = {name: next(d for d in inventory if d['index'] == index)
        for name, index in [('policy', policy_index), ('compute', compute_index), ('sim', sim_index)]}
    if len({d['uuid'] for d in roles.values()}) != 3:
        raise ValueError('This registration uses three explicitly separate devices')
    trials = make_trials(old['registered_trials']); reg = copy.deepcopy(old)
    for key in list(reg):
        if key.endswith('_of') or key in ('registered_trials', 'carried_details', 'carried_evidence',
            'deferred_trials', 'execution_amendment', 'consumed_physical_budget', 'geometry_baseline',
            'report_recovery', 'physical_data_contract', 'unified_registration'):
            reg.pop(key, None)
    reg.update(stage=VERSION, created_unix=time.time(), campaign_started_unix=time.time(),
        deadline_unix=time.time()+14*3600, wall_hours=14, production=False,
        source_snapshot=str(source), execution_source_files=execution, source_identity=digest(execution),
        seed_trials=trials, seeds={t['seed_id']: old['seeds'][t['seed_id']] for t in trials},
        max_physical_trials=24, global_physical_budget=24, max_replay_trials=0,
        recovery_slots_total=0, replay_slots_total=0, carried_results=[], prior_results=[], prerequisite_results=[],
        formal_48_trials_authorized_here=False, training_authorized_here=False,
        automatic_attempt_retries=0, automatic_startup_retries=0, no_automatic_retry=True,
        runtime_module='scripts.curobo_arx.nearfield.runtime_closedloop30',
        worker_module='scripts.curobo_arx.nearfield.closedloop30_worker',
        policy_module='scripts.curobo_arx.nearfield.closedloop30_policy',
        coordinator_module='scripts.curobo_arx.nearfield.closedloop30_campaign',
        requested_devices={k+'_uuid': d['uuid'] for k, d in roles.items()},
        closedloop30_policy=POLICY, historical_results=str(PREVIOUS/'main48_verification.json'),
        execution_authority={'advisor': str(out/'advisor30.md'), 'scope': '24 newly registered C/D replications, including3 historical unknown replacements; no training or retries',
            'criterion_amendment': 'Prospective advisor30 physical task score plus original strict shadow score; never alter historical outcomes'},
        project_root=str(ROOT))
    training = ROOT/'runs/nearfield_lateral_20261109_02/training'
    for method in ('C', 'D'):
        reg['learned_policy']['checkpoints'][method] = str(training/method/'checkpoint-544')
        reg['learned_policy']['training_bindings'][method] = {'registration': str(training/'registration.json'),
                                                            'sha256': sha(training/'registration.json')}
    reg['learned_policy']['gpu_uuid'] = roles['policy']['uuid']
    reg['learned_policy']['training'] = str(training)
    validate_checkpoints(reg)
    deps = copy.deepcopy(old['dependencies'])
    for method, folder in reg['learned_policy']['checkpoints'].items():
        path = Path(folder); cert = read(path/'checkpoint_receipt.json')
        deps.update({str(path/n): h for n, h in cert['files'].items()})
        deps[str(path/'checkpoint_receipt.json')] = sha(path/'checkpoint_receipt.json')
    for p in (PREVIOUS/'registration.json', PREVIOUS/'main48_verification.json', PREVIOUS/'completion_verification.json', training/'registration.json'):
        deps[str(p)] = sha(p)
    for name, p in [('advisor30.md', AUTH), ('PLAN.md', PLAN), ('storage_policy.json', PREVIOUS/'storage_policy.json')]:
        shutil.copy2(p, out/name)
    write(out/'predecessor_exit_receipt.json', idle)
    write(out/'case_manifest.json', {'slots': trials, 'budget': 24, 'automatic_retries': 0,
        'historical48_unchanged': True, 'A_B_reused_as_historical_reference_only': True,
        'shortened_entry_results_separate_from_original80mm_comparison': True})
    write(out/'resource_plan.json', {'roles': roles, 'free_memory_mib': {'policy': 12000, 'compute': 6500, 'sim': 7500},
                                   'stable_samples': 3, 'interval_seconds': 10})
    write(out/'source_change_receipt.json', {'added': [str(p.relative_to(ROOT)) for p in added],
        'inherited_sources_byte_unchanged': True, 'robot_physics_reset_rendering_unchanged': True,
        'new_policy': POLICY, 'original_evaluator_unchanged': True, 'new_versioned_task_evaluation': True,
        'D_checkpoint_replaced_with_unweighted_lateral544': True, 'training_updates': 0})
    (out/'research.md').write_text((out/'PLAN.md').read_text()+'\n预算：一轮、24次、14小时；pause_every=never；成功和失败均归档，不自动提升为默认方法。\n')
    (out/'research_log.md').write_text('Prospectively registered advisor30 mechanism bundle before any new physical attempt.\n')
    write(out/'sources.json', {'advisor': str(out/'advisor30.md'), 'historical_results': reg['historical_results'],
                              'model_training': str(training/'registration.json'), 'method': 'advisor-requested engineering variant; no paper reproduction claim'})
    command = ['podman', 'run', '--pull=never', '--rm', '-i', '--user', '0', '--security-opt=label=disable',
        '-v', str(ROOT.parent)+':'+str(ROOT.parent)+':rw',
        '-v', '/coolas/data/arx_teacher_zhaoyechong:/coolas/data/arx_teacher_zhaoyechong:ro',
        '-w', str(source), '--entrypoint', '/isaac-sim/python.sh', 'localhost/arx-s02-isaac:6.0.1', '-c',
        'import json,pathlib,sys;import scripts.curobo_arx.nearfield.runtime_closedloop30;'
        'paths=json.load(sys.stdin);missing=[p for p in paths if not pathlib.Path(p).is_file()];'
        'print(json.dumps({"missing":missing,"checked":len(paths)}));sys.exit(bool(missing))']
    probe = subprocess.run(command, input=json.dumps(list({*execution, *deps})), capture_output=True, text=True, timeout=90)
    write(out/'container_preflight.json', {'pass': probe.returncode == 0, 'command': command,
        'stdout': probe.stdout, 'stderr': probe.stderr, 'GPU_or_World_started': False, 'image_downloaded': False})
    if probe.returncode: raise RuntimeError('Snapshot container imports/paths failed')
    for p in out.iterdir():
        if p.is_file() and p.name not in ('state.json', 'research_log.md'): deps[str(p)] = sha(p)
    reg['dependencies'] = deps
    ledgers(out, reg); write(out/'catalog.json', reg['catalog']); write(out/'registration.json', reg)
    verify_files({**execution, **deps})
    write(out/'state.json', {'status': 'prepared', 'scheduled_slots': 24, 'physical_trials': 0})
    return out


def run(campaign):
    from . import closure_physical, learned_report
    from .closedloop30_report import report
    root = Path(campaign).resolve(); reg = read(root/'registration.json')
    lock = (root/'advisor30.lock').open('a+'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'STOP').exists() or (root/'workers/policy').exists(): raise RuntimeError('No restart or STOP bypass')
    if reg['closedloop30_policy'] != POLICY or len(reg['seed_trials']) != 24:
        raise RuntimeError('New campaign contract changed')
    validate_checkpoints(reg); verify_files({**reg['execution_source_files'], **reg['dependencies']})
    resource = read(root/'resource_plan.json'); stable = 0
    while stable < 3:
        if (root/'STOP').exists() or time.time() >= reg['deadline_unix']:
            raise RuntimeError('Stopped before resource admission')
        inventory = {d['uuid']: d['free_mb'] for d in devices()}
        stable = stable+1 if all(inventory.get(resource['roles'][role]['uuid'], 0) >= floor
                               for role, floor in resource['free_memory_mib'].items()) else 0
        write(root/'state.json', {'status': 'waiting_for_resources', 'stable_samples': stable, 'wall': time.time()})
        if stable < 3: time.sleep(10)
    learned_report.report = report
    result = closure_physical.run(root)
    write(root/'delivery_receipt.json', {'verification': str(root/'closedloop30_verification.json'),
        'verification_sha256': sha(root/'closedloop30_verification.json'), 'completed': result['completed'],
        'physical_successes': result['physical_successes'], 'strict_successes': result['strict_successes']})
    with (root/'research_log.md').open('a') as stream:
        stream.write(f'Completed {result["completed"]}/24; physical task successes {result["physical_successes"]}; strict successes {result["strict_successes"]}.\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=('prepare', 'run')); p.add_argument('root')
    p.add_argument('--policy', type=int, default=1); p.add_argument('--compute', type=int, default=2); p.add_argument('--sim', type=int, default=3)
    a = p.parse_args()
    if a.action == 'prepare': print(prepare(a.root, a.policy, a.compute, a.sim))
    else: run(a.root)
