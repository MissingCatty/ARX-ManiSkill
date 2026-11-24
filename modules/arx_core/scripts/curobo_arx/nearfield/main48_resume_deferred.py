"""Resume only the seven explicitly authorized, never-started original D slots."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts.curobo_arx.data_factory.common import ROOT, read, write, sha, digest, verify_files, space
from scripts.curobo_arx.data_factory.launch import devices
from training.nearfield_arx.stage_policy import validate_checkpoints
from .main48_continue import ledger_rows
from .main48_report import task_counts
from .main48_spec import validate_schedule
from .prepare_v2 import predecessors


def resumed_registration(old, verification):
    validate_schedule(old)
    if len(old['deferred_trials']) != 7 or any(t['method'] != 'D' for t in old['deferred_trials']):
        raise ValueError('Expected the original seven deferred D slots')
    if not verification['record_integrity_pass'] or verification['unknown_or_incomplete']:
        raise ValueError('Prior report is not independently verified')
    if verification['unexecuted'] != [t['trial_id'] for t in old['deferred_trials']]:
        raise ValueError('Unexpected remaining original scope')
    carried = copy.deepcopy(verification['trials'])
    counts = task_counts(carried)
    if len(carried) != 41 or tuple(counts[k] for k in ('physical_successes', 'physical_failures', 'infrastructure_interruptions')) != (13, 25, 3):
        raise ValueError('Original outcomes changed')
    reg = copy.deepcopy(old)
    reg.update(carried_details=carried, seed_trials=copy.deepcopy(old['deferred_trials']),
               deferred_trials=[], consumed_physical_budget=41, max_physical_trials=7)
    reg['execution_amendment'] = {
        'kind': 'user_resume_deferred_D', 'user_message': '把d跑完吧，然后在写回复',
        'prior_deferral': copy.deepcopy(old['execution_amendment']),
        'prior_deferral_interpretation_was_overbroad': True,
        'actual_order': 'Seven original unstarted D slots after the completed ABC slots; retain original relative D order',
        'original_balanced_execution_order_restored': False,
        'prior_started_or_unknown_slots_retried': 0, 'additional_slots': 0,
        'original48_denominator_and_deadline_unchanged': True}
    validate_schedule(reg)
    return reg


def check_unused_ledgers(directory, pending):
    directory = Path(directory)
    expected = {t['trial_id'] for t in pending}
    if {p.stem for p in directory.glob('*.sqlite3')} != expected:
        raise ValueError('Missing or foreign deferred ledger')
    pins = {}
    for ident in sorted(expected):
        p = directory / (ident + '.sqlite3')
        if ledger_rows(p):
            raise ValueError('Never replay an already claimed deferred slot')
        # A read-only connection to a WAL-mode database may create empty WAL
        # and shared-memory sidecars. Only a nonempty transaction log is data.
        if any(s.exists() and s.stat().st_size for s in (p.with_name(p.name + '-wal'), p.with_name(p.name + '-journal'))):
            raise ValueError('Deferred ledger has unresolved SQLite journal')
        pins[str(p)] = sha(p)
    return pins


def prepare(previous, recovery, out, indices):
    previous, recovery, out = (Path(p).resolve() for p in (previous, recovery, out))
    if out.exists():
        raise FileExistsError('Preserve all earlier registrations')
    old = read(previous / 'registration.json')
    if time.time() >= old['deadline_unix']:
        raise TimeoutError('Original48 deadline expired')
    if read(previous / 'state.json')['status'] != 'complete' or (previous / 'STOP').exists():
        raise ValueError('Predecessor execution is not complete')
    summary = read(previous / 'summary.json')
    if summary['unexecuted'] or summary['incomplete_accounting'] or [t['trial_id'] for t in summary['trials']] != [t['trial_id'] for t in old['seed_trials']]:
        raise ValueError('Prior native scope not completely accounted')
    rr = read(recovery / 'registration.json'); done = read(recovery / 'completion.json')
    if rr['campaign'] != str(previous) or read(recovery / 'state.json')['status'] != 'complete' or not done['pass'] or not done['original_results_unchanged']:
        raise ValueError('Report recovery not bound to completed predecessor')
    verification_path = recovery / 'results/main48_verification.json'
    if done['verification'] != str(verification_path) or done['verification_sha256'] != sha(verification_path):
        raise ValueError('Prior verification binding changed')
    pointer = read(previous / 'recovered_report.json')
    if pointer['completion_sha256'] != sha(recovery / 'completion.json'):
        raise ValueError('Prior recovery pointer changed')
    verification = read(verification_path); reg = resumed_registration(old, verification)
    pins = {**verification['evidence'], **check_unused_ledgers(previous / 'deferred_trial_ledgers', reg['seed_trials'])}
    expected = {d['trial_id']: (d['outcome'], d['physical_success']) for d in old['carried_details']}
    for trial in summary['trials']:
        p = previous / 'trial_ledgers' / (trial['trial_id'] + '.sqlite3')
        if ledger_rows(p) != trial['attempts']:
            raise ValueError('Prior native ledger changed')
        pins[str(p)] = sha(p)
        expected[trial['trial_id']] = (trial['outcome'], trial['success'])
    if {d['trial_id']: (d['outcome'], d['physical_success']) for d in reg['carried_details']} != expected:
        raise ValueError('Carried outcomes differ from native/supplemental evidence')
    pending_ids = {t['trial_id'] for t in reg['seed_trials']}
    if any(p.name in pending_ids for p in (previous / 'groups').glob('*/trials/*')):
        raise ValueError('A deferred trial directory already exists')
    for d in reg['carried_details']:
        v = d.get('video')
        if not v or not v['decode_pass']:
            raise ValueError('Missing verified carried video')
        pins[v['path']] = v['sha256']
    idle = predecessors([previous]); validate_checkpoints(old)
    verify_files({**old['execution_source_files'], **old['dependencies'], **rr['source_files'], **rr['dependencies'], **pins})
    out.mkdir(parents=True); space(out)
    write(out / 'state.json', {'status': 'preparing', 'consumed': 41, 'remaining_slots': 7})
    write(out / 'predecessor_exit_receipt.json', idle)
    for name in ('storage_policy.json', 'advisor29.md', 'PLAN.md', 'case_manifest.json'):
        shutil.copy2(previous / name, out / name)
    prior = Path(rr['source_snapshot']); source = out / 'source_snapshot/COOLAS-ARX'
    shutil.copytree(prior, source, symlinks=True, ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('tmp', 'curobo'):
        (source.parent / name).symlink_to(ROOT.parent / name, target_is_directory=True)
    # Older read-only regression fixtures resolve ROOT/runs directly. Expose
    # the original fixtures without changing their source or copying data.
    fixture_names = ('nearfield_main48_20261110_01', 'nearfield_eef_fit_20261108_02',
        'nearfield_advisor25_diagnosis_20261108_01', 'nearfield_closure_20261109_03',
        'nearfield_lateral_20261109_02', 'nearfield_stage1_20261105_04',
        'nearfield_validation_20261106_03', 'nearfield_stage_recovery_20261109_02',
        'nearfield_main48_20261109_01')
    for name in fixture_names:
        (source / 'runs' / name).symlink_to(ROOT / 'runs' / name, target_is_directory=True)
    execution = {str(source / Path(p).relative_to(prior)) if Path(p).is_relative_to(prior) else p: h for p, h in rr['source_files'].items()}
    added = ('scripts/curobo_arx/nearfield/main48_resume_deferred.py', 'scripts/curobo_arx/nearfield/test_main48_resume.py')
    for name in added:
        shutil.copy2(ROOT / name, source / name); execution[str(source / name)] = sha(source / name)
    verify_files(execution)
    checks = read(recovery / 'host_tests.json')['command'][:-1] + ['scripts.curobo_arx.nearfield.test_main48_resume', '-v']
    env = dict(__import__('os').environ, ARX_TEST_FIXTURE_ROOT=str(ROOT), PYTHONPATH=str(source))
    result = subprocess.run(checks, cwd=source, env=env, capture_output=True, text=True, timeout=180)
    (out / 'offline_checks.log').write_text(result.stdout + result.stderr)
    write(out / 'offline_checks.json', {'pass': result.returncode == 0, 'command': checks, 'log_sha256': sha(out / 'offline_checks.log'), 'GPU_or_World_started': False})
    if result.returncode:
        raise RuntimeError('Frozen-source offline checks failed')
    roles = {k: next(d for d in devices() if d['index'] == n) for k, n in indices.items()}; floors = {}
    for role, floor in [('policy', 12000), ('geometry', 3500), ('compute', 6500), ('sim', 7500)]:
        u = roles[role]['uuid']; floors[u] = floors.get(u, 0) + floor
    write(out / 'resource_plan.json', {'roles': roles, 'combined_free_memory_mib': floors, 'stable_samples': 3, 'interval_seconds': 10, 'other_users_jobs_untouched': True})
    for directory in (previous, recovery):
        for p in directory.iterdir():
            if p.is_file() and (p.suffix == '.json' or p.name == 'STOP'):
                pins[str(p)] = sha(p)
    pins[str(verification_path)] = sha(verification_path)
    deps = {**old['dependencies'], **old['execution_source_files'], **rr['source_files'], **rr['dependencies'], **pins}
    reg.update(created_unix=time.time(), resumed_deferred_of=str(previous), report_recovery=str(recovery),
               source_snapshot=str(source), execution_source_files=execution, source_identity=digest(execution),
               dependencies=deps, carried_evidence=pins, requested_devices={k + '_uuid': v['uuid'] for k, v in roles.items()})
    reg['learned_policy']['gpu_uuid'] = roles['policy']['uuid']; reg['geometry_baseline']['gpu_uuid'] = roles['geometry']['uuid']
    (out / 'trial_ledgers').mkdir()
    for t in reg['seed_trials']:
        original = previous / 'deferred_trial_ledgers' / (t['trial_id'] + '.sqlite3')
        shutil.copy2(original, out / 'trial_ledgers' / original.name)
        if sha(out / 'trial_ledgers' / original.name) != sha(original):
            raise ValueError('Unused ledger copy differs')
    check_unused_ledgers(out / 'trial_ledgers', reg['seed_trials']); validate_schedule(reg)
    write(out / 'source_change_receipt.json', {
        'added_continuation_and_tests': added, 'base': str(prior),
        'readonly_regression_fixture_links': [str(ROOT / 'runs' / n) for n in fixture_names],
        'all_inherited_sources_byte_identical_to_verified_report_snapshot': True,
        'earlier_audit_repair': str(recovery / 'source_change_receipt.json'),
        'physical_execution_models_robot_scene_initial_render_success_unchanged': True,
        'uniform_warmups_unchanged': True, 'old_STOPs_results_terminal_gaps_and_deferral_preserved': True,
        'consumed': 41, 'remaining_original_D': 7, 'original_budget': 48, 'retries': 0,
        'new_training_updates': 0, 'deadline_unix': old['deadline_unix'], 'amendment': reg['execution_amendment']})
    command = ['podman', 'run', '--pull=never', '--rm', '-i', '--network=none', '--user', '0', '--security-opt=label=disable',
        '-v', str(ROOT.parent) + ':' + str(ROOT.parent) + ':rw', '-v', '/coolas/data/arx_teacher_zhaoyechong:/coolas/data/arx_teacher_zhaoyechong:ro',
        '-w', str(source), '--entrypoint', '/isaac-sim/python.sh', 'localhost/arx-s02-isaac:6.0.1', '-c',
        'import os,json,sys,pathlib,unittest;p=json.load(sys.stdin);os.environ["ARX_TEST_FIXTURE_ROOT"]=p["fixture_root"];'
        'import scripts.curobo_arx.nearfield.runtime_main48;'
        't=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromNames(["training.nearfield_arx.test_settling_receipts","scripts.curobo_arx.nearfield.test_main48_resume"]));'
        'missing=[x for x in p["paths"] if not pathlib.Path(x).is_file()];'
        'print(json.dumps({"tests":t.testsRun,"pass":t.wasSuccessful(),"missing":missing}));sys.exit(bool(missing) or not t.wasSuccessful())']
    probe = subprocess.run(command, input=json.dumps({'fixture_root': str(ROOT), 'paths': list({*execution, *deps})}), capture_output=True, text=True, timeout=180)
    write(out / 'container_preflight.json', {'pass': probe.returncode == 0, 'command': command, 'stdout': probe.stdout, 'stderr': probe.stderr, 'GPU_or_World_started': False, 'image_downloaded': False})
    if probe.returncode:
        raise RuntimeError('Frozen Isaac CPU-only preflight failed')
    write(out / 'catalog.json', reg['catalog'])
    for p in out.iterdir():
        if p.is_file() and p.name != 'state.json':
            deps[str(p)] = sha(p)
    write(out / 'registration.json', reg); verify_files({**execution, **deps}); predecessors([previous])
    write(out / 'state.json', {'status': 'prepared', 'consumed': 41, 'remaining_slots': 7, 'deferred_slots': 0, **task_counts(reg['carried_details'])})
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('previous'); p.add_argument('recovery'); p.add_argument('out')
    for role in ('policy', 'geometry', 'compute', 'sim'):
        p.add_argument('--' + role, required=True, type=int)
    a = p.parse_args(); print(prepare(a.previous, a.recovery, a.out, {k: getattr(a, k) for k in ('policy', 'geometry', 'compute', 'sim')}))
