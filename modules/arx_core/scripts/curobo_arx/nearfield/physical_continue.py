"""Carry only unissued physical slots across a recorded sampling correction."""
import argparse
import copy
from pathlib import Path
import shutil
import sqlite3
import time
from scripts.curobo_arx.data_factory.common import ROOT, read, write, sha, digest, verify_files, space
from .prepare_v2 import predecessors
from .physical_startup import startup_expired
from . import campaign as launcher


def remaining(previous):
    previous = Path(previous)
    reg, summary = read(previous/'registration.json'), read(previous/'summary.json')
    if read(previous/'STOP')['reason'] != 'controlled_entry_wait_implementation_correction':
        raise ValueError('Not the recorded sampling correction')
    if reg.get('entry_wait_continuation_of'):
        raise ValueError('No automatic repeated continuation')
    if (read(previous/'state.json')['status'] != 'fault' or
            not read(previous/'postrun_verification.json')['pass'] or summary['incomplete_accounting']):
        raise ValueError('Prior execution is incomplete or unverified')
    done = {t['trial_id']: t for t in summary['trials']}
    if len(done) != len(summary['trials']):raise ValueError('Duplicated terminal slot')
    slots = []
    for trial in reg['seed_trials']:
        ident = trial['trial_id']
        db = sqlite3.connect('file:'+str(previous/'trial_ledgers'/(ident+'.sqlite3'))+'?mode=ro', uri=True)
        db.row_factory = sqlite3.Row
        try: rows = [dict(r) for r in db.execute('SELECT * FROM attempts ORDER BY started,id')]
        finally: db.close()
        if ident in done:
            if rows != done[ident]['attempts'] or len(rows) > 1:
                raise ValueError('Terminal slot/ledger mismatch')
            continue
        if rows or list((previous/'groups').glob('*/trials/'+ident+'/**/requests.json')):
            raise ValueError('Claimed or uncertain slot cannot be replayed')
        slots.append(trial)
    if len(slots)+len(done) != 36 or {t['trial_id'] for t in slots} != set(summary['unexecuted']):
        raise ValueError('Original36slot accounting changed')
    return slots, list(done.values())


def remaining_after_serialization(previous, receipt):
    previous=Path(previous);receipt=Path(receipt);proof=read(receipt)
    reg=read(previous/'registration.json');summary=read(previous/'summary.json')
    if (not proof['pass'] or proof['fault_kind']!='numpy_boolean_event_serialization_before_submit'
            or reg.get('serialization_recovery_receipt') or not reg.get('entry_wait_continuation_of')
            or read(previous/'state.json')['status']!='fault'):
        raise ValueError('Not the one recorded serialization failure')
    verify_files(proof['evidence'])
    if len(summary['incomplete_accounting'])!=1 or summary['incomplete_accounting'][0]!=proof['claim']:
        raise ValueError('Unknown claims differ from preserved fault evidence')
    if proof['first_teacher_request_or_execution_present'] or proof['physical_result']['pass']:
        raise ValueError('Serialization fault was not before first teacher dispatch')
    consumed=proof['trial_id'];slots=[]
    for t in reg['seed_trials']:
        ident=t['trial_id'];db=sqlite3.connect('file:'+str(previous/'trial_ledgers'/(ident+'.sqlite3'))+'?mode=ro',uri=True)
        try:count=db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]
        finally:db.close()
        if ident==consumed:
            if count!=1:raise ValueError('Consumed anomaly has a different claim count')
            continue
        if count or list((previous/'groups').glob('*/trials/'+ident+'/**/requests.json')):
            raise ValueError('Another claimed or uncertain slot cannot be replayed')
        slots.append(t)
    done=copy.deepcopy(reg['preserved_completed_results'])
    done.append({'trial_id':consumed,'object_id':proof['claim']['object_id'],
        'success':False,'admitted':False,'physical_dispatches':proof['claim']['physical'],
        'outcome':'infrastructure_fault_before_first_teacher_dispatch',
        'original_terminal_accounting':'incomplete_preserved','independent_reconciliation':str(receipt),
        'replayed':False})
    if len(slots)+len(done)!=36 or {t['trial_id'] for t in slots}!=set(proof['remaining_original_slots']):
        raise ValueError('Original36slot budget changed')
    return slots,done


def prepare(previous, out, checks, fault_receipt=None):
    previous, out, checks = map(lambda p: Path(p).resolve(), (previous, out, checks))
    if out.exists():raise FileExistsError('Preserve previous registrations and STOPs')
    result = read(checks)
    if not result['pass']:raise ValueError('Missing passing offline fault checks')
    verify_files(result['sources'])
    if fault_receipt:fault_receipt=Path(fault_receipt).resolve()
    slots, done = (remaining_after_serialization(previous,fault_receipt) if fault_receipt else remaining(previous))
    idle = predecessors([previous]); old = read(previous/'registration.json')
    verify_files({**old['execution_source_files'], **old['dependencies']})
    if time.time() >= old['deadline_unix']:raise RuntimeError('Original deadline expired')
    out.mkdir(parents=True);space(out)
    source = out/'source_snapshot/COOLAS-ARX';source.parent.mkdir()
    old_source = Path(old['source_snapshot'])
    shutil.copytree(old_source, source, symlinks=True, ignore=shutil.ignore_patterns('__pycache__'))
    (source.parent/'curobo').symlink_to(ROOT.parent/'curobo', target_is_directory=True)
    reg = copy.deepcopy(old)
    execution = {str(source/Path(p).relative_to(old_source) if Path(p).is_relative_to(old_source) else p): h
                 for p, h in old['execution_source_files'].items()}
    changes = {}
    for name in ('runtime_physical.py', 'physical_spec.py', 'physical_continue.py'):
        local = ROOT/'scripts/curobo_arx/nearfield'/name; dest = source/local.relative_to(ROOT)
        before = sha(dest) if dest.exists() else None
        shutil.copy2(local, dest);execution[str(dest)] = sha(dest)
        changes[name] = {'before': before, 'after': sha(dest)}
    for name in ('storage_policy.json', 'advisor20.md', 'catalog.json', 'offline_checks.json'):
        shutil.copy2(previous/name, out/name)
    for trial in slots:
        origin = previous/'trial_ledgers'/(trial['trial_id']+'.sqlite3')
        dest = out/'trial_ledgers'/origin.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(origin, dest)
    for renderer, origin in old['reference_files'].items():
        dest = out/'references'/(renderer+'.json');dest.parent.mkdir(exist_ok=True);shutil.copy2(origin, dest)
        reg['reference_files'][renderer] = str(dest);reg['dependencies'][str(dest)] = sha(dest)
    write(out/'implementation_change_receipt.json', {
        'predecessor': str(previous), 'idle': idle, 'source_changes': changes,
        'preserved_completed_slots': [t['trial_id'] for t in done],
        'remaining_original_slots': [t['trial_id'] for t in slots],
        'global_slot_budget': 36, 'additional_slots': 0, 'replayed_attempts': 0,
        'original_deadline_unix': old['deadline_unix'], 'original_attempt_seconds': old['attempt_seconds'],
        'change': ('Serialize the teacher final-goal flag as a native Python bool; numerical goals/actions unchanged. Preserve the interrupted slot and continue only unissued slots.' if fault_receipt else
                   'Wait for a fresh stopped live AND rendered observation before each teacher decision, inside original attempt budget; do not fail on the first decelerating frame.'),
        'speed_bound_rad_s_unchanged': .01, 'robot_scene_planner_success_unchanged': True})
    shutil.copy2(checks, out/'entry_wait_offline_checks.json')
    pinned = [previous/n for n in ('registration.json', 'STOP', 'state.json', 'summary.json')]
    pinned += [fault_receipt] if fault_receipt else [previous/n for n in ('postrun_verification.json', 'final_idle_verification.json')]
    pinned += [out/'implementation_change_receipt.json', out/'entry_wait_offline_checks.json']
    for p in pinned:reg['dependencies'][str(p)] = sha(p)
    reg.update(created_unix=time.time(), source_snapshot=str(source), execution_source_files=execution,
        source_identity=digest(execution), entry_wait_continuation_of=str(previous), seed_trials=slots,
        max_physical_trials=len(slots), global_physical_budget=36, consumed_physical_budget=summary_count(done),
        preserved_completed_results=done, coordinator_module='scripts.curobo_arx.nearfield.physical_continue',
        automatic_startup_retries=0, automatic_attempt_retries=0)
    if fault_receipt:reg['serialization_recovery_receipt']=str(fault_receipt)
    write(out/'registration.json', reg)
    for name in ('sources.json', 'research.md', 'research_log.md'):shutil.copy2(previous/name, out/name)
    with (out/'research_log.md').open('a') as f:
        f.write('\nFresh stopped-observation timing correction; preserve completed attempts, carry only unused original slots. See implementation_change_receipt.json.\n')
    verify_files({**execution, **reg['dependencies']})
    write(out/'state.json', {'status': 'prepared', 'scheduled_slots': len(slots), 'physical_trials': 0})
    return str(out)


def summary_count(rows):
    return sum(t['physical_dispatches'] for t in rows)


def run(campaign):
    campaign = Path(campaign).resolve(); reg = read(campaign/'registration.json')
    slots, _ = (remaining_after_serialization(reg['entry_wait_continuation_of'],reg['serialization_recovery_receipt'])
                if reg.get('serialization_recovery_receipt') else remaining(reg['entry_wait_continuation_of']))
    if slots != reg['seed_trials']:raise ValueError('Remaining original slots changed')
    predecessors([Path(reg['entry_wait_continuation_of'])])
    original = launcher.startup_expired
    launcher.startup_expired = lambda group, elapsed: startup_expired(group, elapsed, reg['startup_timeout_seconds'])
    try:launcher.run(campaign, reg['requested_devices']['compute_uuid'], reg['requested_devices']['sim_uuid'])
    finally:launcher.startup_expired = original


if __name__ == '__main__':
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='action', required=True)
    q = sub.add_parser('prepare');q.add_argument('--previous', required=True);q.add_argument('--out', required=True);q.add_argument('--checks', required=True);q.add_argument('--fault-receipt')
    q = sub.add_parser('run');q.add_argument('campaign');a = p.parse_args()
    if a.action == 'prepare':print(prepare(a.previous, a.out, a.checks,a.fault_receipt))
    else:run(a.campaign)
