"""Read-only carry-forward of verified attempts; started slots never restart."""
import json
import sqlite3
from pathlib import Path
from scripts.curobo_arx.data_factory.common import read, sha, verify_files


def rows(path):
    db = sqlite3.connect('file:' + str(Path(path).resolve()) + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    try:return [dict(r) for r in db.execute('SELECT * FROM attempts')]
    finally:db.close()


def relocated_dependencies(files):
    resolved, receipts = {}, []
    for name, expected in files.items():
        p = Path(name); target = p
        if not p.exists() and p.parent.name == 'chat':
            folder = 'gpt回复' if p.name.startswith('gpt') else 'gemini回复'
            target = p.parent / folder / p.name
        if not target.is_file() or sha(target) != expected:
            raise ValueError('Registered dependency missing or changed: ' + name)
        resolved[str(target)] = expected
        if target != p:receipts.append({'old_path': str(p), 'new_path': str(target), 'sha256': expected})
    return resolved, receipts


def admitted_carry(campaign, recheck_file):
    """Verify the preserved failed commit and its independently admitted copy."""
    from scripts.curobo_arx.data_factory.ledger import visual_evidence_files
    campaign, recheck_file = Path(campaign).resolve(), Path(recheck_file).resolve()
    registration = read(campaign/'registration.json'); recheck = read(recheck_file)
    if not recheck.get('pass') or recheck.get('physical_commands_repeated') != 0:
        raise ValueError('Admission recheck was not a verified offline-only repair')
    old_db = Path(recheck['source_ledger']).resolve(); copied_db = Path(recheck['copy']).resolve()
    if old_db.parent != campaign/'trial_ledgers' or copied_db == old_db:
        raise ValueError('Wrong admission provenance')
    if sha(old_db) != recheck['source_ledger_sha256'] or sha(campaign/'STOP') != recheck['old_STOP_sha256']:
        raise ValueError('Original ledger/STOP changed')
    old_rows, copied_rows = rows(old_db), rows(copied_db)
    if len(old_rows) != 1 or len(copied_rows) != 1:
        raise ValueError('Carry-forward expects exactly one original attempt')
    old, accepted = old_rows[0], copied_rows[0]
    if old['state'] != 'writing' or accepted['state'] != 'committed':
        raise ValueError('Admission state transition does not match repair')
    for key in ('id', 'episode', 'object_id', 'physical'):
        if old[key] != accepted[key]:raise ValueError('Copied attempt identity changed')
    if accepted['physical'] != 1:raise ValueError('Carry-forward has no physical attempt')
    episode = Path(accepted['episode']); quality = read(episode/'quality_receipt.json')
    receipt = json.loads(accepted['receipt']) if isinstance(accepted['receipt'], str) else accepted['receipt']
    if not quality['pass'] or not read(episode/'physical_receipt.json')['pass']:
        raise ValueError('Carry-forward does not have verified physical/quality success')
    for name, key in [('quality_receipt.json','quality_sha256'), ('physical_receipt.json','physical_sha256')]:
        if sha(episode/name) != receipt[key]:raise ValueError('Copied admission receipt hash mismatch')
    dependencies = {str(episode/n): h for n, h in quality['files'].items()}
    verify_files(dependencies)
    for name in visual_evidence_files(episode, quality):
        if not read(episode/name)['pass']:raise ValueError('Carried visual certificate failed')
    matches = [t for t in registration['seed_trials'] if old_db.name == t['trial_id']+'.sqlite3']
    if len(matches) != 1 or matches[0]['method'] != 'C':raise ValueError('Wrong carried slot')
    trial = matches[0]
    for name in ('closure_gate_optimization.json', 'interpreter_optimization.json'):
        if not read(episode/name)['pass']:raise ValueError('Carried C interface incomplete')
    for p in (campaign/'registration.json', campaign/'STOP', old_db, copied_db, recheck_file,
              episode/'quality_receipt.json', episode/'physical_receipt.json'):
        dependencies[str(p)] = sha(p)
    result = {**trial, 'completed':True, 'admitted':True, 'success':True,
              'attempts':[accepted], 'physical_dispatches':1, 'new_physical_dispatches':0,
              'carried_from':str(campaign), 'admission_recheck':str(recheck_file)}
    return result, dependencies


def remaining_slots(campaign, carried):
    campaign = Path(campaign); reg = read(campaign/'registration.json'); remaining = []
    done = {r['trial_id'] for r in carried}
    if len(done) != len(carried):raise ValueError('Duplicate carried slot')
    for trial in reg['seed_trials']:
        if trial['trial_id'] in done:continue
        attempts = rows(campaign/'trial_ledgers'/(trial['trial_id']+'.sqlite3'))
        paths = list((campaign/'groups').glob('*/trials/'+trial['trial_id']+'/episodes/*'))
        if attempts or paths:
            raise ValueError('Started or unknown slot cannot be rescheduled: '+trial['trial_id'])
        remaining.append(trial)
    if not done <= {t['trial_id'] for t in reg['seed_trials']}:
        raise ValueError('Unregistered carried slot')
    return remaining


def completed_replays(campaign):
    campaign = Path(campaign); reg = read(campaign/'registration.json')
    if read(campaign/'state.json')['status'] != 'complete':raise ValueError('Replay continuation not complete')
    summary = read(campaign/'summary.json'); new = summary['trials']
    if summary['completed'] != len(reg['seed_trials']):raise ValueError('Replay slots incomplete')
    carried = reg.get('carried_results', [])
    for result in carried:
        current, _ = admitted_carry(result['carried_from'], result['admission_recheck'])
        if current['attempts'][0]['id'] != result['attempts'][0]['id']:
            raise ValueError('Carried attempt changed')
    results = carried + new
    if len(results) != 2 or {r['object_id'] for r in results} != {'household_hope_milk_carton','mustard_bottle'}:
        raise ValueError('Both distinct C interface slots are required')
    for result in results:
        if result['method'] != 'C' or len(result['attempts']) != 1:
            raise ValueError('C interface not physically exercised')
        ep = Path(result['attempts'][0]['episode'])
        for name in ('closure_gate_optimization.json','interpreter_optimization.json'):
            if not read(ep/name)['pass']:raise ValueError('C interface coverage failed')
        if result in new and reg.get('synchronized_capture'):
            from .capture import audit_episode
            audit_episode(ep)
    return results


def authorized_supplement(campaign, authorization_file):
    """A new explicitly authorized slot preserves the already executed fault."""
    campaign,authorization_file=Path(campaign).resolve(),Path(authorization_file).resolve()
    auth=read(authorization_file);proposal=read(campaign/'next_validation_proposal.json')
    if (auth.get('user_message')!='授权' or auth.get('additional_validation_slots')!=1
            or auth.get('recovery_slots')!=24 or auth.get('automatic_retry') is not False
            or Path(auth.get('predecessor','')).resolve()!=campaign
            or auth.get('proposal_sha256')!=sha(campaign/'next_validation_proposal.json')):
        raise ValueError('Supplement does not match the explicit one-slot authorization')
    if (proposal['prior_registration_sha256']!=sha(campaign/'registration.json')
            or proposal['prior_STOP_sha256']!=sha(campaign/'STOP')):
        raise ValueError('Original registration/STOP changed before supplement')
    state=read(campaign/'state.json')
    if state['status']!='fault' or state.get('unconfirmed_world'):
        raise ValueError('Original physical attempt has not stopped conclusively')
    reg=read(campaign/'registration.json');verify_files(reg['execution_source_files'])
    dependencies,_=relocated_dependencies(reg['dependencies'])
    trial=proposal['trial_specification']
    if len(reg['seed_trials'])!=1 or trial!=reg['seed_trials'][0] or trial['object_id']!='mustard_bottle':
        raise ValueError('Supplement altered the fixed mustard specification')
    ledger=campaign/'trial_ledgers'/(trial['trial_id']+'.sqlite3');attempts=rows(ledger)
    if (len(attempts)!=1 or attempts[0]['id']!=proposal['prior_physical_attempt_retained']
            or attempts[0]['physical']!=1):raise ValueError('Original physical attempt missing from accounting')
    from .capture import audit_episode
    from .runtime import audit_actions
    ep=Path(attempts[0]['episode']);audit_episode(ep);audit_actions(ep,read(ep/'requests.json'))
    recheck=read(campaign/'closure_gate_offline_recheck.json')
    if not recheck['pass'] or not recheck['offline_only'] or recheck['new_physical_commands']!=0:
        raise ValueError('Corrected closure gate lacks the offline recheck')
    from scripts.curobo_arx.data_factory.common import ROOT
    if sha(ROOT/'scripts/curobo_arx/nearfield/closure.py')!=recheck['corrected_source_sha256']:
        raise ValueError('Closure implementation changed after recheck')
    for p in [authorization_file,ledger,*[campaign/n for n in ('registration.json','STOP','state.json',
            'stage_receipt.json','next_validation_proposal.json','closure_gate_offline_recheck.json')]]:
        dependencies[str(p)]=sha(p)
    dependencies.update(recheck['inputs'])
    prior={'campaign':str(campaign),'trial_id':trial['trial_id'],'attempt_id':attempts[0]['id'],
        'physical_dispatches':1,'outcome':'interface_fault_before_closure',
        'record':str(campaign/'stage_receipt.json'),'retained_in_prior_denominator':True}
    return trial,prior,dependencies


def unissued_claim(campaign, trial, attempt):
    """Prove an initialization claim stopped before any motion request."""
    if (attempt['state']!='claimed' or attempt['physical']!=0 or attempt.get('episode')
            or attempt.get('outcome') or attempt.get('receipt')):
        raise ValueError('Claim has physical, terminal or ambiguous execution evidence')
    groups=list((campaign/'groups').glob('*/trials/'+trial['trial_id']))
    if len(groups)!=1:raise ValueError('Unissued claim has ambiguous trial directory')
    directory=groups[0];episodes=list((directory/'episodes').iterdir())
    if len(episodes)!=1 or episodes[0].name!=attempt['id']:
        raise ValueError('Unissued claim episode identity mismatch')
    ep=episodes[0];required={'task.json','sampled_entry.json','selected_grasp.json','requests.json'}
    if {p.name for p in ep.iterdir()}!=required or read(ep/'requests.json')!=[]:
        raise ValueError('Unissued claim has missing records or progressed beyond reset')
    task=read(ep/'task.json')
    if any(task.get(k)!=attempt.get(k) for k in ('id','state','physical','started','owner','env','object_id')):
        raise ValueError('Unissued claim/task accounting mismatch')
    group=directory.parent.parent
    if not read(group/'shutdown_receipt.json')['all_trajectories_idle']:
        raise ValueError('Unissued claim idle status unknown')
    if 'CaptureUnavailable' not in read(group/'FAULT.json').get('error',''):
        raise ValueError('Only the identified pre-dispatch capture fault is recoverable')
    files=[ep/n for n in required]+[group/'shutdown_receipt.json',group/'FAULT.json']
    requests=list((campaign/'workers/planner/requests').iterdir())
    if not requests:raise ValueError('Missing planner request history')
    for p in requests:
        if p.suffix!='.done' or read(p).get('action')!='qualify_pool':
            raise ValueError('A motion planning request already occurred or is unresolved')
        response=campaign/'workers/planner/responses'/(read(p)['id']+'.json')
        # WorkerClient consumes replies. The immutable .done request remains;
        # the saved qualified pool binds the accepted non-motion result.
        if response.exists():files.append(response)
        files.append(p)
    pools=list((directory/'blocks').glob('*/qualified_pool.json'))
    if len(pools)!=1:raise ValueError('Missing completed non-motion qualification')
    files+=pools
    mailboxes=list(group.glob('env_*/mailbox'))
    if len(mailboxes)!=1:raise ValueError('Missing or ambiguous original motion mailbox')
    for mailbox in mailboxes:
        for p in mailbox.iterdir():
            if p.name not in {'batched_fingerprint_equivalence.json','fingerprint_progress.json',
                    'additional_asset_runtime_baseline.json','fingerprint_timing.json',
                    'batched_fingerprint_scalar.json','batched_fingerprint_batch.json'}:
                raise ValueError('Motion mailbox has an unknown command or receipt')
            files.append(p)
    return {'attempt_id':attempt['id'],'trial_id':trial['trial_id'],'physical_dispatches':0,
        'classification':'capture_initialization_fault_before_any_motion_planning_or_command',
        'old_claim_retained':True,'files':{str(p):sha(p) for p in files}}


def unused_startup(campaign, recover_unissued_claim=False):
    """Carry unused slots, optionally proving a claim never requested motion."""
    campaign = Path(campaign)
    state = read(campaign/'state.json')
    if state['status'] != 'fault' or state.get('unconfirmed_world'):
        raise ValueError('Startup predecessor has not stopped conclusively')
    reg = read(campaign/'registration.json')
    unissued=[]
    for trial in reg['seed_trials']:
        attempts=rows(campaign/'trial_ledgers'/(trial['trial_id']+'.sqlite3'))
        if attempts:
            if not recover_unissued_claim or len(attempts)!=1:
                raise ValueError('Startup continuation would replay a claimed attempt')
            unissued.append(unissued_claim(campaign,trial,attempts[0]))
    if any((campaign/'groups').glob('*/trials/*/episodes/*/*_command.json')):
        raise ValueError('Startup continuation found an executed or requested command')
    return {'predecessor':str(campaign.resolve()), 'registration_sha256':sha(campaign/'registration.json'),
            'STOP_sha256':sha(campaign/'STOP'), 'claimed_attempts':len(unissued), 'physical_commands':0,
            'unissued_claims':unissued,
            'deadline_unix':reg['deadline_unix'], 'seed_trials':reg['seed_trials']}
