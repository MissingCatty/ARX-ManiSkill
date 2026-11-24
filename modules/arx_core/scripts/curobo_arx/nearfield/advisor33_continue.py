"""Carry terminal slots forward and authorize only the original unissued remainder."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
from scripts.curobo_arx.data_factory.common import ROOT,read,write,sha,digest,verify_files,space
from scripts.curobo_arx.data_factory.launch import devices
from .advisor31_campaign import exited
from .advisor32_spec import validate_trials


def prepare(previous,out,policy_index,compute_index,sim_index,*,stop_history_repair=False,interruption_record=None):
    previous=Path(previous).resolve();out=Path(out).resolve();out.mkdir(exist_ok=False);space(out)
    old=read(previous/'registration.json');state=read(previous/'state.json');verification=read(previous/'advisor33_verification.json')
    if not (previous/'STOP').exists() or state['unconfirmed_world'] or not verification['record_integrity_pass']:
        raise RuntimeError('Predecessor integrity/exit gate failed')
    interruptions=copy.deepcopy(old.get('supplemental_interruptions',{}))
    if interruption_record:
        record_path=Path(interruption_record).resolve();saved=read(record_path);verify_files(saved['evidence'])
        if saved['previous']!=str(previous) or saved['new_physical_trials'] or saved['native_records_or_ledger_rewritten']:
            raise RuntimeError('Invalid supplemental interruption scope')
        detail=saved['detail'];ident=detail['trial_id']
        if ident in interruptions or detail['classification']!='interrupted_unknown' or detail['task_success']:
            raise RuntimeError('Repeated or forged interrupted result')
        interruptions[ident]=saved
    if verification['unknown'] and not interruptions:
        raise RuntimeError('Unknown prior execution cannot be replayed or silently carried')
    if stop_history_repair:
        from training.nearfield_arx.native_mask import configured
        if not configured(old):raise ValueError('Stop-history continuation must retain the native-mask registration')
        diagnosis_path=previous/'stop_history_diagnosis.json'
        diagnosis=read(diagnosis_path);verify_files(diagnosis['evidence'])
        if not diagnosis['confirmed_history_starvation']:raise ValueError('Missing measured history-starvation evidence')
    exit_proof=exited(previous);verify_files({**old['execution_source_files'],**old['dependencies']})
    rows=read(previous/'summary.json')['trials'];done={r['trial_id'] for r in rows}
    if len(done)!=len(rows) or len(rows)!=state['completed']:raise RuntimeError('Predecessor completion accounting mismatch')
    if done&set(interruptions):raise RuntimeError('Interrupted slot was replayed')
    pending_claims=read(previous/'summary.json')['incomplete_accounting']
    if {r['trial_id'] for r in pending_claims}!={i for i,v in interruptions.items() if v['previous']==str(previous)}:
        raise RuntimeError('Unexplained native terminal gap')
    unused=[]
    for trial in old['seed_trials']:
        p=previous/'trial_ledgers'/(trial['trial_id']+'.sqlite3')
        if not p.exists():
            if trial['trial_id'] in old.get('carried_terminal_ids',[]) and trial['trial_id'] in done|set(interruptions):continue
            raise RuntimeError('Missing required attempt ledger')
        db=sqlite3.connect('file:'+str(p)+'?mode=ro',uri=True)
        count=db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0];db.close()
        if trial['trial_id'] not in done|set(interruptions):
            if count or list(previous.glob('groups/*/trials/'+trial['trial_id']+'/episodes/*')):raise RuntimeError('Cannot replay a started/unknown slot')
            unused.append(trial['trial_id'])
    if len(done)+len(interruptions)+len(unused)!=30:raise RuntimeError('Original30-slot budget changed')
    prior=Path(old['source_snapshot']);source=out/'source_snapshot/COOLAS-ARX'
    shutil.copytree(prior,source,symlinks=True,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('tmp','curobo'):(source.parent/name).symlink_to(ROOT.parent/name,target_is_directory=True)
    execution={str(source/Path(p).relative_to(prior)) if Path(p).is_relative_to(prior) else p:h for p,h in old['execution_source_files'].items()}
    changed=['runtime_advisor33.py','advisor33_launch.py','advisor33_report.py','advisor33_continue.py','test_advisor33.py']
    if stop_history_repair:changed+=['test_stop_history.py','stop_history_diagnosis.py']
    if interruption_record:changed+=['advisor33_boundary.py','advisor33_teacher.py','advisor33_interruption.py','test_advisor33_boundary.py']
    source_changes=[]
    for name in changed:
        p=ROOT/'scripts/curobo_arx/nearfield'/name;dest=source/p.relative_to(ROOT);before=sha(dest) if dest.exists() else None
        shutil.copy2(p,dest);execution[str(dest)]=sha(dest)
        if before!=sha(dest):source_changes.append({'path':str(p.relative_to(ROOT)),'old_sha256':before,'new_sha256':sha(dest)})
    if stop_history_repair:
        p=ROOT/'training/nearfield_arx/test_native_mask.py';dest=source/p.relative_to(ROOT);before=sha(dest)
        shutil.copy2(p,dest);execution[str(dest)]=sha(dest)
        source_changes.append({'path':str(p.relative_to(ROOT)),'old_sha256':before,'new_sha256':sha(dest)})
    command=[sys.executable,'-m','unittest','scripts.curobo_arx.nearfield.test_advisor33','scripts.curobo_arx.nearfield.test_advisor32','training.nearfield_arx.test_closure_records','-v']
    if stop_history_repair:command[3:3]=['scripts.curobo_arx.nearfield.test_stop_history','training.nearfield_arx.test_native_mask']
    if interruption_record:command[3:3]=['scripts.curobo_arx.nearfield.test_advisor33_boundary']
    check=subprocess.run(command,cwd=source,env={**os.environ,'ARX_TEST_PROJECT_ROOT':str(ROOT)},capture_output=True,text=True,timeout=180)
    (out/'offline_checks.log').write_text(check.stdout+check.stderr)
    write(out/'offline_checks.json',{'pass':check.returncode==0,'log_sha256':sha(out/'offline_checks.log'),'new_physical_trials':0})
    if check.returncode:raise RuntimeError('Continuation offline tests failed')
    reg=copy.deepcopy(old);deps=dict(old['dependencies']);inventory=devices();by={d['index']:d for d in inventory}
    roles={r+'_uuid':by[i]['uuid'] for r,i in (('policy',policy_index),('compute',compute_index),('sim',sim_index))}
    if roles['sim_uuid'] in (roles['policy_uuid'],roles['compute_uuid']):raise ValueError('Dedicated simulator GPU')
    reg.update(source_snapshot=str(source),execution_source_files=execution,source_identity=digest(execution),
        created_unix=time.time(),continuation_of=str(previous),ancestor_campaigns=list(dict.fromkeys([str(previous),*old.get('ancestor_campaigns',[])])),
        carried_results=rows,carried_terminal_ids=sorted(done|set(interruptions)),supplemental_interruptions=interruptions,remaining_trial_ids=unused,
        max_physical_trials=len(unused),global_physical_budget=30,requested_devices=roles,
        correction_scope='10s terminal stop failures close the old World before scheduling a new slot; all measured-window sampling, thresholds, goals, model inputs and physics unchanged')
    if stop_history_repair:
        reg['correction_scope']='Collect the existing synchronized measured stop samples during commands and retain them at completion. Same 2s/10sample/0.5mm/0.01rad-s window, current-controller-idle requirement, 10 wall-second limit, model, goals, physics and success criterion.'
        reg['advisor33_policy']['stop_history_capture']='continuous recorded tail across command completion, cleared at each episode reset'
    if interruption_record:
        reg['correction_scope']='Retain continuous recorded stop history. The original gripper range ValueError becomes an explicit planning rejection and mandatory World isolation; measured state, bounds, tolerance, model, policy, physics and success criterion unchanged. Preserve one interrupted native terminal gap without replay.'
        reg['advisor33_policy']['measured_hand_range_rejection']='original bounds and1e-5m tolerance, reject and isolate, no clamp/retry'
    reg['learned_policy']['gpu_uuid']=roles['policy_uuid'];validate_trials(reg['seed_trials'])
    receipt={'previous':str(previous),'previous_registration_sha256':sha(previous/'registration.json'),
        'old_STOP_sha256':sha(previous/'STOP'),'completed_retained':sorted(done),'unissued_remaining':unused,
        'global_slots':30,'new_slot_budget':len(unused),'repeat_attempts':0,'unchanged_original_deadline':reg['deadline_unix'],
        'exit':exit_proof,'changed_sources':changed,'sampling_or_model_or_physics_or_success_change':False,
        'earlier_timeout_reused_World_before_correction':'a33_v02_r1_household_hope_milk_carton_plus15_full_stale',
        'following_reset_verified_but_teardown_missing':'a33_v03_r1_household_hope_milk_carton_minus15_fresh'}
    if stop_history_repair:
        for k in ('sampling_or_model_or_physics_or_success_change','earlier_timeout_reused_World_before_correction','following_reset_verified_but_teardown_missing'):receipt.pop(k)
        receipt.update(measured_history_capture_changed=True,thresholds_model_physics_success_unchanged=True,
            diagnostic=str(previous/'stop_history_diagnosis.json'),source_changes=source_changes,
            previous_failed_slots_not_repeated=True,mixed_runtime_versions_report_separately=True)
        for rel in ('scripts/curobo_arx/nearfield/advisor32_rules.py','scripts/curobo_arx/nearfield/advisor33_geometry.py','scripts/curobo_arx/nearfield/advisor33_policy.py'):
            if sha(prior/rel)!=sha(source/rel):raise RuntimeError('Frozen rules/geometry/policy changed')
        shutil.copy2(previous/'stop_history_diagnosis.json',out/'stop_history_diagnosis.json')
        deps[str(previous/'stop_history_diagnosis.json')]=sha(previous/'stop_history_diagnosis.json')
    if interruption_record:
        receipt.update(interrupted_consumed=sorted(interruptions),original_claim_not_replayed=True,
            runtime_changes='Exact original measured gripper range error is an explicit rejection; old unknown retained; failed slot always consumes its original budget')
        for saved in interruptions.values():deps.update(saved['evidence'])
        deps[str(record_path)]=sha(record_path)
        for saved in interruptions.values():
            clip=saved['detail']['video']
            if clip:deps[clip['path']]=clip['sha256']
    write(out/'continuation_receipt.json',receipt)
    for name in ('PLAN.md','research.md','sources.json','storage_policy.json','case_manifest.json','expected_resets.json','catalog.json','geometry_preflight.json'):
        src=previous/name
        if name=='geometry_preflight.json' and not src.exists() and stop_history_repair:
            src=Path(old['mask_diagnostic'])/'geometry_checks/geometry_preflight.json'
            geometry=read(src)
            if not geometry['pass'] or geometry['false_rejections'] or geometry['successful_preclose_states']!=4:
                raise RuntimeError('Native geometry certificate failed')
        shutil.copy2(src,out/name)
    write(out/'resource_plan.json',{'inventory':inventory,'roles':roles,'sim_minimum_free_mib':9000,
        'policy_minimum_free_mib':9500,'compute_minimum_free_mib':5000,'other_users_jobs_untouched':True})
    for p in (previous/'registration.json',previous/'STOP',previous/'summary.json',previous/'advisor33_verification.json'):
        deps[str(p)]=sha(p)
    deps.update(verification['evidence'])
    for p in out.iterdir():
        if p.is_file() and p.suffix in ('.json','.md'):deps[str(p)]=sha(p)
    probe_command=['podman','run','--pull=never','--rm','-i','--user','0','--security-opt=label=disable',
        '-v',str(ROOT.parent)+':'+str(ROOT.parent)+':rw','-v','/coolas/data/arx_teacher_zhaoyechong:/coolas/data/arx_teacher_zhaoyechong:ro',
        '-w',str(source),'--entrypoint','/isaac-sim/python.sh','localhost/arx-s02-isaac:6.0.1','-c',
        'import json,sys,pathlib;import scripts.curobo_arx.nearfield.runtime_advisor33; paths=json.load(sys.stdin); '
        'missing=[p for p in paths if not pathlib.Path(p).is_file()];print(json.dumps({"missing":missing}));sys.exit(bool(missing))']
    probe=subprocess.run(probe_command,input=json.dumps(list({*deps,*execution})),capture_output=True,text=True,timeout=120)
    write(out/'container_preflight.json',{'pass':probe.returncode==0,'stdout':probe.stdout,'stderr':probe.stderr,'new_physical_trials':0})
    if probe.returncode:raise RuntimeError('Container preflight failed')
    (out/'trial_ledgers').mkdir()
    for ident in unused:
        a=sqlite3.connect('file:'+str(previous/'trial_ledgers'/(ident+'.sqlite3'))+'?mode=ro',uri=True)
        b=sqlite3.connect(out/'trial_ledgers'/(ident+'.sqlite3'));a.backup(b);a.close();b.close()
    reg['dependencies']=deps;write(out/'registration.json',reg);verify_files({**execution,**deps})
    write(out/'full_preflight.json',{'pass':True,'checked':len({**execution,**deps}),'carried_completed':len(done),'carried_interrupted':len(interruptions),'unissued':len(unused)})
    (out/'research_log.md').write_text(f'Carry {len(done)} native terminal results and {len(interruptions)} interrupted slots; run only {len(unused)} unissued slots. No repeat; original30 total.\n'+reg['correction_scope']+'\n')
    write(out/'state.json',{'status':'prepared','completed':len(done),'interrupted':len(interruptions),'remaining':len(unused),'scheduled_slots':30,'new_physical_trials':0})
    return str(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('previous');p.add_argument('out');p.add_argument('--policy-index',required=True,type=int)
    p.add_argument('--compute-index',required=True,type=int);p.add_argument('--sim-index',required=True,type=int)
    p.add_argument('--stop-history-repair',action='store_true');p.add_argument('--interruption-record');a=p.parse_args()
    print(prepare(a.previous,a.out,a.policy_index,a.compute_index,a.sim_index,stop_history_repair=a.stop_history_repair,interruption_record=a.interruption_record))
