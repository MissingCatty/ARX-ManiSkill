"""One controlled continuation of original48 after the cross-SciPy audit fault.

Never edits the old ledger, receipts or STOP. The stopped B slot is consumed,
with an explicitly supplemental no-candidate failure, not a fabricated native
terminal receipt. Only the 46 demonstrably unissued slots can be scheduled.
"""
import argparse
import ast
import copy
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
import numpy as np
from scripts.curobo_arx.data_factory.common import ROOT,read,write,sha,digest,verify_files,space
from scripts.curobo_arx.data_factory.launch import devices
from scripts.curobo_arx.lilo_vla.egad_evaluate import evaluate
from training.nearfield_arx.contract import pose_error
from training.nearfield_arx.export import state_time
from training.nearfield_arx.stage_policy import validate_checkpoints
from .main48_audit import audit_baseline,same_selection
from .main48_spec import validate_trials,validate_schedule,select_candidate
from .prepare_v2 import predecessors
from .runtime import audit_actions
from .capture import audit_episode
from .verify_validation import reset_signature
from .verify_physical import entry_matches_raw


def ledger_rows(path):
    db=sqlite3.connect('file:'+str(Path(path).resolve())+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    try:return [dict(r) for r in db.execute('SELECT * FROM attempts ORDER BY started,id')]
    finally:db.close()


def partition(previous):
    previous=Path(previous);reg=read(previous/'registration.json');full=reg['seed_trials'];validate_trials(full)
    if reg.get('registered_trials') or reg.get('carried_details'):raise ValueError('This repair is scoped to the first original48 registration')
    state=read(previous/'state.json')
    if state['status']!='fault' or state.get('unconfirmed_world') or not (previous/'STOP').is_file():
        raise ValueError('A stopped, conclusively exited predecessor is required')
    if read(previous/'STOP')['reason'] not in ('infrastructure/configuration/record stop','main48 infrastructure/record/idle fault'):
        raise ValueError('A manual or unrelated STOP cannot be continued by this repair')
    completed=read(previous/'main48_verification.json')['trials']
    if len(completed)!=1 or completed[0]['trial_id']!=full[0]['trial_id'] or not completed[0]['physical_success']:
        raise ValueError('Unexpected completed-result scope')
    folders={p.name for p in (previous/'groups').glob('*/trials/*') if p.is_dir()}
    if folders!={t['trial_id'] for t in full[:2]}:raise ValueError('Unexpected started trial folder')
    if {p.stem for p in (previous/'trial_ledgers').glob('*.sqlite3')}!={t['trial_id'] for t in full}:
        raise ValueError('Original48 ledgers are incomplete')
    claims=[];pins={}
    for index,trial in enumerate(full):
        path=previous/'trial_ledgers'/(trial['trial_id']+'.sqlite3');rows=ledger_rows(path);pins[str(path)]=sha(path)
        if index>=2 and rows:raise ValueError('A remaining slot was already claimed; never replay it')
        if index<2:
            if len(rows)!=1 or rows[0]['physical']!=1:raise ValueError('Unexpected consumed slot')
            claims.append(rows[0])
    if claims[0]['state']!='committed' or claims[0]['outcome']!='success':raise ValueError('A original result changed')
    if claims[1]['state']!='executing' or claims[1]['outcome'] is not None:raise ValueError('B original native gap changed')
    return {'registered':full,'consumed_claims':claims,'remaining':full[2:],'ledger_files':pins}


def numeric_probe(previous):
    """CPU only: independently recalculate the failed B selection in this env."""
    previous=Path(previous);reg=read(previous/'registration.json');trial=reg['seed_trials'][1]
    claim=ledger_rows(previous/'trial_ledgers'/(trial['trial_id']+'.sqlite3'))[0];episode=Path(claim['episode'])
    result=audit_baseline(episode,trial);ident=result['geometry_queries'][0]['request_id']
    request=read(episode/'geometry'/(ident+'_request.json'));proof=episode/'geometry'/(ident+'_evidence')
    selected=select_candidate(np.load(proof/'filtered_predictions.npy',allow_pickle=False),
        read(request['observation'])['camera_world_optical'],request['G0_world'],np.load(proof/'target_points.npy',allow_pickle=False))
    recorded=read(proof/'candidate_selection.json');differences=[]
    def walk(a,b,path=''):
        if isinstance(a,dict):
            for k in a:walk(a[k],b[k],path+'/'+k)
        elif isinstance(a,list):
            for i,(x,y) in enumerate(zip(a,b)):walk(x,y,path+'/'+str(i))
        elif type(a) is float and a!=b:differences.append({'path':path,'recomputed':a,'recorded':b,'absolute_difference':abs(a-b)})
    if not same_selection(selected,recorded):raise ValueError('Decision or material numeric change')
    walk(selected,recorded)
    if selected['selected'] is not None or selected['accepted']:raise ValueError('Expected an ordinary no-candidate failure')
    import scipy
    return {'pass':True,'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
        'differences':differences,'exact_candidate_decisions':True,'selected':None,'accepted_count':0,
        'rejected_count':len(selected['rejected']),'GPU_or_World_started':False}


def supplement(previous,partitioned):
    previous=Path(previous);reg=read(previous/'registration.json');trial=partitioned['registered'][1]
    claim=partitioned['consumed_claims'][1];episode=Path(claim['episode']);group=episode.parents[3]
    fault=read(group/'FAULT.json');shutdown=read(group/'shutdown_receipt.json')
    if not fault['error'].rstrip().endswith('ValueError: Geometry candidate ranking/conversion changed'):
        raise ValueError('The original fault is outside this repair scope')
    if not shutdown['all_trajectories_idle'] or not shutdown['raw_writer_drained_and_closed']:
        raise ValueError('Cannot supplement an unconfirmed active action/writer')
    if (episode/'physical_receipt.json').exists() or (episode.parent.parent/'trial_result.json').exists():
        raise ValueError('Do not replace a native terminal result')
    requests=read(episode/'requests.json');actions=audit_actions(episode,requests)
    if [r['label'] for r in requests]!=['pregrasp','state_generation'] or set(actions['executed'])!={'pregrasp','state_generation'}:
        raise ValueError('Unexpected actions occurred in the failed B slot')
    if {p.name.removesuffix('_command.json') for p in episode.glob('*_command.json')}!=set(actions['executed']):
        raise ValueError('Unaccounted command file')
    if read(episode/'policy_requests.json') or list((episode/'policy').glob('*')):
        raise ValueError('B must not issue a learned-policy request')
    events=read(episode/'interpreter_events.json')
    if [e['kind'] for e in events]!=['offset_stopped','physical_entry_stopped','geometry_reply']:
        raise ValueError('Unexpected or incomplete B event sequence')
    policy=audit_baseline(episode,trial)
    if policy['geometry_calls']!=1:raise ValueError('Unexpected B query count')
    query=policy['geometry_queries'][0];ident=query['request_id'];folder=episode/'geometry'
    request=read(folder/(ident+'_request.json'));reply=read(folder/(ident+'_reply.json'))
    if (query['selected'] is not None or query['dispatched_event'] or reply['accepted'] or
        reply['rejected']!='no_target_candidate_in_G0_neighborhood' or not reply['projection']['target_visibility_pass'] or
        not read(folder/(ident+'_dispatch_check.json'))['pass']):raise ValueError('No-candidate terminal outcome is not conclusively established')
    if request['registration_sha256']!=sha(previous/'registration.json') or events[-1]['geometry_reply_sha256']!=sha(folder/(ident+'_reply.json')):
        raise ValueError('Wrong request/consumed-reply binding')
    raw=[json.loads(l) for l in (episode/'samples.jsonl').read_text().splitlines()];capture=audit_episode(episode)
    entry=events[1];row=raw[entry['raw_index']];generation=read(episode/'state_generation_execution.json')
    if state_time(row)<=generation['final']['sim_time']+1e-6:raise ValueError('Stale stopped-entry frame')
    if entry['registered_G0_world']!=trial['G0_world'] or entry['registered_H_world']!=trial['entry_world']:
        raise ValueError('Entry or prior changed')
    entry_matches_raw(row,read(episode/'physical_entry_observation.json'))
    entry_matches_raw(raw[request['raw_index']],read(request['observation']))
    if reset_signature(read(episode/'initial_conditions.json'))!=reset_signature(read(Path(reg['seeds'][trial['seed_id']]['episode'])/'initial_conditions.json')):
        raise ValueError('Historical reset changed')
    drift=pose_error(generation['final']['objects']['test_object']['world'],read(episode/'capture_snapshot.json')['objects']['test_object']['world'])
    target=generation['final']['objects']['test_object']['path']
    if drift['translation_m']>.001 or drift['rotation_rad']>.01 or any(c['num_contacts']>0 and target in c['a']+c['b'] for c in generation['contacts']):
        raise ValueError('Entry moved/touched target')
    physical=evaluate(episode,sequence=[('pregrasp','pregrasp'),('approach','state_generation')])
    if physical['pass'] or set(physical['missing_stages'])!={'close','lift','hold'} or physical['converged_stages']!=['pregrasp','state_generation'] or physical['issues'] or physical['forbidden_contact_events']:
        raise ValueError('Original physical evaluator found an additional unresolved issue')
    frozen=list((episode.parent.parent/'blocks').glob('*/frozen_*.json'))
    if not frozen:raise ValueError('Missing runtime robot/scene evidence')
    for file in frozen:
        proof=read(file)
        if not proof['pass'] or proof['mismatches'] or not proof['additional_assets']['pass']:raise ValueError('Frozen configuration changed')
    # Bind every retained input, RGB/depth, action and original failure byte.
    files={str(p):sha(p) for p in episode.rglob('*') if p.is_file()}
    files.update({str(p):sha(p) for p in [*frozen,group/'FAULT.json',group/'shutdown_receipt.json']})
    flags={k:False for k in ('entry_motion','fresh_entry','model_query','geometry_query','policy_motion','baseline_motion','GRASP','closure_check','closure_gate','close','lift','hold')}
    flags.update(entry_motion=True,fresh_entry=True,geometry_query=True)
    detail={'trial_id':trial['trial_id'],'object_id':trial['object_id'],'condition':trial['condition'],'case_id':trial['case_id'],
        'split':trial['source_split'],'method':'B','physical_success':False,'admitted':False,'physical_attempt':True,
        'outcome':'geometry_no_candidate: no_target_candidate_in_G0_neighborhood','record_integrity_pass':True,
        'native_runtime_completed':False,'native_completion_record_present':False,'supplemental_accounting_verified':True,
        'native_gap':'Original ledger remains executing; native physical_receipt and trial_result are absent. No old files repaired or invented.',
        'record_integrity_scope':'All actually occurred actions, captures and geometry requests verified; native terminal gap remains separately reported',
        'wall_seconds':None,'wall_seconds_missing_reason':'Original trial start/end timer did not reach native terminal writing',
        'observed_active_attempt_until_shutdown_seconds':shutdown['wall']-claim['started'],
        'nearfield_wall_seconds':None,'observed_entry_until_shutdown_seconds':shutdown['wall']-entry['wall_unix'],
        'nearfield_sim_seconds':raw[-1]['capture_sim_time']-entry['sim_time'],
        'event_applicability':flags,'capture':capture,'actions':actions,'policy_calls':0,'geometry_calls':1,
        'geometry_queries':policy['geometry_queries'],'convergence':[],'historical_reset_unchanged':True,
        'physical':physical,'video':None,'source_episode':str(episode)}
    return detail,files


def prepare(previous,out):
    previous,out=Path(previous).resolve(),Path(out).resolve()
    if out.exists():raise FileExistsError('Preserve prior registrations')
    old=read(previous/'registration.json')
    if time.time()>=old['deadline_unix']:raise TimeoutError('Original deadline expired')
    partitioned=partition(previous);idle=predecessors([previous]);validate_checkpoints(old)
    verify_files({**old['execution_source_files'],**old['dependencies']})
    prior_report=read(previous/'main48_verification.json');verify_files(prior_report['evidence'])
    b,pins=supplement(previous,partitioned);a=copy.deepcopy(prior_report['trials'][0]);a['native_runtime_completed']=True
    pins.update(prior_report['evidence']);pins.update(partitioned['ledger_files'])
    a_episode=Path(partitioned['consumed_claims'][0]['episode'])
    quality=read(a_episode/'quality_receipt.json')
    if not quality['pass']:raise ValueError('Carried A admission is not valid')
    verify_files({str(a_episode/n):h for n,h in quality['files'].items()})
    pins.update({str(p):sha(p) for p in a_episode.rglob('*') if p.is_file()})
    for p in [previous/n for n in ('registration.json','STOP','state.json','summary.json','main48_verification.json','coordinator_fault.json','FAULT.json','final_idle_verification.json')]:pins[str(p)]=sha(p)
    if a['video']:
        if sha(a['video']['path'])!=a['video']['sha256']:raise ValueError('A video changed')
        pins[a['video']['path']]=a['video']['sha256']
    out.mkdir(parents=True);space(out);write(out/'state.json',{'status':'preparing','carried_slots':2,'remaining_slots':46})
    write(out/'predecessor_exit_receipt.json',idle)
    write(out/'host_numeric_recheck.json',numeric_probe(previous))
    from .report import video
    b['video']=video(Path(b['source_episode']),out/'supplements'/('B_entry_only.mp4'))
    write(out/'supplemental_B_accounting.json',{'pass':True,'detail':b,'original_ledger':partitioned['consumed_claims'][1],
        'evidence':pins,'original_STOP_and_native_gap_preserved':True,'attempt_consumed_never_replayed':True})
    pins[str(out/'supplemental_B_accounting.json')]=sha(out/'supplemental_B_accounting.json')
    if b['video']:pins[b['video']['path']]=b['video']['sha256']
    for name in ('storage_policy.json','advisor29.md','PLAN.md','case_manifest.json','resource_plan.json'):
        shutil.copy2(previous/name,out/name)
    write(out/'resource_inventory.json',{'devices':devices(),'roles_unchanged':old['requested_devices']})
    added=[ROOT/'scripts/curobo_arx/nearfield'/n for n in ('main48_audit.py','main48_spec.py','main48_campaign.py','main48_report.py','main48_continue.py','test_main48.py')]
    cmd=[sys.executable,'-m','unittest','scripts.curobo_arx.nearfield.test_main48',
        'training.nearfield_arx.test_stage_recovery','training.nearfield_arx.test_closure_recovery',
        'training.nearfield_arx.test_closure_records','scripts.curobo_arx.nearfield.test_learned','-v']
    test=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=240)
    (out/'offline_checks.log').write_text(test.stdout+test.stderr)
    write(out/'offline_checks.json',{'pass':test.returncode==0,'command':cmd,'log_sha256':sha(out/'offline_checks.log'),
        'sources':{str(p):sha(p) for p in added},'no_service_or_World_started':True})
    if test.returncode:raise RuntimeError('Offline regression tests failed')
    prior=Path(old['source_snapshot']);source=out/'source_snapshot/COOLAS-ARX'
    shutil.copytree(prior,source,symlinks=True,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('tmp','curobo'):(source.parent/name).symlink_to(ROOT.parent/name,target_is_directory=True)
    execution={str(source/Path(p).relative_to(prior)) if Path(p).is_relative_to(prior) else p:h for p,h in old['execution_source_files'].items()}
    changes={}
    for p in added:
        dest=source/p.relative_to(ROOT);before=sha(dest) if dest.exists() else None
        shutil.copy2(p,dest);execution[str(dest)]=sha(dest)
        changes[str(p.relative_to(ROOT))]={'before':before,'after':sha(dest)}
    for p,h in old['execution_source_files'].items():
        dest=source/Path(p).relative_to(prior) if Path(p).is_relative_to(prior) else Path(p)
        relative=str(Path(p).relative_to(prior)) if Path(p).is_relative_to(prior) else p
        if relative not in changes and sha(dest)!=h:raise ValueError('Unexpected inherited source change: '+relative)
    # The only spec edit adds schedule accounting. Selection and case construction
    # must be exactly the original AST, not merely claim equivalent thresholds.
    def functions(path):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
    before=functions(prior/'scripts/curobo_arx/nearfield/main48_spec.py');after=functions(source/'scripts/curobo_arx/nearfield/main48_spec.py')
    if any(after.get(k)!=v for k,v in before.items()):raise ValueError('Original case/geometry algorithm changed')
    reg=copy.deepcopy(old)
    deps={**old['dependencies'],**old['execution_source_files'],**pins}
    reg.update(created_unix=time.time(),continuation_of=str(previous),source_snapshot=str(source),execution_source_files=execution,
        source_identity=digest(execution),registered_trials=partitioned['registered'],seed_trials=partitioned['remaining'],
        carried_details=[a,b],carried_evidence=pins,consumed_physical_budget=2,max_physical_trials=46,global_physical_budget=48,
        dependencies=deps)
    validate_schedule(reg)
    # Only demonstrably empty ledgers are copied; A/B remain solely in old root.
    (out/'trial_ledgers').mkdir()
    for t in reg['seed_trials']:
        src=previous/'trial_ledgers'/(t['trial_id']+'.sqlite3');dest=out/'trial_ledgers'/src.name
        shutil.copy2(src,dest)
        if ledger_rows(dest) or sha(src)!=sha(dest):raise ValueError('Unissued ledger copy differs')
    write(out/'source_change_receipt.json',{'changes':changes,'scope':'Audit float portability and explicit original48 accounting only',
        'rotation_diagnostic_absolute_tolerance_rad':1e-7,'other_float_absolute_tolerance':1e-12,
        'candidate_membership_order_indices_reasons_exact':True,'original_selection_and_case_functions_AST_identical':True,
        'all_other_execution_sources_byte_identical':True,'C_D_checkpoints_unchanged':True,'new_training_updates':0,
        'old_STOP_sha256':sha(previous/'STOP'),'original_total_budget':48,'consumed':2,'remaining':46,'replayed_attempts':0,
        'deadline_unix':old['deadline_unix'],'native_terminal_gap_retained':b['trial_id']})
    code=('import json,pathlib,sys;import scripts.curobo_arx.nearfield.runtime_main48;'
        'from scripts.curobo_arx.nearfield.main48_continue import numeric_probe;payload=json.load(sys.stdin);'
        'missing=[p for p in payload["paths"] if not pathlib.Path(p).is_file()];'
        'print(json.dumps({"missing":missing,"checked":len(payload["paths"]),"numeric":numeric_probe(payload["previous"])}));sys.exit(bool(missing))')
    command=['podman','run','--pull=never','--rm','-i','--user','0','--security-opt=label=disable',
        '-v',str(ROOT.parent)+':'+str(ROOT.parent)+':rw','-v','/coolas/data/arx_teacher_zhaoyechong:/coolas/data/arx_teacher_zhaoyechong:ro',
        '-w',str(source),'--entrypoint','/isaac-sim/python.sh','localhost/arx-s02-isaac:6.0.1','-c',code]
    probe=subprocess.run(command,input=json.dumps({'paths':list({*execution,*deps}),'previous':str(previous)}),capture_output=True,text=True,timeout=120)
    write(out/'container_preflight.json',{'pass':probe.returncode==0,'command':command,'stdout':probe.stdout,'stderr':probe.stderr,
        'GPU_or_World_started':False,'image_downloaded':False})
    if probe.returncode:raise RuntimeError('Frozen Isaac CPU import/evidence/selection check failed')
    for p in out.iterdir():
        if p.is_file() and p.name!='state.json':deps[str(p)]=sha(p)
    write(out/'catalog.json',reg['catalog']);write(out/'registration.json',reg)
    verify_files({**execution,**deps});predecessors([previous])
    write(out/'state.json',{'status':'prepared','registered_slots':48,'carried_slots':2,'remaining_slots':46,
        'physical_successes_carried':1,'supplemental_failed_slots':1,'new_physical_attempts':0})
    return str(out)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('out');parser.add_argument('--previous',required=True)
    args=parser.parse_args();print(prepare(args.previous,args.out))
