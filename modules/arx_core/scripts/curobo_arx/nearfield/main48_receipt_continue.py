"""Consume known interrupted D slots once; preserve original48 scope and evidence."""
import argparse
import ast
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from scripts.curobo_arx.data_factory.common import ROOT,read,write,sha,digest,verify_files,space
from scripts.curobo_arx.data_factory.launch import devices
from scripts.curobo_arx.lilo_vla.egad_evaluate import evaluate
from training.nearfield_arx import stage_recovery,entry_guard
from training.nearfield_arx.contract import pose_error
from training.nearfield_arx.export import state_time
from training.nearfield_arx.learned_policy import read_bound_reply
from training.nearfield_arx.stage_verification import audit_policy
from training.nearfield_arx.stage_policy import validate_checkpoints
from .main48_continue import ledger_rows
from .main48_report import report,task_counts
from .main48_spec import validate_schedule
from .prepare_v2 import predecessors
from .runtime import audit_actions
from .capture import audit_episode
from .verify_physical import entry_matches_raw
from .verify_validation import reset_signature


def consumed_scope(previous):
    previous=Path(previous);reg=read(previous/'registration.json');validate_schedule(reg)
    if read(previous/'state.json')['status']!='fault' or read(previous/'STOP')['reason']!='infrastructure/configuration/record stop':
        raise ValueError('This continuation requires the preserved infrastructure STOP')
    full=reg['registered_trials']
    carried=len(reg['carried_details'])
    if carried not in (2,4) or reg['seed_trials']!=full[carried:] or [d['trial_id'] for d in reg['carried_details']]!=[t['trial_id'] for t in full[:carried]]:
        raise ValueError('Unexpected original48 continuation scope')
    completed=1 if carried==2 else 2
    consumed=carried+completed+1
    fault_kind='numeric_admission_audit' if carried==2 else 'missing_D_rejection_reason'
    summary=read(previous/'summary.json')
    if [t['trial_id'] for t in summary['trials']]!=[t['trial_id'] for t in full[carried:consumed-1]]:
        raise ValueError('Unexpected completed native slots')
    if len(summary['incomplete_accounting'])!=1 or summary['incomplete_accounting'][0]['trial_id']!=full[consumed-1]['trial_id']:
        raise ValueError('Unexpected incomplete slots')
    trial_dirs={p.name for p in (previous/'groups').glob('*/trials/*') if p.is_dir()}
    if trial_dirs!={t['trial_id'] for t in full[carried:consumed]}:raise ValueError('A remaining slot has already started')
    if {p.stem for p in (previous/'trial_ledgers').glob('*.sqlite3')}!={t['trial_id'] for t in reg['seed_trials']}:
        raise ValueError('Missing/foreign ledgers')
    pins={};claims=[]
    for i,t in enumerate(reg['seed_trials']):
        p=previous/'trial_ledgers'/(t['trial_id']+'.sqlite3');rows=ledger_rows(p);pins[str(p)]=sha(p)
        if i>completed and rows:raise ValueError('Never replay an issued or claimed slot')
        if i<=completed:
            if len(rows)!=1 or rows[0]['physical']!=1:raise ValueError('Unexpected consumed action budget')
            if i<completed and rows!=summary['trials'][i]['attempts']:raise ValueError('Native result and ledger differ')
            claims.append(rows[0])
    expected=['planning_rejected: no_ik_solution'] if carried==2 else ['geometry_no_candidate: no_target_candidate_in_G0_neighborhood','success']
    if [r['outcome'] for r in claims[:-1]]!=expected or claims[-1]['state']!='executing' or claims[-1]['outcome'] is not None:
        raise ValueError('Original native outcomes changed')
    return {'full':full,'claims':claims,'remaining':full[consumed:],'files':pins,'carried':carried,
            'completed_native':completed,'consumed':consumed,'fault_kind':fault_kind}


def interrupted_D(previous,scope):
    previous=Path(previous);reg=read(previous/'registration.json');trial=scope['full'][scope['consumed']-1]
    claim=scope['claims'][-1];episode=Path(claim['episode']);group=episode.parents[3]
    missing_reason=scope['fault_kind']=='missing_D_rejection_reason'
    fault=read(group/'FAULT.json');shutdown=read(group/'shutdown_receipt.json')
    expected_fault="KeyError: 'reason'" if missing_reason else 'RuntimeError: Action-admission receipt differs from deterministic gate'
    if not fault['error'].rstrip().endswith(expected_fault):
        raise ValueError('Unrecognized D interruption')
    if not shutdown['all_trajectories_idle'] or not shutdown['raw_writer_drained_and_closed']:
        raise ValueError('Old execution/writer not idle')
    if (episode/'physical_receipt.json').exists() or (episode.parent.parent/'trial_result.json').exists():
        raise ValueError('Never replace native terminal evidence')
    requests=read(episode/'requests.json');actions=audit_actions(episode,requests)
    labels=['pregrasp','state_generation']+(['d_refine_001'] if missing_reason else [])
    if [r['label'] for r in requests]!=labels or set(actions['executed'])!=set(labels):
        raise ValueError('D has additional actual actions')
    if {f.name.removesuffix('_command.json') for f in episode.glob('*_command.json')}!=set(actions['executed']):
        raise ValueError('Unaccounted command')
    if any(not p.name.startswith('d_refine_001_') or not missing_reason for p in episode.glob('d_refine_*')) or (episode/'closure_model_validation.json').exists():raise ValueError('Unaccounted nearfield planning/closure')
    events=read(episode/'interpreter_events.json')
    kinds=['offset_stopped','physical_entry_stopped']+(['policy_interpreter_decision','target_request','policy_interpreter_decision'] if missing_reason else [])
    if [e['kind'] for e in events]!=kinds:raise ValueError('Unexpected D event sequence')
    policy=audit_policy(episode)
    if policy['policy_calls']!=(2 if missing_reason else 1) or policy['interpreter_decisions']!=(2 if missing_reason else 0) or policy['requests'][-1]['dispatched_event']:
        raise ValueError('Unexpected interrupted policy history')
    if missing_reason:
        if not policy['requests'][0]['dispatched_event'] or not read(episode/'d_refine_001_settling.json')['accepted']:
            raise ValueError('First D correction lacks accepted physical execution')
        last=events[-1]['closure_trigger']
        if last['request_closure'] or last['motion_budget_exhausted']:raise ValueError('Unexpected terminal interpretation')
    ident=policy['requests'][-1]['request_id'];path=episode/'policy'/(ident+'_request.json')
    req,reply=read_bound_reply(path,path.with_name(ident+'_reply.json'))
    if req['registration_sha256']!=sha(previous/'registration.json') or req['method']!='D' or req['inputs']['G0_world']!=trial['G0_world']:
        raise ValueError('Wrong registered policy request')
    if reply['checkpoint']!=reg['learned_policy']['checkpoints']['D'] or not reply['projection']['target_visibility_pass']:
        raise ValueError('Model/view binding differs')
    if not read(path.with_name(ident+'_dispatch_check.json'))['pass']:raise ValueError('Observation moved before validation')
    guard=entry_guard.validate_receipt(path,req,reply,reg['learned_policy']['visual_contract']['depth_scale']['near_m'])
    if guard['pass']==missing_reason:raise ValueError('Interrupted admission result changed')
    rejection=entry_guard.rejection_reason('D',guard) if missing_reason else None
    if missing_reason and rejection!='D_target_outside_original_40mm_region_and_not_inward':raise ValueError('Unexpected D rejection')
    raw=[json.loads(l) for l in (episode/'samples.jsonl').read_text().splitlines()];capture=audit_episode(episode)
    entry=events[1];generation=read(episode/'state_generation_execution.json')
    if state_time(raw[entry['raw_index']])<=generation['final']['sim_time']+1e-6:raise ValueError('Stale entry')
    if entry['registered_G0_world']!=trial['G0_world'] or entry['registered_H_world']!=trial['entry_world']:raise ValueError('Entry changed')
    entry_matches_raw(raw[entry['raw_index']],read(episode/'physical_entry_observation.json'))
    entry_matches_raw(raw[req['raw_index']],read(req['observation']))
    if reset_signature(read(episode/'initial_conditions.json'))!=reset_signature(read(Path(reg['seeds'][trial['seed_id']]['episode'])/'initial_conditions.json')):
        raise ValueError('Historical reset changed')
    drift=pose_error(generation['final']['objects']['test_object']['world'],read(episode/'capture_snapshot.json')['objects']['test_object']['world'])
    target=generation['final']['objects']['test_object']['path']
    if drift['translation_m']>.001 or drift['rotation_rad']>.01 or any(c['num_contacts']>0 and target in c['a']+c['b'] for c in generation['contacts']):
        raise ValueError('Entry moved/touched target')
    physical=evaluate(episode,sequence=[('pregrasp' if n=='pregrasp' else 'approach',n) for n in labels])
    if physical['pass'] or set(physical['missing_stages'])!={'close','lift','hold'} or physical['issues'] or physical['forbidden_contact_events']:
        raise ValueError('Unexpected physical evidence')
    frozen=list((episode.parent.parent/'blocks').glob('*/frozen_*.json'))
    if not frozen:raise ValueError('Missing runtime properties')
    for p in frozen:
        proof=read(p)
        if not proof['pass'] or proof['mismatches'] or not proof['additional_assets']['pass']:raise ValueError('Physical configuration changed')
    pins={str(p):sha(p) for p in episode.rglob('*') if p.is_file()}
    pins.update({str(p):sha(p) for p in [*frozen,group/'FAULT.json',group/'shutdown_receipt.json']})
    flags={k:False for k in ('entry_motion','fresh_entry','model_query','geometry_query','policy_motion','baseline_motion','GRASP','closure_check','closure_gate','close','lift','hold')}
    flags.update(entry_motion=True,fresh_entry=True,model_query=True,policy_motion=missing_reason)
    return {'trial_id':trial['trial_id'],'object_id':trial['object_id'],'condition':trial['condition'],'case_id':trial['case_id'],
        'split':trial['source_split'],'method':'D','physical_attempt':True,'physical_success':False,'admitted':False,
        'task_result':'infrastructure_interrupted','outcome':'infrastructure_interrupted: '+scope['fault_kind'],
        'record_integrity_pass':True,'native_runtime_completed':False,'native_completion_record_present':False,
        'supplemental_accounting_verified':True,'method_grasp_outcome_observed':False,
        'native_gap':'Original ledger remains executing; native terminal receipt absent after pre-dispatch exception; never synthesized',
        'record_integrity_scope':'Occurred action/request/capture chain verified; task interruption and missing terminal receipt remain explicit',
        'source_episode':str(episode),'wall_seconds':None,'nearfield_wall_seconds':None,
        'observed_active_attempt_until_shutdown_seconds':shutdown['wall']-claim['started'],
        'observed_entry_until_shutdown_seconds':shutdown['wall']-entry['wall_unix'],
        'nearfield_sim_seconds':raw[-1]['capture_sim_time']-entry['sim_time'],'event_applicability':flags,
        'capture':capture,'actions':actions,'policy_calls':policy['policy_calls'],'geometry_calls':0,'geometry_queries':[],
        'convergence':[],'historical_reset_unchanged':True,'physical':physical,'video':None,
        'policy_requests_verified':policy['requests'],'posthoc_guard_agrees':True,'last_admission_pass':guard['pass'],
        'verified_rejection_reason':rejection,'interrupted_request_dispatched':False,
        'closure_origins':policy['closure_origins'],'new_action_dispatched_by_recheck':False},pins


def prepare(previous,out,role_indices):
    previous,out=Path(previous).resolve(),Path(out).resolve()
    if out.exists():raise FileExistsError('Preserve every registration')
    old=read(previous/'registration.json')
    if time.time()>=old['deadline_unix']:raise TimeoutError('Original deadline expired')
    scope=consumed_scope(previous);idle=predecessors([previous]);validate_checkpoints(old)
    verify_files({**old['execution_source_files'],**old['dependencies'],**old['carried_evidence']})
    stage_recovery.install();detail,pins=interrupted_D(previous,scope)
    consumed=scope['consumed'];remaining=len(scope['remaining'])
    out.mkdir(parents=True);space(out);write(out/'state.json',{'status':'preparing','consumed':consumed,'remaining_slots':remaining})
    independent=report(previous,out/'predecessor_recheck')
    if [d['trial_id'] for d in independent['trials']]!=[t['trial_id'] for t in scope['full'][:consumed-1]]:
        raise ValueError('Native prefix not independently accounted')
    carried=independent['trials']
    for d in carried[scope['carried']:]:d['native_runtime_completed']=True
    from .report import video
    detail['video']=video(Path(detail['source_episode']),out/'supplements/D_interrupted_prefix.mp4')
    write(out/'supplemental_D_accounting.json',{'detail':detail,'original_claim':scope['claims'][-1],
        'evidence':pins,'old_STOP_and_records_unchanged':True,'consumed_never_retried':True})
    carried.append(detail);pins.update(independent['evidence']);pins.update(scope['files'])
    for claim in scope['claims'][:-1]:pins.update({str(p):sha(p) for p in Path(claim['episode']).rglob('*') if p.is_file()})
    pins.update({str(p):sha(p) for p in (out/'predecessor_recheck').rglob('*') if p.is_file()})
    pins.update({str(p):sha(p) for p in (out/'supplements').rglob('*') if p.is_file()})
    pins[str(out/'supplemental_D_accounting.json')]=sha(out/'supplemental_D_accounting.json')
    write(out/'predecessor_exit_receipt.json',idle)
    for name in ('storage_policy.json','advisor29.md','PLAN.md','case_manifest.json'):shutil.copy2(previous/name,out/name)
    added=['training/nearfield_arx/'+n for n in ('entry_guard.py','stage_verification.py','receipt_numbers.py','test_receipt_numbers.py','test_guard_rejection.py')]
    added+=['scripts/curobo_arx/nearfield/'+n for n in ('main48_report.py','main48_receipt_continue.py','runtime_closure_recovery.py')]
    checks=[sys.executable,'-m','unittest','training.nearfield_arx.test_receipt_numbers','training.nearfield_arx.test_entry_guard',
        'training.nearfield_arx.test_stage_recovery','training.nearfield_arx.test_closure_records','training.nearfield_arx.test_guard_rejection','scripts.curobo_arx.nearfield.test_main48','-v']
    result=subprocess.run(checks,cwd=ROOT,capture_output=True,text=True,timeout=180)
    (out/'offline_checks.log').write_text(result.stdout+result.stderr)
    write(out/'offline_checks.json',{'pass':result.returncode==0,'command':checks,'log_sha256':sha(out/'offline_checks.log'),
        'sources':{str(ROOT/n):sha(ROOT/n) for n in added},'GPU_or_World_started':False})
    if result.returncode:raise RuntimeError('Offline checks failed')
    prior=Path(old['source_snapshot']);source=out/'source_snapshot/COOLAS-ARX'
    shutil.copytree(prior,source,symlinks=True,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('tmp','curobo'):(source.parent/name).symlink_to(ROOT.parent/name,target_is_directory=True)
    execution={str(source/Path(p).relative_to(prior)) if Path(p).is_relative_to(prior) else p:h for p,h in old['execution_source_files'].items()}
    changed={}
    for n in added:
        dest=source/n;changed[n]={'before':sha(dest) if dest.exists() else None,'after':sha(ROOT/n)}
        shutil.copy2(ROOT/n,dest);execution[str(dest)]=sha(dest)
    for p,h in old['execution_source_files'].items():
        dest=source/Path(p).relative_to(prior) if Path(p).is_relative_to(prior) else Path(p)
        rel=str(Path(p).relative_to(prior)) if Path(p).is_relative_to(prior) else p
        if rel not in changed and sha(dest)!=h:raise ValueError('Unexpected inherited source change')
    # Only receipt validation changes in the policy module; guard arithmetic,
    # parsing, target generation and all physical thresholds must be identical.
    def funcs(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    before=funcs(prior/'training/nearfield_arx/entry_guard.py');after=funcs(source/'training/nearfield_arx/entry_guard.py')
    if any(after[k]!=v for k,v in before.items() if k!='validate_receipt'):raise ValueError('Action generation or admission rule changed')
    runtime_rel='scripts/curobo_arx/nearfield/runtime_closure_recovery.py'
    expected_runtime=(prior/runtime_rel).read_text().replace('import validate_receipt\n','import validate_receipt, rejection_reason\n').replace("'reason':guard['reason']","'reason':rejection_reason(method,guard)")
    if (source/runtime_rel).read_text()!=expected_runtime:raise ValueError('Interpreter changed beyond rejection reason logging')
    inv=devices();by_index={d['index']:d for d in inv};roles={k:by_index[v] for k,v in role_indices.items()};floors={}
    for role,n in [('policy',12000),('geometry',3500),('compute',6500),('sim',7500)]:
        u=roles[role]['uuid'];floors[u]=floors.get(u,0)+n
    totals=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.total','--format=csv,noheader,nounits'],text=True,timeout=15)
    capacity={l.split(',')[0].strip():int(l.split(',')[1]) for l in totals.splitlines()}
    if any(v>capacity[u] for u,v in floors.items()):raise ValueError('Insufficient device capacity')
    write(out/'resource_plan.json',{'roles':roles,'combined_free_memory_mib':floors,'stable_samples':3,'interval_seconds':10,'other_users_jobs_untouched':True})
    deps={**old['dependencies'],**old['execution_source_files'],**pins}
    for p in previous.iterdir():
        if p.is_file() and (p.suffix=='.json' or p.name=='STOP'):deps[str(p)]=sha(p)
    reg=copy.deepcopy(old);reg.update(created_unix=time.time(),receipt_continuation_of=str(previous),
        source_snapshot=str(source),execution_source_files=execution,source_identity=digest(execution),dependencies=deps,
        seed_trials=scope['remaining'],carried_details=carried,carried_evidence=pins,consumed_physical_budget=consumed,max_physical_trials=remaining,
        requested_devices={k+'_uuid':v['uuid'] for k,v in roles.items()})
    reg['learned_policy']['gpu_uuid']=roles['policy']['uuid'];reg['geometry_baseline']['gpu_uuid']=roles['geometry']['uuid']
    validate_schedule(reg);(out/'trial_ledgers').mkdir()
    for t in reg['seed_trials']:
        original=previous/'trial_ledgers'/(t['trial_id']+'.sqlite3');target=out/'trial_ledgers'/original.name
        shutil.copy2(original,target)
        if ledger_rows(target) or sha(target)!=sha(original):raise ValueError('Unissued ledger copy changed')
    write(out/'source_change_receipt.json',{'changes':changed,'only_derived_pose_angle_diagnostic_comparison_tolerance_rad':1e-7,
        'goal_request_hashes_action_vectors_gate_flags_counters_and_decisions_exact':True,'action_generation_and_admission_AST_unchanged':True,
        'other_execution_sources_byte_identical':True,'models_robot_scene_initial_render_success_unchanged':True,
        'rejection_record_fix_only_no_admission_rule_change':True,
        'new_training_updates':0,'registered_budget':48,'consumed':consumed,'remaining':remaining,'retries':0,
        'task_counts_carried':task_counts(carried),'deadline_unix':old['deadline_unix']})
    command=['podman','run','--pull=never','--rm','-i','--user','0','--security-opt=label=disable',
        '-v',str(ROOT.parent)+':'+str(ROOT.parent)+':rw','-v','/coolas/data/arx_teacher_zhaoyechong:/coolas/data/arx_teacher_zhaoyechong:ro',
        '-w',str(source),'--entrypoint','/isaac-sim/python.sh','localhost/arx-s02-isaac:6.0.1','-c',
        'import json,sys,pathlib,unittest,os;from scripts.curobo_arx.nearfield.main48_receipt_continue import consumed_scope,interrupted_D;'
        'from training.nearfield_arx import stage_recovery;from training.nearfield_arx.stage_verification import audit_policy;'
        'stage_recovery.install();p=json.load(sys.stdin);os.environ["ARX_TEST_FIXTURE_ROOT"]=p["fixture_root"];scope=consumed_scope(p["previous"]);'
        'd,_=interrupted_D(p["previous"],scope);'
        'tests=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromName("training.nearfield_arx.test_guard_rejection"));'
        'missing=[x for x in p["paths"] if not pathlib.Path(x).is_file()];'
        'print(json.dumps({"missing":missing,"paths":len(p["paths"]),"D_calls":d["policy_calls"],"D_guard":d["posthoc_guard_agrees"],"D_last_not_dispatched":not d["interrupted_request_dispatched"],"consumer_tests":tests.testsRun,"consumer_tests_pass":tests.wasSuccessful()}));sys.exit(bool(missing) or not tests.wasSuccessful())']
    probe=subprocess.run(command,input=json.dumps({'previous':str(previous),'fixture_root':str(ROOT),'paths':list({*execution,*deps})}),capture_output=True,text=True,timeout=120)
    write(out/'container_preflight.json',{'pass':probe.returncode==0,'command':command,'stdout':probe.stdout,'stderr':probe.stderr,'GPU_or_World_started':False,'image_downloaded':False})
    if probe.returncode:raise RuntimeError('Frozen container C/D audit failed')
    for p in out.iterdir():
        if p.is_file() and p.name!='state.json':deps[str(p)]=sha(p)
    write(out/'catalog.json',reg['catalog']);write(out/'registration.json',reg)
    verify_files({**execution,**deps});predecessors([previous])
    write(out/'state.json',{'status':'prepared','registered':48,'consumed':consumed,'remaining_slots':remaining,**task_counts(carried)})
    return str(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out');p.add_argument('--previous',required=True)
    for role in ('policy','geometry','compute','sim'):p.add_argument('--'+role,type=int,required=True)
    a=p.parse_args();print(prepare(a.previous,a.out,{k:getattr(a,k) for k in ('policy','geometry','compute','sim')}))
