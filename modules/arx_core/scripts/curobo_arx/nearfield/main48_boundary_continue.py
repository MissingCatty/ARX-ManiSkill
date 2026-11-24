"""Record the stopped gripper-boundary slot without changing any physical gate."""
import argparse
import ast
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
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
from .main48_report import task_counts
from .main48_spec import validate_schedule
from .prepare_v2 import predecessors
from .runtime import audit_actions
from .capture import audit_episode
from .verify_physical import entry_matches_raw
from .verify_validation import reset_signature
from .closure import validate_reply


def scope(previous):
    previous=Path(previous).resolve();reg=read(previous/'registration.json');validate_schedule(reg)
    full=reg['registered_trials'];summary=read(previous/'summary.json')
    if read(previous/'state.json')['status']!='fault' or read(previous/'STOP')['reason']!='infrastructure/configuration/record stop':
        raise ValueError('Expected the original preserved infrastructure STOP')
    if reg['seed_trials']!=full[7:] or [d['trial_id'] for d in reg['carried_details']]!=[t['trial_id'] for t in full[:7]]:
        raise ValueError('Unexpected inherited original48 scope')
    if [t['trial_id'] for t in summary['trials']]!=[t['trial_id'] for t in full[7:21]]:
        raise ValueError('Completed native prefix changed')
    unknown=summary['incomplete_accounting']
    if len(unknown)!=1 or unknown[0]['trial_id']!=full[21]['trial_id'] or full[21]['method']!='C':
        raise ValueError('Unexpected interrupted slot')
    if {p.name for p in (previous/'groups').glob('*/trials/*') if p.is_dir()}!={t['trial_id'] for t in full[7:22]}:
        raise ValueError('A remaining slot has already started')
    if {p.stem for p in (previous/'trial_ledgers').glob('*.sqlite3')}!={t['trial_id'] for t in full[7:]}:
        raise ValueError('Missing or foreign ledgers')
    pins={}
    for i,t in enumerate(full[7:]):
        p=previous/'trial_ledgers'/(t['trial_id']+'.sqlite3');rows=ledger_rows(p);pins[str(p)]=sha(p)
        if i<14 and rows!=summary['trials'][i]['attempts']:raise ValueError('Native result differs from ledger')
        if i==14:
            if len(rows)!=1 or rows[0]['physical']!=1 or rows[0]['state']!='executing' or rows[0]['outcome'] is not None:
                raise ValueError('Interrupted action budget changed')
            claim=rows[0]
        if i>14 and rows:raise ValueError('Never repeat a claimed or issued slot')
    return {'full':full,'claim':claim,'remaining':full[22:],'files':pins,'consumed':22}


def interrupted_C(previous,accounting):
    previous=Path(previous).resolve();reg=read(previous/'registration.json');trial=accounting['full'][21]
    ep=Path(accounting['claim']['episode']);group=ep.parents[3]
    fault=read(previous/'workers/planner/FAULT.json');group_fault=read(group/'FAULT.json')
    message='ValueError: Measured gripper position is outside the validated model range'
    if fault['ok'] is not False or not fault['error'].rstrip().endswith(message) or not group_fault['error'].rstrip().endswith(message):
        raise ValueError('Unexpected interruption cause')
    shutdown=read(group/'shutdown_receipt.json')
    if not shutdown['all_trajectories_idle'] or not shutdown['raw_writer_drained_and_closed']:raise ValueError('Old execution/writer not idle')
    if (ep/'physical_receipt.json').exists() or (ep.parent.parent/'trial_result.json').exists():raise ValueError('Never replace native terminal evidence')
    labels=['pregrasp','state_generation',*[f'c_step_{i:03d}' for i in range(1,5)],'close']
    requests=read(ep/'requests.json');actions=audit_actions(ep,requests)
    if [r['label'] for r in requests]!=labels or set(actions['executed'])!=set(labels):raise ValueError('Actual action prefix changed')
    for suffix in ('_command.json','_execution.json'):
        if {p.name.removesuffix(suffix) for p in ep.glob('*'+suffix)}!=set(labels):raise ValueError('Unaccounted or missing action record')
    planner_paths=list((previous/'workers/planner/requests').glob('*_'+fault['id']+'.done'))
    if len(planner_paths)!=1:raise ValueError('Missing unique failed lift request')
    planner_path=planner_paths[0];request=read(planner_path)
    response_path=previous/'workers/planner/responses'/(fault['id']+'.json')
    if read(response_path)!=fault or request['id']!=fault['id'] or request['action']!='plan' or request['phase']!='lift' or request['label']!='lift' or Path(request['output'])!=ep:
        raise ValueError('Planner failure is not bound to the interrupted lift')
    close=read(ep/'close_execution.json')['final'];snapshot=request['snapshot']
    for key in ('joint_names','joint_positions','joint_velocities','joint_limits','sim_time'):
        if snapshot[key]!=close[key]:raise ValueError('Lift request differs from measured closed state')
    candidate=read(ep/'lift_goal_search/00/request.json')
    if candidate['snapshot']!=snapshot or candidate['id']!=request['id'] or (ep/'lift_goal_search/00/reply.json').exists():
        raise ValueError('Unexpected interrupted lift search evidence')
    model=previous/'workers/planner/models/model_closed/arx.urdf'
    joints={j.attrib['name']:j for j in ET.parse(model).getroot().findall('joint')};boundary=[]
    for name in ('right_joint27','right_joint28'):
        i=snapshot['joint_names'].index(name);value=snapshot['joint_positions'][i];limit=joints[name].find('limit')
        lo,hi=float(limit.attrib['lower']),float(limit.attrib['upper'])
        if [lo,hi]!=snapshot['joint_limits'][i]:raise ValueError('Runtime and planning gripper limits differ')
        boundary.append({'joint':name,'measured_m':value,'lower_m':lo,'upper_m':hi,'existing_tolerance_m':1e-5,
                         'passes_existing_model_range':lo-1e-5<=value<=hi+1e-5,'over_upper_m':max(0.,value-hi)})
    if [x['joint'] for x in boundary if not x['passes_existing_model_range']]!=['right_joint27']:raise ValueError('Boundary observation changed')
    policy=audit_policy(ep);events=read(ep/'interpreter_events.json')
    if policy['policy_calls']!=5 or policy['interpreter_decisions']!=5 or policy['closure_origins']!=['residual_deadzone']:
        raise ValueError('Unexpected learned action history')
    physical=evaluate(ep,sequence=[('pregrasp' if n=='pregrasp' else 'close' if n=='close' else 'approach',n) for n in labels])
    if physical['pass'] or set(physical['missing_stages'])!={'lift','hold'} or physical['issues'] or physical['forbidden_contact_events'] or not physical['bilateral_closure']:
        raise ValueError('Observed physical prefix differs')
    capture=audit_episode(ep);raw=[json.loads(l) for l in (ep/'samples.jsonl').read_text().splitlines()]
    entry=[e for e in events if e['kind']=='physical_entry_stopped']
    if len(entry)!=1:raise ValueError('Missing unique physical entry')
    entry=entry[0];generation=read(ep/'state_generation_execution.json')
    if entry['registered_G0_world']!=trial['G0_world'] or entry['registered_H_world']!=trial['entry_world'] or state_time(raw[entry['raw_index']])<=generation['final']['sim_time']+1e-6:
        raise ValueError('Changed or stale physical entry')
    entry_matches_raw(raw[entry['raw_index']],read(ep/'physical_entry_observation.json'))
    if reset_signature(read(ep/'initial_conditions.json'))!=reset_signature(read(Path(reg['seeds'][trial['seed_id']]['episode'])/'initial_conditions.json')):
        raise ValueError('Historical reset changed')
    drift=pose_error(generation['final']['objects']['test_object']['world'],read(ep/'capture_snapshot.json')['objects']['test_object']['world'])
    target=generation['final']['objects']['test_object']['path']
    if drift['translation_m']>.001 or drift['rotation_rad']>.01 or any(c['num_contacts']>0 and target in c['a']+c['b'] for c in generation['contacts']):
        raise ValueError('Physical entry moved/touched target')
    convergence=[]
    for item in policy['requests']:
        ident=item['request_id'];path=ep/'policy'/(ident+'_request.json');req,rep=read_bound_reply(path,path.with_name(ident+'_reply.json'))
        if req['registration_sha256']!=sha(previous/'registration.json') or req['inputs']['G0_world']!=trial['G0_world'] or rep['checkpoint']!=reg['learned_policy']['checkpoints']['C']:
            raise ValueError('Policy registration/model/goal binding changed')
        entry_guard.validate_receipt(path,req,rep,reg['learned_policy']['visual_contract']['depth_scale']['near_m'])
        entry_matches_raw(raw[req['raw_index']],read(req['observation']))
        convergence.append({**item,'diagnostic_to_historical_Gstar':pose_error(req['inputs']['eef_world'],trial['source_Gstar_world'])})
    binding=read(ep/'closure_gate_optimization.json')['binding']
    if not validate_reply(binding,read(ep/'closure_model_validation.json'),read(ep/(binding['previous_command_label']+'_command.json'))):
        raise ValueError('Independent closure gate binding changed')
    verify_files(binding['files'])
    frozen=list((ep.parent.parent/'blocks').glob('*/frozen_*.json'))
    if not frozen:raise ValueError('Missing frozen-property proof')
    for p in frozen:
        proof=read(p)
        if not proof['pass'] or proof['mismatches'] or not proof['additional_assets']['pass']:raise ValueError('Physical configuration changed')
    pins={str(p):sha(p) for p in ep.rglob('*') if p.is_file()}
    pins.update({str(p):sha(p) for p in [*frozen,planner_path,response_path,model,previous/'workers/planner/FAULT.json',group/'FAULT.json',group/'shutdown_receipt.json']})
    flags={k:False for k in ('entry_motion','fresh_entry','model_query','geometry_query','policy_motion','baseline_motion','GRASP','closure_check','closure_gate','close','lift','hold')}
    flags.update(entry_motion=True,fresh_entry=True,model_query=True,policy_motion=True,GRASP=True,closure_check=True,closure_gate=True,close=True)
    detail={'trial_id':trial['trial_id'],'object_id':trial['object_id'],'condition':trial['condition'],'case_id':trial['case_id'],
        'split':trial['source_split'],'method':'C','physical_attempt':True,'physical_success':False,'admitted':False,
        'task_result':'infrastructure_interrupted','outcome':'infrastructure_interrupted: measured_gripper_outside_frozen_model_range',
        'record_integrity_pass':True,'native_runtime_completed':False,'native_completion_record_present':False,
        'supplemental_accounting_verified':True,'method_grasp_outcome_observed':False,
        'native_gap':'Native terminal result/physical receipt absent after closed-state model range rejection; old ledger remains executing; no terminal receipt synthesized',
        'record_integrity_scope':'Occurred action/request/capture chain and failed planning request verified; native terminal gap remains explicit',
        'source_episode':str(ep),'wall_seconds':None,'nearfield_wall_seconds':None,
        'observed_active_attempt_until_shutdown_seconds':shutdown['wall']-accounting['claim']['started'],
        'nearfield_sim_seconds':raw[-1]['capture_sim_time']-entry['sim_time'],'event_applicability':flags,
        'capture':capture,'actions':actions,'policy_calls':5,'geometry_calls':0,'geometry_queries':[],
        'convergence':convergence,'historical_reset_unchanged':True,'physical':physical,'video':None,
        'gripper_boundary':boundary,'lift_planning_failed_before_dispatch':True,'new_action_dispatched_by_recheck':False}
    return detail,pins


def record(previous,out):
    previous=Path(previous).resolve();out=Path(out).resolve()
    if out.exists():raise FileExistsError('Preserve prior boundary record')
    accounting=scope(previous);idle=predecessors([previous]);stage_recovery.install()
    detail,pins=interrupted_C(previous,accounting);out.mkdir(parents=True)
    from .report import video
    detail['video']=video(Path(detail['source_episode']),out/'C_interrupted_before_lift.mp4')
    write(out/'boundary_record.json',{'detail':detail,'original_claim':accounting['claim'],'evidence':pins,
        'old_STOP_and_records_unchanged':True,'consumed_never_retried':True,'joint_limits_tolerance_physics_unchanged':True})
    write(out/'scope.json',{'consumed':22,'remaining_trial_ids':[t['trial_id'] for t in accounting['remaining']],
        'ledger_evidence':accounting['files'],'old_execution_exit':idle,'new_physical_trials':0})
    return out


def prepare(previous,out,record_dir,role_indices,defer_d=False):
    """Carry22 original claims; execute only unclaimed slots, with optional D pause."""
    previous=Path(previous).resolve();out=Path(out).resolve();record_dir=Path(record_dir).resolve()
    if out.exists():raise FileExistsError('Preserve all prior continuations')
    old=read(previous/'registration.json')
    if time.time()>=old['deadline_unix']:raise TimeoutError('Original48 deadline expired')
    accounting=scope(previous);idle=predecessors([previous]);validate_checkpoints(old)
    verify_files({**old['execution_source_files'],**old['dependencies'],**old['carried_evidence']})
    independent=read(previous/'main48_verification.json');verify_files(independent['evidence'])
    if [d['trial_id'] for d in independent['trials']]!=[t['trial_id'] for t in accounting['full'][:21]]:
        raise ValueError('Independently verified completed prefix differs')
    if len(independent['unknown_or_incomplete'])!=1 or independent['unknown_or_incomplete'][0]['trial_id']!=accounting['full'][21]['trial_id']:
        raise ValueError('Unexpected independent terminal gaps')
    if independent['actual_attempts']!=22 or independent['physical_successes']!=8 or independent['physical_failures']!=11 or independent['infrastructure_interruptions']!=2:
        raise ValueError('Original outcomes changed')
    saved=read(record_dir/'boundary_record.json');saved_scope=read(record_dir/'scope.json')
    verify_files({**saved['evidence'],**saved_scope['ledger_evidence']})
    if saved['original_claim']!=accounting['claim'] or saved_scope['remaining_trial_ids']!=[t['trial_id'] for t in accounting['remaining']]:
        raise ValueError('Supplement does not account for current claims')
    stage_recovery.install();detail,pins=interrupted_C(previous,accounting)
    detail['video']=saved['detail']['video']
    if detail!=saved['detail'] or pins!=saved['evidence']:raise ValueError('Independent interrupted-prefix recheck differs')
    if not detail['video']['decode_pass']:raise ValueError('Interrupted video did not decode')
    verify_files({detail['video']['path']:detail['video']['sha256']})
    carried=copy.deepcopy(independent['trials'])
    for d in carried[7:]:d['native_runtime_completed']=True
    carried.append(detail)
    pins.update(independent['evidence']);pins.update(accounting['files'])
    for p in record_dir.rglob('*'):
        if p.is_file():pins[str(p)]=sha(p)
    # Pin completed raw records as well as their prior audit, without rewriting it.
    for t in read(previous/'summary.json')['trials']:
        for claim in t['attempts']:
            pins.update({str(p):sha(p) for p in Path(claim['episode']).rglob('*') if p.is_file()})
    pending=[t for t in accounting['remaining'] if not defer_d or t['method']!='D']
    deferred=[t for t in accounting['remaining'] if defer_d and t['method']=='D']
    out.mkdir(parents=True);space(out)
    write(out/'state.json',{'status':'preparing','consumed':22,'remaining_slots':len(pending),'deferred_slots':len(deferred)})
    write(out/'predecessor_exit_receipt.json',idle)
    write(out/'supplemental_C_accounting.json',saved)
    for name in ('storage_policy.json','advisor29.md','PLAN.md','case_manifest.json'):shutil.copy2(previous/name,out/name)
    issue=ROOT/'research_plan/docs/nearfield_main48_gripper_boundary_20261110.md'
    shutil.copy2(issue,out/'INTERRUPTION.md')
    added=['scripts/curobo_arx/nearfield/'+n for n in ('main48_boundary_continue.py','main48_spec.py','main48_report.py',
        'main48_campaign.py','main48_follow.py','test_main48_deferral.py')]
    checks=[sys.executable,'-m','unittest','training.nearfield_arx.test_receipt_numbers','training.nearfield_arx.test_entry_guard',
        'training.nearfield_arx.test_stage_recovery','training.nearfield_arx.test_closure_records','training.nearfield_arx.test_guard_rejection',
        'scripts.curobo_arx.nearfield.test_main48','scripts.curobo_arx.nearfield.test_main48_deferral','-v']
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
        if rel not in changed and sha(dest)!=h:raise ValueError('Unexpected inherited execution-source change')
    def funcs(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    a=funcs(prior/'scripts/curobo_arx/nearfield/main48_spec.py');b=funcs(source/'scripts/curobo_arx/nearfield/main48_spec.py')
    if any(b[k]!=v for k,v in a.items() if k!='validate_schedule'):raise ValueError('Original case generation or B algorithm changed')
    inv=devices();by_index={d['index']:d for d in inv};roles={k:by_index[v] for k,v in role_indices.items()};floors={}
    for role,n in [('policy',12000),('geometry',3500),('compute',6500),('sim',7500)]:
        u=roles[role]['uuid'];floors[u]=floors.get(u,0)+n
    totals=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.total','--format=csv,noheader,nounits'],text=True,timeout=15)
    capacity={l.split(',')[0].strip():int(l.split(',')[1]) for l in totals.splitlines()}
    if any(v>capacity[u] for u,v in floors.items()):raise ValueError('Insufficient combined device capacity')
    write(out/'resource_plan.json',{'roles':roles,'combined_free_memory_mib':floors,'stable_samples':3,'interval_seconds':10,'other_users_jobs_untouched':True})
    deps={**old['dependencies'],**old['execution_source_files'],**pins}
    for p in previous.iterdir():
        if p.is_file() and (p.suffix=='.json' or p.name=='STOP'):deps[str(p)]=sha(p)
    reg=copy.deepcopy(old);reg.update(created_unix=time.time(),boundary_continuation_of=str(previous),
        source_snapshot=str(source),execution_source_files=execution,source_identity=digest(execution),dependencies=deps,
        seed_trials=pending,deferred_trials=deferred,carried_details=carried,carried_evidence=pins,
        consumed_physical_budget=22,max_physical_trials=len(pending),
        requested_devices={k+'_uuid':v['uuid'] for k,v in roles.items()})
    reg['learned_policy']['gpu_uuid']=roles['policy']['uuid'];reg['geometry_baseline']['gpu_uuid']=roles['geometry']['uuid']
    if defer_d:
        reg['execution_amendment']={'kind':'user_defer_remaining_D','deferred_methods':['D'],
            'user_message':'继续，停止d的先跳过去，先记录下来',
            'interpretation':'Temporarily interpret d as method D; continue remaining original A/B/C in relative order',
            'separate_scope_confirmation_received':False,'deferred_are_unexecuted_not_failures':True,
            'preserve_original48_denominator':True,'new_attempts_or_retries':0}
    validate_schedule(reg);(out/'trial_ledgers').mkdir();(out/'deferred_trial_ledgers').mkdir()
    for t in accounting['remaining']:
        original=previous/'trial_ledgers'/(t['trial_id']+'.sqlite3')
        target=out/('deferred_trial_ledgers' if t in deferred else 'trial_ledgers')/original.name
        shutil.copy2(original,target)
        if ledger_rows(target) or sha(target)!=sha(original):raise ValueError('Unissued ledger copy changed')
        if t in deferred:deps[str(target)]=sha(target)
    write(out/'source_change_receipt.json',{'changes':changed,'only_scheduling_reporting_and_interruption_accounting_changed':True,
        'other_execution_sources_byte_identical':True,'models_robot_scene_initial_render_success_unchanged':True,
        'gripper_limits_range_tolerance_and_measured_state_unchanged':True,'old_STOPs_and_native_terminal_gaps_preserved':True,
        'new_training_updates':0,'registered_budget':48,'consumed':22,'remaining_active':len(pending),'deferred_unexecuted':len(deferred),
        'retries':0,'task_counts_carried':task_counts(carried),'deadline_unix':old['deadline_unix'],
        'boundary_record':str(record_dir),'independent_prefix_verification':str(previous/'main48_verification.json')})
    command=['podman','run','--pull=never','--rm','-i','--user','0','--security-opt=label=disable',
        '-v',str(ROOT.parent)+':'+str(ROOT.parent)+':rw','-v','/coolas/data/arx_teacher_zhaoyechong:/coolas/data/arx_teacher_zhaoyechong:ro',
        '-w',str(source),'--entrypoint','/isaac-sim/python.sh','localhost/arx-s02-isaac:6.0.1','-c',
        'import json,sys,pathlib,unittest,os;from scripts.curobo_arx.nearfield.main48_boundary_continue import scope,interrupted_C;'
        'from training.nearfield_arx import stage_recovery;stage_recovery.install();p=json.load(sys.stdin);'
        'os.environ["ARX_TEST_FIXTURE_ROOT"]=p["fixture_root"];d,_=interrupted_C(p["previous"],scope(p["previous"]));'
        'tests=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromName("scripts.curobo_arx.nearfield.test_main48_deferral"));'
        'missing=[x for x in p["paths"] if not pathlib.Path(x).is_file()];'
        'print(json.dumps({"missing":missing,"paths":len(p["paths"]),"C_calls":d["policy_calls"],"C_lift_not_dispatched":d["lift_planning_failed_before_dispatch"],"schedule_tests":tests.testsRun,"schedule_tests_pass":tests.wasSuccessful()}));sys.exit(bool(missing) or not tests.wasSuccessful())']
    probe=subprocess.run(command,input=json.dumps({'previous':str(previous),'fixture_root':str(ROOT),'paths':list({*execution,*deps})}),capture_output=True,text=True,timeout=180)
    write(out/'container_preflight.json',{'pass':probe.returncode==0,'command':command,'stdout':probe.stdout,'stderr':probe.stderr,'GPU_or_World_started':False,'image_downloaded':False})
    if probe.returncode:raise RuntimeError('Frozen container prefix/schedule audit failed')
    for p in out.iterdir():
        if p.is_file() and p.name!='state.json':deps[str(p)]=sha(p)
    write(out/'catalog.json',reg['catalog']);write(out/'registration.json',reg)
    verify_files({**execution,**deps});predecessors([previous])
    write(out/'state.json',{'status':'prepared','registered':48,'consumed':22,'remaining_slots':len(pending),
        'deferred_slots':len(deferred),**task_counts(carried)})
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('previous');p.add_argument('out');p.add_argument('--continue-record');p.add_argument('--defer-d',action='store_true')
    for role in ('policy','geometry','compute','sim'):p.add_argument('--'+role,type=int)
    a=p.parse_args()
    print(prepare(a.previous,a.out,a.continue_record,{k:getattr(a,k) for k in ('policy','geometry','compute','sim')},a.defer_d)
        if a.continue_record else record(a.previous,a.out))
