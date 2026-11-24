"""Queue24 independent physical controls after the existing72-slot campaign."""
import argparse
import copy
import fcntl
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from scripts.curobo_arx.data_factory.common import ROOT,read,write,sha,digest,verify_files,space
from scripts.curobo_arx.data_factory.launch import devices
from scripts.curobo_arx.data_factory.watchdog import exclusive_stop
from training.nearfield_arx.lateral_campaign import ledgers
from .prepare_v2 import predecessors
from .closedloop30_policy import validate_checkpoints
from .visual_grounding_spec import VERSION,OBJECTS,ARMS,make_trials,validate_trials,expected_reset

PARENT=ROOT/'runs/nearfield_g0_attribution_20261110_01'
DONOR=ROOT/'runs/nearfield_closedloop30_20261110_01'
OFFLINE=ROOT/'runs/nearfield_visual_grounding_20261110_01'
PLAN=ROOT/'research_plan/docs/nearfield_visual_grounding_20261110.md'


def prepare(out):
    out=Path(out).resolve()
    if out.exists():raise FileExistsError('Never replace a registration')
    out.mkdir(parents=True);space(out);write(out/'state.json',{'status':'preparing','physical_trials':0})
    old=read(PARENT/'registration.json');donor_reg=read(DONOR/'registration.json')
    if not read(DONOR/'closedloop30_verification.json')['record_integrity_pass']:
        raise RuntimeError('Historical donor record integrity failed')
    trials,seeds=make_trials(donor_reg)
    command=[sys.executable,'-m','unittest','scripts.curobo_arx.nearfield.test_visual_grounding',
             'scripts.curobo_arx.nearfield.test_closedloop30','training.nearfield_arx.test_closure_records','-v']
    check=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=240)
    (out/'offline_checks.log').write_text(check.stdout+check.stderr)
    write(out/'offline_checks.json',{'pass':check.returncode==0,'command':command,'physical_trials':0,'log_sha256':sha(out/'offline_checks.log')})
    if check.returncode:raise RuntimeError('Offline physical-control checks failed')
    prior=Path(old['source_snapshot']);source=out/'source_snapshot/COOLAS-ARX'
    shutil.copytree(prior,source,symlinks=True,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('tmp','curobo'):(source.parent/name).symlink_to(ROOT.parent/name,target_is_directory=True)
    execution={str(source/Path(p).relative_to(prior)) if Path(p).is_relative_to(prior) else p:h for p,h in old['execution_source_files'].items()}
    added=sorted((ROOT/'scripts/curobo_arx/nearfield').glob('visual_grounding_*.py'))
    added += [ROOT/'scripts/curobo_arx/nearfield'/n for n in ('runtime_visual_grounding.py','test_visual_grounding.py')]
    added += [ROOT/'training/nearfield_arx/visual_causality.py']
    for p in added:
        dest=source/p.relative_to(ROOT);shutil.copy2(p,dest);execution[str(dest)]=sha(dest)
    deps=copy.deepcopy(old['dependencies']);donors={}
    for obj in OBJECTS:
        t=next(t for t in donor_reg['seed_trials'] if t['method']=='D' and t['condition']=='normal' and t['object_id']==obj)
        results=list(DONOR.glob('groups/*/trials/'+t['trial_id']+'/trial_result.json'))
        if len(results)!=1:raise RuntimeError('Missing unique historical donor trial')
        result=read(results[0]);ep=Path(result['attempts'][0]['episode']);first=read(ep/'policy_requests.json')[0]
        request=Path(first['request']);req=read(request);prompt=request.with_name(req['request_id']+'_input.json')
        images=read(prompt)['images'];files={str(p):sha(p) for p in [request,prompt,Path(req['observation']),*map(Path,images)]}
        donors[obj]={'source_trial_id':t['trial_id'],'source_episode':str(ep),'request':str(request),
            'images':images,'files':files,'G0_world':t['G0_world'],'entry_world':t['entry_world'],
            'scope':'Original-placement real first-decision wrist views, same object/model/camera/nominal entrance'}
        deps.update(files)
        for trial in trials:
            if trial['object_id']==obj and (trial['G0_world']!=t['G0_world'] or trial['entry_world']!=t['entry_world']):
                raise ValueError('Counterfactual prior or entrance changed')
    for p in (PARENT/'registration.json',DONOR/'registration.json',DONOR/'closedloop30_verification.json',OFFLINE/'registration.json'):
        deps[str(p)]=sha(p)
    reg=copy.deepcopy(old)
    for key in ('carried_results','prior_results','prerequisite_results'):reg[key]=[]
    reg.update(stage=VERSION,created_unix=time.time(),campaign_started_unix=time.time(),deadline_unix=time.time()+36*3600,
        wall_hours=36,source_snapshot=str(source),execution_source_files=execution,source_identity=digest(execution),
        seed_trials=trials,seeds=seeds,visual_donors=donors,max_physical_trials=24,global_physical_budget=24,
        max_replay_trials=0,replay_slots_total=0,recovery_slots_total=0,
        runtime_module='scripts.curobo_arx.nearfield.runtime_visual_grounding',
        policy_module='scripts.curobo_arx.nearfield.visual_grounding_policy',
        coordinator_module='scripts.curobo_arx.nearfield.visual_grounding_campaign',
        policy_memory_floor_mib=8000,training_authorized_here=False,no_automatic_retry=True,
        automatic_attempt_retries=0,automatic_startup_retries=0,visual_causality_parent=str(OFFLINE),
        wait_for_existing_campaign=str(PARENT),
        execution_authority={'user_message':'你能验证吗','scope':'Independent visual controls and new target-position tests; historical72 registration unchanged',
             'new_scenario_only_override':'Two original objects translated ±15mm along worldX; shared by all comparison arms; all other reset fields/physics fixed'},
        control_pose_matching={'translation_m':.005,'rotation_degrees':2.,'finger_joint_m':.0005,
             'apply_equally_to_all_arms':True,'failure_is_explicit_control_inapplicability':True})
    validate_checkpoints(reg);validate_trials(trials)
    shutil.copy2(PLAN,out/'PLAN.md');shutil.copy2(PARENT/'storage_policy.json',out/'storage_policy.json')
    write(out/'authorization.json',reg['execution_authority'])
    write(out/'case_manifest.json',{'slots':trials,'budget':24,'arms':ARMS,'automatic_retries':0,
          'no_existing_trial_rerun':True,'only_new_target_X_placements':True})
    write(out/'expected_resets.json',{t['case_id']:expected_reset(reg['seeds'][t['seed_id']],t) for t in trials})
    write(out/'source_change_receipt.json',{'added':[str(p.relative_to(ROOT)) for p in added],
        'inherited_execution_and_physical_evaluator_bytes_unchanged':True,
        'robot_and_scene_physics_unchanged':True,'new_registered_target_X_offsets_m':[-.015,.015],
        'historical_cases_and_current72_unchanged':True,'no_training':True,
        'stale_first_only_first_decision':True,'all_later_decisions_fresh':True})
    write(out/'resource_plan.json',{'roles':reg['requested_devices'],'free_memory_mib':{'policy':8000,'compute':6500,'sim':7500},
        'stable_samples':3,'interval_seconds':10,'other_users_jobs_untouched':True})
    (out/'research.md').write_text((out/'PLAN.md').read_text()+'\nPhysical phase:24 new slots, no automatic retries, only after parent72 completes and exits.\n')
    (out/'research_log.md').write_text('Registered24 new counterfactual slots; awaiting independent parent completion and offline record integrity.\n')
    write(out/'sources.json',{'hypothesis':'Fresh wrist evidence improves displaced-target actions and physical success over fixedG0 and a matched stale first view',
        'primary_sources':['https://arxiv.org/abs/2004.07780','https://arxiv.org/abs/2608.06154'],
        'checkpoint':reg['learned_policy']['checkpoints']['D'],'offline_diagnostic':str(OFFLINE),
        'new_object_generalization':False})
    # CPU-only imports in the existing image; no GPU/World or image download.
    command=['podman','run','--pull=never','--rm','-i','--user','0','--security-opt=label=disable',
        '-v',str(ROOT.parent)+':'+str(ROOT.parent)+':rw','-v','/coolas/data/arx_teacher_zhaoyechong:/coolas/data/arx_teacher_zhaoyechong:ro',
        '-w',str(source),'--entrypoint','/isaac-sim/python.sh','localhost/arx-s02-isaac:6.0.1','-c',
        'import scripts.curobo_arx.nearfield.runtime_visual_grounding;print("CPU_IMPORT_OK")']
    probe=subprocess.run(command,capture_output=True,text=True,timeout=120)
    write(out/'container_preflight.json',{'pass':probe.returncode==0,'command':command,'stdout':probe.stdout,'stderr':probe.stderr,'GPU_or_World_started':False})
    if probe.returncode:raise RuntimeError('Container import failed')
    for p in out.iterdir():
        if p.is_file() and p.name not in ('state.json','research_log.md'):deps[str(p)]=sha(p)
    reg['dependencies']=deps;ledgers(out,reg);write(out/'catalog.json',reg['catalog']);write(out/'registration.json',reg)
    verify_files({**execution,**deps});write(out/'state.json',{'status':'prepared','scheduled_slots':24,'physical_trials':0})
    return out


def prerequisites(parent,offline):
    """Read-only prerequisite gate. A completed marker alone is insufficient."""
    parent,offline=Path(parent),Path(offline)
    if (parent/'STOP').exists() or (offline/'STOP').exists():raise RuntimeError('Predecessor STOP retained; no bypass')
    p=read(parent/'state.json');o=read(offline/'state.json')
    if p['status']=='fault' or o['status']=='fault':raise RuntimeError('Predecessor fault')
    if not (parent/'delivery_receipt.json').exists() or o['status']!='complete':return False
    receipt=read(parent/'delivery_receipt.json');verification=parent/'attribution_verification.json'
    if receipt['sha256']!=sha(verification):raise RuntimeError('Predecessor delivery changed')
    old=read(verification);diagnostic=read(offline/'verification.json')
    if (not receipt['complete'] or not old['complete'] or old['completed']!=72 or not old['record_integrity_pass']
            or old['unexecuted'] or old['unknown_or_incomplete']):
        raise RuntimeError('Existing72 incomplete or damaged')
    if not diagnostic['complete'] or not diagnostic['record_integrity_pass'] or o['report_sha256']!=sha(offline/'verification.json'):
        raise RuntimeError('Offline diagnostic integrity incomplete')
    return True


def run(campaign):
    from . import closure_physical,learned_report
    from .visual_grounding_report import report
    root=Path(campaign).resolve();reg=read(root/'registration.json')
    lock=(root/'run.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'STOP').exists() or (root/'STARTED.json').exists():raise RuntimeError('No restart or STOP bypass')
    write(root/'STARTED.json',{'wall':time.time(),'registration_sha256':sha(root/'registration.json')})
    try:
        verify_files({**reg['execution_source_files'],**reg['dependencies']})
        while not prerequisites(reg['wait_for_existing_campaign'],reg['visual_causality_parent']):
            if (root/'STOP').exists() or time.time()>=reg['deadline_unix']:raise RuntimeError('Stopped or prerequisite deadline')
            write(root/'state.json',{'status':'waiting_for_existing72_and_visual_diagnostic','physical_trials':0,'wall':time.time()})
            time.sleep(30)
        idle=predecessors([Path(reg['wait_for_existing_campaign'])]);write(root/'predecessor_exit_receipt.json',idle)
        plan=read(root/'resource_plan.json');stable=0
        while stable<3:
            if (root/'STOP').exists() or time.time()>=reg['deadline_unix']:raise RuntimeError('Stopped or resource deadline')
            inventory={d['uuid']:d['free_mb'] for d in devices()}
            stable=stable+1 if all(inventory.get(reg['requested_devices'][k+'_uuid'],0)>=v for k,v in plan['free_memory_mib'].items()) else 0
            write(root/'state.json',{'status':'waiting_for_resources','stable_samples':stable,'physical_trials':0,'wall':time.time()})
            if stable<3:time.sleep(10)
        verify_files({**reg['execution_source_files'],**reg['dependencies']});validate_trials(reg['seed_trials'])
        learned_report.report=report;result=closure_physical.run(root)
        write(root/'delivery_receipt.json',{'complete':result['complete'],'completed':result['completed'],
             'sha256':sha(root/'visual_grounding_verification.json')})
        with (root/'research_log.md').open('a') as f:f.write(f'Completed {result["completed"]}/24; outcomes retained; no promotion.\n')
    except BaseException:
        error=traceback.format_exc();exclusive_stop(root/'STOP',{'reason':'visual_grounding_coordinator_failure','error':error})
        write(root/'FAULT.json',{'error':error});write(root/'state.json',{'status':'fault','wall':time.time()});raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run'));p.add_argument('root');a=p.parse_args()
    if a.action=='prepare':print(prepare(a.root))
    else:run(a.root)
