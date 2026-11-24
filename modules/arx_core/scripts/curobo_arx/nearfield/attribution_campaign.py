"""One72-slot matched experiment; immutable inherited execution and source receipts."""
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
from .closedloop30_rules import POLICY
from .attribution_spec import VERSION,make_trials,validate_trials,identity,ARMS

PREVIOUS=ROOT/'runs/nearfield_closedloop30_20261110_01'
PLAN=ROOT/'research_plan/docs/nearfield_g0_attribution_20261110.md'


def prepare(out,policy_index=0,compute_index=2,sim_index=3):
    out=Path(out).resolve()
    if out.exists():raise FileExistsError('Never overwrite a registration')
    old=read(PREVIOUS/'registration.json');result=read(PREVIOUS/'closedloop30_verification.json')
    if read(PREVIOUS/'state.json')['status']!='complete' or not result['record_integrity_pass'] or result['completed']!=24:
        raise RuntimeError('Previous experiment is not complete and verified')
    idle=predecessors([PREVIOUS]);trials=make_trials(old,result)
    out.mkdir(parents=True);space(out);write(out/'state.json',{'status':'preparing','physical_trials':0})
    added=sorted((ROOT/'scripts/curobo_arx/nearfield').glob('attribution_*.py'))
    added += [ROOT/'scripts/curobo_arx/nearfield/runtime_attribution.py',ROOT/'scripts/curobo_arx/nearfield/test_attribution.py']
    command=[sys.executable,'-m','unittest','scripts.curobo_arx.nearfield.test_attribution',
        'scripts.curobo_arx.nearfield.test_closedloop30','training.nearfield_arx.test_closure_recovery',
        'training.nearfield_arx.test_closure_records','scripts.curobo_arx.nearfield.test_learned','-v']
    check=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=240)
    (out/'offline_checks.log').write_text(check.stdout+check.stderr)
    write(out/'offline_checks.json',{'pass':check.returncode==0,'command':command,'log_sha256':sha(out/'offline_checks.log'),
        'sources':{str(p):sha(p) for p in added},'no_GPU_service_or_World_started':True})
    if check.returncode:raise RuntimeError('Offline checks failed')
    verify_files({**old['execution_source_files'],**old['dependencies']});verify_files(result['evidence'])
    prior=Path(old['source_snapshot']);source=out/'source_snapshot/COOLAS-ARX'
    shutil.copytree(prior,source,symlinks=True,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('tmp','curobo'):(source.parent/name).symlink_to(ROOT.parent/name,target_is_directory=True)
    execution={str(source/Path(p).relative_to(prior)) if Path(p).is_relative_to(prior) else p:h for p,h in old['execution_source_files'].items()}
    for p in added:
        dest=source/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);execution[str(dest)]=sha(dest)
    for p,h in old['execution_source_files'].items():
        dest=source/Path(p).relative_to(prior) if Path(p).is_relative_to(prior) else Path(p)
        if sha(dest)!=h:raise RuntimeError('Inherited execution bytes changed: '+str(dest))
    inventory=devices();roles={name:next(d for d in inventory if d['index']==index)
        for name,index in [('policy',policy_index),('compute',compute_index),('sim',sim_index)]}
    if len({d['uuid'] for d in roles.values()})!=3:raise ValueError('Separate policy/planner/simulator devices required')
    reg=copy.deepcopy(old)
    for key in ('carried_results','prior_results','prerequisite_results'):reg[key]=[]
    reg.update(stage=VERSION,created_unix=time.time(),campaign_started_unix=time.time(),
        deadline_unix=time.time()+30*3600,wall_hours=30,source_snapshot=str(source),
        execution_source_files=execution,source_identity=digest(execution),seed_trials=trials,
        max_physical_trials=72,global_physical_budget=72,max_replay_trials=0,replay_slots_total=0,recovery_slots_total=0,
        runtime_module='scripts.curobo_arx.nearfield.runtime_attribution',
        policy_module='scripts.curobo_arx.nearfield.attribution_policy',
        coordinator_module='scripts.curobo_arx.nearfield.attribution_campaign',
        requested_devices={k+'_uuid':v['uuid'] for k,v in roles.items()},policy_memory_floor_mib=12000,
        closedloop30_policy=POLICY,attribution_version=VERSION,training_authorized_here=False,
        no_automatic_retry=True,automatic_attempt_retries=0,automatic_startup_retries=0,
        execution_authority={'user_message':'测一下','scope':'G0/ARX-trained D/foundation model;12conditions x3arms x2prospective repeats=72, no training, no G0 removal'},
        previous=str(PREVIOUS),project_root=str(ROOT))
    reg['learned_policy']['gpu_uuid']=roles['policy']['uuid'];validate_checkpoints(reg)
    deps=copy.deepcopy(old['dependencies']);training=read(reg['learned_policy']['training_bindings']['D']['registration'])
    deps.update(training['model_files'])
    for p in (PREVIOUS/'registration.json',PREVIOUS/'closedloop30_verification.json',PREVIOUS/'final_idle_verification.json'):
        deps[str(p)]=sha(p)
    shutil.copy2(PLAN,out/'PLAN.md');shutil.copy2(PREVIOUS/'storage_policy.json',out/'storage_policy.json')
    write(out/'predecessor_exit_receipt.json',idle)
    write(out/'authorization.json',reg['execution_authority'])
    write(out/'case_manifest.json',{'slots':trials,'budget':72,'identities':{a:identity(reg,a) for a in ARMS},
        'entry_source':'PreviousD actual selected entry, identical for all arms and repetitions; no new search',
        'automatic_retries':0,'repetitions_registered_before_execution':True})
    write(out/'resource_plan.json',{'roles':roles,'free_memory_mib':{'policy':12000,'compute':6500,'sim':7500},
        'stable_samples':3,'interval_seconds':10,'other_users_jobs_untouched':True})
    write(out/'source_change_receipt.json',{'added':[str(p.relative_to(ROOT)) for p in added],
        'inherited_sources_byte_unchanged':True,'robot_physics_resets_rendering_unchanged':True,
        'task_and_strict_evaluators_unchanged_from_advisor30':True,'training_updates':0,
        'shared_change':'Fix previous selectedH across all three arms, disabling entry search',
        'variable':'Only local target generator: savedG0 versus trainedD versus foundationD'})
    (out/'research.md').write_text((out/'PLAN.md').read_text()+'\nEvaluator: scripts.curobo_arx.nearfield.attribution_report; keep_policy: no_automatic_promotion; max_iterations: 1; pause_every: never.\n')
    (out/'research_log.md').write_text('Registered three matched generators and72 slots before new execution.\n')
    write(out/'sources.json',{'primary_model_source':'https://huggingface.co/Qwen/Qwen3.5-2B',
        'checked_on':'2026-11-10','model_revision':training['model_revision'],
        'local_training':reg['learned_policy']['training_bindings']['D'],
        'hypothesis':'ARX-trained visual correction has positive paired task-success increment over the sharedG0 controller',
        'previous_results':str(PREVIOUS/'closedloop30_verification.json'),'classification':'User-authorized engineering ablation, no paper reproduction claim'})
    command=['podman','run','--pull=never','--rm','-i','--user','0','--security-opt=label=disable',
        '-v',str(ROOT.parent)+':'+str(ROOT.parent)+':rw',
        '-v','/coolas/data/arx_teacher_zhaoyechong:/coolas/data/arx_teacher_zhaoyechong:ro',
        '-w',str(source),'--entrypoint','/isaac-sim/python.sh','localhost/arx-s02-isaac:6.0.1','-c',
        'import json,pathlib,sys;import scripts.curobo_arx.nearfield.runtime_attribution;'
        'paths=json.load(sys.stdin);missing=[p for p in paths if not pathlib.Path(p).is_file()];'
        'print(json.dumps({"missing":missing,"checked":len(paths)}));sys.exit(bool(missing))']
    import json
    probe=subprocess.run(command,input=json.dumps(list({*execution,*deps})),capture_output=True,text=True,timeout=90)
    write(out/'container_preflight.json',{'pass':probe.returncode==0,'command':command,'stdout':probe.stdout,
        'stderr':probe.stderr,'GPU_or_World_started':False,'image_downloaded':False})
    if probe.returncode:raise RuntimeError('Frozen container imports or paths failed')
    for p in out.iterdir():
        if p.is_file() and p.name not in ('state.json','research_log.md'):deps[str(p)]=sha(p)
    reg['dependencies']=deps;ledgers(out,reg);write(out/'catalog.json',reg['catalog']);write(out/'registration.json',reg)
    verify_files({**execution,**deps});write(out/'state.json',{'status':'prepared','scheduled_slots':72,'physical_trials':0})
    return out


def run(campaign):
    from . import closure_physical,learned_report
    from .attribution_report import report
    root=Path(campaign).resolve();reg=read(root/'registration.json')
    lock=(root/'attribution.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'STOP').exists() or (root/'workers/policy').exists():raise RuntimeError('No restart or STOP bypass')
    validate_trials(reg['seed_trials']);validate_checkpoints(reg)
    verify_files({**reg['execution_source_files'],**reg['dependencies']})
    resource=read(root/'resource_plan.json');stable=0
    while stable<3:
        if (root/'STOP').exists() or time.time()>=reg['deadline_unix']:raise RuntimeError('Stopped before resource admission')
        available={d['uuid']:d['free_mb'] for d in devices()}
        stable=stable+1 if all(available.get(resource['roles'][k]['uuid'],0)>=v for k,v in resource['free_memory_mib'].items()) else 0
        write(root/'state.json',{'status':'waiting_for_resources','stable_samples':stable,'wall':time.time()})
        if stable<3:time.sleep(10)
    learned_report.report=report
    try:result=closure_physical.run(root)
    except BaseException:
        error=traceback.format_exc();exclusive_stop(root/'STOP',{'reason':'attribution_fault','error':error})
        write(root/'attribution_fault.json',{'error':error});raise
    write(root/'delivery_receipt.json',{'verification':str(root/'attribution_verification.json'),
        'sha256':sha(root/'attribution_verification.json'),'completed':result['completed'],'complete':result['complete']})
    with (root/'research_log.md').open('a') as f:f.write(f'Completed {result["completed"]}/72, successes {result["task_successes"]}; paired results in attribution_verification.json. No promotion.\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run'));p.add_argument('root')
    p.add_argument('--policy',type=int,default=0);p.add_argument('--compute',type=int,default=2);p.add_argument('--sim',type=int,default=3)
    a=p.parse_args()
    if a.action=='prepare':print(prepare(a.root,a.policy,a.compute,a.sim))
    else:run(a.root)
