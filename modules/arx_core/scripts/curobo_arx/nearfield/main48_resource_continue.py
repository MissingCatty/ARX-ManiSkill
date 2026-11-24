"""Manual resource-only continuation; no retry of a claimed original48 slot."""
import argparse
import copy
from pathlib import Path
import shutil
import subprocess
import sys
import time
from scripts.curobo_arx.data_factory.common import ROOT,read,write,sha,verify_files,digest,space
from scripts.curobo_arx.data_factory.launch import devices
from training.nearfield_arx.lateral_campaign import ledgers
from training.nearfield_arx.stage_policy import validate_checkpoints
from .main48_continue import ledger_rows
from .main48_spec import validate_schedule
from .prepare_v2 import predecessors


def unissued(previous):
    previous=Path(previous);reg=read(previous/'registration.json');validate_schedule(reg)
    if read(previous/'state.json')['status']!='fault' or not (previous/'STOP').is_file():
        raise ValueError('Preserve the concluded resource STOP')
    stopped=read(previous/'STOP')
    if stopped['reason']!='main48 infrastructure/record/idle fault' or 'ERROR_OUT_OF_DEVICE_MEMORY' not in stopped['error']:
        raise ValueError('Manual/unrelated stops cannot be resumed here')
    if not read(previous/'resource_failure_observed.json')['before_next_physical_dispatch']:
        raise ValueError('Not a pre-dispatch resource failure')
    if len(reg['seed_trials'])!=46 or len(reg['carried_details'])!=2 or reg['max_physical_trials']!=46:
        raise ValueError('Only the original remaining46 are authorized')
    if {p.stem for p in (previous/'trial_ledgers').glob('*.sqlite3')}!={t['trial_id'] for t in reg['seed_trials']}:
        raise ValueError('Missing or foreign remaining-slot ledger')
    pins={}
    for t in reg['seed_trials']:
        path=previous/'trial_ledgers'/(t['trial_id']+'.sqlite3')
        if ledger_rows(path):raise ValueError('A claimed slot cannot be repeated: '+t['trial_id'])
        pins[str(path)]=sha(path)
    if read(previous/'summary.json')['trials'] or read(previous/'summary.json')['incomplete_accounting']:
        raise ValueError('A completed or unresolved slot cannot be replayed')
    # First placement metadata can precede qualification; no observation,
    # proposal, episode, action or task result may have been produced.
    trial_files=[f for f in (previous/'groups').glob('*/trials/**/*') if f.is_file()]
    if len(trial_files)!=1 or trial_files[0].name!='collection_placement.json':
        raise ValueError('Further trial preparation/execution requires separate accounting')
    abort=read(previous/'initialization_abort_receipt.json')
    if abort['physical_attempts']!=0 or abort['new_action_command_files']!=0 or any(abort['new_claims'].values()):
        raise ValueError('Forced initializer exit did not prove zero issued actions')
    verify_files({abort['only_trial_preparation_file']:abort['only_preparation_file_sha256']})
    if str(trial_files[0])!=abort['only_trial_preparation_file']:raise ValueError('Unregistered preparation artifact')
    for kind in ('policy','geometry','planner'):
        for queue in ('queue','done'):
            if list((previous/'workers'/kind/queue).glob('*')):raise ValueError('Unexpected actual service request')
    pins[str(trial_files[0])]=sha(trial_files[0])
    return {'pass':True,'new_physical_attempts':0,'new_claims':0,'remaining_slots':46,'files':pins,
        'preparation_metadata_only':str(trial_files[0]),'original_two_results_carried_without_replay':True}


def prepare(previous,out,role_indices):
    previous,out=Path(previous).resolve(),Path(out).resolve()
    if out.exists():raise FileExistsError('Keep all prior resource records')
    old=read(previous/'registration.json')
    if time.time()>=old['deadline_unix']:raise TimeoutError('Original deadline expired')
    proof=unissued(previous);idle=predecessors([previous]);validate_checkpoints(old)
    verify_files({**old['execution_source_files'],**old['dependencies'],**old['carried_evidence']})
    final=read(previous/'main48_verification.json')
    if final['completed']!=2 or final['trials']!=old['carried_details'] or not final['record_integrity_pass']:
        raise ValueError('Carried outcomes changed or new evidence is incomplete')
    verify_files(final['evidence'])
    inventory=devices();by_index={d['index']:d for d in inventory}
    roles={name:by_index[index] for name,index in role_indices.items()};requirements={}
    for role,floor in [('policy',12000),('geometry',3500),('compute',6500),('sim',7500)]:
        uuid=roles[role]['uuid'];requirements[uuid]=requirements.get(uuid,0)+floor
    totals=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.total','--format=csv,noheader,nounits'],text=True,timeout=15)
    capacity={line.split(',')[0].strip():int(line.split(',')[1]) for line in totals.splitlines()}
    if any(n>capacity[u] for u,n in requirements.items()):raise ValueError('Role sum exceeds physical memory')
    if roles['sim']['uuid']==old['requested_devices']['sim_uuid']:raise ValueError('This resource continuation moves the renderer to another authorized GPU')
    out.mkdir(parents=True);space(out);write(out/'state.json',{'status':'preparing','remaining_slots':46,'new_physical_attempts':0})
    command=[sys.executable,'-m','unittest','scripts.curobo_arx.nearfield.test_main48_resource','-v']
    checks=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=30)
    (out/'resource_offline_checks.log').write_text(checks.stdout+checks.stderr)
    write(out/'resource_offline_checks.json',{'pass':checks.returncode==0,'command':command,
        'log_sha256':sha(out/'resource_offline_checks.log'),'sources':{str(p):sha(p) for p in
            (Path(__file__),Path(__file__).with_name('test_main48_resource.py'))},'GPU_or_World_started':False})
    if checks.returncode:raise RuntimeError('Resource continuation regression failed')
    pins={**old['dependencies'],**old['execution_source_files'],**proof['files'],**final['evidence']}
    for p in previous.iterdir():
        if p.is_file() and (p.suffix in ('.json','.md') or p.name=='STOP'):pins[str(p)]=sha(p)
    for file in (previous/'workers').glob('*/process_exit.json'):pins[str(file)]=sha(file)
    for file in (previous/'group_registrations').glob('*.process.json'):pins[str(file)]=sha(file)
    write(out/'predecessor_exit_receipt.json',idle);write(out/'zero_new_attempts.json',proof)
    prior=Path(old['source_snapshot']);source=out/'source_snapshot/COOLAS-ARX'
    shutil.copytree(prior,source,symlinks=True,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('tmp','curobo'):(source.parent/name).symlink_to(ROOT.parent/name,target_is_directory=True)
    execution={str(source/Path(p).relative_to(prior)) if Path(p).is_relative_to(prior) else p:h for p,h in old['execution_source_files'].items()}
    verify_files(execution)
    for name in ('storage_policy.json','advisor29.md','PLAN.md','case_manifest.json','offline_checks.json','offline_checks.log','container_preflight.json'):
        shutil.copy2(previous/name,out/('inherited_'+name if name.startswith(('offline_','container_')) else name))
    shutil.copy2(__file__,out/'prepare_resource_continuation.py')
    write(out/'resource_plan.json',{'roles':roles,'combined_free_memory_mib':requirements,'stable_samples':3,'interval_seconds':10,
        'other_users_jobs_untouched':True})
    write(out/'resource_continuation.json',{'previous':str(previous),'old_STOP_sha256':sha(previous/'STOP'),
        'old_roles':old['requested_devices'],'new_roles':{k+'_uuid':v['uuid'] for k,v in roles.items()},
        'all_execution_source_bytes_unchanged':True,'C_D_checkpoints_unchanged':True,
        'physical_scene_robot_renderer_warmup_and_success_unchanged':True,'original_total_slots':48,
        'carried_slots':2,'remaining_slots':46,'new_physical_attempts_in_predecessor':0,'automatic_retry':False,
        'reinitialized_ledger_scope':'Only provably unclaimed46; release abandoned object-preparation lease in fresh ledgers; no consumed slot is copied',
        'deadline_unix':old['deadline_unix'],'reason':'Manual continuation on a different renderer GPU after confirmed zero-action RTX allocation failure'})
    reg=copy.deepcopy(old);reg.update(created_unix=time.time(),resource_continuation_of=str(previous),
        source_snapshot=str(source),execution_source_files=execution,source_identity=digest(execution),dependencies=pins,
        requested_devices={k+'_uuid':v['uuid'] for k,v in roles.items()})
    reg['learned_policy']['gpu_uuid']=roles['policy']['uuid'];reg['geometry_baseline']['gpu_uuid']=roles['geometry']['uuid']
    validate_schedule(reg);ledgers(out,reg)
    for p in out.iterdir():
        if p.is_file() and p.name!='state.json':pins[str(p)]=sha(p)
    write(out/'catalog.json',reg['catalog']);write(out/'registration.json',reg)
    verify_files({**execution,**pins});predecessors([previous])
    write(out/'state.json',{'status':'prepared','registered_slots':48,'carried_slots':2,'remaining_slots':46,'new_physical_attempts':0})
    return str(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out');p.add_argument('--previous',required=True)
    for role in ('policy','geometry','compute','sim'):p.add_argument('--'+role,type=int,required=True)
    a=p.parse_args();print(prepare(a.previous,a.out,{k:getattr(a,k) for k in ('policy','geometry','compute','sim')}))
