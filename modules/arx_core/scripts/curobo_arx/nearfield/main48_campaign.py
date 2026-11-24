"""Original48 preregistration and bounded service coordinator; no training/retries."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from scripts.curobo_arx.data_factory.common import ROOT,read,write,sha,digest,verify_files,space
from scripts.curobo_arx.data_factory.watchdog import exclusive_stop,process_info
from scripts.curobo_arx.data_factory.launch import devices,environment
from training.nearfield_arx.stage_policy import validate_checkpoints
from training.nearfield_arx.lateral_campaign import ledgers
from .prepare_physical import source_inventory,TRAIN,HELD
from .prepare_v2 import predecessors
from .main48_spec import VERSION,GEOMETRY,make_trials,validate_trials,validate_schedule
from .learned_campaign import registered_launcher_project
from .physical_startup import startup_expired
from . import campaign as launcher


def prepare(previous,out,policy_index,compute_index,sim_index,geometry_index):
    previous,out=Path(previous).resolve(),Path(out).resolve()
    if out.exists():raise FileExistsError('Preserve every prior registration')
    final=read(previous/'final_verification.json');lock=read(previous/'model_lock.json')
    if not final['qualified_for_registered_48'] or not final['record_integrity_pass'] or final['physical']['physical_successes']<3:
        raise RuntimeError('Advisor29 original full-success transition gate not met')
    if read(previous/'state.json')['status']!='complete':raise RuntimeError('Predecessor incomplete')
    physical=previous/'training/physical';old=read(physical/'registration.json')
    if lock['checkpoints']!=old['learned_policy']['checkpoints'] or lock['training_bindings']!=old['learned_policy']['training_bindings']:
        raise RuntimeError('Model lock differs from executed models')
    idle=predecessors([physical,TRAIN,HELD]);validate_checkpoints(old)
    verify_files(final['evidence']);verify_files({**old['execution_source_files'],**old['dependencies']})
    sources,seeds,families,geometries=source_inventory();trials=make_trials(sources,seeds)
    inventory=devices();select=lambda n:next(d for d in inventory if d['index']==n)
    roles={name:select(n) for name,n in [('policy',policy_index),('compute',compute_index),('sim',sim_index),('geometry',geometry_index)]}
    requirements={}
    for role,floor in [('policy',12000),('compute',6500),('sim',7500),('geometry',3500)]:
        gpu=roles[role]['uuid'];requirements[gpu]=requirements.get(gpu,0)+floor
    totals=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.total','--format=csv,noheader,nounits'],text=True,timeout=15)
    capacity={line.split(',')[0].strip():int(line.split(',')[1]) for line in totals.splitlines()}
    if any(v>capacity[u] for u,v in requirements.items()):
        raise ValueError('Combined role memory estimate exceeds device capacity')
    out.mkdir(parents=True);space(out);write(out/'state.json',{'status':'preparing','physical_trials':0})
    added=sorted((ROOT/'scripts/curobo_arx/nearfield').glob('main48*.py'))
    added += [ROOT/'scripts/curobo_arx/nearfield/runtime_main48.py',ROOT/'scripts/curobo_arx/nearfield/test_main48.py']
    cmd=[sys.executable,'-m','unittest','scripts.curobo_arx.nearfield.test_main48',
         'training.nearfield_arx.test_stage_recovery','training.nearfield_arx.test_closure_recovery',
         'training.nearfield_arx.test_closure_records','scripts.curobo_arx.nearfield.test_learned','-v']
    test=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=240)
    (out/'offline_checks.log').write_text(test.stdout+test.stderr)
    write(out/'offline_checks.json',{'pass':test.returncode==0,'command':cmd,'log_sha256':sha(out/'offline_checks.log'),
        'sources':{str(p):sha(p) for p in added},'no_service_or_World_started':True})
    if test.returncode:raise RuntimeError('Offline checks failed; no registration or execution')
    prior=Path(old['source_snapshot']);source=out/'source_snapshot/COOLAS-ARX'
    shutil.copytree(prior,source,symlinks=True,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('tmp','curobo'):(source.parent/name).symlink_to(ROOT.parent/name,target_is_directory=True)
    execution={str(source/Path(p).relative_to(prior)) if Path(p).is_relative_to(prior) else p:h
               for p,h in old['execution_source_files'].items()}
    for p in added:
        dest=source/p.relative_to(ROOT);dest.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(p,dest);execution[str(dest)]=sha(dest)
    for p in (ROOT/'scripts/curobo_arx/graspness_teacher').glob('*.py'):
        dest=source/p.relative_to(ROOT)
        if not dest.exists():dest.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(p,dest);execution[str(dest)]=sha(dest)
    # Assert all pre-existing execution bytes, including C/D, are unchanged.
    for p,h in old['execution_source_files'].items():
        dest=source/Path(p).relative_to(prior) if Path(p).is_relative_to(prior) else Path(p)
        if sha(dest)!=h:raise RuntimeError('Changed inherited execution source: '+str(dest))
    deps=copy.deepcopy(old['dependencies'])
    selected_seeds={t['seed_id']:seeds[t['seed_id']] for t in trials}
    for seed in selected_seeds.values():deps.update(seed['files'])
    # Retain exactly the two original C/D warmups. Geometry gets two old ARX
    # training-object views, selected by raw chronology, not prediction quality.
    trial=next(t for t in final['physical']['trials'] if t['method']=='C' and t['object_id']=='mustard_bottle')
    actual=next(t for t in read(physical/'summary.json')['trials'] if t['trial_id']==trial['trial_id'])
    episode=Path(actual['attempts'][0]['episode']);rows=[json.loads(l) for l in (episode/'samples.jsonl').read_text().splitlines()]
    asset=next(a for a in old['catalog']['objects'] if a['object_id']=='mustard_bottle')
    warmups=[]
    for i in (0,1):
        row=out/f'geometry_warmups/{i}_observation.json';write(row,rows[i])
        paths=[row,episode/f'raw/{i:06d}.png',episode/f'raw/{i:06d}.npy']
        pins={str(p):sha(p) for p in paths};deps.update(pins)
        warmups.append({'observation':str(row),'rgb':str(paths[1]),'depth':str(paths[2]),
            'asset':asset['usd_path'],'G0_world':actual['G0_world'],'files':pins})
    # Pin all pretrained Graspness inference sources and compiled extensions.
    for p in Path(old['graspness']['source']).rglob('*'):
        if p.is_file() and p.suffix in ('.py','.so','.cpp','.cu','.h','.hpp'):deps[str(p)]=sha(p)
    deps[old['graspness']['checkpoint']]=old['graspness']['checkpoint_sha256']
    for p in (previous/'registration.json',previous/'final_verification.json',previous/'model_lock.json',
              physical/'registration.json',TRAIN/'registration.json',HELD/'registration.json'):
        deps[str(p)]=sha(p)
    for n,p in [('storage_policy.json',previous/'storage_policy.json'),('advisor29.md',previous/'advisor29.md'),
                ('PLAN.md',ROOT/'research_plan/docs/nearfield_main48_20261109.md')]:
        shutil.copy2(p,out/n)
    write(out/'case_manifest.json',{'slots':trials,'selection':'First two training objects in pre-existing prepare_v2.OBJECTS; both original held-out objects; first sorted seed per object',
        'selection_timing':'First full48 manifest frozen now after four-slot model lock, not retrospectively claimed as frozen before training',
        'Gbase':'Original saved front-end proposal; measured historical Gstar only for diagnostics',
        'old_four_results_in_main48':False,'automatic_retries':0})
    write(out/'predecessor_exit_receipt.json',idle)
    write(out/'resource_plan.json',{'roles':roles,'combined_free_memory_mib':requirements,'stable_samples':3,'interval_seconds':10,
        'other_users_jobs_untouched':True})
    write(out/'source_change_receipt.json',{'version':VERSION,'added':[str(p.relative_to(ROOT)) for p in added],
        'inherited_execution_source_bytes_unchanged':True,'C_D_checkpoints_unchanged':True,'new_training_updates':0,
        'new_physical_budget':48,'retries':0,'Gstar_generated_actions':False,'B_contract':GEOMETRY,
        'frozen':'robot, calibration, physics, resets, rendering, success, C/D motion/closure/lift gates'})
    reg=copy.deepcopy(old)
    for key in ('physical_data_contract','entry_wait_continuation_of','startup_continuation_of','unified_registration',
                'resource_continuation','preserved_completed_results','consumed_physical_budget'):
        reg.pop(key,None)
    reg.update(stage=VERSION,created_unix=time.time(),campaign_started_unix=time.time(),deadline_unix=time.time()+18*3600,
        wall_hours=18,production=False,source_snapshot=str(source),execution_source_files=execution,source_identity=digest(execution),
        dependencies=deps,seeds=selected_seeds,seed_trials=trials,project_root=str(ROOT),
        runtime_module='scripts.curobo_arx.nearfield.runtime_main48',coordinator_module='scripts.curobo_arx.nearfield.main48_campaign',
        worker_module='scripts.curobo_arx.nearfield.closure_worker',requested_devices={k+'_uuid':v['uuid'] for k,v in roles.items()},
        geometry_baseline={'contract':GEOMETRY,'gpu_uuid':roles['geometry']['uuid'],'warmups':warmups},
        max_physical_trials=48,global_physical_budget=48,max_replay_trials=0,replay_slots_total=0,recovery_slots_total=0,
        formal_48_trials_authorized_here=True,training_authorized_here=False,automatic_startup_retries=0,automatic_attempt_retries=0,
        carried_results=[],prior_results=[],prerequisite_results=[],source_family_splits=families,source_geometry_splits=geometries,
        execution_authority={'advisor':str(out/'advisor29.md'),'model_lock':str(previous/'model_lock.json'),
            'scope':'Exactly original48 after verified3of4; user continue; no further local model tuning'})
    reg['learned_policy']['gpu_uuid']=roles['policy']['uuid']
    reg['geometry_baseline']['checkpoint_provenance']=old['graspness']['weight_provenance']
    # No GPU, World or image download in this frozen import/path check.
    cmd=['podman','run','--pull=never','--rm','-i','--user','0','--security-opt=label=disable',
         '-v',str(ROOT.parent)+':'+str(ROOT.parent)+':rw','-v','/coolas/data/arx_teacher_zhaoyechong:/coolas/data/arx_teacher_zhaoyechong:ro',
         '-w',str(source),'--entrypoint','/isaac-sim/python.sh','localhost/arx-s02-isaac:6.0.1','-c',
         'import json,pathlib,sys;import scripts.curobo_arx.nearfield.runtime_main48;'
         'paths=json.load(sys.stdin);missing=[p for p in paths if not pathlib.Path(p).is_file()];'
         'print(json.dumps({"missing":missing,"checked":len(paths)}));sys.exit(bool(missing))']
    probe=subprocess.run(cmd,input=json.dumps(list({*execution,*deps})),capture_output=True,text=True,timeout=90)
    write(out/'container_preflight.json',{'pass':probe.returncode==0,'command':cmd,'stdout':probe.stdout,'stderr':probe.stderr,
        'GPU_or_World_started':False,'image_downloaded':False})
    if probe.returncode:raise RuntimeError('Frozen container imports/paths failed')
    for p in out.iterdir():
        if p.is_file() and p.name!='state.json':deps[str(p)]=sha(p)
    reg['dependencies']=deps
    ledgers(out,reg);write(out/'catalog.json',reg['catalog']);write(out/'registration.json',reg)
    verify_files({**execution,**deps})
    write(out/'state.json',{'status':'prepared','scheduled_slots':48,'physical_trials':0})
    return out


def run(campaign):
    campaign=Path(campaign).resolve();reg=read(campaign/'registration.json');space(campaign)
    lock=(campaign/'main48.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (campaign/'STOP').exists() or (campaign/'services_started.json').exists():raise RuntimeError('No retries or STOP bypass')
    validate_schedule(reg);validate_checkpoints(reg)
    verify_files({**reg['execution_source_files'],**reg['dependencies']})
    requirements=read(campaign/'resource_plan.json')['combined_free_memory_mib'];stable=0
    processes={};finished=threading.Event();error=None
    def stopped():return (campaign/'STOP').exists() or time.time()>=reg['deadline_unix']
    def interrupt(signum,frame):exclusive_stop(campaign/'STOP',{'reason':'coordinator signal','signal':signum})
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    try:
        while stable<3:
            if stopped():raise RuntimeError('Stopped before service launch')
            inventory=devices();free={d['uuid']:d['free_mb'] for d in inventory}
            stable=stable+1 if all(free.get(u,0)>=v for u,v in requirements.items()) else 0
            write(campaign/'state.json',{'status':'waiting_for_resources','stable_samples':stable,'inventory':inventory,'wall':time.time()})
            if stable<3:time.sleep(10)
        from .resources import probe_cuda
        for device in inventory:
            if device['uuid'] in requirements:
                proof=probe_cuda(device,environment(device));write(campaign/('cuda_'+str(device['index'])+'.json'),proof)
                if not proof['pass']:raise RuntimeError('Fresh CUDA initialization failed')
        write(campaign/'services_started.json',{'wall':time.time(),'registration_sha256':sha(campaign/'registration.json')})
        for name,module,python in [('policy',reg['policy_module'],sys.executable),
            ('geometry','scripts.curobo_arx.nearfield.main48_geometry',reg['graspness']['python'])]:
            directory=campaign/'workers'/name;directory.mkdir(parents=True);(directory/'queue').mkdir();(directory/'done').mkdir()
            device=next(d for d in inventory if d['uuid']==reg['requested_devices'][name+'_uuid'])
            env=environment(device);env['PYTHONPATH']=reg['source_snapshot']
            if name=='geometry':env['LD_LIBRARY_PATH']=os.pathsep.join([*reg['graspness'].get('library_directories',[]),env.get('LD_LIBRARY_PATH','')])
            cmd=[str(python),'-m',module,str(campaign)]
            with (directory/'process.log').open('x') as log:
                p=subprocess.Popen(cmd,cwd=reg['source_snapshot'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            processes[name]=p;write(directory/'process.json',{'pid':p.pid,'command':cmd,'owned':True,**(process_info(p.pid) or {})})
            deadline=time.monotonic()+600
            while not (directory/'ready.json').exists():
                if p.poll() is not None or time.monotonic()>deadline or stopped():raise RuntimeError(name+' service failed before ready')
                time.sleep(.25)
            ready=read(directory/'ready.json')
            if ready['registration_sha256']!=sha(campaign/'registration.json'):raise RuntimeError('Foreign service ready receipt')
        def monitor():
            while not finished.wait(.5):
                for name,p in processes.items():
                    if p.poll() is not None:
                        exclusive_stop(campaign/'STOP',{'reason':'owned service exited','service':name});return
        threading.Thread(target=monitor,daemon=True).start()
        previous=launcher.startup_expired;launcher.startup_expired=lambda g,t:startup_expired(g,t,1800)
        old_summary=launcher.summarize
        def summarize(root,output_name='summary.json'):
            result=old_summary(root,output_name='base_summary.json')
            result['by_method']={m:{'completed':sum(t['method']==m for t in result['trials']),
                'success':sum(t['method']==m and t['success'] for t in result['trials']),
                'admitted':sum(t['method']==m and t['admitted'] for t in result['trials'])} for m in 'ABCD'}
            result['formal_comparison_trials']=len(result['trials'])
            result['trained_policy_trials']=sum(t['method'] in ('C','D') and t['physical_dispatches']>0 for t in result['trials'])
            carried=reg.get('carried_details',[])
            result['original48_cumulative']={'registered':48,'accounted':len(result['trials'])+len(carried),
                'actual_attempts':result['actual_attempts']+sum(t['physical_attempt'] for t in carried),
                'successes':sum(t['success'] for t in result['trials'])+sum(t['physical_success'] for t in carried),
                'carried':len(carried),'remaining_unexecuted':len(result['unexecuted'])+len(reg.get('deferred_trials',[])),
                'user_deferred_unexecuted':[t['trial_id'] for t in reg.get('deferred_trials',[])],
                'native_terminal_gaps_carried':[t['trial_id'] for t in carried if not t.get('native_runtime_completed',True)]}
            write(Path(root)/output_name,result);return result
        launcher.summarize=summarize
        try:
            with registered_launcher_project(reg):launcher.run(campaign,reg['requested_devices']['compute_uuid'],reg['requested_devices']['sim_uuid'])
        finally:launcher.startup_expired=previous;launcher.summarize=old_summary
        if read(campaign/'state.json')['status']!='complete' or (campaign/'STOP').exists():raise RuntimeError('Original48 execution stopped; retain all results')
    except BaseException:
        error=traceback.format_exc();exclusive_stop(campaign/'STOP',{'reason':'main48 infrastructure/record/idle fault','error':error})
        write(campaign/'coordinator_fault.json',{'error':error})
        if not (campaign/'launch.json').exists():write(campaign/'state.json',{'status':'fault','error':error,'physical_trials':0,'unconfirmed_world':False})
    finally:
        finished.set()
        for name,p in processes.items():
            directory=campaign/'workers'/name;write(directory/'STOP',{'reason':'bounded main48 service shutdown'})
            try:p.wait(timeout=300)
            except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=30)
            write(directory/'process_exit.json',{'pid':p.pid,'exit_code':p.returncode,'wall':time.time()})
    if (campaign/'summary.json').exists():
        from .main48_report import report
        result=report(campaign)
        write(campaign/'delivery_receipt.json',{'document':str(campaign/'ACCEPTANCE.md'),'sha256':sha(campaign/'ACCEPTANCE.md'),
            'verification_sha256':sha(campaign/'main48_verification.json'),'complete':len(result['trials'])==48,
            'authorized_scope_accounted':len(result['trials'])+len(reg.get('deferred_trials',[]))==48,
            'deferred_unexecuted':result['deferred_unexecuted']})
    if error:raise RuntimeError(error)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run'));p.add_argument('root');p.add_argument('--previous')
    for role in ('policy','compute','sim','geometry'):p.add_argument('--'+role,type=int)
    a=p.parse_args()
    if a.action=='prepare':print(prepare(a.previous,a.root,a.policy,a.compute,a.sim,a.geometry))
    else:run(a.root)
