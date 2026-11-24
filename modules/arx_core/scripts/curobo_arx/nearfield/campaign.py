"""Independent, immutable eight-slot first-stage registration and ARX launcher."""
from arx_project.layout import canonical_path
import argparse
import copy
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
import numpy as np
from scripts.curobo_arx.data_factory.common import ROOT,read,write,sha,digest,verify_files,space
from scripts.curobo_arx.data_factory.ledger import Ledger
from scripts.curobo_arx.seed_augmentation.contract import source_seed
from training.nearfield_arx.export import ledger_rows,episode_labels


def startup_expired(group,elapsed):
    progress=Path(group)/'initialization.json'
    phase=read(progress).get('phase') if progress.exists() else None
    return elapsed>600 and phase!='ready_for_uniform_warmup'


def prepare(parent,out,predecessor=None,review_of=None):
    parent=Path(parent).resolve();out=Path(out).resolve()
    if out.exists():raise FileExistsError('Never replace a prior registration')
    old=read(parent/'registration.json');prior=None;proof=None;review=None
    if review_of:
        review_of=Path(review_of).resolve();review=read(review_of/'registration.json')
        if read(review_of/'state.json')['status']!='complete' or read(review_of/'summary.json')['completed']!=8:
            raise RuntimeError('Finish and preserve all eight first-version slots before implementation review')
        for file in (review_of/'group_registrations').glob('*.process.json'):
            if subprocess.run(['podman','container','exists',read(file)['container']],timeout=30).returncode!=1:
                raise RuntimeError('Prior World absence unconfirmed')
        for file in (review_of/'groups').glob('*/shutdown_receipt.json'):
            if not read(file)['all_trajectories_idle']:raise RuntimeError('Prior robot idle unconfirmed')
        from scripts.curobo_arx.data_factory.watchdog import process_info
        for file in (review_of/'workers').glob('*/process.json'):
            if process_info(read(file)['pid']) is not None:raise RuntimeError('Prior planner process remains alive')
    if predecessor:
        predecessor=Path(predecessor).resolve();prior=read(predecessor/'registration.json')
        state=read(predecessor/'state.json')
        if state['status']!='fault' or state.get('unconfirmed_world'):
            raise RuntimeError('Controlled startup correction requires a concluded, confirmed exit')
        from scripts.curobo_arx.data_factory.watchdog import process_info
        processes=list((predecessor/'group_registrations').glob('*.process.json'))
        processes+=list((predecessor/'workers').glob('*/process.json'))
        for file in processes:
            record=read(file);live=process_info(record['pid'])
            if live and live.get('start_ticks')==record.get('start_ticks'):
                raise RuntimeError('Old owned process remains alive')
            if record.get('container') and subprocess.run(['podman','container','exists',record['container']],timeout=30).returncode!=1:
                raise RuntimeError('Old World removal unconfirmed')
        import sqlite3
        for dbpath in (predecessor/'trial_ledgers').glob('*.sqlite3'):
            db=sqlite3.connect('file:'+str(dbpath)+'?mode=ro',uri=True)
            try:
                if db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]:
                    raise RuntimeError('Startup-only continuation cannot repeat any claimed attempt')
            finally:db.close()
        if any((predecessor/'groups').glob('*/trials/*/episodes/*/*_command.json')):
            raise RuntimeError('Startup-only continuation found a physical command')
        proof={'predecessor':str(predecessor),'prior_registration_sha256':sha(predecessor/'registration.json'),
            'prior_STOP_sha256':sha(predecessor/'STOP'),'claimed_attempts':0,'physical_commands':0,
            'old_owned_processes_exited':True,'old_Worlds_absent':True,
            'change':'Startup600s ends at actual scene readiness; first qualification retains original900s budget',
            'failed_or_unknown_physical_attempts_retried':0,'original_eight_slot_budget_unchanged':True}
    out.mkdir(parents=True)
    write(out/'state.json',{'status':'preparing','physical_trials':0})
    verify_files({**old['execution_source_files'],**old['dependencies']});space(out)
    source=out/'source_snapshot/COOLAS-ARX';source.parent.mkdir()
    previous=Path(old['source_snapshot'])
    shutil.copytree(previous,source,symlinks=True,ignore=shutil.ignore_patterns('__pycache__'))
    (source.parent/'curobo').symlink_to(ROOT.parent/'curobo',target_is_directory=True)
    for package in ('scripts/curobo_arx/seed_augmentation','scripts/curobo_arx/nearfield',
                    'training/nearfield_arx','training/show_harness_arx'):
        shutil.copytree(ROOT/package,source/package,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    reg=copy.deepcopy(old);execution={}
    for file,value in old['execution_source_files'].items():
        p=Path(file);execution[str(source/p.relative_to(previous) if p.is_relative_to(previous) else p)]=value
    for file in source.rglob('*.py'):execution[str(file)]=sha(file)
    reg['execution_source_files']=execution;reg['source_snapshot']=str(source)
    for key in ('bulk_storage','resume','lineage','controlled_resource_pause'):reg.pop(key,None)
    seeds={};trials=[];rows=ledger_rows(parent/'ledger.sqlite3')
    for object_id in ('household_hope_milk_carton','mustard_bottle'):
        for row in rows:
            if row['object_id']!=object_id:continue
            try:seed=source_seed(row)
            except ValueError:continue
            if read(Path(seed['episode'])/'episode.json')['split']!='train':continue
            break
        else:raise ValueError('No stationary training seed for '+object_id)
        seeds[seed['seed_id']]=seed;reg['dependencies'].update(seed['files'])
        record={'id':row['id'],'episode':row['episode'],'object_id':object_id,'family':row['id'],
                'split':'train','geometry_group':read(Path(row['episode'])/'episode.json')['geometry_group'],'policy':'native'}
        _,star,priors,samples,excluded=episode_labels(record)
        tokens=[s['labels']['C'] for s in samples if s['prior_id']==0 and s['labels']['C']!='GRASP']
        if not tokens:raise ValueError('Source has no representable C actions')
        # Four independent slots per source. Recovery offsets retreat along the
        # registered grasp approach axis; no object/camera/robot changes.
        for method,offset in [('C',None),('D',None),('recovery',.005),('recovery',.020)]:
            if review and (method=='D' or method=='recovery' and object_id!='household_hope_milk_carton'):continue
            target=star.copy()
            if offset:target[:3,3]-=star[:3,2]*offset
            trial={'trial_id':f'{len(trials):02d}__{object_id}__{method}'+(f'_{int(offset*1000)}mm' if offset else ''),
                'seed_id':seed['seed_id'],'object_id':object_id,'renderer':seed['renderer'],
                'method':method,'policy':method,'placement':seed['placement'],'entry_world':seed['anchor_world'],
                'perturbation':{'offset_m':offset},'source_Gstar_world':star.tolist(),'G0_world':priors[1]['world'],
                'C_tokens':tokens if method=='C' else [],'offset_goal_world':target.tolist() if offset else None,
                'exercise_skip_reobserve':method=='D' and object_id=='mustard_bottle',
                'source_transitions_excluded':excluded,'truth_replay_only_no_policy_training':method in ('C','D')}
            if review and method=='C':
                trial['C_waypoints']=[s['offline_target']['C_next_world'] for s in samples if s['prior_id']==0 and s['labels']['C']!='GRASP']
            trials.append(trial)
    for trial in trials:
        catalog=copy.deepcopy(reg['catalog'])
        for asset in catalog['objects']:
            asset['collection_enabled']=asset['object_id']==trial['object_id']
            if asset['collection_enabled']:asset['collection_placements']=[trial['placement']]
        path=out/'trial_ledgers'/(trial['trial_id']+'.sqlite3');path.parent.mkdir(exist_ok=True)
        ledger=Ledger(path);ledger.initialize(catalog,{**reg['quotas'],'first_object':trial['object_id'],
            'max_physical_attempts':1,'max_total_dispatches':1,'physical_per_bin':1,
            'planning_rejections_per_bin':1,'goals_per_object':1,'successes_per_bin':1});ledger.close()
    for name,p in list(reg['reference_files'].items()):
        new=out/'references'/(name+'.json');new.parent.mkdir(exist_ok=True);shutil.copy2(p,new)
        reg['reference_files'][name]=str(new);reg['dependencies'][str(new)]=sha(new)
    policy=parent/'storage_policy.json'
    if not policy.exists():raise FileNotFoundError('Inherited storage policy missing')
    shutil.copy2(policy,out/'storage_policy.json');reg['dependencies'][str(out/'storage_policy.json')]=sha(out/'storage_policy.json')
    for name in ('gpt回应1.md','gpt回应2.md','gpt回应3.md','gpt回应4.md','gemini回复4.md'):
        p=ROOT/'research_plan/chat'/name;dest=out/'frozen_spec'/name;dest.parent.mkdir(exist_ok=True);shutil.copy2(p,dest)
        reg['dependencies'][str(dest)]=sha(dest)
    reg.update(stage='nearfield_cd_stage1',production=False,preparation_complete=True,
        created_unix=time.time(),deadline_unix=time.time()+4*3600,seeds=seeds,seed_trials=trials,
        source_identity=digest(execution),envs_per_gpu=1,object_block_attempts=1,
        max_physical_trials=8,max_replay_trials=4,max_recovery_trials=4,no_automatic_retry=True,
        minimum_available_host_memory_bytes=40*1024**3,project_root=str(ROOT),
        formal_48_trials_authorized_here=False,training_authorized_here=False,
        recovery_supervision='Exclude H-to-offset state generation. Start after fresh stopped-offset observation.')
    if prior:
        reg['deadline_unix']=prior['deadline_unix'];reg['predecessor']=str(predecessor)
        if digest(trials)!=digest(prior['seed_trials']):raise RuntimeError('Remaining registered cases changed')
        reg['dependencies'][str(predecessor/'registration.json')]=sha(predecessor/'registration.json')
        reg['dependencies'][str(predecessor/'STOP')]=sha(predecessor/'STOP')
        write(out/'startup_correction_receipt.json',proof)
    if review:
        reg.update(deadline_unix=review['deadline_unix'],max_physical_trials=4,max_replay_trials=2,max_recovery_trials=2,
            C_truth_driver='current_state_to_recorded_next_waypoint',C_intermediate_phase='local',
            recovery_motion_phase='local',implementation_review_of=str(review_of))
        reg['dependencies'][str(review_of/'registration.json')]=sha(review_of/'registration.json')
        reg['dependencies'][str(review_of/'summary.json')]=sha(review_of/'summary.json')
        write(out/'implementation_change_receipt.json',{
            'first_version_trials_preserved':8,'new_variant_slots':4,'same_source_seeds':True,
            'C_driver':'Recompute EE STEP from actual current state to each recorded next waypoint; oracle only',
            'intermediate_motion':'Original open-hand local path checks; nominal closed-goal check still required at final approach before GRASP',
            'state_generation':'Original local-approach planning; still reject any actual target touch or object movement',
            'unchanged_robot_physics_and_success_gates':True,'no_automatic_retry':True})
    write(out/'registration.json',reg);write(out/'catalog.json',reg['catalog'])
    write(out/'sources.json',{'spec':'frozen_spec/gemini回复4.md','data_parent':str(parent),
        'method':'ARX teacher-forced C/D interface test and expert recovery; no trained neural model'})
    (out/'research.md').write_text('# First-stage C/D execution\n\n'+
        ('Four separately bounded implementation-review slots: two state-conditioned C oracle replays and two milk5/20mm recoveries. First eight outcomes remain separate. See implementation_change_receipt.json. ' if review else
         'Eight bounded slots: four teacher-label interpreter replays and four5/20mm stopped-offset recoveries. ')+
        'Original ARX, physics, placements, renderers, convergence, contacts and >6 cm / 3 s all-samples success gates remain fixed. Ordinary task failures continue. Resource, configuration, record-chain and idle-uncertainty faults stop all later dispatch. No retries, no training or formal48 trials.\n')
    (out/'research_log.md').write_text('Registered before any new physical execution.\n')
    verify_files({**execution,**reg['dependencies']})
    write(out/'state.json',{'status':'prepared','scheduled_slots':len(trials),'physical_trials':0})
    return out


def summarize(campaign,output_name='summary.json'):
    rows=[];reg=read(campaign/'registration.json')
    for p in (campaign/'groups').glob('*/results.json'):rows.extend(read(p)['trials'])
    done={r['trial_id'] for r in rows}
    # A failure during commit can leave complete physical evidence without a
    # trial_result.json. The sole-claim ledgers still record those attempts.
    import sqlite3
    claimed=[]
    for trial in reg['seed_trials']:
        path=campaign/'trial_ledgers'/(trial['trial_id']+'.sqlite3')
        if not path.exists():continue
        db=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
        try:attempts=[dict(r) for r in db.execute('SELECT * FROM attempts')]
        finally:db.close()
        for attempt in attempts:
            ep=Path(attempt['episode']) if attempt.get('episode') else None
            receipt=ep/'physical_receipt.json' if ep else None
            claimed.append({'trial_id':trial['trial_id'],'method':trial['method'],**attempt,
                'verified_physical_success':bool(receipt and receipt.exists() and read(receipt).get('pass'))})
    begun={r['trial_id'] for r in claimed}
    result={'scheduled':len(reg['seed_trials']),'completed':len(rows),
        'actual_attempts':sum(r['physical'] for r in claimed),
        'verified_physical_successes':sum(r['verified_physical_success'] for r in claimed),
        'ledger_admitted':sum(r['state']=='committed' for r in claimed),
        'incomplete_accounting':[r for r in claimed if r['trial_id'] not in done],
        'by_method':{m:{'completed':sum(r['method']==m for r in rows),
                        'success':sum(r['method']==m and r['success'] for r in rows),
                        'admitted':sum(r['method']==m and r['admitted'] for r in rows)} for m in ('C','D','recovery')},
        'unexecuted':[t['trial_id'] for t in reg['seed_trials'] if t['trial_id'] not in done|begun],
        'trials':rows,'carried_results':reg.get('carried_results',[]),
        'preserved_prior_results':reg.get('prior_results',[]),
        'trained_policy_trials':0,'formal_comparison_trials':0}
    write(campaign/output_name,result);return result


def run(campaign,compute_uuid,sim_uuid):
    from scripts.curobo_arx.data_factory.launch import devices,allocate,environment,seed_kit_cache
    from scripts.curobo_arx.data_factory.watchdog import process_info,exclusive_stop
    from scripts.curobo_arx.seed_augmentation.campaign import available_memory
    campaign=Path(campaign).resolve();reg=read(campaign/'registration.json')
    lock=(campaign/'coordinator.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (campaign/'launch.json').exists() or (campaign/'STOP').exists():raise RuntimeError('No automatic retry or STOP bypass')
    verify_files({**reg['execution_source_files'],**reg['dependencies']});space(campaign)
    compute,sims=allocate(devices(),1,compute_uuid,[sim_uuid]);sim=sims[0]
    if available_memory()<reg['minimum_available_host_memory_bytes']:raise RuntimeError('Insufficient available host memory')
    from .resources import probe_cuda
    resource_receipts=[]
    for device in {d['uuid']:d for d in (compute,sim)}.values():
        receipt=probe_cuda(device,environment(device));resource_receipts.append(receipt)
        write(campaign/'cuda_preflight.json',{'devices':resource_receipts,'pass':all(r['pass'] for r in resource_receipts)})
        if not receipt['pass']:
            exclusive_stop(campaign/'STOP',{'reason':'CUDA initialization failed before service launch','gpu_uuid':device['uuid']})
            write(campaign/'state.json',{'status':'fault','completed':0,'unconfirmed_world':False,
                'physical_trials':0,'error':'CUDA preflight failed; no service or simulator was started'})
            raise RuntimeError('CUDA preflight failed: '+str(receipt))
    # Fresh runtime inventory; empty/healthy podman is required, not a stale PID.
    inventory=subprocess.check_output(['podman','ps','--format','{{.Names}}'],timeout=30,text=True)
    source=Path(reg['source_snapshot']);worker_dir=campaign/'workers/planner';worker_dir.mkdir(parents=True)
    write(campaign/'launch.json',{'wall':time.time(),'compute':compute,'simulator':sim,
        'containers_before':inventory.splitlines(),'registration_sha256':sha(campaign/'registration.json')})
    worker=None;active=None;container=None;error=None;unresolved=False
    try:
        cmd=[str(ROOT.parent/'curobo/.venv/bin/python'),'-m',reg.get('worker_module','scripts.curobo_arx.seed_augmentation.worker'),
             '--directory',str(worker_dir),'--registration',str(campaign/'registration.json')]
        with (worker_dir/'process.log').open('x') as log:
            worker=subprocess.Popen(cmd,cwd=source,env=environment(compute),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(worker_dir/'process.json',{'pid':worker.pid,'command':cmd,'owned':True,**(process_info(worker.pid) or {})})
        deadline=time.monotonic()+600
        while not (worker_dir/'ready.json').exists():
            if worker.poll() is not None or time.monotonic()>deadline:raise RuntimeError('New owned planner startup failure')
            time.sleep(.25)
        for renderer in sorted({t['renderer'] for t in reg['seed_trials']}):
            if (campaign/'STOP').exists() or time.time()>=reg['deadline_unix']:break
            if available_memory()<reg['minimum_available_host_memory_bytes']:raise RuntimeError('Host memory below floor')
            if next(d['free_mb'] for d in devices() if d['uuid']==sim_uuid)<7500:raise RuntimeError('Simulator memory below floor')
            ident='nearfield_'+renderer.lower();container='arx-'+campaign.name+'-'+renderer.lower()
            if subprocess.run(['podman','container','exists',container],timeout=30).returncode!=1:raise RuntimeError('World absence unconfirmed')
            path=campaign/'group_registrations'/(ident+'.json')
            write(path,{'group_id':ident,'renderer':renderer,'gpu_uuid':sim_uuid,'env_count':1,
                'planner_directory':str(worker_dir),'candidate_directory':str(campaign/'no_neural_requests'),
                'trial_ids':[t['trial_id'] for t in reg['seed_trials'] if t['renderer']==renderer],
                'stop_unix':reg['deadline_unix'],'registration_sha256':sha(campaign/'registration.json')})
            cache=ROOT/'runs/data_factory_runtime_cache';kit=seed_kit_cache(cache/sim_uuid)
            cmd=['podman','run','--rm','--name',container,'--user','0','--device','nvidia.com/gpu='+sim_uuid,
                '--security-opt=label=disable','--network=host','--shm-size=2g']
            for v in ('ACCEPT_EULA=Y','PRIVACY_CONSENT=N','OMNI_KIT_ALLOW_ROOT=1','CUDA_VISIBLE_DEVICES='+sim_uuid,
                      'CUDA_DEVICE_ORDER=PCI_BUS_ID','NVIDIA_VISIBLE_DEVICES=void','PYTHONUNBUFFERED=1','PYTHONDONTWRITEBYTECODE=1'):
                cmd+=['-e',v]
            for mount in (str(cache/'ov')+':/root/.cache/ov:rw',str(cache/'nvidia')+':/root/.cache/nvidia:rw',
                str(kit)+':/isaac-sim/kit/cache:rw',str(ROOT.parent)+':'+str(ROOT.parent)+':rw',
                '/coolas/data/arx_teacher_zhaoyechong:/coolas/data/arx_teacher_zhaoyechong:ro',
                str(canonical_path(ROOT/'asset'))+':/workspace/asset:ro',str(ROOT)+':/workspace/COOLAS-ARX:ro'):
                cmd+=['-v',mount]
            cmd+=['-w',str(source),'--entrypoint','/isaac-sim/python.sh','localhost/arx-s02-isaac:6.0.1',
                '-m',reg.get('runtime_module','scripts.curobo_arx.nearfield.runtime'),'--campaign',str(campaign),'--launch',str(path)]
            with path.with_suffix('.log').open('x') as log:
                active=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(path.with_suffix('.process.json'),{'pid':active.pid,'container':container,'command':cmd,
                'owned':True,**(process_info(active.pid) or {})})
            started=time.monotonic()
            while active.poll() is None:
                write(campaign/'state.json',{'status':'running','wall':time.time(),'renderer':renderer})
                if worker.poll() is not None:exclusive_stop(campaign/'STOP',{'reason':'owned planner exited'})
                if startup_expired(campaign/'groups'/ident,time.monotonic()-started):
                    exclusive_stop(campaign/'STOP',{'reason':'runtime startup exceeded 600 seconds'})
                if (campaign/'STOP').exists() or time.time()>=reg['deadline_unix']:
                    subprocess.run(['podman','kill','--signal','TERM',container],capture_output=True,timeout=30)
                time.sleep(2)
            if active.returncode!=0:raise RuntimeError('ARX runtime exited with failure; inspect group log')
            if (campaign/'STOP').exists():
                raise RuntimeError('ARX startup/execution stopped: '+str(read(campaign/'STOP')))
            shutdown=read(campaign/'groups'/ident/'shutdown_receipt.json')
            if not shutdown['all_trajectories_idle'] or subprocess.run(['podman','container','exists',container],timeout=30).returncode!=1:
                raise RuntimeError('Robot idle / World removal unconfirmed')
            active=None
        verify_files({**reg['execution_source_files'],**reg['dependencies']})
    except BaseException:
        error=traceback.format_exc();write(campaign/'FAULT.json',{'error':error,'wall':time.time()})
        exclusive_stop(campaign/'STOP',{'reason':'infrastructure/configuration/record stop','error':error})
    finally:
        if active is not None and active.poll() is None:
            subprocess.run(['podman','kill','--signal','TERM',container],capture_output=True,timeout=30)
            try:active.wait(timeout=reg['attempt_seconds']+60)
            except subprocess.TimeoutExpired:unresolved=True
        if not unresolved and worker is not None:
            write(worker_dir/'STOP',{'reason':'bounded stage finished'})
            try:worker.wait(timeout=30)
            except subprocess.TimeoutExpired:worker.terminate();worker.wait(timeout=30)
        result=summarize(campaign)
        write(campaign/'state.json',{'status':'fault' if error or unresolved else 'complete',
              'error':error,'unconfirmed_world':unresolved,'completed':result['completed'],'finished_unix':time.time()})


if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='action',required=True)
    q=s.add_parser('prepare');q.add_argument('--parent',required=True);q.add_argument('--out',required=True);q.add_argument('--predecessor');q.add_argument('--review-of')
    q=s.add_parser('run');q.add_argument('campaign');q.add_argument('--compute-uuid',required=True);q.add_argument('--sim-uuid',required=True)
    a=p.parse_args()
    if a.action=='prepare':print(prepare(a.parent,a.out,a.predecessor,a.review_of))
    else:run(a.campaign,a.compute_uuid,a.sim_uuid)
