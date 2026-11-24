"""Bounded production coordinator; fresh UUID binding, no other-user cleanup."""
from arx_project.layout import canonical_path
import argparse
import csv
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import uuid
from .common import ROOT,read,sha,space,verify_files,write
from .ledger import Ledger
from .report import report

MODULE='scripts.curobo_arx.data_factory'


def wait_for_groups(active, campaign, timeout):
    """Drain in-flight ARX attempts before stopping their resident planner."""
    deadline=time.monotonic()+timeout
    while True:
        unresolved=[g['id'] for g in active.values() if g['process'].poll() is None]
        write(Path(campaign)/'cleanup_progress.json',{'wall':time.time(),'groups':unresolved,
            'seconds_remaining':max(0.,deadline-time.monotonic()),'new_dispatch_allowed':False})
        if not unresolved or time.monotonic()>=deadline:return unresolved
        time.sleep(.2)


def seed_kit_cache(cache):
    cache=Path(cache)
    if (cache/'seed_receipt.json').exists():return cache/'kit'
    cache.mkdir(parents=True,exist_ok=True)
    lock=(cache/'seed.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX)
    try:
        if (cache/'seed_receipt.json').exists():return cache/'kit'
        name='arx-data-factory-cache-seed-'+uuid.uuid4().hex[:10]
        subprocess.run(['podman','create','--name',name,'--entrypoint','/bin/true','localhost/arx-s02-isaac:6.0.1'],check=True,capture_output=True)
        try:
            (cache/'kit').mkdir(exist_ok=True)
            copied=subprocess.run(['podman','cp',name+':/isaac-sim/kit/cache/.',str(cache/'kit')],capture_output=True,text=True)
            if copied.returncode and 'no such file or directory' not in copied.stderr.lower():
                raise RuntimeError('Kit cache initialization failed: '+copied.stderr)
        finally:subprocess.run(['podman','rm',name],check=True,capture_output=True)
        write(cache/'seed_receipt.json',{'wall':time.time(),'image':'localhost/arx-s02-isaac:6.0.1',
            'source':'Image filesystem only; seed container was never started',
            'base_cache_present':copied.returncode==0,
            'initial_state':'Copied image cache' if copied.returncode==0 else 'Empty; Kit generates this directory at first startup',
            'purpose':'Persist derived geometry/program caches, never trajectories or images'})
        return cache/'kit'
    finally:lock.close()


def devices():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,compute_cap','--format=csv,noheader,nounits'],text=True,timeout=15)
    return [{'index':int(r[0]),'uuid':r[1].strip(),'free_mb':int(r[2]),'arch':r[3].strip()} for r in csv.reader(raw.splitlines())]


def allocate(inventory,count,compute=None,sim=None):
    """Conservative startup estimates, not a memory-fit guarantee."""
    ranked=sorted(inventory,key=lambda r:-r['free_mb'])
    def find(token):
        return next(d for d in inventory if str(d['index'])==str(token) or d['uuid']==token)
    compute=find(compute) if compute is not None else next((d for d in ranked if d['free_mb']>=6500),None)
    if compute is None or compute['free_mb']<6500:
        raise RuntimeError('No GPU has the 6500 MiB startup budget for resident grasp inference and both ARX planners')
    candidates=[find(token) for token in sim] if sim else [d for d in ranked if d['uuid']!=compute['uuid'] and d['free_mb']>=7500]
    if not candidates and not sim and compute['free_mb']>=6500+7500:
        candidates=[compute]
    if not candidates:raise RuntimeError('No GPU fits the registered ARX simulator startup budget, including shared compute')
    selected=[]
    for d in candidates:
        available=d['free_mb']-(6500 if d['uuid']==compute['uuid'] else 0)
        env_count=min(count,1+max(0,(available-7500)//1200))
        if available<7500:raise RuntimeError('Combined registered GPU memory estimate does not fit: '+d['uuid'])
        selected.append({**d,'env_count':env_count})
    return compute,selected


def environment(device):
    env={k:v for k,v in os.environ.items() if not k.startswith(('ARX_LILO_','GRASP_','ARX_CUROBO_'))}
    base=canonical_path(ROOT.parent/'tmp/arx-fullchain')
    env.update(CUDA_VISIBLE_DEVICES=device['uuid'],CUDA_DEVICE_ORDER='PCI_BUS_ID',TORCH_CUDA_ARCH_LIST=device['arch'],
        TORCH_EXTENSIONS_DIR=str(base/('torch_extensions_sm'+device['arch'].replace('.',''))),
        CUDA_HOME=str(base/'cuda'),CUDA_PATH=str(base/'cuda'),CC='/usr/bin/gcc',CXX='/usr/bin/g++',CUDAHOSTCXX='/usr/bin/g++',
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',OCNN_DISABLE_TRITON='1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',PYTHONUNBUFFERED='1',
        PATH=str(base/'cuda/bin')+':'+os.environ.get('PATH',''),CUROBO_SWEPT_MESH_BOUNDS='1')
    return env


def start_video_encoding(campaign,source):
    selection=read(campaign/'video_selection.json')
    missing=[r for r in selection['selected'] if not (campaign/'videos'/(r['id']+'.mp4')).exists()]
    if not missing:return None
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    log=(campaign/'video_encoding.log').open('a')
    process=subprocess.Popen([str(canonical_path(ROOT.parent/'tmp/arx-fullchain/env/bin/python')),'-m',MODULE+'.video',str(campaign)],
        cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    log.close();return process


def start_worker(campaign,kind,device,source):
    directory=campaign/'workers'/kind
    directory.mkdir(parents=True,exist_ok=False)
    py=canonical_path(ROOT.parent/('curobo/.venv/bin/python' if kind in ('planner','sdg') else 'tmp/arx-fullchain/env/bin/python'))
    if kind=='anygrasp':py=Path(read(campaign/'registration.json')['anygrasp']['python'])
    if kind=='graspness':py=Path(read(campaign/'registration.json')['graspness']['python'])
    log=(directory/'process.log').open('x')
    env=environment(device)
    if kind=='graspness':
        library_directories=read(campaign/'registration.json')['graspness'].get('library_directories',[])
        env['LD_LIBRARY_PATH']=os.pathsep.join([*library_directories,env.get('LD_LIBRARY_PATH','')])
    if kind=='sdg':env['CUDA_VISIBLE_DEVICES']=''
    p=subprocess.Popen([str(py),'-m',MODULE+'.workers',kind,'--directory',str(directory),'--registration',str(campaign/'registration.json')],
        cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    log.close()
    write(directory/'process.json',{'pid':p.pid,'gpu_uuid':device['uuid'] if kind!='sdg' else None,
        'device_kind':'cpu' if kind=='sdg' else 'cuda','kind':kind,'owned':True})
    return p,directory


def next_object(reg,ledger,excluded):
    return ledger.available_object(excluded)


def worker_startup_timeout(reg):
    seconds=float(reg.get('worker_startup_timeout_seconds',180))
    if not 0<seconds<=1800:
        raise ValueError('Registered worker startup timeout must be positive and at most 1800 seconds')
    return seconds


def wait_for_workers(processes,directories,timeout_seconds):
    """Bound only model initialization; inference and physical gates are unchanged."""
    started=time.monotonic();deadline=started+timeout_seconds
    while not all((d/'ready.json').exists() for d in directories):
        if any(p.poll() is not None for p in processes):
            raise RuntimeError('Resident worker startup failed: process exited; inspect workers/*/process.log')
        if time.monotonic()>deadline:
            raise RuntimeError('Resident worker startup failed: registered initialization timeout; inspect workers/*/process.log')
        time.sleep(.2)
    return time.monotonic()-started


def launch(campaign, *, compute_gpu=None,sim_gpus=None):
    campaign=Path(campaign).absolute()
    lock=(campaign/'coordinator.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    reg=read(campaign/'registration.json')
    candidate_kind=reg.get('candidate_backend','zerograsp')
    if candidate_kind not in ('zerograsp','sdg','anygrasp','graspness'):raise ValueError('Unknown registered candidate source')
    if candidate_kind in ('anygrasp','graspness') and candidate_kind not in reg:
        raise RuntimeError('Neural teacher registration is incomplete; no simulator or worker may start')
    if reg.get('production_ready') is False:raise RuntimeError('Registration preparation is incomplete')
    startup_timeout=worker_startup_timeout(reg)
    verify_files({**reg.get('execution_source_files',reg['source_files']),**reg['dependencies']})
    space(campaign)
    from .storage import container_storage_mounts
    storage_mounts=container_storage_mounts(campaign,reg)
    inventory=devices()
    try:
        compute,sim=allocate(inventory,reg['envs_per_gpu'],compute_gpu,sim_gpus)
    except Exception:
        write(campaign/('resource_preflight_'+str(time.time_ns())+'.json'),{'pass':False,'inventory':inventory,'error':traceback.format_exc()})
        write(campaign/'state.json',{'status':'resource_wait',
            'successes':len(reg.get('resume',{}).get('inherited_success_ids',[])),
            'physical_attempts':reg.get('resume',{}).get('inherited_physical_attempts',0),'quota_complete':False})
        report(campaign)
        raise
    if (campaign/'launch.json').exists():
        raise RuntimeError('Campaign has an existing launch; preserve it and explicitly recover interrupted claims before a new launch registration')
    started=time.time()
    launch_reg={'version':reg['version'],'started_unix':started,'stop_unix':min(started+3600*reg['wall_hours'],reg.get('deadline_unix',float('inf'))),
        'inventory':inventory,'compute':compute,'simulators':sim,'registration_sha256':sha(campaign/'registration.json'),
        'worker_startup_timeout_seconds':startup_timeout,
        'memory_estimates':'6500 MiB combined resident compute; simulator 7500 + 1200 per extra environment; no fit guarantee',
        'prior_results_modified':False,'authorized_task':'User requested independent implementation and production per automated_teacher_pipeline_20261029'}
    write(campaign/'launch.json',launch_reg)
    write(campaign/'state.json',{'status':'running','started_unix':started,
        'successes':len(reg.get('resume',{}).get('inherited_success_ids',[])),'new_successes':0,'quota_complete':False})
    processes=[];active={};excluded=set();stop=False;fault=None;video_process=None;last_video=0.;last_resources=0.
    def stop_request(*_):
        nonlocal stop
        stop=True
        write(campaign/'STOP',{'reason':'user/process stop request','wall':time.time()})
    signal.signal(signal.SIGINT,stop_request);signal.signal(signal.SIGTERM,stop_request)
    ledger=Ledger(campaign/'ledger.sqlite3')
    try:
        source=Path(reg['source_snapshot'])
        planner,planner_dir=start_worker(campaign,'planner',compute,source);processes.append(planner)
        neural,neural_dir=start_worker(campaign,candidate_kind,compute,source);processes.append(neural)
        startup_elapsed=wait_for_workers(processes,(planner_dir,neural_dir),startup_timeout)
        write(campaign/'worker_startup_receipt.json',{'wall':time.time(),'ready':True,
            'elapsed_seconds':startup_elapsed,'registered_timeout_seconds':startup_timeout,
            'inference_or_trajectory_timeouts_changed':False,'physical_attempts_during_startup':0})
        while True:
            if time.time()>=launch_reg['stop_unix'] or (campaign/'STOP').exists():stop=True
            for device in sim:
                gpu=device['uuid']
                if gpu in active:
                    group=active[gpu]
                    if group['process'].poll() is not None:
                        write(campaign/'groups'/group['id']/'process_exit_receipt.json',
                            {'wall':time.time(),'returncode':group['process'].returncode,'podman_process_exited':True,
                             'idle_or_clean_shutdown_not_inferred_from_exit_code':True})
                        result_path=campaign/'groups'/group['id']/'block_result.json'
                        if group['process'].returncode!=0 or not result_path.exists():
                            raise RuntimeError('Simulator group failed: '+group['id'])
                        result=read(result_path)
                        if not result.get('idle',result['status']=='candidate_exhausted'):
                            raise RuntimeError('Environment idle status not confirmed at object block boundary')
                        if result['status'] in ('candidate_exhausted','object_budget_exhausted'):excluded.add(group['object_id'])
                        del active[gpu]
                if stop or gpu in active:continue
                asset=next_object(reg,ledger,excluded | {g['object_id'] for g in active.values()})
                if asset is None:continue
                # Re-evaluate current free memory instead of relying on startup inventory.
                free=next(d for d in devices() if d['uuid']==gpu)['free_mb']
                if free<7500:raise RuntimeError('Simulator resource floor unavailable at task block boundary')
                env_count=min(device['env_count'],1+max(0,(free-7500)//1200))
                ident='group_'+uuid.uuid4().hex[:12]
                if ledger.acquire_object(ident,object_id=asset['object_id']) is None:continue
                group_launch={'group_id':ident,'gpu_uuid':gpu,'env_count':env_count,'renderer':asset['renderer'],
                    'object_id':asset['object_id'],'planner_directory':str(planner_dir),'candidate_directory':str(neural_dir),
                    'candidate_backend':candidate_kind,
                    'stop_unix':launch_reg['stop_unix'],'registration_sha256':launch_reg['registration_sha256']}
                path=campaign/'group_registrations'/(ident+'.json');write(path,group_launch)
                log_path=campaign/'group_registrations'/(ident+'.log')
                log=log_path.open('x')
                cache_root=ROOT/'runs/data_factory_runtime_cache'
                for name in ('ov','nvidia'):(cache_root/name).mkdir(parents=True,exist_ok=True)
                kit_cache=seed_kit_cache(cache_root/gpu)
                command=['podman','run','--rm','--name','arx-data-factory-'+ident,'--user','0',
                    '--device','nvidia.com/gpu='+gpu,'--security-opt=label=disable','--network=host','--shm-size=2g',
                    '-e','ACCEPT_EULA=Y','-e','PRIVACY_CONSENT=N','-e','OMNI_KIT_ALLOW_ROOT=1',
                    '-e','CUDA_VISIBLE_DEVICES='+gpu,'-e','CUDA_DEVICE_ORDER=PCI_BUS_ID','-e','NVIDIA_VISIBLE_DEVICES=void',
                    '-e','PYTHONUNBUFFERED=1','-e','PYTHONDONTWRITEBYTECODE=1',
                    '-v',str(cache_root/'ov')+':/root/.cache/ov:rw',
                    '-v',str(cache_root/'nvidia')+':/root/.cache/nvidia:rw',
                    '-v',str(kit_cache)+':/isaac-sim/kit/cache:rw',
                    '-v',str(ROOT.parent)+':'+str(ROOT.parent)+':rw','-v',str(canonical_path(ROOT/'asset'))+':/workspace/asset:ro',
                    '-v',str(ROOT)+':/workspace/COOLAS-ARX:ro',*storage_mounts,'-w',str(source),'--entrypoint','/isaac-sim/python.sh',
                    'localhost/arx-s02-isaac:6.0.1','-m',MODULE+'.runtime','--campaign',str(campaign),'--launch',str(path)]
                p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close()
                active[gpu]={'id':ident,'object_id':asset['object_id'],'process':p}
                write(path.with_suffix('.process.json'),{'pid':p.pid,'container':'arx-data-factory-'+ident,'command':command})
            if any(p.poll() is not None for p in processes):raise RuntimeError('Resident compute worker exited; stop dispatch')
            summary=report(campaign)
            write(campaign/'state.json',{'status':'running','started_unix':started,
                'successes':summary['successful_demonstrations'],'new_successes':summary['new_successful_demonstrations'],
                'physical_attempts':summary['physical_attempts'],'quota_complete':summary['quota_complete'],
                'updated_unix':time.time()})
            if time.time()-last_resources>=30:
                with (campaign/'resources.jsonl').open('a') as stream:
                    stream.write(__import__('json').dumps({'wall':time.time(),'devices':devices(),
                        'successful_demonstrations':summary['successful_demonstrations'],'physical_attempts':summary['physical_attempts']})+'\n')
                last_resources=time.time()
            if (video_process is None or video_process.poll() is not None) and time.time()-last_video>=60:
                video_process=start_video_encoding(campaign,source);last_video=time.time()
            if summary['quota_complete']:stop=True
            if not active:
                if stop or next_object(reg,ledger,excluded) is None:break
            if stop:
                for group in active.values():
                    # Owned new groups only. Signal requests graceful no-more-dispatch;
                    # runtime completes current trajectories before closing the World.
                    subprocess.run(['podman','kill','--signal','TERM','arx-data-factory-'+group['id']],capture_output=True)
            time.sleep(2)
        status='quota_complete' if summary['quota_complete'] else ('stopped' if (campaign/'STOP').exists() else 'budget_exhausted')
    except BaseException:
        fault=traceback.format_exc()
        status='fault'
        write(campaign/'FAULT.json',{'error':fault,'active_groups':{k:{n:v for n,v in g.items() if n!='process'} for k,g in active.items()},'wall':time.time()})
        write(campaign/'STOP',{'reason':'infrastructure/record/resource fault','wall':time.time()})
    finally:
        # Infrastructure stop: preserve evidence, ask only our groups to stop.
        for group in active.values():
            subprocess.run(['podman','kill','--signal','TERM','arx-data-factory-'+group['id']],capture_output=True)
        drain_seconds=(max(180.,reg['attempt_seconds']+60.) if reg.get('stage') in ('sdg_stage1','graspness_teacher','anygrasp_teacher') else 180.)
        unresolved=wait_for_groups(active,campaign,drain_seconds)
        if unresolved:
            status='fault'
            write(campaign/'cleanup_unknown.json',{'groups':unresolved,'automatic_reuse':False})
        for kind in ('planner',candidate_kind):
            directory=campaign/'workers'/kind
            if directory.exists():write(directory/'STOP',{'wall':time.time()})
        for p in processes:
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGTERM)
        ledger.close()
        write(campaign/'state.json',{'status':status,'finished_unix':time.time(),'quota_complete':status=='quota_complete','error':fault,'unresolved_groups':unresolved})
        report(campaign)
        if video_process is not None:
            try:video_process.wait(timeout=60)
            except subprocess.TimeoutExpired:write(campaign/'video_encoding_pending.json',{'pid':video_process.pid})
        if video_process is None or video_process.poll() is not None:
            start_video_encoding(campaign,Path(reg['source_snapshot']))
        lock.close()
    if fault:raise RuntimeError(fault)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('campaign',type=Path)
    p.add_argument('--compute-gpu');p.add_argument('--sim-gpus',nargs='+')
    a=p.parse_args();launch(a.campaign,compute_gpu=a.compute_gpu,sim_gpus=a.sim_gpus)
