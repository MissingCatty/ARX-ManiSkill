"""Bounded serial Worlds; isolate a timed-out slot without retrying it."""
from arx_project.layout import canonical_path
import argparse
import fcntl
from pathlib import Path
import subprocess
import time
import traceback
from scripts.curobo_arx.data_factory.common import ROOT,read,write,sha,verify_files,space
from scripts.curobo_arx.data_factory.launch import devices,environment,seed_kit_cache
from scripts.curobo_arx.data_factory.watchdog import process_info,exclusive_stop
from scripts.curobo_arx.seed_augmentation.campaign import available_memory
from .advisor31_campaign import exited
from .advisor32_spec import validate_trials
from .advisor34_audit import audit_partial
from .campaign import summarize
from .physical_startup import startup_expired
from .resources import probe_cuda
from .advisor34_boundary import ISOLATED_OUTCOMES


def terminal_isolation(results,selected,marker):
    if not results or [r['trial_id'] for r in results]!=selected[:len(results)]:raise RuntimeError('Non-prefix terminal slots')
    last=results[-1]
    if (not marker or marker['trial_id']!=last['trial_id'] or last['outcome'] not in ISOLATED_OUTCOMES
        or not marker['terminal_records_complete'] or not marker['configuration_intact'] or marker['retry']
        or not marker['next_slot_requires_new_World']):raise RuntimeError('Unbound terminal isolation request')
    return True


def summary(root,reg):
    value=summarize(root);carried=reg.get('carried_results',[])
    if carried:
        ids={r['trial_id'] for r in carried}
        if ids&{r['trial_id'] for r in value['trials']}:raise RuntimeError('Carried trial repeated')
        value['new_completed']=value['completed'];value['carried_completed']=len(carried)
        value['trials']=carried+value['trials'];value['completed']+=len(carried)
        value['unexecuted']=[x for x in value['unexecuted'] if x not in ids]
        value['actual_attempts']+=sum(bool(r['physical_dispatches']) for r in carried)
        value['verified_physical_successes']+=sum(bool(r['success']) for r in carried)
        value['ledger_admitted']+=sum(bool(r['admitted']) for r in carried)
        value['by_method']={m:{'completed':sum(r['method']==m for r in value['trials']),
            'success':sum(r['method']==m and r['success'] for r in value['trials']),
            'admitted':sum(r['method']==m and r['admitted'] for r in value['trials'])} for m in ('C','D','recovery')}
        write(root/'summary.json',value)
    interrupted=reg.get('supplemental_interruptions',{})
    if interrupted:
        if set(interrupted)&{r['trial_id'] for r in value['trials']}:raise RuntimeError('Interrupted slot was replayed')
        value['unexecuted']=[x for x in value['unexecuted'] if x not in interrupted]
        value['actual_attempts']+=sum(bool(x['detail']['physical_attempt']) for x in interrupted.values())
        value['supplemental_interrupted']=list(interrupted)
        write(root/'summary.json',value)
    return value


def owned_process(command,cwd,env,directory):
    directory.mkdir(parents=True,exist_ok=False)
    with (directory/'process.log').open('x') as log:
        process=subprocess.Popen(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(directory/'process.json',{'pid':process.pid,'command':command,'owned':True,**(process_info(process.pid) or {})})
    return process


def wait_ready(process,directory,root):
    deadline=time.monotonic()+600
    while not (directory/'ready.json').exists():
        if process.poll() is not None or (directory/'FAULT.json').exists() or (root/'STOP').exists() or time.monotonic()>deadline:
            raise RuntimeError('Owned service startup failed: '+str(directory))
        time.sleep(.25)


def healthy(root,processes,require_idle=False):
    for role,p in processes.items():
        directory=root/'workers'/role
        if p.poll() is not None or (directory/'FAULT.json').exists():raise RuntimeError('Owned '+role+' service failed')
        if require_idle:
            if role=='planner' and (list((directory/'requests').glob('*.json')) or list((directory/'requests').glob('*.running'))):return False
            if role=='policy' and any(not (directory/'done'/p.name).exists() for p in (directory/'queue').glob('*.json')):return False
    return True


def container_command(root,reg,sim,path,name):
    project=Path(reg['project_root']);cache=project/'runs/data_factory_runtime_cache';kit=seed_kit_cache(cache/sim)
    cmd=['podman','run','--pull=never','--rm','--name',name,'--user','0','--device','nvidia.com/gpu='+sim,
        '--security-opt=label=disable','--network=host','--shm-size=2g']
    for v in ('ACCEPT_EULA=Y','PRIVACY_CONSENT=N','OMNI_KIT_ALLOW_ROOT=1','CUDA_VISIBLE_DEVICES='+sim,
        'CUDA_DEVICE_ORDER=PCI_BUS_ID','NVIDIA_VISIBLE_DEVICES=void','PYTHONUNBUFFERED=1','PYTHONDONTWRITEBYTECODE=1'):cmd+=['-e',v]
    for mount in (str(cache/'ov')+':/root/.cache/ov:rw',str(cache/'nvidia')+':/root/.cache/nvidia:rw',
        str(kit)+':/isaac-sim/kit/cache:rw',str(project.parent)+':'+str(project.parent)+':rw',
        '/coolas/data/arx_teacher_zhaoyechong:/coolas/data/arx_teacher_zhaoyechong:ro',
        str(canonical_path(project/'asset'))+':/workspace/asset:ro',str(project)+':/workspace/COOLAS-ARX:ro'):cmd+=['-v',mount]
    return cmd+['-w',reg['source_snapshot'],'--entrypoint','/isaac-sim/python.sh','localhost/arx-s02-isaac:6.0.1',
        '-m',reg['runtime_module'],'--campaign',str(root),'--launch',str(path)]


def run(root):
    root=Path(root).resolve();reg=read(root/'registration.json');roles=reg['requested_devices'];project=Path(reg['project_root'])
    lock=(root/'run.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'STOP').exists() or (root/'STARTED.json').exists():raise RuntimeError('No in-place retry or STOP bypass')
    write(root/'STARTED.json',{'wall':time.time(),'registration_sha256':sha(root/'registration.json')})
    processes={};active=None;name=None;error=None;unresolved=False;terminals=set(reg.get('carried_terminal_ids',[]));groups=0
    try:
        validate_trials(reg['seed_trials']);verify_files({**reg['dependencies'],**reg['execution_source_files']})
        write(root/'fresh_predecessor_exit.json',exited(reg.get('continuation_of',project/'runs/nearfield_advisor31_20261110_01')))
        stable=0
        while stable<3:
            if (root/'STOP').exists() or time.time()>reg['deadline_unix']:raise RuntimeError('Stopped/resource deadline')
            inventory={d['uuid']:d for d in devices()};space(root)
            needed={roles['policy_uuid']:9500,roles['sim_uuid']:9000}
            if roles['compute_uuid']==roles['policy_uuid']:needed[roles['policy_uuid']]=12000
            else:needed[roles['compute_uuid']]=5000
            ok=all(inventory[u]['free_mb']>=floor for u,floor in needed.items()) and available_memory()>=reg['minimum_available_host_memory_bytes']
            stable=stable+1 if ok else 0
            write(root/'state.json',{'status':'waiting_for_resources','stable_samples':stable,'required_mib':needed,
                'free_mib':{u:d['free_mb'] for u,d in inventory.items()},'wall':time.time()})
            if stable<3:time.sleep(5)
        receipts=[]
        for uuid in sorted(set(roles.values())):
            proof=probe_cuda(inventory[uuid],environment(inventory[uuid]));receipts.append(proof)
            write(root/'cuda_preflight.json',{'pass':all(r['pass'] for r in receipts),'devices':receipts})
            if not proof['pass']:raise RuntimeError('CUDA preflight failed')
        for role,python,module,args in (
            ('policy',project.parent/'tmp/arx-show-harness-env/bin/python',reg['policy_module'],[str(root)]),
            ('planner',project.parent/'curobo/.venv/bin/python',reg['worker_module'],['--directory',str(root/'workers/planner'),'--registration',str(root/'registration.json')])):
            directory=root/'workers'/role;env=environment(inventory[roles['compute_uuid' if role=='planner' else 'policy_uuid']]);env['PYTHONPATH']=reg['source_snapshot'];env['PYTHONUNBUFFERED']='1'
            processes[role]=owned_process([str(python),'-m',module,*args],reg['source_snapshot'],env,directory)
            if role=='policy':
                (directory/'queue').mkdir();(directory/'done').mkdir()
            wait_ready(processes[role],directory,root)
        write(root/'launch.json',{'wall':time.time(),'roles':roles,'registration_sha256':sha(root/'registration.json')})
        while len(terminals)<30:
            healthy(root,processes);space(root)
            if (root/'STOP').exists() or time.time()>reg['deadline_unix']:raise RuntimeError('Stopped/deadline')
            pending=[t for t in reg['seed_trials'] if t['trial_id'] not in terminals];renderer=pending[0]['renderer'];selected=[]
            for t in pending:
                if t['renderer']!=renderer:break
                selected.append(t['trial_id'])
            if next(d['free_mb'] for d in devices() if d['uuid']==roles['sim_uuid'])<9000:raise RuntimeError('Simulator memory below floor')
            if available_memory()<reg['minimum_available_host_memory_bytes']:raise RuntimeError('Host memory below floor')
            ident=f'advisor34_{groups:02d}_{renderer.lower()}';groups+=1;name='arx-'+root.name+'-'+ident
            if subprocess.run(['podman','container','exists',name],capture_output=True,timeout=30).returncode!=1:raise RuntimeError('Old World absence unconfirmed')
            path=root/'group_registrations'/(ident+'.json');group=root/'groups'/ident
            write(path,{'group_id':ident,'renderer':renderer,'gpu_uuid':roles['sim_uuid'],'env_count':1,
                'planner_directory':str(root/'workers/planner'),'candidate_directory':str(root/'no_neural_requests'),
                'trial_ids':selected,'stop_unix':reg['deadline_unix'],'registration_sha256':sha(root/'registration.json')})
            cmd=container_command(root,reg,roles['sim_uuid'],path,name)
            with path.with_suffix('.log').open('x') as log:active=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(path.with_suffix('.process.json'),{'pid':active.pid,'container':name,'command':cmd,'owned':True,**(process_info(active.pid) or {})})
            began=time.monotonic()
            while active.poll() is None:
                healthy(root,processes)
                if (root/'STOP').exists() or time.time()>reg['deadline_unix']:raise RuntimeError('Stopped during execution')
                if startup_expired(group,time.monotonic()-began,1800):raise RuntimeError('Simulator startup exceeded registered1800s')
                completed=read(group/'results.json')['trials'] if (group/'results.json').exists() else []
                write(root/'state.json',{'status':'running','wall':time.time(),'group':ident,'renderer':renderer,
                    'completed':len(terminals)+len(completed),'scheduled_slots':30,
                    'current_results':[{'trial_id':r['trial_id'],'success':r['success'],'outcome':r['outcome']} for r in completed]})
                time.sleep(2)
            code=active.returncode
            if subprocess.run(['podman','container','exists',name],capture_output=True,timeout=30).returncode!=1:raise RuntimeError('World removal unconfirmed')
            active=None
            results=read(group/'results.json')['trials'] if (group/'results.json').exists() else []
            for r in results:
                if r['trial_id'] in terminals:raise RuntimeError('Repeated physical slot')
                terminals.add(r['trial_id'])
            timeouts=list(group.glob('trials/*/ISOLATED_TIMEOUT.json'))
            if code or (group/'FAULT.json').exists():
                if len(timeouts)!=1:raise RuntimeError('Simulator infrastructure/record fault; inspect '+str(group))
                timeout=timeouts[0];receipt=read(timeout);proof=audit_partial(timeout.parent)
                if receipt['trial_id'] in terminals:raise RuntimeError('Repeated timeout slot')
                deadline=time.monotonic()+120
                while not healthy(root,processes,require_idle=True):
                    if time.monotonic()>deadline:raise RuntimeError('Workers did not become idle after isolated slot')
                    time.sleep(.25)
                frozen=read(timeout.parent/'frozen_after.json')
                if not frozen['pass']:raise RuntimeError('Frozen configuration failure during timeout')
                verify_files({**reg['execution_source_files'],**reg['dependencies']})
                write(timeout.parent/'isolation_receipt.json',{'pass':True,'old_World_absent':True,'services_healthy_and_idle':True,
                    'partial_record_audit':proof,'configuration_intact':True,'new_World_reset_required_before_next_dispatch':True,
                    'old_STOPs_retained':True,'retry':False,'wall':time.time()})
                terminals.add(receipt['trial_id'])
            else:
                shutdown=read(group/'shutdown_receipt.json')
                if not shutdown['all_trajectories_idle']:raise RuntimeError('Missing idle receipt')
                markers=list(group.glob('trials/*/ISOLATED_TERMINAL.json'))
                if len(results)!=len(selected) or markers:
                    if len(markers)!=1:raise RuntimeError('Missing unique terminal isolation receipt')
                    terminal_isolation(results,selected,read(markers[0]))
                    if not healthy(root,processes,require_idle=True):raise RuntimeError('Services not idle after terminal isolation')
                    verify_files({**reg['execution_source_files'],**reg['dependencies']})
                    write(markers[0].parent/'isolation_receipt.json',{'pass':True,'old_World_absent':True,
                        'services_healthy_and_idle':True,'terminal_record_complete':True,'configuration_intact':True,
                        'retry':False,'next_slot_requires_verified_new_World_reset':True,'wall':time.time()})
            write(root/'terminal_slots.json',{'trial_ids':sorted(terminals),'scheduled':30,'retry':False})
            summary(root,reg)
        verify_files({**reg['execution_source_files'],**reg['dependencies']})
    except BaseException:
        error=traceback.format_exc();write(root/'coordinator_fault.json',{'error':error,'wall':time.time()})
        exclusive_stop(root/'STOP',{'reason':'resource/configuration/record/infrastructure fault','error':error})
    finally:
        if active is not None and active.poll() is None:
            subprocess.run(['podman','kill','--signal','TERM',name],capture_output=True,timeout=30)
            try:active.wait(timeout=30)
            except subprocess.TimeoutExpired:
                subprocess.run(['podman','kill','--signal','KILL',name],capture_output=True,timeout=30)
                try:active.wait(timeout=30)
                except subprocess.TimeoutExpired:unresolved=True
            if subprocess.run(['podman','container','exists',name],capture_output=True,timeout=30).returncode!=1:unresolved=True
        for role,p in processes.items():
            directory=root/'workers'/role;write(directory/'STOP',{'reason':'bounded advisor34 campaign ended'})
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=30)
            write(directory/'process_exit.json',{'pid':p.pid,'exit_code':p.returncode,'wall':time.time()})
        result=summary(root,reg)
        write(root/'state.json',{'status':'fault' if error or unresolved else 'complete','completed':result['completed'],
            'terminal_slots':len(terminals),'error':error,'unconfirmed_world':unresolved,'finished_unix':time.time()})
    if not unresolved:
        from .advisor34_report import report
        report(root)
    if error:raise RuntimeError(error)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('campaign');run(p.parse_args().campaign)
