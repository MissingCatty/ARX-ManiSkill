"""Stdlib augmentation queue watched independently of native-recovery waits."""
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

try:
    from . import watchdog as wd
except ImportError:
    import watchdog as wd

AUTO='watchdog: seed augmentation unexpected exit'
AUTO_REASONS={AUTO,'Owned seed planner exited','augmentation deadline reached'}
FINAL={'complete','fault','stopped'}
INTEGRITY=wd.INTEGRITY_ERRORS+('registered successful command changed','successful tail command changed',
    'seed identity','seed ledger acceptance','source tail dispatched','source admission changed',
    'executed path lacks','unregistered result','reconciled trial conflicts','ledger is corrupt')


def owned(meta,campaign,role):
    info=wd.process_info(meta['pid'])
    if not info or info['uid']!=os.getuid():return None
    scope=str(Path(campaign).resolve());args=info['argv']
    if not any(a==scope or a.startswith(scope+'/') for a in args):return None
    module=('scripts.curobo_arx.seed_augmentation.entrypoint' if role=='coordinator'
            else 'scripts.curobo_arx.seed_augmentation.worker' if role=='worker' else 'podman')
    if role=='group':
        if Path(args[0]).name!='podman':return None
    elif module not in args:return None
    if role=='group' and meta['container'] not in args:return None
    if any(k in meta and meta[k]!=info[k] for k in ('boot_id','start_ticks')):return None
    return info


def group_records(campaign):
    campaign=Path(campaign);registration=wd.sha(campaign/'registration.json')
    expected={t['trial_id'] for t in wd.read(campaign/'registration.json')['seed_trials']}
    for path in campaign.glob('group_registrations/*.json'):
        if path.name.endswith('.process.json'):continue
        group=wd.read(path)
        if group.get('registration_sha256')!=registration or group['group_id']!=path.stem:
            raise ValueError('Augmentation group registration identity mismatch')
        if group['env_count']!=1 or not set(group['trial_ids'])<=expected:
            raise ValueError('Unregistered augmentation World/trial scope')
        container='arx-seed-augmentation-'+campaign.name+'-'+group['renderer'].lower()
        yield group,path.with_name(path.stem+'.process.json'),container


def manual_stop(campaign):
    path=Path(campaign)/'STOP'
    return path.exists() and wd.read(path).get('reason') not in AUTO_REASONS


def prove_exited(campaign):
    campaign=Path(campaign);processes=[];containers=[]
    files=[(campaign/'coordinator_process.json','coordinator')]
    files += [(p,'worker') for p in campaign.glob('workers/*/process.json')]
    groups=list(group_records(campaign))
    files += [(p,'group') for _,p,_ in groups]
    if (campaign/'launch.json').exists() and not files[0][0].exists():
        raise ValueError('Missing augmentation coordinator ownership evidence')
    for path,role in files:
        if not path.exists():continue
        meta=wd.read(path)
        if owned(meta,campaign,role):return None
        processes.append({'path':str(path),'pid':meta['pid'],'exited':True})
    for _,_,name in groups:
        if wd.container_running(name) is not None:return None
        containers.append({'container':name,'absent':True})
    # Exclude the bootstrap/Popen-to-metadata crash window as well.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        info=wd.process_info(int(proc.name))
        if not info or info['uid']!=os.getuid():continue
        args=info['argv'];scope=str(campaign.resolve())
        if scope in args and any('seed_augmentation' in a or '--exec-augmentation'==a for a in args):return None
    return {'wall':time.time(),'all_owned_processes_exited':True,'all_virtual_Worlds_absent':True,
        'processes':processes,'containers':containers,'pre_crash_convergence_not_inferred':True}


class AugmentationWatchdog:
    def __init__(self,config,watchdog_config,stop_requested):
        self.config_path=Path(config);self.cfg=wd.read(config);self.home=Path(self.cfg['home'])
        self.watchdog_config=Path(watchdog_config);self.stop_requested=stop_requested
        path=self.home/'state.json'
        self.state=wd.read(path) if path.exists() else {'phase':'idle','allocated_slots':0,
            'reserved_seed_ids':list(self.cfg.get('excluded_seed_ids',[])),
            'batches':[],'physical_attempts':0,'admitted':0,'admitted_by_policy':{},'restart_times':[]}
        if self.state.get('phase')=='draining' and self.state.get('drain_boot_id')!=wd.boot_id():
            self.state.update(drain_started=time.time(),drain_boot_id=wd.boot_id())

    def save(self,**values):
        self.state.update(values,checked_unix=time.time(),watchdog_pid=os.getpid())
        wd.write(self.home/'state.json',self.state)

    def event(self,name,**fields):
        with (self.home/'events.jsonl').open('a') as f:
            import json
            f.write(json.dumps({'wall':time.time(),'event':name,**fields})+'\n');f.flush();os.fsync(f.fileno())

    def spawn_job(self,kind,**fields):
        ident=str(time.time_ns());request=self.home/'jobs'/(ident+'.json')
        values={'kind':kind,'config':str(self.config_path),'result':str(request.with_suffix('.result.json')),
            'process_metadata':str(request.with_suffix('.process.json')),**fields}
        wd.write(request,values)
        self.save(phase='helper',job=str(request),job_kind=kind,job_started=time.time())
        self.spawn(['--exec-aug-job',str(request)],request.with_suffix('.log'))

    def spawn(self,arguments,log):
        command=[self.cfg['monitor_python'],str(Path(__file__).with_name('watchdog.py')),
                 '--config',str(self.watchdog_config),*arguments]
        with Path(log).open('a') as f:
            subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,
                start_new_session=True,close_fds=True)

    def recovery_allowed(self,reason):
        now=time.time()
        if now<self.state.get('recovery_after',0):return False
        history=[t for t in self.state.get('restart_times',[]) if now-t<3600]
        if len(history)>=self.cfg.get('max_restarts_per_hour',4):
            self.save(health='recovery_backoff',recovery_after=history[0]+3600);return False
        history.append(now)
        self.save(restart_times=history,recovery_after=now+self.cfg.get('backoff_seconds',60))
        self.event('unexpected_exit_recovery',reason=reason)
        return True

    def helper(self):
        request=Path(self.state['job']);job=wd.read(request);result=Path(job['result'])
        if result.exists():
            value=wd.read(result);status=value['status']
            if status=='blocked':self.save(phase='blocked',reason=value['error']);return
            if status=='waiting_seeds':self.save(phase='idle',retry_after=time.time()+60,selection=value);return
            if status=='budget_complete':self.save(phase='complete');return
            if status=='prepared':
                campaign=value['campaign'];batch={'campaign':campaign,'slots':value['new_allocated_slots'],
                    'seed_ids':value['seed_ids'],'accounted':False}
                if any(b['campaign']==campaign for b in self.state['batches']):raise ValueError('Duplicate augmentation batch allocation')
                allocated=self.state['allocated_slots']+batch['slots']
                if allocated>self.cfg['max_slots']:raise ValueError('Augmentation global slot budget exceeded')
                self.state['batches'].append(batch)
                self.state['reserved_seed_ids']=sorted(set(self.state['reserved_seed_ids'])|set(value['seed_ids']))
                self.save(phase='prepared',campaign=campaign,allocated_slots=allocated);return
            if status=='finished':
                batch=next(b for b in self.state['batches'] if b['campaign']==value['campaign'])
                if batch['accounted']:raise ValueError('Augmentation batch accounted twice')
                batch.update(accounted=True,result=str(result))
                policies=dict(self.state.get('admitted_by_policy',{}))
                for name,count in value['admitted_by_policy'].items():policies[name]=policies.get(name,0)+count
                self.state.update(physical_attempts=self.state['physical_attempts']+value['physical_attempts'],
                    admitted=self.state['admitted']+value['admitted'],admitted_by_policy=policies)
                child=value.get('continuation')
                if child:
                    self.state['batches'].append({'campaign':child['campaign'],'slots':0,'seed_ids':child['seed_ids'],'accounted':False})
                    self.save(phase='prepared',campaign=child['campaign']);return
                self.save(phase='idle',campaign=None,retry_after=time.time()+1);return
            raise ValueError('Incomplete augmentation helper result')
        meta=Path(job['process_metadata'])
        if meta.exists():
            data=wd.read(meta);info=wd.process_info(data['pid'])
            if info and all(data.get(k)==info[k] for k in ('boot_id','start_ticks')):
                if str(request) in info['argv']:return
                raise ValueError('Augmentation helper ownership mismatch')
        if time.time()-self.state['job_started']<60:return
        # No helper dispatches robot commands. Preserve its partial output and
        # repeat only preparation/postprocessing, never a started trial.
        if self.recovery_allowed('Read-only helper exited without durable result'):
            fields={k:v for k,v in job.items() if k not in ('kind','config','result','process_metadata')}
            fields['out']=str(Path(self.cfg['runs_root'])/(self.cfg['campaign_prefix']+'_recovered_'+str(time.time_ns())))
            self.spawn_job(job['kind'],**fields)

    def start(self):
        campaign=Path(self.state['campaign'])
        if manual_stop(campaign):self.save(phase='manual_stop');return
        meta=campaign/'coordinator_process.json'
        if meta.exists() and owned(wd.read(meta),campaign,'coordinator'):
            self.save(phase='watching',bad=0);return
        if (campaign/'launch.json').exists():
            self.save(phase='draining',drain_started=time.time(),drain_boot_id=wd.boot_id(),resume=True);return
        if not self.resources():return
        intent=campaign/'watchdog_launch_intent.json'
        if intent.exists() and time.time()-intent.stat().st_mtime<60:return
        wd.write(intent,{'wall':time.time(),'config':str(self.config_path)})
        self.spawn(['--exec-augmentation',str(campaign)],campaign/'coordinator.log')
        self.save(phase='watching',launch_started=time.time(),bad=0)

    def resources(self):
        result=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
        if result.returncode:raise RuntimeError('GPU inventory unavailable')
        values={r.split(',')[0].strip():int(r.split(',')[1]) for r in result.stdout.splitlines()}
        available=int(next(line for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')).split()[1])*1024
        selected=next((u for u in self.cfg['preferred_gpu_uuids'] if values.get(u,0)>=14000),None)
        if selected is None or available<40*1024**3:
            self.save(health='waiting_resources',reason='Wait for original combined GPU and 40-GiB host startup floors');return False
        self.state['selected_gpu_uuid']=selected;return True

    def monitor(self):
        campaign=Path(self.state['campaign'])
        if manual_stop(campaign):self.save(phase='manual_stop');return
        for path in [campaign/'FAULT.json',campaign/'BOOTSTRAP_FAULT.json',*campaign.glob('workers/*/FAULT.json'),*campaign.glob('groups/*/FAULT.json')]:
            if path.exists() and any(t in str(wd.read(path).get('error','')).lower() for t in INTEGRITY):
                self.save(phase='blocked',reason='Augmentation record/configuration fault: '+str(path));return
        state=wd.read(campaign/'state.json')
        meta=campaign/'coordinator_process.json'
        alive=meta.exists() and owned(wd.read(meta),campaign,'coordinator')
        if alive:
            for worker in campaign.glob('workers/*/process.json'):
                if not owned(wd.read(worker),campaign,'worker'):alive=None;break
            for group,path,_ in group_records(campaign):
                shutdown=campaign/'groups'/group['group_id']/'shutdown_receipt.json'
                if shutdown.exists() and wd.read(shutdown).get('all_trajectories_idle'):continue
                if path.exists() and not owned(wd.read(path),campaign,'group'):alive=None;break
        if alive and time.time()<self.cfg['deadline_unix']:
            screen=campaign/'model_screen.json';stage='model_activation'
            if list(campaign.glob('group_registrations/*.json')):stage='ARX_execution'
            self.save(health='healthy',bad=0,stage=stage,
                model_checks_completed=len(wd.read(screen).get('rows',[])) if screen.exists() else 0);return
        if not meta.exists() and time.time()-self.state.get('launch_started',0)<60:return
        bad=self.state.get('bad',0)+1;self.save(bad=bad,health='failed')
        if bad<self.cfg.get('failure_confirmations',2):return
        normal=state.get('status')=='complete' or state.get('model_activation_rejected')
        resume=not normal and time.time()<self.cfg['deadline_unix']
        wd.exclusive_stop(campaign/'STOP',{'reason':AUTO,'wall':time.time()})
        self.event('batch_ended_or_exited',campaign=str(campaign),normal=bool(normal),resume_unused=resume)
        self.save(phase='draining',drain_started=time.time(),drain_boot_id=wd.boot_id(),resume=resume)

    def drain(self):
        campaign=Path(self.state['campaign'])
        if manual_stop(campaign):self.save(phase='manual_stop');return
        if time.time()-self.state['drain_started']>self.cfg.get('drain_seconds',1080):
            self.save(phase='blocked',reason='Old augmentation World exit remains unconfirmed');return
        meta=campaign/'coordinator_process.json'
        if meta.exists() and owned(wd.read(meta),campaign,'coordinator'):return
        present=False
        for _,_,name in group_records(campaign):
            value=wd.container_running(name)
            if value is not None:
                present=True
                subprocess.run(['podman','kill','--signal','TERM',name] if value else ['podman','rm',name],capture_output=True,timeout=15)
        if present:return
        for path in campaign.glob('workers/*/process.json'):
            wd.exclusive_stop(path.parent/'STOP',{'reason':AUTO,'wall':time.time()})
            info=owned(wd.read(path),campaign,'worker')
            if info:
                if time.time()-self.state['drain_started']>30:os.kill(info['pid'],signal.SIGTERM)
                return
        proof=prove_exited(campaign)
        if not proof:return
        wd.write(campaign/'watchdog_exit_proof.json',proof)
        if self.state.get('resume') and not self.recovery_allowed('Augmentation controller/worker/World exited'):
            return
        child=Path(self.cfg['runs_root'])/(self.cfg['campaign_prefix']+'_resume_'+str(time.time_ns()))
        self.spawn_job('finish',campaign=str(campaign),resume=self.state.get('resume',False),out=str(child))

    def step(self):
        if not wd.storage_ready(self.cfg):self.save(health='waiting_storage');return
        if (self.home/'STOP').exists():self.save(phase='manual_stop');return
        phase=self.state['phase']
        if phase=='helper':self.helper()
        elif phase=='prepared':self.start()
        elif phase=='watching':self.monitor()
        elif phase=='draining':self.drain()
        elif phase=='idle':
            if time.time()>=self.cfg['deadline_unix'] or self.cfg['max_slots']-self.state['allocated_slots']<6:
                self.save(phase='complete');return
            if time.time()<self.state.get('retry_after',0):return
            if not self.resources():return
            out=Path(self.cfg['runs_root'])/(self.cfg['campaign_prefix']+'_'+str(time.time_ns()))
            self.spawn_job('prepare',out=str(out),excluded_seed_ids=self.state['reserved_seed_ids'],
                remaining_slots=self.cfg['max_slots']-self.state['allocated_slots'])
        self.save()

    def run(self):
        with (self.home/'manager.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            while not self.stop_requested():
                try:self.step()
                except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as error:
                    self.event('check_error',error=repr(error));self.save(phase='blocked',reason=repr(error))
                for _ in range(self.cfg.get('poll_seconds',20)):
                    if self.stop_requested():break
                    time.sleep(1)


def bootstrap(watchdog_config,campaign=None,job=None):
    outer=wd.read(watchdog_config)
    if wd.sha(outer['augmentation']['config'])!=outer['augmentation']['sha256']:
        raise ValueError('Registered input/source changed: augmentation configuration')
    cfg=wd.read(outer['augmentation']['config']);home=Path(cfg['home'])
    if (Path(watchdog_config).parent/'STOP').exists() or (home/'STOP').exists():return
    source=Path(cfg['source_snapshot'])
    for name,expected in cfg['bootstrap_source_hashes'].items():
        if wd.sha(name)!=expected:raise ValueError('Registered input/source changed: '+name)
    if job:
        request=Path(job).resolve()
        if not request.is_relative_to(home/'jobs'):raise ValueError('Unscoped augmentation helper request')
        task=wd.read(request);metadata=Path(task['process_metadata'])
        command=[cfg['collector_python'],'-m','scripts.curobo_arx.seed_augmentation.entrypoint','job','--request',str(request)]
        lockpath=request.with_suffix('.spawn.lock')
    else:
        campaign=Path(campaign).resolve()
        if campaign.parent!=Path(cfg['runs_root']) or not campaign.name.startswith(cfg['campaign_prefix']):raise ValueError('Unscoped augmentation batch')
        if (campaign/'launch.json').exists() or manual_stop(campaign):return
        state=wd.read(home/'state.json');gpu=state['selected_gpu_uuid']
        command=[cfg['collector_python'],'-m','scripts.curobo_arx.seed_augmentation.entrypoint','run',str(campaign),
                 '--compute-uuid',gpu,'--sim-uuid',gpu]
        metadata=campaign/'coordinator_process.json';lockpath=campaign/'watchdog_spawn.lock'
    lock=lockpath.open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    info=wd.process_info(os.getpid())
    wd.write(metadata,{'pid':os.getpid(),'boot_id':info['boot_id'],'start_ticks':info['start_ticks'],
        'uid':info['uid'],'command':command,'cwd':str(source),'wall':time.time(),'owned':True})
    os.set_inheritable(lock.fileno(),True);os.chdir(source)
    env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    os.execve(command[0],command,env)
