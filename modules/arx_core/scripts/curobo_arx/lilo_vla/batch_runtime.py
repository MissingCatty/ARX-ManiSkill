"""Bounded ARX teacher execution and positive proof of instance idleness."""
import gzip
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from batch_ledger import journal, motion_sent, process_identity, write
from batch_manifest import ROOT, pipeline_command
from interaction_contract import prepare_environment, resolve, validate_episode, read

sys.path.insert(0,str(ROOT/'scripts/curobo_arx'))
from client import Client
from gpu_policy import COMPUTE_GPU


class UnsafeInstance(RuntimeError):pass
class FrozenMismatch(RuntimeError):pass
class PipelineFailed(RuntimeError):pass
class InfrastructureFailure(RuntimeError):pass


def status(mailbox,fixture,session=None,now=None):
    value=read(Path(mailbox)/'stream_state.json')['collection_status']
    age=(time.time() if now is None else now)-value['wall_time']
    if value['version']!='arx_collection_status_v1' or not 0<=age<=5:raise UnsafeInstance('Missing/future/stale collection status')
    if session is not None and value['session_id']!=session:raise UnsafeInstance('Simulator session changed')
    if value['input_sha256']!=fixture['input_sha256'] or value['renderer']!=fixture['renderer']:
        raise FrozenMismatch('Simulator inputs/renderer differ from registered fixture')
    if value['gpu_uuid']!=fixture.get('gpu_uuid',COMPUTE_GPU):raise UnsafeInstance('Simulator GPU differs from registered fixture UUID')
    for key in ('controller_busy','streaming'):
        if type(value[key]) is not bool:raise UnsafeInstance('Incomplete collection status: '+key)
    for key in ('active_trajectory_request_id','setup_pending','recorders'):value[key]
    if set(value['recorders'])!={'teacher','video'}:raise UnsafeInstance('Incomplete recorder status')
    return value


def idle(value):
    return not any(value[k] for k in ('active_trajectory_request_id','controller_busy','setup_pending','streaming'))


def confirm_idle(mailbox,fixture,session,deadline,trial=None,allow_owned_recorders=False,heartbeat=lambda:None,
                 require_recorders_stopped=False):
    """Never send a request to probe idleness. Retain uncertain mailbox commands."""
    mailbox=Path(mailbox);last=None;consecutive=0;next_heartbeat=time.monotonic()+30
    while time.monotonic()<deadline:
        if time.monotonic()>=next_heartbeat:heartbeat();next_heartbeat=time.monotonic()+30
        value=status(mailbox,fixture,session)
        events=journal(trial) if trial else []
        intents={e['request_id']:e for e in events if e['event']=='intent'}
        answered={e['request_id'] for e in events if e['event']=='response'}
        pending=mailbox/'request.json'
        if pending.exists() and read(pending)['id'] not in intents:raise UnsafeInstance('Unknown pending mailbox request; preserve it')
        if (mailbox/'stream_command.json').exists():raise UnsafeInstance('Unowned streaming command file; preserve it')
        response=mailbox/'response.json'
        if response.exists():
            reply=read(response);rid=reply['id']
            if rid in intents and rid not in answered:
                Client(mailbox,journal=Path(trial)/'requests.jsonl',explicit_root=True).event(
                    {'event':'response','request_id':rid,'action':intents[rid]['action'],'ok':reply['ok'],'late':True})
                answered.add(rid)
                write(Path(trial)/('late_response_'+rid+'.json'),reply)
        for kind,path in value['recorders'].items():
            expected=Path(trial)/('teacher_raw' if kind=='teacher' else 'frames') if trial else None
            if path and (not allow_owned_recorders or Path(path)!=expected):raise UnsafeInstance('Unowned active recorder: '+str(path))
        recorders_stopped=not any(value['recorders'].values())
        safe=(idle(value) and not pending.exists() and not (set(intents)-answered)
              and (not require_recorders_stopped or recorders_stopped))
        if safe and value['wall_time']!=last:
            consecutive+=1;last=value['wall_time']
            if consecutive>=2:return value
        elif not safe:consecutive=0;last=None
        time.sleep(.1)
    raise UnsafeInstance('No consecutive fresh idle states with resolved request identities before cleanup deadline')


def run_owned(command,env,log,deadline,heartbeat=lambda:None,kill_grace=10):
    """Kill only the owned process group. This is not a simulator stop."""
    if time.monotonic()>=deadline:raise TimeoutError('Shared wall-clock budget exhausted')
    log=Path(log)
    with log.open('x') as stream:
        child=subprocess.Popen(list(map(str,command)),cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        next_heartbeat=time.monotonic()
        try:
            write(log.with_suffix('.process.json'),{'owner':process_identity(),'child':process_identity(child.pid),'process_group':child.pid,'command':list(map(str,command))})
            while child.poll() is None:
                remaining=deadline-time.monotonic()
                if remaining<=0:raise TimeoutError('Shared wall-clock budget exhausted')
                if time.monotonic()>=next_heartbeat:heartbeat();next_heartbeat=time.monotonic()+30
                try:child.wait(timeout=min(1,remaining))
                except subprocess.TimeoutExpired:pass
            return child.returncode
        except BaseException:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:child.wait(timeout=kill_grace)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait()
            # Parent may have exited before descendants. Its group is still ours.
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            raise


class Runtime:
    def __init__(self,registration,session):
        self.registration=registration;self.fixture=registration['fixture'];self.session=session

    def begin(self,trial,case,goal,deadline,heartbeat):
        self.trial=trial;self.case=case;self.deadline=deadline;self.heartbeat=heartbeat
        self.client=Client(self.fixture['mailbox'],journal=trial/'requests.jsonl',deadline=deadline,session=self.session,explicit_root=True,heartbeat=heartbeat)
        self.scenario={'name':case['case_id'],'target':'test_object','asset':case['object_name'],'placement':case['placement']}
        write(trial/'scenario.json',self.scenario)
        env={k:v for k,v in os.environ.items() if not k.startswith(('ARX_LILO_','GRASP_','ARX_REQUEST_','ARX_COLLECTION_','ARX_ATTEMPT_'))}
        env.pop('ARX_RESEARCH_GUARD_FILE',None)
        env.update(ARX_CUROBO_VALIDATION_DIR=self.fixture['mailbox'],ARX_OPS_URL=self.fixture['ops_url'],
            ARX_OBJECT_LABEL=case['object_name'],ARX_RESEARCH_AUDIT='0',ARX_BATCH_MANAGED_CLEANUP='1',
            ARX_LILO_TEACHER_DIR=str(trial/'teacher_raw'),ARX_REQUEST_JOURNAL=str(trial/'requests.jsonl'),
            ARX_COLLECTION_SESSION=self.session,ARX_ATTEMPT_DEADLINE=str(deadline),PYTHONUNBUFFERED='1',
            ARX_ZEROGRASP_CHECKPOINT=self.registration['resources']['zerograsp_checkpoint'])
        contract=self.registration['contracts'][case['contract']]
        self.env=prepare_environment(env,trial,resolve(contract,case['object_name'],'teacher',delta=case['entry_delta_world'],goal_source=goal))
        self.command=pipeline_command(self.registration,trial)
        if goal:self.command+=['--fixed-selection',str(goal)]

    def check_idle(self):
        confirm_idle(self.fixture['mailbox'],self.fixture,self.session,min(self.deadline,time.monotonic()+5))

    def reset(self):
        from egad49 import compare_configuration
        setup=self.client.request('dynamic_setup',scenario=self.scenario,timeout=240)
        write(self.trial/'setup.json',setup)
        bundle=Path(self.fixture['bundle'])
        historical=json.loads(gzip.decompress((bundle/'setup.json.gz').read_bytes()))
        mismatch=compare_configuration(historical['fingerprint'],setup['fingerprint'])
        initial=read(bundle/'initial_setup.json');bounds=setup['snapshot']['joint_limits']
        clipped=[min(max(q,lo),hi) for q,(lo,hi) in zip(initial['joint_positions'],bounds)]
        if setup['initial_conditions']['joint_names']!=initial['joint_names'] or setup['initial_conditions']['joint_positions']!=clipped:mismatch.append('historical robot reset')
        if setup['physics_steps']!=360 or abs(setup['captured_sim_time']-setup['reset_sim_time']-3)>.001:mismatch.append('historical 3s settle')
        pose=setup['initial_conditions']['objects']['asset:'+self.case['object_name']]
        if pose['position']!=self.case['placement']['position'] or pose['quaternion_wxyz']!=self.case['placement']['orientation']:mismatch.append('registered object placement')
        write(self.trial/'setup_verification.json',{'pass':not mismatch,'mismatches':mismatch,'historical_physical_properties_equal':not mismatch})
        if mismatch:raise FrozenMismatch('Frozen scene/reset differs: '+str(mismatch[:12]))
        write(self.trial/'active_asset.json',setup['asset'])
        write(self.trial/'guard.json',{'invariant_sha256':'initial fingerprint archived in setup.json','file_sha256':{}})

    def viewport(self):
        expected=read(Path(self.fixture['bundle'])/'viewport_initial.json')
        self.client.request('grasp_viewport_state',restore=expected)
        actual=self.client.request('grasp_viewport_state');write(self.trial/'viewport_initial.json',actual)
        if actual['camera_path']!=expected['camera_path'] or actual['attributes']!=expected['attributes'] or any(
                abs(a-b)>1e-8 for key in ('world_matrix','local_matrix')
                for ar,br in zip(actual[key],expected[key]) for a,b in zip(ar,br)):
            raise FrozenMismatch('Viewport differs from the registered camera')

    def pipeline(self):
        code=run_owned(self.command,self.env,self.trial/'pipeline.log',self.deadline,self.heartbeat)
        if code:
            stage=read(self.trial/'pipeline_stage.json')['stage'] if (self.trial/'pipeline_stage.json').exists() else 'startup'
            logs=[self.trial/'network/logs/stderr.log',self.trial/(stage+'.log')]
            if any(p.exists() and any(word in p.read_text(errors='replace') for word in ('OutOfMemoryError','No space left','Connection refused')) for p in logs):
                raise InfrastructureFailure('Pipeline infrastructure error at '+stage)
            raise PipelineFailed('Teacher pipeline exit '+str(code)+' at '+stage)

    def cleanup(self,deadline):
        value=confirm_idle(self.fixture['mailbox'],self.fixture,self.session,deadline,self.trial,True,self.heartbeat)
        cleanup=Client(self.fixture['mailbox'],journal=self.trial/'requests.jsonl',deadline=deadline,session=self.session,explicit_root=True,heartbeat=self.heartbeat)
        for kind,action,filename in (('teacher','grasp_lilo_record_stop','teacher_recording.json'),('video','grasp_video_stop','recording_stop.json')):
            if value['recorders'][kind]:write(self.trial/filename,cleanup.request(action,timeout=max(0,deadline-time.monotonic())))
        # The stop reply can precede the next telemetry frame. Continue to
        # recognize only this trial's recorder, but require two fresh stopped
        # states; foreign recorders and unresolved requests still fail closed.
        value=confirm_idle(self.fixture['mailbox'],self.fixture,self.session,deadline,self.trial,
                           allow_owned_recorders=True,require_recorders_stopped=True)
        write(self.trial/'boundary_status.json',value)

    def evaluate(self):
        from egad_evaluate import evaluate
        return evaluate(self.trial)

    def quality(self):
        command=[self.registration['resources']['neural_python'],ROOT/'scripts/curobo_arx/lilo_vla/episode_quality.py',
                 '--source',self.trial,'--output',self.trial/'quality.json']
        env={**os.environ,'CUDA_VISIBLE_DEVICES':''}
        code=run_owned(command,env,self.trial/'quality.log',self.deadline,self.heartbeat)
        if not (self.trial/'quality.json').exists():raise InfrastructureFailure('Quality process exited without report: '+str(code))
        return read(self.trial/'quality.json')

    def planned_goal(self):
        path=self.trial/'resolved_interaction.json'
        if not path.exists() or not read(path).get('grasp_world'):return None
        validate_episode(self.trial,read(path)['contract'])
        return str(self.trial/'selected_grasp.json')


def execute_attempt(ledger,case,runtime,goal=None,seen_recordings=None):
    trial,start=ledger.begin(case);began=time.monotonic();deadline=began+900
    error=None;cleanup_errors=[];failed_stage=None;stage='prepare';timings={};halt=False
    physical={'pass':False};quality={'pass':False,'error':'Not evaluated'};planned=None;timed_out=False
    def heartbeat():
        current=stage
        if stage=='pipeline' and (trial/'pipeline_stage.json').exists():current=read(trial/'pipeline_stage.json')['stage']
        ledger.progress('running',case_id=case['case_id'],stage=current,elapsed_seconds=time.monotonic()-began)
    def invoke(name,fn):
        nonlocal stage
        stage=name;heartbeat();then=time.monotonic()
        if then>=deadline:raise TimeoutError('Shared attempt deadline exhausted before '+name)
        try:return fn()
        finally:timings[name]=time.monotonic()-then
    try:
        runtime.begin(trial,case,goal,deadline,heartbeat)
        invoke('idle',runtime.check_idle);invoke('reset',runtime.reset);invoke('viewport',runtime.viewport)
        invoke('pipeline',runtime.pipeline)
    except BaseException as exc:
        failed_stage=stage;error={'type':type(exc).__name__,'message':str(exc)}
        timed_out=isinstance(exc,TimeoutError) or time.monotonic()>=deadline
        halt=isinstance(exc,(UnsafeInstance,FrozenMismatch,KeyboardInterrupt,SystemExit))
    # Native trajectories can outlive a killed child. Cleanup is bounded separately
    # only when the execution deadline has expired. Successful cleanup is charged.
    stage='cleanup';heartbeat()
    then=time.monotonic();cleanup_deadline=min(deadline,then+60) if then<deadline else deadline+60
    try:runtime.cleanup(cleanup_deadline)
    except BaseException as exc:
        cleanup_errors.append({'type':type(exc).__name__,'message':str(exc)});halt=True
    timings['cleanup']=time.monotonic()-then
    try:
        physical=runtime.evaluate()
        planned=runtime.planned_goal()
        if not error and not cleanup_errors:
            quality=invoke('quality',runtime.quality)
            identity=quality.get('content_sha256')
            if quality['pass'] and identity in (seen_recordings or set()):quality={**quality,'pass':False,'error':'Duplicate recording content'}
    except BaseException as exc:
        if not error:error={'type':type(exc).__name__,'message':str(exc)};failed_stage=stage
        timed_out=timed_out or isinstance(exc,TimeoutError)
        halt=halt or isinstance(exc,(FrozenMismatch,KeyboardInterrupt,SystemExit))
    timed_out=timed_out or time.monotonic()>=deadline
    def evidence(path,default,jsonl=False):
        nonlocal error,halt,failed_stage
        if not path.exists():return default
        try:return [json.loads(line) for line in path.read_text().splitlines()] if jsonl else read(path)
        except Exception as exc:
            if error is None:error={'type':type(exc).__name__,'message':str(exc)};failed_stage='evidence'
            halt=True
            return default
    recording=evidence(trial/'teacher_recording.json',{'error':'Missing recording receipt'})
    pipeline_timings=evidence(trial/'pipeline_timing.jsonl',[],True)
    executions={p.name:evidence(p,{}) for p in trial.glob('*_execution.json')}
    detail={'pipeline':pipeline_timings,
            'motion_wall_seconds':sum(v.get('wall_duration_s',0) for v in executions.values()),
            'motion_sim_seconds':sum(v.get('sim_duration_s',0) for v in executions.values()),
            'planner_initialization':evidence(trial/'planner_initialization.json',None),
            'candidate_checks':evidence(trial/'candidate_diagnostics.json',None)}
    timed_out=timed_out or time.monotonic()>=deadline
    ready=bool(not error and not cleanup_errors and not timed_out and physical['pass'] and not recording.get('error') and quality['pass'])
    status_name='timeout' if timed_out else ('interrupted' if error and error['type'] in ('KeyboardInterrupt','SystemExit') else
        'infrastructure_failure' if cleanup_errors or error and error['type'] in ('UnsafeInstance','InfrastructureFailure','ConnectionError','FileNotFoundError') else
        'failed' if error or not physical['pass'] else 'quality_rejected' if not ready else 'completed')
    result={**start,'source':str(trial),'status':status_name,'failed_stage':failed_stage,'error':error,'cleanup_errors':cleanup_errors,
            'pass':physical['pass'],'demonstration_ready':ready,'evaluation':physical,'recording':recording,'quality':quality,
            'elapsed_seconds':time.monotonic()-began,'timings':timings,'timing_details':detail,'motion_sent':motion_sent(trial),
            'budget_expires_monotonic':deadline,
            'planned_goal':planned,'goal_physical_qualification':bool(physical['pass']),
            'instance_session':runtime.session,'video_status':'pending_offline_encoding','stop_campaign':halt}
    write(trial/'quality_admission.json',{'pass':ready,'quality':quality,'error':error,'timeout':timed_out})
    ledger.finish(trial,result)
    return result
