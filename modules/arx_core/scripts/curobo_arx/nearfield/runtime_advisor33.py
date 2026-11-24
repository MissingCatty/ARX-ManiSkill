"""Measured-state ARX runtime; after H, D never reads a coarse/teacher pose."""
import argparse
import copy
from dataclasses import asdict
from pathlib import Path
import time
import types
import uuid
import numpy as np
from scripts.curobo_arx.data_factory.common import read,write,sha,digest
from scripts.curobo_arx.data_factory import runtime as factory
from scripts.curobo_arx.settling import assess_settling
from training.nearfield_arx.history import before
from training.nearfield_arx.learned_policy import read_bound_reply,dispatch_drift
from .runtime import LocalEnvironment
from .runtime_v2 import ActionEnvironment,ActionGroup
from .advisor31_io import StatusWriter
from .closedloop30_rules import forbidden_contacts
from .closure import observation_timing
from .advisor32_rules import StopSample,settle
from . import advisor33_audit
from .advisor33_boundary import ISOLATED_OUTCOMES


class Environment(ActionEnvironment):
    def reset(self,asset):
        self.stop_samples=[]
        super().reset(asset)
        from .visual_grounding_spec import expected_reset
        from .verify_validation import reset_signature
        expected=expected_reset(self.group.active_seed,self.group.active_trial)
        actual=reset_signature(self.reset_initial)
        proof={'pass':actual==expected,'actual':actual,'expected':expected,'wall':time.time()}
        write(self.group.out/'reset_verification.json',proof)
        if not proof['pass']:raise RuntimeError('Registered reset mismatch')

    def record(self):
        if not self.recorder:return
        stage='approach' if (self.stage or '').startswith(('d_refine_','c_step_')) else self.stage
        row,rgb,depth=self.bridge.observation(self.id,self.task['id'],self.asset['object_id'],None,self.last_command,stage)
        if self.group.registration.get('visual_geometry_policy'):
            from scripts.curobo_arx.data_factory.visual_geometry import inspect_row
            row['visual_geometry']=inspect_row(self.visual_geometry,row,depth.shape)
            if not row['visual_geometry']['pass']:self.visual_issues.append({'raw_index':self.recorder.rows,**row['visual_geometry']})
        extras=getattr(self.bridge.context.cameras['right_wrist'],'capture_arrays',None)
        self.recorder.add(row,rgb,depth,extras=extras);self.last_observation=row
        # Keep the measured tail while commands execute. A completed hold can
        # already contain a valid two-second stationary window; discarding it
        # makes the 10 wall-second fallback depend on simulator throughput.
        self.sample_stop()

    def submit(self,label,goal,phase='local',action='plan'):
        path=self.episode/(label+'_scene.json');write(path,self.bridge.export_scene());snap=self.bridge.snapshot()
        self.future=self.group.planner.submit(action,phase=phase,label=label,
            seed_id=self.group.active_seed['seed_id'],snapshot=snap,scene=str(path),
            goal_world=np.asarray(goal).tolist(),grasp_world=snap['link_world']['right_eef_link'] if phase in ('local','lift') else self.task['goal']['grasp_world'],
            output=str(self.episode),visual_context=self.visual_context(),command_id=uuid.uuid4().hex,
            minimum_duration=6. if label=='pregrasp' else 3.)
        self.state='planning_'+phase

    def plan(self,phase,world):
        if phase=='reach':return self.submit('pregrasp',self.group.active_seed['anchor_world'],'reach','seed_connect')
        if phase=='lift':
            pose=np.array(self.bridge.snapshot()['link_world']['right_eef_link']);pose[2,3]+=.12
            return self.submit('lift',pose,'lift')
        if phase!='approach':return factory.Environment.plan(self,phase,world)
        self.actual_entry=self.last_observation;self.local_index=0;self.events=[];self.traces=[]
        self.calls=0;self.prior_event=None;self.grasp_seen=False;self.session=types.SimpleNamespace(refines=0,calls=0)
        self.pending_policy=None;self.policy_requests=[];self.stop_samples=[];self.neural_calls=0;self.entry_saved=False
        self.last_command=read(self.episode/'pregrasp_command.json');self.previous_execution=read(self.episode/'pregrasp_execution.json')
        self.previous_goal=np.asarray(self.group.active_seed['anchor_world']);write(self.episode/'policy_requests.json',[])
        return self.submit('state_generation',self.group.active_trial['entry_world'],'local')

    def sample_stop(self):
        row=self.last_observation;snap=row['snapshot'];names=snap['joint_names'];idx=[names.index(n) for n in snap['right_arm']]
        q=np.asarray(snap['joint_positions'])[idx];limits=np.asarray(snap['joint_limits'])[idx]
        if np.max(np.maximum(limits[:,0]-q,q-limits[:,1]))>1e-4+1e-7:raise RuntimeError('Actual joint limit violation')
        sample=StopSample(float(snap['sim_time']),tuple(np.array(snap['link_world']['right_eef_link'])[:3,3]),
                          tuple(q),tuple(np.array(snap['joint_velocities'])[idx]))
        if not self.stop_samples or sample.sim_time>self.stop_samples[-1].sim_time+1e-9:self.stop_samples.append(sample)
        self.stop_samples=self.stop_samples[-30:]

    def proof(self,elapsed):
        self.sample_stop();answer=settle(self.stop_samples,active=self.bridge.active,fault=self.bridge.fault,elapsed_wall_s=elapsed)
        label=self.last_command['label'];paths=[self.episode/(label+'_'+s+'.json') for s in ('command','execution','settling')]
        return {**answer,'label':label,'snapshot':self.bridge.snapshot(),'samples':[asdict(s) for s in self.stop_samples],
                'wait_elapsed_wall_s':elapsed,'files':{str(p):sha(p) for p in paths}}

    def decide(self):
        self.sample_stop()
        if self.pending_policy is not None:return self.consume_prediction(self.bridge.snapshot())
        if self.last_observation['frame_time']<=self.previous_execution['final']['sim_time']+1e-6:return
        if not self.proof(0.)['pass']:return
        trial=self.group.active_trial;method=trial['method'];row=self.last_observation
        if not self.entry_saved:
            self.actual_entry=row
            write(self.episode/'physical_entry_observation.json',row);write(self.episode/'actual_entry.json',row);self.entry_saved=True
            self.log_event({'kind':'H_visual_takeover','blind_axial_preparation':False})
        mode='final_gate' if self.local_index >= (2 if method=='D' else 12) else 'decision'
        self.group.writer.drain();ident=uuid.uuid4().hex;index=self.recorder.rows-1
        inputs={'eef_world':row['snapshot']['link_world']['right_eef_link'],'stage':'local_observation',
            'decision_history_by_method':{method:before(method,self.calls,self.session.refines,self.prior_event,row['frame_time'])}}
        if method=='C' or trial['vision_arm']=='g0':inputs['G0_world']=trial['G0_world']
        folder=self.episode/'policy';observation=folder/(ident+'_observation.json');write(observation,row)
        rgb=self.episode/f'raw/{index:06d}.png';depth=self.episode/f'raw/{index:06d}.npy'
        from .advisor31_rules import fingers_touching
        touching=bool(method=='C' and self.local_index and fingers_touching(self.previous_execution))
        req={'request_id':ident,'method':method,'trial_id':trial['trial_id'],'object_id':trial['object_id'],
            'vision_arm':trial['vision_arm'],'mode':mode,'motions_completed':self.local_index,'contact_closure':touching,
            'raw_index':index,'observation':str(observation),'rgb':str(rgb),'depth':str(depth),'asset':self.asset['usd_path'],
            'inputs':inputs,'registration_sha256':sha(self.group.campaign/'registration.json'),
            'files':{str(p):sha(p) for p in (observation,rgb,depth)}}
        from training.nearfield_arx import native_mask
        if native_mask.configured(self.group.registration):
            req['native_mask_binding']=native_mask.bind(row,rgb)['native_mask_binding']
            req['files'].update(req['native_mask_binding']['files'])
        path=folder/(ident+'_request.json');write(path,req);self.policy_requests.append({'request_id':ident,'raw_index':index,'request':str(path)})
        write(self.episode/'policy_requests.json',self.policy_requests)
        self.pending_policy={'path':path,'request':req,'snapshot':copy.deepcopy(row['snapshot']),'started':time.monotonic()}
        write(self.group.campaign/'workers/policy/queue'/(ident+'.json'),{'request':str(path)})

    def consume_prediction(self,snapshot):
        pending=self.pending_policy;path=pending['path'];req=pending['request'];reply_path=path.with_name(req['request_id']+'_reply.json')
        if (self.group.campaign/'workers/policy/FAULT.json').exists():raise RuntimeError('Policy infrastructure fault')
        if not reply_path.exists():
            if time.monotonic()-pending['started']>self.group.registration['learned_policy']['request_timeout_seconds']:raise TimeoutError('Policy request timeout')
            return
        _,reply=read_bound_reply(path,reply_path);self.pending_policy=None
        from .advisor33_policy import stationary_dispatch
        drift=stationary_dispatch(pending['snapshot'],snapshot,self.proof(0.))
        write(path.with_name(req['request_id']+'_dispatch_check.json'),drift)
        if not drift['pass']:return self.finish('observation_changed_before_dispatch')
        token=reply['prediction'].strip()
        trace={'decision_id':uuid.uuid4().hex,'raw_index':req['raw_index'],'observation_time_s':read(req['observation'])['frame_time'],
            'observation_sha256':digest(read(req['observation'])),'history_before':req['inputs']['decision_history_by_method'][req['method']],
            'method':req['method'],'action':token,'policy_request_id':req['request_id']}
        self.calls+=1;trace['history_after_action']={'calls_completed':self.calls};self.traces.append(trace)
        self.neural_calls+=int(reply['model_inference']);write(self.episode/'decision_traces.json',self.traces)
        common={'token':token,'policy_request_id':req['request_id'],'decision_id':trace['decision_id'],'model_inference':reply['model_inference']}
        action=reply['action'];self.log_event({**common,'kind':'policy_interpreter_decision','action':action,'finger_box':reply['finger_box']})
        if action['kind']=='reject':return self.finish(action['reason'])
        if action['kind']=='close':
            self.visual_gate=reply['finger_box'];self.grasp_seen=True
            self.log_event({**common,'kind':'grasp_request','closure_origin':action['closure_origin'],'model_explicit_GRASP':bool(reply['model_inference'] and token=='GRASP')})
            return self.request_closure(trace['decision_id'])
        self.local_index+=1
        if req['method']=='D':self.session.refines+=1
        label=('d_refine_' if req['method']=='D' else 'c_step_')+f'{self.local_index:03d}'
        self.log_event({**common,'kind':'target_request','goal_world':action['goal_world'],'command_label':label})
        return self.submit(label,action['goal_world'])

    def request_closure(self,ident):
        self.group.writer.drain();snap=self.bridge.snapshot();proof=self.proof(0.)
        if not proof['pass']:return self.finish('closure_not_stationary')
        timing=observation_timing(self.last_observation);names=snap['right_arm'];q=[float(snap['joint_positions'][snap['joint_names'].index(n)]) for n in names]
        binding={'pass':True,'version':'advisor33','grasp_request_id':ident,'hold_joint_names':names,'hold_q':q,
            'snapshot_sha256':digest(snap),'measured_hold_snapshot':snap,'files':proof['files'],'arrival':proof,
            'visual_gate':self.visual_gate,'previous_command_id':self.last_command['id'],'previous_command_label':self.last_command['label'],
            'observation_timing':timing,'configuration_identity':self.group.registration['source_identity']}
        write(self.episode/'closure_request.json',binding);self.closure_binding=binding;self.closure_snapshot=snap
        scene=self.episode/'closure_scene.json';write(scene,self.bridge.export_scene())
        self.future=self.group.planner.submit('closure_validate',binding=binding,snapshot=snap,scene=str(scene),output=str(self.episode))
        self.state='validating_closure'

    def tick(self):
        if self.bridge.fault:raise RuntimeError('Controller fault: '+str(self.bridge.fault))
        if self.task and time.monotonic()-self.started>self.group.registration['attempt_seconds']:raise TimeoutError('Slot wall budget exhausted')
        if self.state=='validating_closure':
            reply=self.group.planner.poll(self.future)
            if reply is None:return
            if not reply['ok']:raise RuntimeError(reply['error'])
            cert=reply['result']
            if not cert['pass']:return self.finish('closure_model_rejected: '+cert['rejected'])
            if cert['binding_sha256']!=digest(self.closure_binding):raise RuntimeError('Closure binding changed')
            snap=self.bridge.snapshot();prior=dict(zip(self.closure_snapshot['joint_names'],self.closure_snapshot['joint_positions']))
            now=dict(zip(snap['joint_names'],snap['joint_positions']));drift=max(abs(now[n]-prior[n]) for n in snap['right_arm'])
            if drift>.0005:return self.finish('closure_snapshot_drift')
            if not self.proof(0.)['pass']:return self.finish('closure_dispatch_not_stationary')
            self.closure_certificate=cert
            write(self.episode/'closure_gate_optimization.json',{'pass':True,'binding':self.closure_binding,'maximum_joint_drift_rad':drift,'planner_response':reply})
            q=self.closure_binding['hold_q'];cmd={'id':uuid.uuid4().hex,'label':'close','joint_names':snap['right_arm'],
                'position':[q,q],'time_from_start':[0.,3.],'gripper_target':.1,'position_tolerance_rad':.005,'planner':'Measured stationary hold after independent closure validation'}
            return self.dispatch(cmd,cert['hold_world'])
        if self.state=='waiting_idle':
            proof=self.proof(time.monotonic()-self.wait_started)
            if proof['timeout']:return self.finish('stationary_wait_10s_timeout')
            if not proof['pass']:return
            write(self.episode/(self.last_command['label']+'_waypoint_stop.json'),proof)
            self.previous_execution=self.completed_waypoint;self.previous_goal=self.desired.copy()
            if self.last_command['label']=='close':
                from .advisor31_rules import fingers_touching
                pos=dict(zip(proof['snapshot']['joint_names'],proof['snapshot']['joint_positions']))
                closed=float(self.bridge.context.controller._gripper_config('right')['closed_m'])
                if len(fingers_touching(self.completed_waypoint))<2 and np.mean([pos['right_joint27'],pos['right_joint28']])-closed<=.001:
                    return self.finish('close_without_grip_evidence')
                return self.plan('lift',self.desired)
            self.outcome('accepted',True);self.ready_time=self.group.world.current_time;self.state='local_decision';return
        label=(self.last_command or {}).get('label','')
        if self.state=='executing' and (label in ('state_generation','close') or label.startswith(('d_refine_','c_step_'))):
            result=self.bridge.result()
            if result is None:return
            write(self.episode/(label+'_execution.json'),result);write(self.episode/(label+'_settling.json'),assess_settling(result,self.desired))
            self.stages.append(label);self.task['state']='executing'
            if forbidden_contacts(result):return self.finish('forbidden_contact')
            if label!='close' and self.visual_issues:return self.finish('visual_geometry_rejected_before_close')
            self.completed_waypoint=result;self.wait_started=time.monotonic();self.state='waiting_idle';return
        if self.state=='local_decision':return self.decide()
        if self.state=='executing' and label in ('lift','hold'):
            result=self.bridge.result()
            if result is None:return
            assessment=assess_settling(result,self.desired)
            write(self.episode/(label+'_execution.json'),result);write(self.episode/(label+'_settling.json'),assessment)
            self.stages.append(label);self.task['state']='executing'
            if forbidden_contacts(result):return self.finish('forbidden_contact')
            if not assessment['accepted']:return self.finish('original_convergence_rejected')
            if label=='lift':return self.dwell('hold',self.desired)
            return self.finish()
        return factory.Environment.tick(self)

    def finish(self,reason=None):
        if reason in ISOLATED_OUTCOMES:
            self.requires_world_isolation=True;self.isolation_outcome=reason
        if not hasattr(self,'events'):self.events=[];self.traces=[];self.policy_requests=[]
        write(self.episode/'interpreter_events.json',self.events)
        if not (self.episode/'policy_requests.json').exists():write(self.episode/'policy_requests.json',[])
        sequence=[('approach' if r['label'] not in ('pregrasp','close','lift','hold') else r['label'],r['label']) for r in self.requests if r['label'] in self.stages]
        write(self.episode/'execution_roles.json',{'sequence':sequence,'basis':'All actually executed open motions'})
        return LocalEnvironment.finish(self,reason)


class Group(ActionGroup):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.envs=[Environment(i,e.bridge,self) for i,e in enumerate(self.envs)]
    def sample_handoff(self,task):
        return {'world':self.active_trial['entry_world'],'distance_mm':80.,'sample':0,'seed':self.registration['seed'],'policy':'advisor33_registered_H'}
    def run_seed_trial(self,asset):
        env=self.envs[0];env.events=[];env.traces=[];env.policy_requests=[];env.entry_saved=False;env.requires_world_isolation=False
        try:
            result=super().run_seed_trial(asset)
            from scripts.curobo_arx.data_factory.frozen import fingerprint
            proof=fingerprint(self.envs[0].bridge,read(self.registration['reference_files'][self.renderer]))
            write(self.out/'frozen_after.json',proof)
            if not proof['pass']:raise RuntimeError('Post-slot frozen configuration mismatch')
            if env.requires_world_isolation:
                write(self.out/'ISOLATED_TERMINAL.json',{'trial_id':self.active_trial['trial_id'],
                    'outcome':env.isolation_outcome,'terminal_records_complete':True,
                    'configuration_intact':True,'retry':False,'next_slot_requires_new_World':True,'wall':time.time()})
                self.stop=True
            return result
        except TimeoutError as exc:
            env=self.envs[0];self.writer.drain()
            from scripts.curobo_arx.data_factory.frozen import fingerprint
            frozen=fingerprint(env.bridge,read(self.registration['reference_files'][self.renderer]))
            write(self.out/'frozen_after.json',frozen)
            if not frozen['pass']:raise RuntimeError('Frozen configuration changed at timeout') from exc
            if env.recorder:
                write(env.episode/'interpreter_events.json',env.events)
                write(env.episode/'execution_roles.json',{'sequence':[(s,s) for s in env.stages]})
            files={str(p):sha(p) for p in self.out.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl')}
            write(self.out/'ISOLATED_TIMEOUT.json',{'error':str(exc),'trial_id':self.active_trial['trial_id'],
                'state':env.state,'controller_active_at_timeout':bool(env.bridge.active),'task':env.task,
                'episode':str(env.episode) if env.task else None,'files':files,'known_partial_record':True,
                'success':False,'outcome':'interrupted_unknown','retry':False,'wall':time.time()})
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--launch',required=True);a=p.parse_args()
    from . import runtime_v2
    runtime_v2.write=StatusWriter(write);factory.physical=advisor33_audit.physical;factory.quality=advisor33_audit.record_quality
    Group(a.campaign,a.launch).run()
