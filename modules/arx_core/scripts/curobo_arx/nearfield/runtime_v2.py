"""Explicit-action near-field executor; teacher waypoints never choose gates."""
import argparse
import copy
from pathlib import Path
import time
import uuid
import numpy as np
from scripts.curobo_arx.data_factory.common import read,write,digest
from scripts.curobo_arx.data_factory.runtime import Environment
from scripts.curobo_arx.settling import assess_settling
from training.nearfield_arx.contract import DSession,encode,decode,parse_action,pose_error
from training.nearfield_arx.history import before
from training.nearfield_arx.lookahead import select
from .runtime import LocalEnvironment,LocalGroup
from .closure import bind,validate_reply


class ActionEnvironment(LocalEnvironment):
    def plan(self,phase,world):
        if phase=='lift':
            goal=np.array(self.closure_certificate['hold_world']);goal[2,3]+=.12
            return Environment.plan(self,'lift',goal)
        if phase!='approach':return super().plan(phase,world)
        self.actual_entry=self.last_observation
        write(self.episode/'actual_entry.json',self.actual_entry)
        self.local_index=0;self.waypoint_cursor=0;self.events=[];self.traces=[]
        self.calls=0;self.prior_event=None;self.grasp_seen=False;self.recovered=False
        self.session=DSession(self.group.active_trial['G0_world'])
        # Original reaching clears last_command when the wrist recorder starts.
        # Restore its real accepted evidence so an early GRASP is also checked.
        self.last_command=read(self.episode/'pregrasp_command.json')
        self.previous_execution=read(self.episode/'pregrasp_execution.json')
        self.previous_goal=np.asarray(self.group.active_seed['anchor_world'])
        self.ready_time=self.group.world.current_time;self.state='local_decision'
        if self.group.active_trial['method']=='recovery':
            self.submit('state_generation',self.group.active_trial['offset_goal_world'],'local')

    def trace_before(self,method,token):
        row=copy.deepcopy(self.last_observation)
        history=before(method,self.calls,self.session.refines,self.prior_event,row['frame_time'])
        entry={'decision_id':uuid.uuid4().hex,'raw_index':self.recorder.rows-1,
            'observation_time_s':row['frame_time'],'observation_sha256':digest(row),
            'history_before':history,'method':method,'action':token}
        self.traces.append(entry);self.calls+=1
        entry['history_after_action']={'calls_completed':self.calls}
        write(self.episode/'decision_traces.json',self.traces)
        return entry

    def outcome(self,result,dispatched):
        self.prior_event={'outcome':result,'dispatched':dispatched,'sim_time':self.group.world.current_time}
        if self.traces:
            self.traces[-1]['outcome']=self.prior_event.copy()
            write(self.episode/'decision_traces.json',self.traces)

    def decide(self):
        trial=self.group.active_trial;method=trial['method']
        ee=np.asarray(self.bridge.snapshot()['link_world']['right_eef_link'])
        if method=='recovery':
            if not self.recovered:
                self.log_event({'kind':'expert_recovery_start','Gstar_offline_world':trial['source_Gstar_world'],
                    'source_segment':'positive supervision starts at this fresh stopped view'})
                self.recovered=True
                return self.submit('approach',trial['source_Gstar_world'],'local')
            return self.apply_action('C','GRASP')
        if method=='C':
            if self.waypoint_cursor==len(trial['C_waypoints']):return self.apply_action('C','GRASP')
            try:index=select(ee,trial['C_waypoints'],self.waypoint_cursor)
            except ValueError:return self.finish('lookahead_no_admissible_source_waypoint')
            token=encode('C',ee,trial['C_waypoints'][index]);self.waypoint_cursor=index+1
            self.teacher_provenance={'future_waypoint_index':index,'source_raw_index':trial['C_waypoint_raw_indices'][index]}
            return self.apply_action('C',token)
        raise RuntimeError('Only registered C replay and recovery teachers are used in this batch')

    def apply_action(self,method,token):
        """Policy-facing action interpreter: never inspects teacher sequence size."""
        if self.bridge.active or self.bridge.fault:raise RuntimeError('Action requires confirmed idle healthy robot')
        if self.grasp_seen:raise RuntimeError('Policy action after terminal GRASP')
        parse_action(method,token);trace=self.trace_before(method,token)
        if token=='GRASP':
            self.grasp_seen=True
            self.log_event({'kind':'grasp_request','token':token,'decision_id':trace['decision_id']})
            return self.request_closure(trace['decision_id'])
        ee=np.asarray(self.bridge.snapshot()['link_world']['right_eef_link'])
        if method=='D':
            event=self.session.request(ee,token)
            trace['history_after_action'].update(refines_processed=self.session.refines)
            if event['kind']=='budget_exhausted':return self.finish('D_correction_budget_exhausted')
            goal=np.asarray(event['goal_world'])
        else:goal=decode('C',ee,token)
        self.local_index+=1;label=f'c_step_{self.local_index:03d}' if method=='C' else f'd_refine_{self.local_index:03d}'
        self.log_event({'kind':'target_request','token':token,'goal_world':goal.tolist(),
            'decision_id':trace['decision_id'],'driver':'current_state_discrete_source_lookahead',
            'offline_teacher_provenance':getattr(self,'teacher_provenance',None),'execution_phase':'local'})
        return self.submit(label,goal,'local')

    def request_closure(self,ident):
        self.group.writer.drain()
        snapshot=self.bridge.snapshot()
        binding=bind(self.episode,snapshot,self.last_observation,self.last_command,self.previous_goal,
            active=self.bridge.active,fault=self.bridge.fault,request_id=ident,
            configuration_identity=self.group.registration['source_identity'])
        write(self.episode/'closure_request.json',binding)
        if not binding['pass']:
            self.outcome(binding['rejected'],False);return self.finish(binding['rejected'])
        self.closure_binding=binding;self.closure_snapshot=snapshot
        scene=self.episode/'closure_scene.json';write(scene,self.bridge.export_scene())
        self.future=self.group.planner.submit('closure_validate',binding=binding,snapshot=snapshot,
            scene=str(scene),output=str(self.episode))
        self.state='validating_closure'

    def tick(self):
        if self.bridge.fault:raise RuntimeError(str(self.bridge.fault))
        if self.task and time.monotonic()-self.started>self.group.registration['attempt_seconds']:
            raise TimeoutError('Original per-attempt resource budget exceeded')
        if self.state=='validating_closure':
            if self.bridge.active:raise RuntimeError('Unexpected motion during closure validation')
            reply=self.group.planner.poll(self.future)
            if reply is None:return
            if not reply['ok']:raise RuntimeError(reply['error'])
            cert=reply['result']
            if not validate_reply(self.closure_binding,cert,self.last_command):
                self.outcome(cert['rejected'],False);return self.finish('closure_gate_rejected: '+cert['rejected'])
            snap=self.bridge.snapshot()
            fresh=bind(self.episode,snap,self.last_observation,self.last_command,self.previous_goal,
                active=self.bridge.active,fault=self.bridge.fault,request_id=self.closure_binding['grasp_request_id'],
                configuration_identity=self.group.registration['source_identity'])
            if not fresh['pass']:return self.finish(fresh['rejected'])
            names=snap['right_arm'];prior=dict(zip(self.closure_snapshot['joint_names'],self.closure_snapshot['joint_positions']))
            now=dict(zip(snap['joint_names'],snap['joint_positions']))
            drift=max(abs(now[n]-prior[n]) for n in names)
            if drift>.0005:raise RuntimeError('Closed-model validation became stale: joint state drift')
            self.closure_certificate=cert
            write(self.episode/'closure_gate_optimization.json',{'pass':True,'planner_response':reply,
                'binding':self.closure_binding,'fresh_dispatch_check':fresh,'maximum_joint_drift_rad':drift,
                'scope':'Independent GRASP gate; no simulated movement receipt'})
            self.outcome('closure_gate_accepted',True)
            return self.dwell('close',np.asarray(cert['hold_world']))
        label=(self.last_command or {}).get('label','')
        if self.state!='executing' or not(label in ('approach','state_generation') or label.startswith(('c_step_','d_refine_'))):
            return super().tick()
        result=self.bridge.result()
        if result is None:return
        write(self.episode/(label+'_execution.json'),result)
        assessment=assess_settling(result,self.desired);write(self.episode/(label+'_settling.json'),assessment)
        self.stages.append(label);self.task['state']='executing'
        target=result['final']['objects']['test_object']['path']
        contacts=[c for c in result['contacts'] if c['num_contacts']>0]
        forbidden=[c for c in contacts if not('/World/Robot/' in c['a'] and '/World/Robot/' in c['b'])
            and not(target in c['a']+c['b'] and any(n in c['a']+c['b'] for n in ('right_link27','right_link28')))]
        if forbidden:return self.finish('forbidden_contact')
        if not assessment['accepted']:return self.finish('original_convergence_rejected')
        if self.visual_issues:return self.finish('visual_geometry_rejected_before_close')
        self.previous_execution=result;self.previous_goal=self.desired.copy()
        if label=='state_generation':
            initial=read(self.episode/'capture_snapshot.json')['objects']['test_object']['world']
            actual=result['final']['objects']['test_object']['world'];drift=pose_error(actual,initial)
            self.log_event({'kind':'offset_stopped','measured_offset':pose_error(result['final']['link_world']['right_eef_link'],self.group.active_trial['source_Gstar_world']),
                'object_drift':drift,'positive_supervision_begins_after_this':True})
            if any(target in c['a']+c['b'] for c in contacts) or drift['translation_m']>.001 or drift['rotation_rad']>.01:
                return self.finish('state_generation_contact_or_object_motion')
        else:self.outcome('accepted',True)
        self.ready_time=self.group.world.current_time;self.state='local_decision'

    def finish(self,reason=None):
        if self.bridge.active:raise RuntimeError('Cannot finalize active robot')
        sequence=[(r['label'] if r['label'] in ('pregrasp','approach','close','lift','hold') else 'approach',r['label'])
            for r in self.requests if r['label'] in self.stages]
        write(self.episode/'execution_roles.json',{'sequence':sequence,'basis':'actual request order; all open local motions have approach semantics'})
        return super().finish(reason)


class ActionGroup(LocalGroup):
    def __init__(self,campaign,launch):
        super().__init__(campaign,launch)
        self.envs=[ActionEnvironment(i,e.bridge,self) for i,e in enumerate(self.envs)]
        self.captures = []
        if self.registration.get('synchronized_capture'):
            from .capture import SynchronizedCapture
            self.captures = [SynchronizedCapture(e.bridge,
                render_every_update=self.registration['synchronized_capture'].get('render_every_update',False)) for e in self.envs]
            write(self.out/'capture_render_schedule.json',{'cameras':[c.render_schedule for c in self.captures],
                'sample_seconds':self.registration['sample_seconds'],
                'physics_dt':self.world.get_physics_dt(),
                'reference':'https://docs.isaacsim.omniverse.nvidia.com/6.0.1/sensors/isaacsim_sensors_multitick_rendering.html'})

    def step(self, record=True):
        if not getattr(self, 'captures', None):return super().step(record)
        before = [e.bridge.snapshot() for e in self.envs]
        super().step(False)
        failures = []
        for env, capture, previous in zip(self.envs, self.captures, before):
            from .capture import CaptureUnavailable
            # reset() is also called after the task is claimed. Honor that
            # existing settling interval; never publish an unready frame.
            remaining = max(0., env.settle_until - self.world.current_time)
            warming = env.state == 'settling' and remaining > 1e-6
            try: capture.capture(previous, env.bridge.snapshot())
            except CaptureUnavailable as exc:
                failures.append(exc.diagnostics)
                write(self.out/'capture_preflight.json', {'pass':False,
                    'settling_seconds_remaining':remaining,'failures':failures})
                if not warming or env.bridge.active or env.recorder:raise
                continue
            if record and env.recorder:env.record()
        if not failures:
            write(self.out/'capture_preflight.json', {'pass':True,
                'settling_seconds_remaining':remaining,'failures':[]})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--launch',required=True)
    a=p.parse_args();ActionGroup(a.campaign,a.launch).run()
