"""A/B add-on; inherited locked C/D and all shared physical gates are unchanged."""
import argparse
import copy
from pathlib import Path
import time
import uuid
import numpy as np
from scripts.curobo_arx.data_factory.common import read,write,sha
from scripts.curobo_arx.settling import assess_settling
from training.nearfield_arx.contract import pose_error
from training.nearfield_arx.learned_policy import read_bound_reply,dispatch_drift
from . import runtime_stage_recovery,runtime_learned
from .runtime_physical import PhysicalEnvironment
from .waypoint_continuation import forbidden_contacts
from .main48_spec import GEOMETRY
from .main48_audit import audit_baseline


class MainEnvironment(runtime_stage_recovery.StageEnvironment):
    def log_event(self,event):
        return super().log_event({**event,'wall_unix':time.time()})

    def plan(self,phase,world):
        if phase=='approach':
            self.baseline_queries=0;self.baseline_motions=0;self.geometry_requests=[];self.pending_geometry=None
        return super().plan(phase,world)

    def record(self):
        saved=self.stage
        if saved and saved.startswith(('a_approach','b_refine_')):self.stage='approach'
        try:return super().record()
        finally:self.stage=saved

    def decide(self):
        trial=self.group.active_trial;method=trial['method']
        if method in ('C','D'):return super().decide()
        snapshot=self.bridge.snapshot()
        if self.bridge.active or self.bridge.fault:raise RuntimeError('Baseline requires confirmed healthy idle')
        if self.pending_geometry is not None:return self.consume_geometry(snapshot)
        row=self.last_observation
        completed=self.entry_stopped_sim_time if not self.physical_entry_ready else self.previous_execution['final']['sim_time']
        if row['frame_time']<=completed+1e-6:return
        for state in (snapshot,row['snapshot']):
            speeds=dict(zip(state['joint_names'],state['joint_velocities']))
            if max(abs(speeds[n]) for n in state['right_arm'])>.01:return
        if not self.physical_entry_ready:
            self.actual_entry=row
            write(self.episode/'physical_entry_observation.json',row);write(self.episode/'actual_entry.json',row)
            self.log_event({'kind':'physical_entry_stopped','source_anchor_not_positive':True,
                'registered_G0_world':trial['G0_world'],'registered_H_world':trial['entry_world'],
                'actual_entry_error':pose_error(snapshot['link_world']['right_eef_link'],trial['entry_world'])})
            self.physical_entry_ready=True
        if (method=='A' and self.baseline_motions==1) or (method=='B' and self.baseline_motions==GEOMETRY['max_queries']):
            ident=uuid.uuid4().hex;self.grasp_seen=True
            self.log_event({'kind':'grasp_request','decision_id':ident,'baseline_method':method,
                'closure_origin':'baseline_'+method,'model_explicit_GRASP':False,
                'driver':'registered_baseline_after_accepted_target','token':'GRASP'})
            return self.request_closure(ident)
        if method=='A':
            self.baseline_motions+=1
            self.log_event({'kind':'target_request','baseline_method':'A','driver':'fixed_registered_G0',
                'command_label':'a_approach','goal_world':trial['G0_world']})
            return self.submit('a_approach',trial['G0_world'],'local')
        if method!='B' or self.baseline_queries>=GEOMETRY['max_queries']:raise RuntimeError('Baseline action/query budget violated')
        self.group.writer.drain();ident=uuid.uuid4().hex;index=self.recorder.rows-1
        folder=self.episode/'geometry'; observation=folder/(ident+'_observation.json');write(observation,row)
        rgb,depth=self.episode/f'raw/{index:06d}.png',self.episode/f'raw/{index:06d}.npy'
        request={'request_id':ident,'trial_id':trial['trial_id'],'method':'B','raw_index':index,
            'G0_world':trial['G0_world'],'observation':str(observation),'rgb':str(rgb),'depth':str(depth),
            'asset':self.asset['usd_path'],'registration_sha256':sha(self.group.campaign/'registration.json'),
            'files':{str(p):sha(p) for p in (observation,rgb,depth)}}
        path=folder/(ident+'_request.json');self.baseline_queries+=1
        self.geometry_requests.append({'request_id':ident,'raw_index':index,'request':str(path)})
        write(self.episode/'geometry_requests.json',self.geometry_requests);write(path,request)
        self.pending_geometry={'path':path,'request':request,'snapshot':copy.deepcopy(row['snapshot']),'started':time.monotonic()}
        write(self.group.campaign/'workers/geometry/queue'/(ident+'.json'),{'request':str(path)})

    def consume_geometry(self,snapshot):
        p=self.pending_geometry;path=p['path'];request=p['request'];ident=request['request_id']
        reply_path=path.with_name(ident+'_reply.json')
        if (self.group.campaign/'workers/geometry/FAULT.json').exists():raise RuntimeError('Geometry service fault')
        if not reply_path.exists():
            if time.monotonic()-p['started']>GEOMETRY['request_timeout_seconds']:raise TimeoutError('Geometry inference timed out; no retry')
            return
        _,reply=read_bound_reply(path,reply_path);self.pending_geometry=None
        drift=dispatch_drift(p['snapshot'],snapshot);write(path.with_name(ident+'_dispatch_check.json'),drift)
        self.log_event({'kind':'geometry_reply','geometry_request_id':ident,'selected':reply['selected'],
            'geometry_reply_sha256':sha(reply_path),'dispatch_check':drift,'reason':reply['rejected']})
        if not drift['pass']:return self.finish('geometry_observation_changed_before_dispatch')
        if not reply['projection']['target_visibility_pass']:return self.finish('geometry_target_visibility_rejected')
        if reply['selected'] is None:return self.finish('geometry_no_candidate: '+reply['rejected'])
        self.baseline_motions+=1;label=f'b_refine_{self.baseline_motions:03d}'
        goal=reply['selected']['goal_world']
        self.log_event({'kind':'target_request','baseline_method':'B','geometry_request_id':ident,
            'driver':'fresh_wrist_graspness_G0_neighborhood','command_label':label,'goal_world':goal})
        return self.submit(label,goal,'local')

    def tick(self):
        label=(self.last_command or {}).get('label','')
        if self.state!='executing' or not label.startswith(('a_approach','b_refine_')):return super().tick()
        if self.bridge.fault:raise RuntimeError(str(self.bridge.fault))
        if self.task and time.monotonic()-self.started>self.group.registration['attempt_seconds']:
            raise TimeoutError('Original per-attempt resource budget exceeded')
        result=self.bridge.result()
        if result is None:return
        write(self.episode/(label+'_execution.json'),result)
        assessment=assess_settling(result,self.desired);write(self.episode/(label+'_settling.json'),assessment)
        self.stages.append(label);self.task['state']='executing'
        if forbidden_contacts(result):return self.finish('forbidden_contact')
        if not assessment['accepted']:return self.finish('original_convergence_rejected')
        if self.visual_issues:return self.finish('visual_geometry_rejected_before_close')
        self.previous_execution=result;self.previous_goal=self.desired.copy()
        self.outcome('accepted',True);self.ready_time=self.group.world.current_time;self.state='local_decision'

    def finish(self,reason=None):
        if self.group.active_trial['method'] in ('C','D'):return super().finish(reason)
        if self.recorder:
            self.group.writer.drain();write(self.episode/'baseline_integrity.json',audit_baseline(self.episode,self.group.active_trial))
        # Skip only the learned-output provenance check for A/B. The original
        # ActionEnvironment/LocalEnvironment and full physical evaluator remain.
        return PhysicalEnvironment.finish(self,reason)


def install(campaign):
    runtime_stage_recovery.install(campaign)
    reg=read(Path(campaign)/'registration.json')
    if reg['geometry_baseline']['contract']!=GEOMETRY:raise ValueError('Baseline contract changed')
    ready=read(Path(campaign)/'workers/geometry/ready.json')
    if ready['registration_sha256']!=sha(Path(campaign)/'registration.json'):raise ValueError('Foreign geometry worker')
    runtime_learned.LearnedEnvironment=MainEnvironment


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--launch',required=True)
    a=p.parse_args();install(a.campaign);runtime_learned.LearnedGroup(a.campaign,a.launch).run()
