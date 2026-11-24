"""Real ARX execution with teacher labels; no learned-policy claims."""
import argparse
from pathlib import Path
import time
import traceback
import uuid
import numpy as np
from scripts.curobo_arx.data_factory.common import read,write
from scripts.curobo_arx.data_factory.runtime import Environment
from scripts.curobo_arx.data_factory.ledger import Ledger
from scripts.curobo_arx.seed_augmentation.runtime import SeedGroup
from scripts.curobo_arx.settling import assess_settling
from training.nearfield_arx.contract import DSession,encode,decode,pose_error


def audit_actions(root,requests):
    requested={r['label'] for r in requests}
    executed={p.name.removesuffix('_execution.json') for p in root.glob('*_execution.json')}
    if executed-requested:raise RuntimeError('Orphan execution receipt')
    for label in requested|executed:
        for suffix in ('_command.json','_execution.json','_settling.json'):
            if not (root/(label+suffix)).is_file():raise RuntimeError('Missing occurred-action record: '+label+suffix)
        if label not in ('close','hold'):
            cert=read(root/(label+'_optimization.json'))
            if not cert.get('success') or cert['invalid_samples'] or cert['max_limit_exceed_rad']>1e-6:
                raise RuntimeError('Executed movement has no original planning certificate')
    return {'pass':True,'requested':sorted(requested),'executed':sorted(executed)}


def skip_arrival(execution,previous_goal,goal,snapshot,command):
    if execution is None or previous_goal is None:return {'accepted':False,'reason':'no_previous_execution'}
    match=pose_error(previous_goal,goal)
    if match['translation_m']>=2e-6 or match['rotation_rad']>=4e-6:
        return {'accepted':False,'reason':'different_command_target'}
    positions=dict(zip(snapshot['joint_names'],snapshot['joint_positions']))
    q=[positions[n] for n in command['joint_names']];target=command['position'][-1]
    latest={'time':execution['records'][-1]['time']+max(0.,snapshot['sim_time']-execution['final']['sim_time']),
            'position':q,'target':target,'ee_world':snapshot['link_world']['right_eef_link']}
    measured={**execution,'records':execution['records']+[latest],
              'settled':bool(execution['settled'] and np.max(np.abs(np.asarray(q)-target))<=command['position_tolerance_rad'])}
    return {**assess_settling(measured,previous_goal),'scope':'fresh measured arrival, not a fabricated command/execution'}


class LocalEnvironment(Environment):
    def record(self):
        saved=self.stage
        if saved and (saved.startswith('c_step_') or saved.startswith('d_refine_')):self.stage='approach'
        try:return super().record()
        finally:self.stage=saved

    def submit(self,label,goal,phase='approach',action='plan'):
        path=self.episode/(label+'_scene.json');write(path,self.bridge.export_scene())
        self.future=self.group.planner.submit(action,phase=phase,label=label,
            seed_id=self.group.active_seed['seed_id'],snapshot=self.bridge.snapshot(),scene=str(path),
            goal_world=np.asarray(goal).tolist(),grasp_world=self.task['goal']['grasp_world'],
            output=str(self.episode),visual_context=self.visual_context(),
            command_id=uuid.uuid4().hex,minimum_duration=6. if label=='pregrasp' else 3.)
        self.state='planning_'+phase

    def plan(self,phase,world):
        if phase=='reach':
            return self.submit('pregrasp',self.group.active_seed['anchor_world'],'reach','seed_connect')
        if phase!='approach':return super().plan(phase,world)
        self.actual_entry=self.last_observation
        write(self.episode/'actual_entry.json',self.actual_entry)
        self.local_index=0;self.events=[];self.session=DSession(self.group.active_trial['G0_world'])
        self.state='local_decision';self.ready_time=self.group.world.current_time
        self.previous_execution=None;self.previous_goal=None
        if self.group.active_trial['method']=='recovery':
            self.submit('state_generation',self.group.active_trial['offset_goal_world'],
                        self.group.registration.get('recovery_motion_phase','reach'))

    def dispatch(self,command,desired):
        saved=self.actual_entry
        super().dispatch(command,desired)
        if saved is not None:
            self.actual_entry=saved;write(self.episode/'actual_entry.json',saved)

    def log_event(self,event):
        self.events.append({**event,'sim_time':self.group.world.current_time,
                            'raw_index':self.recorder.rows-1,'snapshot':self.bridge.snapshot()})
        write(self.episode/'interpreter_events.json',self.events)

    def decide(self):
        trial=self.group.active_trial;method=trial['method']
        ee=np.array(self.bridge.snapshot()['link_world']['right_eef_link'])
        star=np.array(trial['source_Gstar_world'])
        if method=='C':
            tokens=trial['C_tokens']
            token=tokens[self.local_index] if self.local_index<len(tokens) else 'GRASP'
            if token=='GRASP':
                self.log_event({'kind':'grasp_request','token':token,'driver':'recorded_teacher_actions'})
                return self.dwell('close',self.previous_goal)
            driver=self.group.registration.get('C_truth_driver','recorded_teacher_actions')
            if driver=='current_state_to_recorded_next_waypoint':
                # The oracle is an OFFLINE validation driver, not a deployed
                # policy input. Labels are conditioned on this actual state.
                token=encode('C',ee,trial['C_waypoints'][self.local_index])
            goal=decode('C',ee,token);self.local_index+=1
            label='approach' if self.local_index==len(tokens) else f'c_step_{self.local_index:03d}'
            self.log_event({'kind':'target_request','token':token,'goal_world':goal.tolist(),'driver':driver})
            phase='approach' if label=='approach' else self.group.registration.get('C_intermediate_phase','approach')
            return self.submit(label,goal,phase)
        if method=='recovery':
            self.log_event({'kind':'expert_recovery_start','Gstar_offline_world':star.tolist(),
                'source_segment':'only states at/after this event are positive recovery supervision'})
            return self.submit('approach',star)
        # Teacher forcing tests the exact codec, fixed-prior handling, dispatch
        # and closed-loop receipt semantics. It is not a VLA performance trial.
        repeats=2 if trial['exercise_skip_reobserve'] else 1
        token=encode('D',ee,star,trial['G0_world']) if self.session.refines<repeats else 'GRASP'
        event=self.session.request(ee,token);self.log_event({**event,'driver':'offline_Gstar_teacher_forcing'})
        if event['kind']=='budget_exhausted':return self.finish('D_correction_budget_exhausted')
        if event['kind']=='grasp_request':return self.dwell('close',self.previous_goal)
        goal=np.array(event['goal_world'])
        # A strict joint-settled flag is meaningful only for its ORIGINAL target.
        # Never reuse it to declare a different decoded goal already reached.
        arrival=skip_arrival(self.previous_execution,self.previous_goal,goal,self.bridge.snapshot(),self.last_command)
        if arrival['accepted']:
            self.log_event({'kind':'motion_not_dispatched','reason':'skipped_already_arrived',
                'goal_world':goal.tolist(),'actual_goal_error':pose_error(ee,goal),
                'arrival_receipt':self.last_command['label']+'_settling.json','fresh_arrival_check':arrival,
                'execution_receipt_fabricated':False})
            self.ready_time=self.group.world.current_time;self.state='local_decision'
            return
        label='approach' if self.session.refines==1 else f'd_refine_{self.session.refines}'
        return self.submit(label,goal)

    def tick(self):
        if self.task and time.monotonic()-self.started>self.group.registration['attempt_seconds']:
            raise TimeoutError('Original per-attempt resource budget exceeded')
        if self.bridge.fault:raise RuntimeError(str(self.bridge.fault))
        if self.state=='local_decision':
            if self.bridge.active:raise RuntimeError('Decision requested while robot active')
            if self.group.world.current_time<=self.ready_time+1e-6:return
            return self.decide()
        local=(self.last_command or {}).get('label','')
        if self.state!='executing' or not(local in ('approach','state_generation') or local.startswith(('c_step_','d_refine_'))):
            return super().tick()
        result=self.bridge.result()
        if result is None:return
        write(self.episode/(local+'_execution.json'),result)
        assessment=assess_settling(result,self.desired);write(self.episode/(local+'_settling.json'),assessment)
        self.stages.append(local);self.task['state']='executing'
        target=result['final']['objects']['test_object']['path']
        contacts=[c for c in result['contacts'] if c['num_contacts']>0]
        forbidden=[c for c in contacts if not('/World/Robot/' in c['a'] and '/World/Robot/' in c['b'])
            and not(target in c['a']+c['b'] and any(n in c['a']+c['b'] for n in ('right_link27','right_link28')))]
        if forbidden:return self.finish('forbidden_contact')
        if not assessment['accepted']:return self.finish('original_convergence_rejected')
        if self.visual_issues:return self.finish('visual_geometry_rejected_before_close')
        self.previous_execution=result;self.previous_goal=self.desired.copy()
        if local=='state_generation':
            initial=read(self.episode/'capture_snapshot.json')['objects']['test_object']['world']
            actual=result['final']['objects']['test_object']['world'];drift=pose_error(actual,initial)
            self.log_event({'kind':'offset_stopped','measured_offset':pose_error(result['final']['link_world']['right_eef_link'],self.group.active_trial['source_Gstar_world']),
                            'object_drift':drift,'positive_supervision_begins_after_this':True})
            if any(target in c['a']+c['b'] for c in contacts) or drift['translation_m']>.001 or drift['rotation_rad']>.01:
                return self.finish('state_generation_contact_or_object_motion')
        elif self.group.active_trial['method']=='recovery':
            self.log_event({'kind':'grasp_request','driver':'successful_expert_recovery','token':'GRASP'})
            return self.dwell('close',self.previous_goal)
        self.ready_time=self.group.world.current_time;self.state='local_decision'

    def finish(self,reason=None):
        if self.bridge.active:raise RuntimeError('Cannot finalize active robot')
        write(self.episode/'interpreter_optimization.json',audit_actions(self.episode,self.requests))
        return super().finish(reason)


class LocalGroup(SeedGroup):
    def __init__(self,campaign,launch):
        super().__init__(campaign,launch)
        self.envs=[LocalEnvironment(i,e.bridge,self) for i,e in enumerate(self.envs)]

    def run(self):
        base=self.out;results=[];self.ledger.close()
        try:
            for ident in self.launch['trial_ids']:
                if self.stop or (self.campaign/'STOP').exists() or time.time()>=self.launch['stop_unix']:break
                trial=next(t for t in self.registration['seed_trials'] if t['trial_id']==ident)
                self.active_trial=trial;self.active_seed=self.registration['seeds'][trial['seed_id']]
                self.out=base/'trials'/ident;self.out.mkdir(parents=True,exist_ok=False)
                self.ledger=Ledger(self.campaign/'trial_ledgers'/(ident+'.sqlite3'))
                asset=self.ledger.acquire_object(self.owner,object_id=trial['object_id'],renderer=self.renderer)
                if asset is None:raise RuntimeError('Registered exact placement unavailable')
                started=time.time();block=self.run_seed_trial(asset);rows=self.ledger.rows()
                if len(rows)>1:raise RuntimeError('One attempt per slot')
                receipt={**trial,'completed':True,'started_unix':started,'finished_unix':time.time(),
                    'admitted':bool(rows and rows[0]['state']=='committed'),
                    'success':bool(rows and read(Path(rows[0]['episode'])/'physical_receipt.json')['pass']),
                    'physical_dispatches':sum(r['physical'] for r in rows),
                    'outcome':rows[0]['outcome'] if rows else block['status'],'attempts':rows,'block':block}
                write(self.out/'trial_result.json',receipt);results.append(receipt)
                write(base/'results.json',{'trials':results});self.ledger.release_object(self.owner);self.ledger.close()
                print('NEARFIELD_TRIAL',ident,receipt['outcome'],receipt['admitted'],flush=True)
        except BaseException:
            write(base/'FAULT.json',{'error':traceback.format_exc(),'completed':[r['trial_id'] for r in results]});raise
        finally:
            self.writer.close();self.ledger.close();self.diagnostics.close()
            write(base/'shutdown_receipt.json',{'all_trajectories_idle':not any(e.bridge.active for e in self.envs),
                  'raw_writer_drained_and_closed':True,'wall':time.time()})
            self.app.close(wait_for_replicator=False,skip_cleanup=True,exit_code=1 if (base/'FAULT.json').exists() else 0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--launch',required=True)
    a=p.parse_args();LocalGroup(a.campaign,a.launch).run()
