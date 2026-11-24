"""Advisor38 uniformly camera-gated seed checks: axial insertion, measured closure, no VLM."""
import argparse
import copy
from pathlib import Path
import time
import traceback
import uuid
import numpy as np
from scripts.curobo_arx.data_factory.common import read,write,sha,digest
from scripts.curobo_arx.data_factory import runtime as factory
from scripts.curobo_arx.data_factory.frozen import fingerprint
from scripts.curobo_arx.data_factory.ledger import Ledger
from training.nearfield_arx.contract import pose_error
from .runtime_advisor34 import Environment as MeasuredEnvironment
from .runtime_v2 import ActionGroup
from .closure import observation_timing
from .advisor31_io import StatusWriter
from . import advisor36_audit


class SeedEnvironment(MeasuredEnvironment):
    def submit(self,label,goal,phase='local',action='plan'):
        from .advisor37_surfaces import export_geometry
        path=self.episode/(label+'_scene.json');write(path,self.bridge.export_scene());snap=self.bridge.snapshot()
        geometry=self.episode/'original_cooked_collision_surfaces.json'
        if not geometry.exists():write(geometry,export_geometry(self.bridge))
        if label=='state_generation':action='axial_seed_insert'
        self.future=self.group.planner.submit(action,phase=phase,label=label,
            seed_id=self.group.active_seed['seed_id'],snapshot=snap,scene=str(path),
            goal_world=np.asarray(goal).tolist(),
            grasp_world=snap['link_world']['right_eef_link'] if phase in ('local','lift') else self.task['goal']['grasp_world'],
            output=str(self.episode),visual_context=self.visual_context(),command_id=uuid.uuid4().hex,
            surface_geometry=str(geometry),minimum_duration=6. if label=='pregrasp' else 8.)
        self.state='planning_'+phase

    def plan(self,phase,world):
        from .advisor37_motion import prepose
        if phase=='reach':
            return self.submit('pregrasp',prepose(self.group.active_trial['entry_world']),'reach','plan')
        result=super().plan(phase,world)
        if phase=='approach':self.previous_goal=prepose(self.group.active_trial['entry_world'])
        return result

    def decide(self):
        if self.bridge.active or self.bridge.fault:raise RuntimeError('Seed closure requires confirmed idle')
        if self.last_observation['frame_time']<=self.previous_execution['final']['sim_time']+1e-6:return
        proof=self.proof(0.)
        if not proof['pass']:return
        trial=self.group.active_trial;snapshot=self.bridge.snapshot()
        actual=np.asarray(snapshot['link_world']['right_eef_link']);target=np.asarray(trial['entry_world'])
        error=pose_error(actual,target)
        initial=read(self.episode/'capture_snapshot.json')['objects']['test_object']['world']
        drift=pose_error(snapshot['objects']['test_object']['world'],initial)
        from .advisor37_motion import seed_arrival
        passed=seed_arrival(error,drift)
        camera_check=None
        if True:  # Same 10 mm camera rule for every object.
            from .advisor37_geometry import clearance_bound
            row=self.last_observation
            camera_check=self.visual_geometry.inspect(row['snapshot']['objects']['test_object']['world'],
                row['camera_world_optical'],row['intrinsics'],self.visual_context()['shape'],row['camera_clipping_range_m'],
                {**self.group.registration['visual_geometry_policy'],'minimum_surface_distance_m':.010})
            passed=bool(passed and camera_check['pass'] and clearance_bound(camera_check)>=.010)
        check={'pass':bool(passed),'boundary_case':trial['boundary_case'],'requested_goal_world':target.tolist(),
            'actual_eef_world':actual.tolist(),'target_error':error,'object_drift_before_close':drift,
            'actual_residual_to_canonical':pose_error(actual,trial['Gcanonical_world']),
            'actual_preclosure_camera_check':camera_check,
            'original_pose_arrival_limits_retained':False,'authorized_arrival_limits_m_deg':[.003,1.5],'new_model_inference':False}
        write(self.episode/'seed_pose_verification.json',check)
        if not passed:return self.finish('canonical_seed_pose_or_object_drift_rejected')
        self.actual_entry=self.last_observation;write(self.episode/'actual_entry.json',self.actual_entry)
        self.log_event({'kind':'canonical_seed_close_request','model_inference':False,
            'seed_case':trial['boundary_case'],'requested_goal_world':target.tolist()})
        self.grasp_seen=True
        return self.request_closure(uuid.uuid4().hex)

    def request_closure(self,ident):
        self.group.writer.drain();snap=self.bridge.snapshot();proof=self.proof(0.)
        if not proof['pass']:return self.finish('closure_not_stationary')
        names=snap['right_arm'];q=[float(snap['joint_positions'][snap['joint_names'].index(n)]) for n in names]
        binding={'pass':True,'version':'advisor38_seed','grasp_request_id':ident,
            'closure_source':'canonical_seed_preflight','hold_joint_names':names,'hold_q':q,
            'snapshot_sha256':digest(snap),'measured_hold_snapshot':snap,'files':proof['files'],
            'arrival':proof,'previous_command_id':self.last_command['id'],'previous_command_label':self.last_command['label'],
            'observation_timing':observation_timing(self.last_observation),'configuration_identity':self.group.registration['source_identity']}
        write(self.episode/'closure_request.json',binding);self.closure_binding=binding;self.closure_snapshot=snap
        scene=self.episode/'closure_scene.json';write(scene,self.bridge.export_scene())
        self.future=self.group.planner.submit('closure_validate',binding=binding,snapshot=snap,scene=str(scene),output=str(self.episode))
        self.state='validating_closure'


class Group(ActionGroup):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.envs=[SeedEnvironment(i,e.bridge,self) for i,e in enumerate(self.envs)]

    def sample_handoff(self,task):
        return {'world':self.active_trial['entry_world'],'distance_mm':0.,'sample':0,
            'seed':self.registration['seed'],'policy':'direct_canonical_seed_preflight',
            'boundary_case':self.active_trial['boundary_case'],'not_a_VLM_trial':True}

    def run(self):
        base=self.out;results=[];failed_objects=set();self.ledger.close()
        try:
            for ident in self.launch['trial_ids']:
                if self.stop or (self.campaign/'STOP').exists() or time.time()>=self.launch['stop_unix']:break
                trial=next(t for t in self.registration['seed_trials'] if t['trial_id']==ident)
                self.active_trial=trial;self.active_seed=self.registration['seeds'][trial['seed_id']]
                self.out=base/'trials'/ident;self.out.mkdir(parents=True,exist_ok=False)
                if trial['object_id'] in failed_objects:
                    receipt={**trial,'completed':False,'executed':False,'success':False,'admitted':False,
                        'record_integrity_pass':True,'seed_pose_verified':False,'physical_dispatches':0,
                        'outcome':'unexecuted_after_failed_seed_case','attempts':[]}
                else:
                    env=self.envs[0];env.events=[];env.traces=[];env.policy_requests=[];env.entry_saved=False
                    self.ledger=Ledger(self.campaign/'trial_ledgers'/(ident+'.sqlite3'))
                    asset=self.ledger.acquire_object(self.owner,object_id=trial['object_id'],renderer=self.renderer)
                    if asset is None:raise RuntimeError('Registered placement unavailable')
                    started=time.time();block=self.run_seed_trial(asset);rows=self.ledger.rows()
                    if len(rows)>1:raise RuntimeError('More than one seed attempt')
                    proof=fingerprint(env.bridge,read(self.registration['reference_files'][self.renderer]))
                    write(self.out/'frozen_after.json',proof)
                    if not proof['pass'] or env.bridge.active or env.bridge.fault:raise RuntimeError('Configuration or idle failure')
                    episode=Path(rows[0]['episode']) if rows and rows[0].get('episode') else None
                    success=bool(episode and (episode/'physical_receipt.json').exists() and read(episode/'physical_receipt.json')['pass'])
                    seed_pose=bool(episode and (episode/'seed_pose_verification.json').exists() and read(episode/'seed_pose_verification.json')['pass'])
                    if episode:
                        from .runtime import audit_actions
                        audit_actions(episode,read(episode/'requests.json'))
                    receipt={**trial,'completed':True,'executed':bool(rows and rows[0]['physical']),
                        'started_unix':started,'finished_unix':time.time(),'success':success,
                        'admitted':bool(rows and rows[0]['state']=='committed'),'record_integrity_pass':True,
                        'seed_pose_verified':seed_pose,'physical_dispatches':sum(bool(r['physical']) for r in rows),
                        'outcome':rows[0]['outcome'] if rows else block['status'],'attempts':rows,'block':block}
                    self.ledger.release_object(self.owner);self.ledger.close()
                    if not success or not seed_pose:failed_objects.add(trial['object_id'])
                write(self.out/'trial_result.json',receipt);results.append(receipt)
                write(base/'results.json',{'trials':results,'failed_objects':sorted(failed_objects)})
                print('ADVISOR38_SEED',ident,receipt['outcome'],receipt['success'],flush=True)
        except BaseException:
            write(base/'FAULT.json',{'error':traceback.format_exc(),'completed':[r['trial_id'] for r in results]});raise
        finally:
            self.writer.close();self.ledger.close();self.diagnostics.close()
            write(base/'shutdown_receipt.json',{'all_trajectories_idle':not any(e.bridge.active for e in self.envs),
                'raw_writer_drained_and_closed':True,'wall':time.time()})
            self.app.close(wait_for_replicator=False,skip_cleanup=True,exit_code=1 if (base/'FAULT.json').exists() else 0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--launch',required=True);a=p.parse_args()
    from . import runtime_v2
    runtime_v2.write=StatusWriter(write)
    factory.physical=advisor36_audit.physical;factory.quality=advisor36_audit.install_quality()
    Group(a.campaign,a.launch).run()
