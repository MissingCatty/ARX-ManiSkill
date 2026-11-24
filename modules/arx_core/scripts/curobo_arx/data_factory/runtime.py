"""Nonblocking per-environment physical teacher state machines."""
import argparse
import copy
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import uuid
import numpy as np
from PIL import Image
from .common import ROOT,VERSION,STAGES,digest,read,sha,space,verify_files,write
from .geometry import handoff,pose_equal
from .ledger import Ledger
from .recording import Recording,Writer,physical,quality
from .workers import WorkerClient
from .vector_env import create_group


def advance_render_tick(world, expected_seconds):
    """Verify the actual public World step, including its physics substeps."""
    before=world.current_time
    world.step(render=True)
    delta=world.current_time-before
    # Match the unchanged recording-quality window. Kit can align the first
    # render with one 1/120 s physics substep (observed 0.1916666767 s).
    # Requiring microsecond-exact render advance invents a stricter contract.
    if not expected_seconds-.01<=delta<=expected_seconds+.01:
        raise RuntimeError(f'World render step advanced {delta}s; expected frozen {expected_seconds}s')


class Environment:
    def __init__(self, index, bridge, group):
        self.bridge,self.group = bridge,group
        self.id = group.owner+'/env_'+str(index)
        self.state,self.task,self.recorder = 'idle',None,None
        self.last_command,self.stage,self.actual_entry = None,None,None
        self.requests,self.stages = [],[]
        self.block_count = 0
        self.settle_until = 0.
        self.future = None
        self.visual_issues=[]

    def reset(self, asset):
        if self.bridge.active:
            raise RuntimeError('Unconfirmed idle robot at reset boundary')
        self.asset = asset
        if self.group.registration.get('visual_geometry_policy'):
            from .visual_geometry import TargetGeometry
            if not hasattr(self,'visual_asset') or self.visual_asset!=asset['asset_sha256']:
                self.visual_geometry=TargetGeometry.from_asset(asset);self.visual_asset=asset['asset_sha256']
        initial = self.bridge.reset_episode(asset)
        if asset.get('collection_placement'):initial['collection_placement']=asset['collection_placement']
        self.reset_initial = initial
        self.settle_until = self.group.world.current_time+3.
        self.state = 'settling'

    def record(self):
        if self.recorder:
            row,rgb,depth = self.bridge.observation(self.id,self.task['id'],self.asset['object_id'],self.task['goal']['grasp_world'],self.last_command,self.stage)
            if self.group.registration.get('visual_geometry_policy'):
                from .visual_geometry import inspect_row
                row['visual_geometry']=inspect_row(self.visual_geometry,row,depth.shape)
                if not row['visual_geometry']['pass']:
                    self.visual_issues.append({'raw_index':self.recorder.rows,'frame_time':row['frame_time'],**row['visual_geometry']})
            extras = getattr(self.bridge.context.cameras['right_wrist'], 'capture_arrays', None)
            self.recorder.add(row,rgb,depth,extras=extras)
            self.last_observation = row

    def visual_context(self):
        if not self.group.registration.get('visual_geometry_policy'):return None
        row,rgb,_=self.bridge.observation(self.id,self.task['id'] if self.task else 'qualification',self.asset['object_id'],None)
        return {'asset':{'usd_path':self.asset['usd_path'],'asset_sha256':self.asset['asset_sha256']},
            'eef_from_camera':(np.linalg.inv(row['snapshot']['link_world']['right_eef_link'])@np.asarray(row['camera_world_optical'])).tolist(),
            'intrinsics':row['intrinsics'],'shape':list(rgb.shape[:2]),'clipping_range_m':row['camera_clipping_range_m'],
            'calibration_frame_time':row['frame_time'],'camera_source_path':row['camera_source_path']}

    def plan(self, phase, world):
        snapshot = self.bridge.snapshot()
        scene = self.bridge.export_scene()
        scene_path = self.episode/(phase+'_scene.json')
        write(scene_path,scene)
        self.command_id = uuid.uuid4().hex
        self.future = self.group.planner.submit('plan',phase=phase,label=phase if phase!='reach' else 'pregrasp',
            snapshot=snapshot,scene=str(scene_path),goal_world=np.asarray(world).tolist(),output=str(self.episode),
            grasp_world=self.task['goal']['grasp_world'],
            ik_seed=self.task['goal'].get('qualification',{}).get('q_pregrasp') if phase=='reach' else None,
            visual_context=self.visual_context(),
            command_id=self.command_id,minimum_duration=3. if phase!='reach' else 6.)
        self.state = 'planning_'+phase

    def dispatch(self, command, desired):
        label = command['label']
        if any(r['label']==label for r in self.requests):
            raise RuntimeError('Duplicate phase dispatch')
        request = {**command,'requested_sim_time':self.group.world.current_time,'env_id':self.id,'attempt_id':self.task['id']}
        self.requests.append(request)
        write(self.episode/'requests.json',self.requests)
        write(self.episode/(label+'_command.json'),command)
        self.desired = np.asarray(desired)
        if label=='pregrasp':
            self.group.ledger.transition(self.task['id'],self.group.owner,'executing',physical=True,episode=self.episode)
        if label=='approach':
            # Snapshot from the last real, pre-command camera tick.
            self.actual_entry = self.last_observation
            write(self.episode/'actual_entry.json',self.actual_entry)
        self.bridge.command(command)
        self.last_command,self.stage = command,label
        self.state = 'executing'

    def dwell(self, label, desired):
        names = self.bridge.snapshot()['right_arm']
        q = self.last_command['position'][-1]
        cmd = {'id':uuid.uuid4().hex,'label':label,'joint_names':names,'position':[q,q],
               'time_from_start':[0.,3.],'gripper_target':.1,'position_tolerance_rad':.005,
               'planner':'Hold the preceding actual accepted ARX command for original 3-second dwell'}
        self.dispatch(cmd,desired)

    def finish(self, reason=None):
        if self.bridge.active:
            raise RuntimeError('Cannot finalize while trajectory is active')
        g = self.group
        g.writer.drain()
        for path in (self.episode/'raw').glob('*'):
            with path.open('rb') as stream:os.fsync(stream.fileno())
        samples_path=self.episode/'samples.jsonl'
        if samples_path.exists():
            with samples_path.open('rb') as stream:os.fsync(stream.fileno())
        if (self.episode/'raw').exists():
            raw_dir=os.open(self.episode/'raw',os.O_DIRECTORY)
            try:os.fsync(raw_dir)
            finally:os.close(raw_dir)
        write(self.episode/'grasp_report.json',{'success':reason is None,'error':reason,
            'stage_timing':{s:{} for s in self.stages},'planner':'cuRobo IK + PRM + trajectory optimization'})
        verification = physical(self.episode)
        meta = {'version':VERSION,'attempt_id':self.task['id'],'env_id':self.id,'environment_root':self.bridge.environment_root,
            'env_origin':self.bridge.roles.origin.tolist(),'object_id':self.asset['object_id'],'split':self.asset['split'],
            'geometry_group':self.asset['geometry_group'],'asset_sha256':self.asset['asset_sha256'],
            'scene_initial':self.reset_initial,'goal':self.task['goal'],'sampled_entry':self.handoff,
            'settled_initial_object_world':read(self.episode/'capture_snapshot.json')['objects']['test_object']['world'],
            'collection_placement':self.asset.get('collection_placement'),
            'visual_geometry_required':bool(g.registration.get('visual_geometry_policy')),
            'synchronized_capture_required':bool(g.registration.get('synchronized_capture')),
            'actual_entry':self.actual_entry,'samples':self.recorder.rows if self.recorder else 0,
            'write_complete':True,'source_identity':g.registration['source_identity'],'renderer':g.renderer,
            'gpu_uuid':g.launch['gpu_uuid'],'elapsed_wall_seconds':time.monotonic()-self.started}
        write(self.episode/'episode.json',meta)
        visual={'pass':not self.visual_issues,'checked':bool(g.registration.get('visual_geometry_policy')),
            'invalid_frames':self.visual_issues,'task_success':verification['pass'],
            'scope':'Additional training-image admission; original physical success is unchanged'}
        if visual['checked']:write(self.episode/'visual_receipt.json',visual)
        if self.task['state']=='claimed':
            g.ledger.transition(self.task['id'],g.owner,'rejected',outcome=reason or 'planning_rejected',episode=self.episode)
        else:
            g.ledger.transition(self.task['id'],g.owner,'writing',episode=self.episode)
            if verification['pass']:
                if visual['checked'] and not visual['pass']:
                    write(self.episode/'quality_receipt.json',{'pass':False,'reason':'visual_geometry_rejected','checked':False})
                    g.ledger.transition(self.task['id'],g.owner,'visual_rejected',outcome='physical_success_with_invalid_visual_observations',episode=self.episode)
                    outcome='visual_rejected'
                else:
                    quality(self.episode)  # A data failure raises and stops scheduling; never counted as grasp failure.
                    outcome = g.ledger.commit(self.task['id'],g.owner,self.episode)
            else:
                write(self.episode/'quality_receipt.json',{'pass':False,'reason':'physical_failure_not_admitted','checked':False})
                g.ledger.transition(self.task['id'],g.owner,'failed',outcome=reason or 'physical_grasp_failure',episode=self.episode)
                outcome = 'failed'
            print('EPISODE',self.task['id'],self.asset['object_id'],outcome,flush=True)
        self.task,self.recorder,self.last_command,self.stage,self.actual_entry = None,None,None,None,None
        self.requests,self.stages = [],[]
        self.visual_issues=[]
        self.block_count += 1
        self.state = 'idle'

    def tick(self):
        g = self.group
        if self.bridge.fault:
            raise RuntimeError(str(self.bridge.fault))
        if self.task and time.monotonic()-self.started>g.registration['attempt_seconds']:
            # Never issue another reset while the previous execution status is unknown.
            raise TimeoutError('Attempt wall budget exceeded; preserve partial attempt and stop group')
        if self.state=='idle':
            if g.stop or g.block_dispatches>=g.registration['object_block_attempts'] or self.block_count>=g.registration['object_block_attempts'] or not g.writer.dispatch_ready:
                return
            task = g.ledger.claim(self.asset['object_id'],g.owner,self.id,active_pool=g.pool['candidates'])
            if task is None:
                self.block_count = g.registration['object_block_attempts']
                return
            self.task,self.started = task,time.monotonic()
            g.block_dispatches+=1
            self.episode = g.out/'episodes'/task['id']
            self.episode.mkdir(parents=True,exist_ok=False)
            self.recorder = None
            self.handoff = g.sample_handoff(task)
            write(self.episode/'task.json',task)
            write(self.episode/'sampled_entry.json',self.handoff)
            write(self.episode/'selected_grasp.json',task['goal'])
            write(self.episode/'requests.json',[])
            self.reset(self.asset)
            return
        if self.state=='settling' and g.world.current_time>=self.settle_until-1e-6:
            if not self.task:
                self.state='idle'
                return
            snapshot = self.bridge.snapshot()
            write(self.episode/'capture_snapshot.json',snapshot)
            write(self.episode/'initial_conditions.json',self.reset_initial)
            self.task['state']='executing'  # Used after a successful physical dispatch below; restored on rejection.
            self.plan('reach',self.handoff['world'])
            return
        if self.state.startswith('planning_'):
            reply = g.planner.poll(self.future)
            if reply is None:
                return
            if not reply['ok']:
                raise RuntimeError(reply['error'])
            result = reply['result']
            if result.get('rejected'):
                self.task['state'] = 'claimed' if not self.stages else 'executing'
                self.finish('planning_rejected: '+result['rejected'])
                return
            self.dispatch(result['command'],result['desired_world'])
            return
        if self.state=='executing':
            result = self.bridge.result()
            if result is None:
                return
            label = self.last_command['label']
            write(self.episode/(label+'_execution.json'),result)
            sys.path.insert(0,str(ROOT/'scripts/curobo_arx'))
            from settling import assess_settling
            assessment = assess_settling(result,self.desired)
            write(self.episode/(label+'_settling.json'),assessment)
            self.stages.append(label)
            self.task['state']='executing'
            target = result['final']['objects']['test_object']['path']
            forbidden = [c for c in result['contacts'] if c['num_contacts']>0
                and not ('/World/Robot/' in c['a'] and '/World/Robot/' in c['b'])
                and not (target in c['a']+c['b'] and any(n in c['a']+c['b'] for n in ('right_link27','right_link28')))]
            if forbidden or not assessment['accepted']:
                self.finish('forbidden_contact' if forbidden else 'original_convergence_rejected')
                return
            goal = np.asarray(self.task['goal']['grasp_world'])
            if label=='pregrasp':
                if any(c['num_contacts']>0 and target in c['a']+c['b'] for c in result['contacts']):
                    self.finish('target_touched_during_reaching')
                    return
                row,rgb,depth = self.bridge.observation(self.id,self.task['id'],self.asset['object_id'],goal)
                target_camera = np.linalg.inv(np.asarray(row['camera_world_optical'])) @ np.asarray(row['snapshot']['objects']['test_object']['world'])[:,3]
                K = np.asarray(row['intrinsics'])
                uv = (K@target_camera[:3])[:2]/target_camera[2]
                if target_camera[2]<=0 or not (32<uv[0]<rgb.shape[1]-32 and 32<uv[1]<rgb.shape[0]-32):
                    self.finish('target_outside_registered_wrist_frustum')
                    return
                self.stage,self.last_command = 'entry',None
                self.recorder=Recording(self.episode,self.task,g.writer,existing=True)
                self.record()
                self.plan('approach',goal)
            elif label=='approach':
                if self.visual_issues:
                    self.finish('visual_geometry_rejected_before_close')
                    return
                self.dwell('close',goal)
            elif label=='close':
                fingers = {n for c in result['contacts'] if c['num_contacts']>0 and target in c['a']+c['b']
                    for n in ('right_link27','right_link28') if n in c['a']+c['b']}
                if len(fingers)!=2:
                    self.finish('no_bilateral_closure')
                    return
                lift = goal.copy()
                lift[2,3]+=.12
                self.plan('lift',lift)
            elif label=='lift':
                self.dwell('hold',self.desired)
            elif label=='hold':
                self.finish()


class Group:
    def __init__(self, campaign, launch):
        self.process_started=time.monotonic()
        self.campaign = Path(campaign)
        self.registration = read(self.campaign/'registration.json')
        self.launch = read(launch)
        self.owner = self.launch['group_id']
        self.renderer = self.launch['renderer']
        self.out = self.campaign/'groups'/self.owner
        self.out.mkdir(parents=True,exist_ok=False)
        from .diagnostics import FaultDiagnostics
        self.diagnostics=FaultDiagnostics(self.out/'python_stack.log')
        self.ledger = Ledger(self.campaign/'ledger.sqlite3')
        self.writer = Writer(capacity=128)
        self.planner = WorkerClient(self.launch['planner_directory'])
        self.neural = WorkerClient(self.launch.get('candidate_directory') or self.launch['zerograsp_directory'])
        self.stop = False
        signal.signal(signal.SIGTERM,lambda *_:setattr(self,'stop',True))
        signal.signal(signal.SIGINT,lambda *_:setattr(self,'stop',True))
        self.registration['envs_per_gpu'] = self.launch['env_count']
        self.app,self.world,bridges,self.subscription = create_group(self.registration,self.renderer,self.launch['env_count'],self.out)
        if self.registration.get('fingerprint_optimization'):
            for bridge in bridges:
                bridge.fingerprint_optimization = self.registration['fingerprint_optimization']
        self.envs = [Environment(i,b,self) for i,b in enumerate(bridges)]

    def sample_handoff(self,task):
        seed=self.registration['seed']
        if self.registration.get('entry_sampler')=='per_object_G_seed_v1':
            seed=int(digest([seed,task['object_id'],task['goal_id']])[:8],16)
        result=handoff(task['goal']['grasp_world'],task['distance'],task['sample'],seed)
        result['global_seed']=self.registration['seed']
        return result

    def step(self, record=True):
        self.diagnostics.tick()
        for env in self.envs:
            for camera in env.bridge.context.cameras.values():
                camera.sync_pose()
        advance_render_tick(self.world,self.registration['sample_seconds'])
        for env in self.envs:
            for camera in env.bridge.context.cameras.values():
                camera.refresh(self.world.current_time)
            env.bridge.after_step()
            if record and env.recorder:
                env.record()

    def run_block(self,asset,index):
        started=time.monotonic()
        block=self.out/'blocks'/f'{index:05d}__{asset["object_id"]}'
        block.mkdir(parents=True,exist_ok=False)
        if asset.get('collection_placement'):write(block/'collection_placement.json',asset['collection_placement'])
        self.block_dispatches=0
        preparation={}
        reset_started=time.monotonic()
        for env in self.envs:
            if env.task or env.bridge.active:raise RuntimeError('Object switch requires every environment idle')
            env.block_count=0
            env.reset(asset)
        preparation['reset_seconds']=time.monotonic()-reset_started
        settle_started=time.monotonic()
        for _ in range(15):self.step(False)
        preparation['settle_seconds']=time.monotonic()-settle_started
        from .frozen import fingerprint
        for i,env in enumerate(self.envs):
            check_started=time.monotonic()
            review=fingerprint(env.bridge,read(self.registration['reference_files'][self.renderer]))
            write(block/('frozen_'+str(i)+'.json'),review)
            preparation['environment_'+str(i)+'_check_seconds']=time.monotonic()-check_started
            if not review['pass']:raise RuntimeError('Frozen physical configuration mismatch: '+str(review['mismatches'][:20]))
            env.state='idle'
        preparation.update(total_seconds=time.monotonic()-started,
            fingerprint_optimization=self.registration.get('fingerprint_optimization'))
        write(block/'preparation_timing.json',preparation)
        first=self.envs[0]
        perception=block/'perception';perception.mkdir()
        perception_started=time.monotonic()
        row,rgb,depth=first.bridge.observation(first.id,'candidate_pool',asset['object_id'],None,camera_role='head')
        write(perception/'head_observation.json',row)
        Image.fromarray(rgb).save(perception/'head.png');np.save(perception/'head_depth.npy',depth)
        job=self.neural.submit('infer',directory=str(perception),asset=asset,visual_context=first.visual_context())
        while True:
            if self.stop or (self.campaign/'STOP').exists():
                self.stop=True
                return {'status':'stopped','object_id':asset['object_id'],'idle':True,'dispatches':0}
            reply=self.neural.poll(job)
            if reply is not None:break
            self.step(False)
            if time.monotonic()-perception_started>900:raise TimeoutError('Resident perception budget exceeded')
        if not reply['ok']:raise RuntimeError(reply['error'])
        from .placements import tag_pool
        raw_pool=tag_pool(reply['result'],asset)
        scene=first.bridge.export_scene();scene_path=block/'qualification_scene.json';write(scene_path,scene)
        job=self.planner.submit('qualify_pool',pool=raw_pool,snapshot=first.bridge.snapshot(),scene=str(scene_path),
            visual_context=first.visual_context())
        qualified_started=time.monotonic()
        while True:
            if self.stop or (self.campaign/'STOP').exists():
                self.stop=True
                return {'status':'stopped','object_id':asset['object_id'],'idle':True,'dispatches':0}
            qualified=self.planner.poll(job)
            if qualified is not None:break
            self.step(False)
            if time.monotonic()-qualified_started>900:raise TimeoutError('Resident G qualification budget exceeded')
        if not qualified['ok']:raise RuntimeError(qualified['error'])
        write(block/'qualified_pool.json',qualified['result'])
        self.pool=self.ledger.add_goals(asset['object_id'],qualified['result'])
        write(block/'pool.json',self.pool)
        if not self.pool['candidates']:
            result={'status':'candidate_exhausted','object_id':asset['object_id'],'idle':True,'dispatches':0,'reason':self.pool.get('rejected')}
            write(block/'result.json',result);return result
        while self.app.is_running():
            if time.time()>self.launch['stop_unix'] or (self.campaign/'STOP').exists():self.stop=True
            space(self.out);self.writer.check()
            for env in self.envs:env.tick()
            active=any(env.task is not None for env in self.envs)
            if not active and (self.stop or self.block_dispatches>=self.registration['object_block_attempts'] or all(env.block_count>=self.registration['object_block_attempts'] for env in self.envs)):break
            self.step()
            write(self.out/'heartbeat.json',{'wall':time.time(),'object_id':asset['object_id'],'block':index,
                'states':{env.id:env.state for env in self.envs},'active':[env.task['id'] for env in self.envs if env.task]})
        self.writer.drain()
        if any(env.task or env.bridge.active for env in self.envs):raise RuntimeError('Object block ended without confirmed idle')
        result={'status':'stopped' if self.stop else ('block_complete' if self.block_dispatches else 'object_budget_exhausted'),
            'object_id':asset['object_id'],'wall_seconds':time.monotonic()-started,'idle':True,'dispatches':self.block_dispatches}
        write(block/'result.json',result);return result

    def run(self):
        index=0;asset=None;result=None
        try:
            while not self.stop and not (self.campaign/'STOP').exists() and time.time()<self.launch['stop_unix']:
                asset=self.ledger.acquire_object(self.owner,object_id=self.launch['object_id'] if index==0 else None,renderer=self.renderer)
                if asset is None:break
                result=self.run_block(asset,index)
                reason=result['status'] if result['status'] in ('candidate_exhausted','object_budget_exhausted') else None
                self.ledger.release_object(self.owner,exhausted_reason=reason,evidence=result)
                index+=1
            self.writer.drain()
            if any(env.task or env.bridge.active for env in self.envs):raise RuntimeError('Application exit without confirmed idle')
            self.ledger.release_object(self.owner)
            write(self.out/'block_result.json',{'status':'stopped' if self.stop or (self.campaign/'STOP').exists() else 'renderer_rotation',
                'object_id':asset['object_id'] if asset else self.launch['object_id'],'completed_object_blocks':index,
                'wall_seconds':time.monotonic()-self.process_started,'env_count':len(self.envs),'idle':True})
        except BaseException:
            error = traceback.format_exc()
            write(self.out/'FAULT.json',{'error':error,'wall':time.time(),'active':[env.task for env in self.envs if env.task]})
            for env in self.envs:
                if env.task:
                    try:
                        self.ledger.transition(env.task['id'],self.owner,'fault',outcome='infrastructure_or_record_fault',episode=env.episode)
                    except Exception:
                        pass
            raise
        finally:
            self.writer.close()
            self.ledger.close()
            self.diagnostics.close()
            idle=not any(env.bridge.active for env in self.envs)
            write(self.out/'shutdown_receipt.json',{'wall':time.time(),'all_trajectories_idle':idle,
                'raw_writer_drained_and_closed':True,'ledger_closed':True,
                'sdk_exit':'Documented skip_cleanup after durable evidence; no Replicator workflow remains to wait for'})
            self.app.close(wait_for_replicator=False,skip_cleanup=True,
                exit_code=1 if (self.out/'FAULT.json').exists() else 0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',required=True)
    p.add_argument('--launch',required=True)
    args=p.parse_args()
    Group(args.campaign,args.launch).run()
