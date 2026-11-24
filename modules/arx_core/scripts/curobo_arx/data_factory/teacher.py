"""Resident, serial cuRobo IK/PRM/trajopt teacher with frozen ARX models.

Planning requests are serialized here; independent physical execution/rendering
continues concurrently in the simulator. This is not fully batched planning.
"""
import copy
import os
from pathlib import Path
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .common import ROOT, digest, read, write, sha
from .geometry import equivalent_grasp


class Rejection(Exception):
    def __init__(self, reason, details=None):
        super().__init__(reason)
        self.details = details or {}


def ik_failure_details(result):
    """Explain solver rejection without turning a converged pose into acceptance."""
    details = {}
    for name in ('position_error', 'rotation_error', 'feasible'):
        value = getattr(result, name, None)
        if value is None:
            continue
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        if name == 'feasible':
            details['feasible_seeds'] = int(array.sum())
            details['reported_seeds'] = int(array.size)
        elif array.size and np.isfinite(array).any():
            details['minimum_' + name] = float(array[np.isfinite(array)].min())
    return details


def bounded_planning_start(raw, limits):
    """Existing boundary-noise allowance, before the existing interior margin."""
    raw,limits=np.asarray(raw,dtype=float),np.asarray(limits,dtype=float)
    violation=np.maximum(np.maximum(limits[:,0]-raw,raw-limits[:,1]),0.)
    if not np.isfinite(raw).all() or violation.max()>1e-4+1e-7:
        raise Rejection('measured_start_outside_frozen_limits')
    return np.maximum(np.minimum(raw,limits[:,1]-1e-4),limits[:,0]+1e-4)


class Teacher:
    def __init__(self, registration, directory):
        self.registration,self.directory,self.context = registration,Path(directory),None
        sys.path.insert(0,str(ROOT/'scripts/curobo_arx'))
        os.environ['GRASP_TRAJOPT_POLICY'] = 'interactive'
        import torch
        self.torch = torch
        torch.manual_seed(registration['seed'])

    def visual_target(self, request):
        if not self.registration.get('visual_geometry_policy'):return None
        context=request.get('visual_context')
        if context is None:raise RuntimeError('Registered visual gate has no measured camera context')
        from .visual_geometry import TargetGeometry
        asset=context['asset'];key=asset['asset_sha256']
        if not hasattr(self,'visual_targets'):self.visual_targets={}
        if key not in self.visual_targets:
            if sha(asset['usd_path'])!=key:raise RuntimeError('Visual target source asset changed')
            self.visual_targets[key]=TargetGeometry.from_asset(asset)
        return self.visual_targets[key]

    def visual_goal(self, request, eef_world):
        geometry=self.visual_target(request)
        if geometry is None:return None
        context=request['visual_context']
        return geometry.inspect(request['snapshot']['objects']['test_object']['world'],
            eef_world@np.asarray(context['eef_from_camera']),context['intrinsics'],context['shape'],
            context['clipping_range_m'],self.registration['visual_geometry_policy'])

    def visual_path(self, request, planner, q, out):
        geometry=self.visual_target(request)
        if geometry is None:return
        from curobo.types import JointState
        from .visual_geometry import camera_path
        torch=self.torch;context=request['visual_context'];snapshot=request['snapshot']
        joint=JointState.from_position(torch.as_tensor(q,device='cuda',dtype=torch.float32)[None],joint_names=self.context['names'])
        fk=planner.compute_kinematics(joint).tool_poses
        positions=fk.position.detach().cpu().numpy().reshape(-1,3)
        quaternions=fk.quaternion.detach().cpu().numpy().reshape(-1,4)
        matrices=np.tile(np.eye(4),(len(positions),1,1))
        matrices[:,:3,:3]=Rotation.from_quat(quaternions[:,[1,2,3,0]]).as_matrix()
        matrices[:,:3,3]=positions
        world=np.asarray(snapshot['link_world']['body4'])@matrices
        obj=np.asarray(snapshot['objects']['test_object']['world'])
        relative_object=np.linalg.inv(snapshot['link_world']['right_eef_link'])@obj
        mount=np.asarray(context['eef_from_camera']);checked=0;failure=None
        for camera in camera_path(world,mount,self.registration['visual_geometry_policy']):
            # During lift the prediction assumes an object rigidly held at its
            # measured relative pose. Actual frames independently check slip.
            target=(camera@np.linalg.inv(mount)@relative_object) if request['phase']=='lift' else obj
            assessment=geometry.inspect(target,camera,context['intrinsics'],context['shape'],
                context['clipping_range_m'],self.registration['visual_geometry_policy'])
            checked+=1
            if not assessment['pass']:failure={'index':checked-1,**assessment};break
        receipt={'pass':failure is None,'camera_poses_checked':checked,'first_failure':failure,
            'source':'Actual unchanged ARX model FK of optimized joint path, densified optical poses',
            'lift_object_prediction':'Rigid measured EEF/object attachment; actual frame checks still required',
            'continuous_execution_certificate':False,'original_robot_camera_physics_changed':False}
        write(out/(request['label']+'_visual_validation.json'),receipt)
        if failure is not None:raise Rejection('camera_visual_path_rejected',failure)

    @staticmethod
    def scene_meshes(scene, target_path, include_target=False):
        return [m for m in scene['meshes'] if not m['path'].startswith('/World/Robot/')
            and (include_target or not (m['path']==target_path or m['path'].startswith(target_path+'/')))]

    def initialize(self, scene, snapshot, include_target=False):
        import torch
        import yaml
        import trimesh
        from curobo.motion_planner import MotionPlanner,MotionPlannerCfg
        from curobo.collision_checking import RobotCollisionChecker,RobotCollisionCheckerCfg
        from model_archive import relocate
        from gripper_update import WarmedGripperUpdate
        target_path=snapshot['objects']['test_object']['path']
        # Match scripts/curobo_arx/prepare.py's established scene preparation.
        meshes = self.scene_meshes(scene,target_path,include_target)
        key = digest(meshes)
        if self.context:
            if self.context['scene_key'] != key:
                # Refresh actual exported geometry; same fixed-capacity planners stay resident.
                from curobo._src.geom.types import SceneCfg
                cfg = self.scene_files(meshes,key)
                for mode in ('open','closed'):
                    planning = copy.deepcopy(cfg)
                    if mode == 'closed':
                        invbase = np.linalg.inv(np.asarray(snapshot['link_world']['body4']))
                        for name,mesh in planning['mesh'].items():
                            if 'breakfast_table' in meshes[int(name.split('_')[-1])]['path']:
                                mesh['pose'][:3] = (np.asarray(mesh['pose'][:3])+invbase[:3,2]*.008).tolist()
                    config = SceneCfg.create(planning)
                    self.context[mode][0].update_world(config)
                    self.context[mode][1].update_world(config)
                self.context['scene_key'] = key
            return
        started = time.monotonic()
        names = snapshot['right_arm']
        cfg = self.scene_files(meshes,key)
        model = self.directory/'models'
        reference = Path(self.registration['model_reference'])
        relocate(reference/'model',model/'model',snapshot)
        relocate(reference/'model_closed',model/'model_closed',snapshot,measured_gripper=False)
        closed_path = model/'model_closed/arx.yml'
        closed = yaml.safe_load(closed_path.read_text())
        closed['robot_cfg']['kinematics']['lock_joints'].update(right_joint27=.022,right_joint28=.022)
        closed_path.write_text(yaml.safe_dump(closed,sort_keys=False))
        context = {'scene_key':key,'names':names,'models_root':str(model)}
        for mode,folder in (('open','model'),('closed','model_closed')):
            scene_cfg = copy.deepcopy(cfg)
            if mode == 'closed':
                invbase = np.linalg.inv(np.asarray(snapshot['link_world']['body4']))
                for name,mesh in scene_cfg['mesh'].items():
                    if 'breakfast_table' in meshes[int(name.split('_')[-1])]['path']:
                        mesh['pose'][:3] = (np.asarray(mesh['pose'][:3])+invbase[:3,2]*.008).tolist()
            robot = str(model/folder/'arx.yml')
            task = yaml.safe_load((ROOT.parent/'curobo/source/curobo/content/configs/task/trajopt/lbfgs_bspline_trajopt.yml').read_text())
            task['rollout']['cost_cfg']['cspace_cfg']['weight'][0] *= 100
            config = MotionPlannerCfg.create(robot=robot,scene_model=scene_cfg,self_collision_check=False,
                trajopt_optimizer_configs=[task],collision_cache={'mesh':self.registration['planning_mesh_capacity'],'primitive':8},
                interpolation_dt=.025,optimizer_collision_activation_distance=.003,position_tolerance=.002,orientation_tolerance=.03)
            for rollout in (config.graph_planner_config.rollout_config,config.ik_solver_config.core_cfg.metrics_rollout_config,config.trajopt_solver_config.core_cfg.metrics_rollout_config):
                for cost in rollout.get_cost_manager_configs():
                    cost.disable_self_collision()
            planner = MotionPlanner(config)
            current = planner.default_joint_state.clone().unsqueeze(0)
            planner.ik_solver.solve_pose(planner.compute_kinematics(current).tool_poses.as_goal(),current_state=current,return_seeds=1)
            planner.trajopt_solver.solve_cspace(current,current,num_seeds=1,initial_iters=25,finetune_attempts=0)
            planner.graph_planner.warmup(num_warmup_iterations=1,max_batch_size=1)
            planner.reset_seed()
            checker = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(robot_config=robot,scene_model=scene_cfg,
                n_meshes=self.registration['planning_mesh_capacity'],collision_activation_distance=0.))
            checker.self_collision_cost = None
            update = WarmedGripperUpdate(model/folder/'arx.yml',planner.kinematics)
            models = [r.transition_model.robot_model for r in (planner.ik_solver.get_all_rollout_instances()+planner.trajopt_solver.get_all_rollout_instances()+planner.graph_planner.get_all_rollout_instances())]+[checker.kinematics,planner.kinematics]
            context[mode] = (planner,checker,update,models)
        self.context = context
        write(self.directory/'warmup_receipt.json',{'seconds':time.monotonic()-started,'open_and_closed':True,
            'uniform_workload':'IK, cspace trajectory optimization, PRM, fixed nominal gripper update',
            'physical_commands':0,'geometry_source':str(reference),'geometry_modified':False})

    def scene_files(self, meshes, key):
        import trimesh
        if len(meshes)>=self.registration['planning_mesh_capacity']:
            raise RuntimeError('Scene exceeds registered collision cache; cannot truncate obstacles')
        out = self.directory/'scene_versions'/key
        out.mkdir(parents=True,exist_ok=True)
        cfg = {'mesh':{}}
        for i,mesh in enumerate(meshes):
            if 'geometry_file' in mesh:
                from .common import sha
                path=Path(mesh['geometry_file'])
                if not hasattr(self,'verified_geometry_files'):self.verified_geometry_files={}
                info=path.stat();stamp=(info.st_size,info.st_mtime_ns)
                if self.verified_geometry_files.get(str(path))!=(stamp,mesh['geometry_sha256']):
                    if sha(path)!=mesh['geometry_sha256']:raise RuntimeError('Shared collision geometry changed')
                    self.verified_geometry_files[str(path)]=(stamp,mesh['geometry_sha256'])
                cfg['mesh'][f'collider_{i}']={'file_path':str(path),'pose':mesh['pose']}
                continue
            path = out/f'{i}.obj'
            if not path.exists():
                trimesh.Trimesh(vertices=mesh['vertices'],faces=mesh['faces'],process=False).export(path)
            cfg['mesh'][f'collider_{i}'] = {'file_path':str(path),'pose':[0,0,0,1,0,0,0]}
        return cfg

    def js(self, q):
        from curobo.types import JointState
        return JointState.from_position(q[None],joint_names=self.context['names'])

    def ik(self, world, start, planner, checker, inverse, seed=None):
        from curobo.types import GoalToolPose
        torch = self.torch
        t = inverse @ world
        q = Rotation.from_matrix(t[:3,:3]).as_quat()
        goal = GoalToolPose(tool_frames=['right_eef_link'],position=torch.tensor(t[:3,3],dtype=torch.float32,device='cuda').reshape(1,1,1,1,3),
            quaternion=torch.tensor(np.r_[q[3],q[:3]],dtype=torch.float32,device='cuda').reshape(1,1,1,1,4))
        options={}
        if seed is not None:
            numerical=torch.as_tensor(seed,device=start.device,dtype=start.dtype)
            if numerical.shape!=start.shape or not torch.isfinite(numerical).all():
                raise RuntimeError('Invalid qualified numerical IK seed')
            options['seed_config']=numerical.reshape(1,1,-1)
        result = planner.ik_solver.solve_pose(goal,current_state=self.js(start),return_seeds=4,**options)
        if not result.success.any():
            raise Rejection('no_ik_solution', ik_failure_details(result))
        values = result.solution[result.success].reshape(-1,len(start))
        values = values[checker.validate(values[:,None]).flatten()]
        if not len(values):
            raise Rejection('ik_goal_collision')
        return values[(values-start).square().sum(-1).argmin()].clone()

    def handle(self, request):
        if request['action']=='qualify_pool':return self.qualify_pool(request)
        import torch
        from curobo._src.graph_planner.graph_planner_prm import TrajInterpolationType
        from optimize_path import optimize_path,flush_planning_artifacts
        snapshot = request['snapshot']
        scene=read(request['scene'])
        self.initialize(scene,snapshot)
        if snapshot['right_arm'] != self.context['names']:
            raise RuntimeError('ARX planner joint identity changed')
        names = self.context['names']
        limits = torch.tensor([snapshot['joint_limits'][snapshot['joint_names'].index(n)] for n in names],device='cuda')
        raw_start=[snapshot['joint_positions'][snapshot['joint_names'].index(n)] for n in names]
        raw_limits=[snapshot['joint_limits'][snapshot['joint_names'].index(n)] for n in names]
        try:projected=bounded_planning_start(raw_start,raw_limits)
        except Rejection as exc:return {'rejected':str(exc)}
        start=torch.tensor(projected,dtype=torch.float32,device='cuda')
        phase = request['phase']
        mode = 'closed' if phase=='lift' else 'open'
        planner,checker,updater,models = self.context[mode]
        measured = dict(zip(snapshot['joint_names'],snapshot['joint_positions']))
        updater.apply(models,{n:float(measured[n]) for n in updater.names})
        inverse = np.linalg.inv(np.asarray(snapshot['link_world']['body4']))
        actual = inverse @ np.asarray(snapshot['link_world']['right_eef_link'])
        state = planner.compute_kinematics(self.js(start)).tool_poses
        xyz = state.position.detach().cpu().numpy().reshape(3)
        quat = state.quaternion.detach().cpu().numpy().reshape(4)
        rot = Rotation.from_quat(np.r_[quat[1:],quat[0]]).as_matrix()
        fk = {'position_error_m':float(np.linalg.norm(xyz-actual[:3,3])),
              'rotation_error_rad':float(Rotation.from_matrix(rot@actual[:3,:3].T).magnitude())}
        if fk['position_error_m']>=.002 or fk['rotation_error_rad']>=.01:
            raise RuntimeError('Live ARX/planner FK mismatch: '+str(fk))
        try:
            if not checker.validate(start[None,None]).all():
                raise Rejection('measured_start_collision')
            desired = np.asarray(request['goal_world'])
            numerical_seed=request.get('ik_seed') if self.registration.get('qualified_pregrasp') and phase=='reach' else None
            goal = self.ik(desired,start,planner,checker,inverse,seed=numerical_seed)
            if phase in ('reach','approach'):
                certificate=self.nominal_closed(np.asarray(request['grasp_world']),start,inverse,
                    q_grasp_override=goal if phase=='approach' else None)
                write(Path(request['output'])/(request['label']+'_nominal_closed_validation.json'),certificate)
            if phase=='reach' and self.registration.get('reach_avoid_target'):
                # Grasp/lift qualification permits intended target contact. Reaching
                # H must instead avoid the unchanged live target geometry.
                self.initialize(scene,snapshot,include_target=True)
                write(Path(request['output'])/'reach_target_avoidance.json',{
                    'target_path':snapshot['objects']['test_object']['path'],
                    'target_mesh_paths':[m['path'] for m in self.scene_meshes(scene,snapshot['objects']['test_object']['path'],True)
                        if m['path']==snapshot['objects']['test_object']['path'] or m['path'].startswith(snapshot['objects']['test_object']['path']+'/')],
                    'phase':'reach','original_target_geometry_and_pose':True,
                    'robot_geometry_limits_margins_and_physics_changed':False,
                    'scope':'IK, PRM and dense trajectory optimization all avoid the live target on the route to H; no physical success inferred'})
                if not checker.validate(start[None,None]).all():raise Rejection('measured_start_collision_with_target')
                goal=self.ik(desired,start,planner,checker,inverse,seed=numerical_seed)
            result = planner.graph_planner.find_path(start[None].clone(),goal[None].clone(),interpolate_waypoints=True,
                interpolation_steps=201,interpolation_type=TrajInterpolationType.LINEAR,validate_interpolated_trajectory=True)
            if not result.success.all():
                raise Rejection('prm_route_rejected')
            route = result.interpolated_waypoints.reshape(-1,len(names)).clone()
            out = Path(request['output'])
            q,times = optimize_path(planner,checker,route,names,limits,request.get('minimum_duration',3.),out,request['label'])
            flush_planning_artifacts()
            self.visual_path(request,planner,q,out)
            return {'command':{'id':request['command_id'],'joint_names':names,'position':q.tolist(),
                'time_from_start':times.tolist(),'gripper_target':.1 if phase=='lift' else -3.2,
                'position_tolerance_rad':.005,'label':request['label'],'planner':'cuRobo IK + PRM + trajectory optimization'},
                'desired_world':desired.tolist(),'fk':fk,'phase':phase,'model_state':{n:measured[n] for n in updater.names}}
        except (Rejection,AssertionError) as exc:
            return {'rejected':str(exc),'phase':phase,'fk':fk}
        except RuntimeError as exc:
            if 'Trajectory optimizer rejected all attempts' in str(exc):
                return {'rejected':str(exc),'phase':phase,'fk':fk}
            raise

    def nominal_closed(self, grasp, start, inverse, q_grasp_override=None):
        """Original nominal 22mm-finger / 8mm-table gate before any closure."""
        planner,checker,updater,models=self.context['closed']
        updater.apply(models,{n:.022 for n in updater.names})
        opened,open_checker,_,_=self.context['open']
        qg=self.ik(grasp,start,opened,open_checker,inverse) if q_grasp_override is None else q_grasp_override
        if not checker.validate(qg[None,None]).all():
            raise Rejection('nominal_closed_grasp_collision')
        lift=grasp.copy();lift[2,3]+=.12
        ql=self.ik(lift,qg,planner,checker,inverse)
        if not checker.validate(ql[None,None]).all():
            raise Rejection('nominal_closed_lift_collision')
        return {'pass':True,'nominal_gripper_m':.022,'table_margin_m':.008,
                'q_grasp':qg.tolist(),'q_lift':ql.tolist(),'geometry_changed':False,
                'scope':'Actual frozen open-grasp IK and nominal closed-model grasp/lift validity, before physical closure'}

    def qualify_pool(self,request):
        """Screen fixed G once, before spending H perturbation attempts."""
        import torch
        snapshot=request['snapshot']
        self.initialize(read(request['scene']),snapshot)
        names=self.context['names']
        start=torch.tensor([snapshot['joint_positions'][snapshot['joint_names'].index(n)] for n in names],device='cuda',dtype=torch.float32)
        inverse=np.linalg.inv(np.asarray(snapshot['link_world']['body4']))
        _,_,updater,models=self.context['open']
        measured=dict(zip(snapshot['joint_names'],snapshot['joint_positions']))
        updater.apply(models,{n:float(measured[n]) for n in updater.names})
        accepted=[];rejected=[]
        sources=request['pool']['candidates'];limit=self.registration.get('qualification_limit')
        examined=0
        for source in sources:
            if limit and len(accepted)>=limit:break
            examined+=1
            trials=[]
            for flip in (False,True):
                goal=equivalent_grasp(source,flip)
                try:
                    visual=self.visual_goal(request,goal)
                    if visual is not None and not visual['pass']:
                        raise Rejection('camera_visual_goal_rejected',visual)
                    certificate=self.nominal_closed(goal,start,inverse)
                    if visual is not None:certificate['visual_geometry']=visual
                    pregrasp=None
                    if self.registration.get('qualified_pregrasp'):
                        from .geometry import handoff
                        pregrasp=handoff(goal,source['pregrasp_distance_mm'],0,0)['world']
                        visual_h=self.visual_goal(request,np.asarray(pregrasp))
                        if visual_h is not None and not visual_h['pass']:
                            raise Rejection('camera_visual_pregrasp_rejected',visual_h)
                        context=request.get('visual_context')
                        if context is not None:
                            camera=np.asarray(pregrasp)@np.asarray(context['eef_from_camera'])
                            center=np.linalg.inv(camera)@np.asarray(snapshot['objects']['test_object']['world'])[:,3]
                            uv=(np.asarray(context['intrinsics'])@center[:3])[:2]/center[2] if center[2]>0 else [float('-inf')]*2
                            height,width=context['shape']
                            if center[2]<=0 or not (32<uv[0]<width-32 and 32<uv[1]<height-32):
                                raise Rejection('target_outside_pregrasp_wrist_frustum',{'optical_z_m':float(center[2])})
                        qh=self.ik(np.asarray(pregrasp),start,self.context['open'][0],self.context['open'][1],inverse,
                            seed=certificate['q_grasp'])
                        certificate['q_pregrasp']=qh.tolist()
                        certificate['pregrasp_scope']='Endpoint IK with original open model; full target-avoiding route and actual execution still required'
                    selected={**source,'raw_grasp_world':source['grasp_world'],'grasp_world':goal.tolist(),
                        'parallel_jaw_equivalent_flip':flip,'qualification':certificate}
                    if pregrasp is not None:selected['pregrasp_world']=pregrasp
                    if 'pad_center_eef_m' in source:
                        selected['contact_symmetry']='contact_center_5mm_10deg_parallel_jaw_v2'
                        if flip and 'object_local_contact_frame' in source:
                            local=np.asarray(source['object_local_contact_frame']).copy()
                            local[:3,:3]=local[:3,:3]@np.diag([-1.,-1.,1.])
                            selected['raw_object_local_contact_frame']=source['object_local_contact_frame']
                            selected['object_local_contact_frame']=local.tolist()
                    accepted.append(selected)
                    break
                except Rejection as exc:trials.append({'equivalent_flip':flip,'reason':str(exc),'details':exc.details})
            else:rejected.append({'goal_id':source['goal_id'],'raw_index':source['raw_index'],'trials':trials})
        return {**request['pool'],'candidates':accepted,'planning_pool_rejections':rejected,
            'examined_candidates':examined,'deferred_candidate_ids':[g['goal_id'] for g in sources[examined:]],
            'qualification_scope':'Frozen ARX IK and nominal closed grasp/lift gates; no physical motion, no H attempt consumed; real execution still required'}
