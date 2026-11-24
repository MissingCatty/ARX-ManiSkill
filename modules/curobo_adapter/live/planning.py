"""Resident v0.9 planning with the unchanged ARX models and local action gates."""
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace
import time

from paths import BASE, MS, ARX, read, save, sha
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml

import adapter
from adapter import ARXPlanner, NAMES
from planner_cli import build_frozen_models, pose7
from curobo._src.geom.types import SceneCfg
from curobo._src.robot.types.kinematics_params import KinematicsParams
from gripper_update import WarmedGripperUpdate
from lilo_vla.observations import smooth_segment
from lilo_vla.tracking import residual

FINGERS = ['right_joint27', 'right_joint28']


@contextmanager
def capacity():
    old_motion, old_checker = adapter.MotionPlannerCfg, adapter.RobotCollisionCheckerCfg
    def motion(*args, **kwargs):
        kwargs['collision_cache'] = {'mesh':8, 'primitive':8}
        return old_motion.create(*args, **kwargs)
    def checker(*args, **kwargs):
        kwargs.update(n_meshes=8, n_cuboids=8)
        return old_checker.load_from_config(*args, **kwargs)
    adapter.MotionPlannerCfg = SimpleNamespace(create=motion)
    adapter.RobotCollisionCheckerCfg = SimpleNamespace(load_from_config=checker)
    try:
        yield
    finally:
        adapter.MotionPlannerCfg, adapter.RobotCollisionCheckerCfg = old_motion, old_checker


def parameter_views(*roots):
    pending, seen, found = list(roots), set(), []
    while pending:
        obj = pending.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, KinematicsParams):
            found.append(SimpleNamespace(config=SimpleNamespace(kinematics_config=obj)))
        elif isinstance(obj, dict):
            pending.extend(obj.values())
        elif isinstance(obj, (tuple, list)):
            pending.extend(obj)
        elif type(obj).__module__.startswith('curobo.') and hasattr(obj, '__dict__'):
            pending.extend(vars(obj).values())
        if len(seen)>100000:
            raise RuntimeError('Unexpected cuRobo object graph')
    if not found:
        raise RuntimeError('No resident kinematics parameters found')
    return found


def scene_for(state, closed, include_target):
    base = np.linalg.inv(np.asarray(state['links']['body4']))
    scene = {'cuboid':{}}
    for name, geometry in state['scene_cuboids'].items():
        world = np.eye(4)
        world[:3,3] = geometry['center']
        if closed and name == 'table':
            world[2,3] += .008
        scene['cuboid'][name] = {'pose':pose7(base@world), 'dims':geometry['dims']}
    meshes = {n:g for n,g in state.get('scene_meshes',{}).items() if include_target or not g['target']}
    if len(meshes)>8 or len(scene['cuboid'])>8:
        raise RuntimeError('Registered scene cache capacity exceeded')
    for geometry in meshes.values():
        assert sha(geometry['file_path']) == geometry['file_sha256']
    if meshes:
        scene['mesh'] = {n:{'pose':pose7(base@np.asarray(g['world'])), 'file_path':g['file_path']} for n,g in meshes.items()}
    return scene


def content_scene(scene):
    scene = deepcopy(scene)
    if 'mesh' in scene:
        scene['mesh'] = {n+'_'+sha(g['file_path'])[:20]:g for n,g in scene['mesh'].items()}
    return SceneCfg.create(scene)


class Context:
    def __init__(self, backend):
        self.backend = backend
    def __getattr__(self, name):
        return getattr(self.backend, name)
    def ik(self, world, start):
        q, result = self.backend.ik(np.asarray(world), start)
        return q


class Pool:
    def __init__(self, root, reg):
        self.root, self.reg = root, reg
        self.contexts, self.views, self.updaters, self.loads = {}, {}, {}, {}
        self.initial_limits = None
        self.bind_id = 0
        self.ik_queries = 0
        warm = __import__('pathlib').Path(reg['warmup_trial'])
        for closed, filename in [(False,'state.json'), (True,'closed_state.json')]:
            state = read(warm/filename)
            role = 'closed' if closed else 'open'
            source = root/'inputs'/role
            build_frozen_models(state, source)
            if closed:
                ref = ARX/'scripts/curobo_arx/static_pick/reference/model_closed'
                cfg = yaml.safe_load((ref/'arx.yml').read_text())
                cfg['robot_cfg']['kinematics']['urdf_path'] = str(ref/'arx.urdf')
                values = dict(zip(state['joint_names'],state['qpos']))
                cfg['robot_cfg']['kinematics']['lock_joints'].update({n:values[n] for n in FINGERS})
                (source/'plan/arx.yml').write_text(yaml.safe_dump(cfg,sort_keys=False))
                save(source/'plan/scene.json',scene_for(state,True,False))
            with capacity():
                context = Context(ARXPlanner(source,root/'models'/role))
            self.contexts[closed] = context
            self.views[closed] = parameter_views(context.planner,context.checker)
            self.updaters[closed] = WarmedGripperUpdate(context.model,context.planner.kinematics)
            self.loads[role] = 1
            before = time.monotonic()
            start = context.measured(state)
            q = context.ik(np.asarray(state['links']['right_eef_link']),start)
            if q is None:
                raise RuntimeError('Resident startup zero-motion IK failed: '+role)
            path = context.path(start,start)
            if path is None:
                raise RuntimeError('Resident startup zero-motion PRM failed: '+role)
            context.optimize('resident_startup',path,6.)
            context.planner.reset_seed()
            save(root/(role+'_warmup.json'),{'passed':True,'model_load_count':1,
                'warmup_seconds':time.monotonic()-before,'kinematics_parameter_views':len(self.views[closed]),
                'runtime_robot_rebuilds':0,'physical_commands':0,'version':'0.9.0'})
        self.velocity = np.array([read(MS/'assets/arx/manifest.json')['joints'][n]['limits']['velocity'] for n in NAMES])

    def bind(self, state, closed=False, include_target=False):
        p = self.contexts[closed]
        initial = p.backend.state
        if state['joint_names'] != initial['joint_names'] or state['joint_limits'] != initial['joint_limits']:
            raise RuntimeError('Frozen measured topology/limits changed')
        if state['scene_cuboids'] != initial['scene_cuboids']:
            raise RuntimeError('Frozen cuboid geometry changed')
        self.bind_id += 1
        directory = self.root/'bindings'/f'{self.bind_id:04d}'
        directory.mkdir(parents=True,exist_ok=False)
        p.backend.out = directory
        p.backend.base = np.asarray(state['links']['body4'])
        p.backend.invbase = np.linalg.inv(p.backend.base)
        values = dict(zip(state['joint_names'],state['qpos']))
        self.updaters[closed].apply(self.views[closed],values)
        scene = scene_for(state,closed,include_target)
        p.planner.clear_scene_cache()
        p.planner.update_world(content_scene(scene))
        p.checker.clear_scene_cache()
        p.checker.update_world(content_scene(scene))
        # The open context's auxiliary checker remains nominally closed, as in
        # the original representation. It is never used as an H-entry gate.
        nominal = deepcopy(scene)
        nominal['cuboid']['table']['pose'][:3] = (np.asarray(nominal['cuboid']['table']['pose'][:3])+p.invbase[:3,2]*.008).tolist()
        p.closed_checker.clear_scene_cache()
        p.closed_checker.update_world(content_scene(nominal))
        p.planner.reset_seed()
        actual = torch.tensor([state['qpos'][state['joint_names'].index(n)] for n in NAMES],device='cuda',dtype=torch.float32)
        error = residual(p.checker.kinematics,actual.cpu().numpy(),p.invbase@np.asarray(state['links']['right_eef_link']),NAMES)
        assert error['position_error_m']<2e-5 and error['orientation_error_rad']<2e-4,error
        save(directory/'binding.json',{'closed':closed,'include_target':include_target,'model_load_count':1,
            'runtime_robot_rebuilds':0,'measured_fingers':{n:values[n] for n in FINGERS},
            'scene':scene,'FK':error,'source_state':state})
        return p

    def request(self, message):
        kind = message['kind']
        if kind == 'observe':
            directory = __import__('pathlib').Path(message['directory'])
            state = read(directory/'state.json')
            p = self.bind(state,include_target=True)
            start = p.measured(state)
            if not p.checker.validate(start[None,None]).all():
                return {'success':False,'reason':'observation_initial_state_collision'}
            q = torch.tensor(read(directory/'observation_request.json')['q'],device='cuda',dtype=torch.float32)
            path = p.path(start,q)
            result = {'success':False,'reason':'observation_path_failed'} if path is None else {'success':True,'paths':{'observe':p.optimize('observe',path,6.)}}
            save(directory/'observation_plan_result.json',result)
            return result
        if kind == 'entry':
            directory = __import__('pathlib').Path(message['directory'])
            state = read(directory/'state.json')
            fixed = self.reg['fixed_entries'][message['case']]
            goal = np.asarray(fixed['pregrasp_world'])
            p = self.bind(state)
            start = p.measured(state)
            if not p.checker.validate(start[None,None]).all():
                return {'success':False,'reason':'entry_initial_state_collision'}
            q = self.ik(p,goal,start)
            if q is None:
                return {'success':False,'reason':'registered_H_ik_rejected'}
            path = p.path(start,q)
            if path is None:
                return {'success':False,'reason':'registered_H_path_rejected'}
            result = {'success':True,'planning_scope':'pregrasp_entry_only','full_grasp_validated':False,
                'selected':{'index':fixed['index'],'pregrasp_world':goal.tolist(),'q_pre':q.tolist()},
                'paths':{'pregrasp':p.optimize('pregrasp',path,6.)},'G_used_by_local_policy':False,
                'fixed_archived_entry':True,'runtime_robot_rebuilds':0}
            save(directory/'plan_result.json',result)
            return result
        state = message['state']
        if kind == 'initialize':
            for closed in (False,True):
                p = self.bind(state,closed=closed)
                if not p.checker.validate(p.measured(state)[None,None]).all():
                    return {'ready':False,'reason':'local_initial_state_collision','closed':closed}
            return {'ready':True,'warm_seconds':0.,'resident':True,'model_load_count':self.loads,'runtime_robot_rebuilds':0}
        if kind == 'motion':
            return self.motion(state,message['world_goal'],message['closed'])
        if kind == 'gripper_sweep':
            return self.sweep(state,message['opening'])
        raise RuntimeError('Unknown resident planning command')

    def motion(self,state,goal,closed):
        p = self.bind(state,closed)
        start = p.measured(state)
        goal = np.asarray(goal,dtype=float)
        if not p.checker.validate(start[None,None]).all():
            return {'accepted':False,'reason':'vlm_current_state_collision'}
        qgoal = self.ik(p,goal,start)
        if qgoal is None:
            return {'accepted':False,'reason':'vlm_motion_unreachable'}
        error = residual(p.checker.kinematics,qgoal.cpu().numpy(),p.invbase@goal,NAMES)
        if error['position_error_m']>.002 or error['orientation_error_rad']>.03:
            return {'accepted':False,'reason':'vlm_ik_precision','pose_error':error}
        q,times = smooth_segment(start.cpu().numpy(),qgoal.cpu().numpy())
        scale = max(1.,float(np.max(1.875*np.abs(q[-1]-q[0])/times[-1]/self.velocity)))
        times = times*scale
        bounds = p.limits.cpu().numpy()
        if ((q<bounds[:,0]-1e-6)|(q>bounds[:,1]+1e-6)).any():
            return {'accepted':False,'reason':'vlm_joint_limits'}
        alpha = np.arange(10)/10
        dense = np.concatenate([(q[:-1,None]*(1-alpha[None,:,None])+q[1:,None]*alpha[None,:,None]).reshape(-1,6),q[-1:]])
        tensor = torch.tensor(dense,device='cuda',dtype=torch.float32)
        valid = torch.cat([p.checker.validate(chunk[:,None]).flatten() for chunk in tensor.split(64)])
        if not valid.all():
            return {'accepted':False,'reason':'vlm_dense_scene_collision','invalid_samples':int((~valid).sum())}
        return {'accepted':True,'q':q.tolist(),'times':times.tolist(),'dense_samples':len(dense),'pose_error':error,
                'closed_original_model':closed,'measured_fingers':{n:state['qpos'][state['joint_names'].index(n)] for n in FINGERS},
                'command_time_scale_for_original_velocity_limits':scale}

    def sweep(self,state,opening):
        closed = opening<=.05
        p = self.bind(state,closed)
        values = dict(zip(state['joint_names'],state['qpos']))
        arm = p.measured(state)
        try:
            for ratio in np.linspace(0,1,17):
                measured = {**values,**{n:(1-ratio)*values[n]+ratio*.044*opening for n in FINGERS}}
                self.updaters[closed].apply(self.views[closed],measured)
                if not p.checker.validate(arm[None,None]).all():
                    return {'accepted':False,'reason':'vlm_gripper_sweep_scene_collision','rejected_ratio':float(ratio)}
        finally:
            self.updaters[closed].apply(self.views[closed],values)
        return {'accepted':True,'sweep_samples':17,'closed_original_model':closed}

    def close(self):
        adapter.flush_planning_artifacts()
        for p in self.contexts.values():
            p.planner.destroy()

    def ik(self,p,goal,start):
        if self.ik_queries>=self.reg['maximum_native_ik_queries']:
            raise RuntimeError('Registered native IK query budget exhausted')
        self.ik_queries += 1
        return p.ik(goal,start)
