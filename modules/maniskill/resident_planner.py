"""Reuse frozen ARX cuRobo models while updating measured state and scene data."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import yaml

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / 'arx_pipeline'))
sys.path.insert(0, str(CODE))
import planner as frozen
from collision_profile import require_current_profile
from graspcl_planner import GraspCLPlanner
from curobo._src.geom.types import SceneCfg
from curobo._src.robot.types.kinematics_params import KinematicsParams
from curobo._src.types.robot import RobotCfg


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def all_kinematics_params(*roots):
    """Find solver/rollout tensor copies, keeping every CUDA graph address stable."""
    pending, seen, result = list(roots), set(), []
    while pending:
        obj = pending.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, KinematicsParams):
            result.append(obj)
        elif isinstance(obj, dict):
            pending.extend(obj.values())
        elif isinstance(obj, (list, tuple)):
            pending.extend(obj)
        elif type(obj).__module__.startswith('curobo.') and hasattr(obj, '__dict__'):
            pending.extend(vars(obj).values())
        if len(seen) > 100000:
            raise RuntimeError('Unexpected planner object graph')
    if not result:
        raise RuntimeError('No kinematics tensor copies found')
    return result


@contextmanager
def fixed_cache_capacity():
    """Cache capacity only; original meshes, primitive sizes and margins retained."""
    old_motion = frozen.MotionPlannerCfg
    old_checker = frozen.RobotCollisionCheckerCfg

    def motion(*args, **kwargs):
        kwargs['collision_cache'] = {'mesh': 8, 'primitive': 8}
        return old_motion.create(*args, **kwargs)

    def checker(*args, **kwargs):
        kwargs['n_meshes'] = 8
        kwargs['n_cuboids'] = 8
        return old_checker.load_from_config(*args, **kwargs)

    frozen.MotionPlannerCfg = SimpleNamespace(create=motion)
    frozen.RobotCollisionCheckerCfg = SimpleNamespace(load_from_config=checker)
    try:
        yield
    finally:
        frozen.MotionPlannerCfg = old_motion
        frozen.RobotCollisionCheckerCfg = old_checker


def bind_inputs(context, out, state, closed, include_target):
    """Reproduce the original adapter's model and scene receipts for this state."""
    out.mkdir(parents=True, exist_ok=True)
    context.out = out
    context.base = np.asarray(state['links']['body4'])
    context.invbase = np.linalg.inv(context.base)
    reference = frozen.ARX / 'scripts/curobo_arx/static_pick/reference' / ('model_closed' if closed else 'model')
    cfg = yaml.safe_load((reference / 'arx.yml').read_text())
    save(out / 'collision_profile.json', require_current_profile(cfg))
    cfg['robot_cfg']['kinematics']['urdf_path'] = str(reference / 'arx.urdf')
    measured = dict(zip(state['joint_names'], state['qpos']))
    for name in frozen.FINGERS:
        cfg['robot_cfg']['kinematics']['lock_joints'][name] = measured[name]
    context.model = out / 'arx.yml'
    context.model.write_text(yaml.safe_dump(cfg, sort_keys=False))
    scene = {'cuboid': {}}
    for key, geometry in state['scene_cuboids'].items():
        transform = np.eye(4)
        transform[:3, 3] = geometry['center']
        if closed and key == 'table':
            transform[2, 3] += .008
        scene['cuboid'][key] = {'pose': frozen.pose7(context.invbase @ transform), 'dims': geometry['dims']}
    meshes = {n: g for n, g in state.get('scene_meshes', {}).items() if include_target or not g['target']}
    if len(meshes) > 8 or len(scene['cuboid']) > 8:
        raise RuntimeError('Registered resident collision-cache capacity exceeded')
    for geometry in meshes.values():
        if 'file_sha256' in geometry:
            assert hashlib.sha256(Path(geometry['file_path']).read_bytes()).hexdigest() == geometry['file_sha256']
    if meshes:
        scene['mesh'] = {n: {'pose': frozen.pose7(context.invbase @ np.asarray(g['world'])),
                             'file_path': g['file_path']} for n, g in meshes.items()}
    save(out / 'scene.json', scene)
    nominal_scene = None
    if not closed:
        nominal = yaml.safe_load((frozen.ARX / 'scripts/curobo_arx/static_pick/reference/model_closed/arx.yml').read_text())
        require_current_profile(nominal)
        nominal['robot_cfg']['kinematics']['urdf_path'] = str(frozen.ARX / 'scripts/curobo_arx/static_pick/reference/model_closed/arx.urdf')
        for name in frozen.FINGERS:
            nominal['robot_cfg']['kinematics']['lock_joints'][name] = .022
        (out / 'nominal_closed.yml').write_text(yaml.safe_dump(nominal, sort_keys=False))
        nominal_scene = deepcopy(scene)
        nominal_scene['cuboid']['table']['pose'][:3] = (np.array(nominal_scene['cuboid']['table']['pose'][:3]) + context.invbase[:3, 2] * .008).tolist()
    return cfg, scene, nominal_scene


def collision_scene_config(scene):
    """Keep geometry immutable when names repeat between benchmark scenes."""
    content = deepcopy(scene)
    if 'mesh' in content:
        content['mesh'] = {name + '_' + hashlib.sha256(Path(g['file_path']).read_bytes()).hexdigest()[:20]: g
                           for name, g in content['mesh'].items()}
    return SceneCfg.create(content)


class PlannerPool:
    def __init__(self, root, warmup_trial):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.contexts = {}
        self.loads = {'open': 0, 'closed': 0}
        self.parameters = {}
        self.warmup = {}
        for closed, filename in [(False, 'state.json'), (True, 'closed_state.json')]:
            state = json.loads((warmup_trial / filename).read_text())
            role = 'closed' if closed else 'open'
            started = time.perf_counter()
            with fixed_cache_capacity():
                context = GraspCLPlanner(root / role, state, closed=closed)
            self.contexts[role] = context
            self.parameters[role] = all_kinematics_params(context.planner, context.checker)
            self.loads[role] += 1
            construction = time.perf_counter() - started
            started = time.perf_counter()
            current = context.measured(state)
            if context.ik(np.asarray(state['links']['right_eef_link']), current) is None:
                raise RuntimeError('Resident startup IK failed at recorded measured state')
            path = context.path(current, current)
            if path is None:
                raise RuntimeError('Resident startup PRM failed at recorded measured state')
            context.optimize('resident_startup_warmup', path, 6.)
            torch.cuda.synchronize()
            context.planner.reset_seed()
            self.warmup[role] = {'construction_seconds': construction,
                'warmup_seconds': time.perf_counter() - started,
                'kinematics_tensor_copies': len(self.parameters[role]), 'robot_trajectories_executed': 0}

    def bind(self, directory, state, closed=False, include_target=False):
        role = 'closed' if closed else 'open'
        context = self.contexts[role]
        started = time.perf_counter()
        cfg, scene, nominal_scene = bind_inputs(context, directory, state, closed, include_target)
        params = RobotCfg.create(cfg['robot_cfg'], context.planner.device_cfg).kinematics.kinematics_config
        static_fields = ['joint_map', 'joint_map_type', 'joint_offset_map', 'link_map', 'tool_frame_map',
                         'link_sphere_idx_map', 'link_chain_data', 'link_chain_offsets', 'joint_links_data',
                         'joint_links_offsets', 'joint_affects_endeffector']
        for target in self.parameters[role]:
            assert target.num_dof == params.num_dof and target.joint_names == params.joint_names
            for field in static_fields:
                old, new = getattr(target, field), getattr(params, field)
                if old is not None and new is not None:
                    assert torch.equal(old, new), 'Static kinematics changed: ' + field
            for field in ['position', 'velocity', 'acceleration', 'jerk']:
                old, new = getattr(target.joint_limits, field, None), getattr(params.joint_limits, field, None)
                if old is not None and new is not None:
                    assert torch.equal(old, new), 'Frozen joint limit changed: ' + field
            target.copy_(params)
        context.planner.clear_scene_cache()
        context.planner.update_world(collision_scene_config(scene))
        context.checker.clear_scene_cache()
        context.checker.update_world(collision_scene_config(scene))
        if nominal_scene is not None:
            context.closed_checker.clear_scene_cache()
            context.closed_checker.update_world(collision_scene_config(nominal_scene))
        context.planner.reset_seed()
        context.limits = torch.tensor([state['joint_limits'][state['joint_names'].index(n)] for n in frozen.NAMES], device='cuda')
        q = torch.tensor([state['qpos'][state['joint_names'].index(n)] for n in frozen.NAMES], dtype=torch.float32, device='cuda')
        kin = context.planner.compute_kinematics(context.js(q)).tool_poses
        xyz = kin.position.detach().cpu().numpy().reshape(-1, 3)[0]
        quat = kin.quaternion.detach().cpu().numpy().reshape(-1, 4)[0]
        expected = context.invbase @ np.asarray(state['links']['right_eef_link'])
        rotation = frozen.Rotation.from_quat([*quat[1:], quat[0]]).as_matrix()
        error = {'position_error_m': float(np.linalg.norm(xyz - expected[:3, 3])),
                 'rotation_error_rad': float(frozen.Rotation.from_matrix(rotation @ expected[:3, :3].T).magnitude())}
        save(directory / 'fk_validation.json', error)
        assert error['position_error_m'] < 2e-5 and error['rotation_error_rad'] < 2e-4, error
        current = context.measured(state)
        assert bool(context.checker.validate(current[None, None]).all()), 'Measured state collides in unchanged ARX model'
        save(directory / 'start_state.json', {'actual': q.tolist(), 'planning_start': current.tolist(),
            'max_clamp_rad': float((current-q).abs().max()), 'policy': 'Original 1e-4 rad inset'})
        torch.cuda.synchronize()
        save(directory / 'resident_rebind.json', {'role': role, 'model_load_count': self.loads[role],
            'measured_finger_joints': {n: state['qpos'][state['joint_names'].index(n)] for n in frozen.FINGERS},
            'kinematics_tensor_copies_updated': len(self.parameters[role]), 'world_replaced': True,
            'search_state_reset': True, 'mesh_cache_key_includes_geometry_sha256': True,
            'bind_seconds': time.perf_counter() - started})
        return context

    def request(self, directory, observe=False, approach=False, lift=False, selector='graspcl'):
        directory = Path(directory)
        name = 'observation_plan' if observe else 'lift_plan' if lift else 'approach_plan' if approach else 'plan'
        state_file = 'closed_state.json' if lift else 'pregrasp_state.json' if approach else 'state.json'
        state = json.loads((directory / state_file).read_text())
        started = time.perf_counter()
        context = self.bind(directory / name, state, closed=lift, include_target=observe)
        if observe:
            q = torch.tensor(json.loads((directory / 'observation_request.json').read_text())['q'], device='cuda', dtype=torch.float32)
            path = context.path(context.measured(state), q)
            result = {'success': True, 'paths': {'observe': context.optimize('observe', path, 6.)}} if path is not None else {'success': False, 'reason': 'observation_path_failed'}
        elif approach or lift:
            selected = json.loads((directory / 'plan_result.json').read_text())['selected']
            current = context.measured(state)
            goal = context.ik(np.asarray(selected['lift_world']), current) if lift else torch.tensor(selected['q_grasp'], device='cuda', dtype=torch.float32)
            path = context.path(current, goal) if goal is not None else None
            label = 'lift' if lift else 'approach'
            result = ({'success': True, 'paths': {label: context.optimize(label, path, 6. if lift else 5.)}} if path is not None else
                      {'success': False, 'reason': 'measured_closure_lift_unreachable' if lift else 'measured_pregrasp_approach_unreachable'})
        else:
            rows = np.load(directory / 'raw_grasps.npy', allow_pickle=False)
            rows = rows[np.argsort(-rows[:, 0], kind='stable')]
            result = (context.select(state, rows, np.asarray(state['camera_world_optical'])) if selector == 'graspcl'
                      else frozen.Planner.select(context, state, rows, np.asarray(state['camera_world_optical'])))
        frozen.flush_planning_artifacts()
        torch.cuda.synchronize()
        result['wall_seconds'] = time.perf_counter() - started
        save(directory / (name + '_result.json'), result)
        return result
