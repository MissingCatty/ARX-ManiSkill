"""Reuse a validated collision world when only measured finger locks change."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import yaml

from arx_project.layout import write_json
from modules.graspgenx.native_postprocessing.width_planning import FINGERS, COLLISION_MESSAGE
from .lock_state import copy_lock_dependent_fields


def state_signature(state, include_target):
    value = deepcopy(state)
    for name in FINGERS:
        value['qpos'][value['joint_names'].index(name)] = None
    # This field declares the sweep's destination, not collision geometry.
    value.pop('planned_opening', None)
    for geometry in value.get('scene_meshes', {}).values():
        actual = hashlib.sha256(Path(geometry['file_path']).read_bytes()).hexdigest()
        if 'file_sha256' in geometry:
            if actual != geometry['file_sha256']:
                raise RuntimeError('Measured scene mesh differs from its registered content')
        geometry['_current_content_sha256'] = actual
    text = json.dumps([bool(include_target), value], sort_keys=True, allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


class WidthOnlyBinder:
    def __init__(self, pool):
        self.pool = pool
        self.original = pool.bind
        self.clear()

    def clear(self):
        self.last = None
        self.hits = self.full_binds = 0

    def __call__(self, directory, state, closed=False, include_target=False):
        if not self.pool.width_refresh_enabled or closed:
            self.last = None
            self.full_binds += 1
            return self.original(directory, state, closed=closed, include_target=include_target)
        urdf = Path(self.pool.views['open']['cfg']['kinematics']['urdf_path'])
        source_key = tuple(hashlib.sha256(p.read_bytes()).hexdigest() for p in [urdf,urdf.with_name('arx.yml')])
        key = (state_signature(state, include_target),source_key)
        with self.pool.stage_timing.stage('world_and_robot_bind'):
            try:return self.refresh(Path(directory),state,key,include_target)
            except BaseException:
                self.last=None
                raise

    def refresh(self, directory, state, key, include_target):
        import torch
        import resident_planner as native
        frozen=native.frozen
        began = time.perf_counter()
        pool, cached = self.pool, self.last
        same_world=cached is not None and cached['key']==key
        context = pool.contexts['open']
        directory.mkdir(parents=True, exist_ok=True)
        measured = dict(zip(state['joint_names'], state['qpos']))
        if same_world:
            context.out=directory;context.base=np.asarray(state['links']['body4'])
            context.invbase=np.linalg.inv(context.base);cfg=deepcopy(cached['cfg'])
            for name in FINGERS:cfg['robot_cfg']['kinematics']['lock_joints'][name]=measured[name]
            context.model=directory/'arx.yml'
            context.model.write_text(yaml.dump(cfg,Dumper=yaml.CSafeDumper,sort_keys=False))
            for name,text in cached['files'].items():(directory/name).write_text(text)
        else:
            self.last=None
            with pool.binding_input_scope.activate():
                cfg,scene,nominal_scene=native.bind_inputs(context,directory,state,False,include_target)
        try:
            # This original method checks every non-lock robot configuration
            # value against its unchanged URDF-derived reference.
            params = pool._measured_robot_view(cfg['robot_cfg'], context.planner.device_cfg).kinematics.kinematics_config
            for target in pool.parameters['open']:
                copy_lock_dependent_fields(target,params)
            if not same_world:
                context.planner.clear_scene_cache()
                context.planner.update_world(native.collision_scene_config(scene))
                context.checker.clear_scene_cache()
                context.checker.update_world(native.collision_scene_config(scene))
                if nominal_scene is not None:
                    context.closed_checker.clear_scene_cache()
                    context.closed_checker.update_world(native.collision_scene_config(nominal_scene))
            context.planner.reset_seed()
            context.limits = torch.tensor([state['joint_limits'][state['joint_names'].index(n)]
                                          for n in frozen.NAMES], device='cuda')
            q = torch.tensor([measured[n] for n in frozen.NAMES], dtype=torch.float32, device='cuda')
            kin = context.planner.compute_kinematics(context.js(q)).tool_poses
            xyz = kin.position.detach().cpu().numpy().reshape(-1,3)[0]
            quat = kin.quaternion.detach().cpu().numpy().reshape(-1,4)[0]
            expected = context.invbase @ np.asarray(state['links']['right_eef_link'])
            rotation = frozen.Rotation.from_quat([*quat[1:],quat[0]]).as_matrix()
            error = {'position_error_m': float(np.linalg.norm(xyz-expected[:3,3])),
                     'rotation_error_rad': float(frozen.Rotation.from_matrix(rotation @ expected[:3,:3].T).magnitude())}
            write_json(directory/'fk_validation.json',error)
            if not (error['position_error_m'] < 2e-5 and error['rotation_error_rad'] < 2e-4):
                raise AssertionError(error)
            current = context.measured(state)
            if not pool.validate_start_state(context,state,current,False,include_target):
                raise AssertionError(COLLISION_MESSAGE)
            write_json(directory/'start_state.json',{'actual': q.tolist(),'planning_start': current.tolist(),
                'max_clamp_rad': float((current-q).abs().max()),'policy': 'Original 1e-4 rad inset'})
            torch.cuda.synchronize()
        except BaseException:
            self.last = None
            raise
        if same_world:self.hits+=1
        else:
            self.full_binds+=1
            self.last={'key':key,'cfg':deepcopy(cfg),'files':{name:(directory/name).read_text() for name in
                ['collision_profile.json','scene.json','nominal_closed.yml'] if (directory/name).exists()}}
        write_json(directory/'resident_rebind.json',{'role':'open','model_load_count':pool.loads['open'],
            'measured_finger_joints':{n:measured[n] for n in FINGERS},
            'kinematics_tensor_copies_updated':len(pool.parameters['open']),
            'world_replaced':not same_world,'world_reused_after_complete_state_signature':same_world,
            'only_URDF_proven_lock_dependent_fields_refreshed':True,
            'search_state_reset':True,'mesh_cache_key_includes_geometry_sha256':True,
            'runtime_robot_configuration_rebuilds':0,'bind_seconds':time.perf_counter()-began})
        return context


def attach(pool):
    if hasattr(pool,'width_only_binder'):
        raise RuntimeError('Width-only binder already attached')
    pool.width_refresh_enabled = False
    binder = WidthOnlyBinder(pool)
    pool.width_only_binder = binder
    pool.bind = binder
    return pool
