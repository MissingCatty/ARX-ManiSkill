"""Bind measured state to existing solver storage without configuration files."""
from copy import deepcopy
from pathlib import Path
import time

import numpy as np

from arx_project.layout import sha256, write_json
from .artifacts import ReceiptWriter, ModelTemplate
from .scenes import UnionSceneUpdates
from .snapshots import GeometryFiles, InputBuilder, collision_inputs


STATIC_FIELDS = ['joint_map', 'joint_map_type', 'joint_offset_map', 'link_map', 'tool_frame_map',
                 'link_sphere_idx_map', 'link_chain_data', 'link_chain_offsets', 'joint_links_data',
                 'joint_links_offsets', 'joint_affects_endeffector']


class MeasuredStartCollision(AssertionError):
    def __init__(self):
        super().__init__('Measured state collides in unchanged ARX model')


class MemoryBindings:
    def __init__(self, pool):
        import resident_planner as native
        from curobo.types import Pose
        self.pool, self.native = pool, native
        if any(c.closed_checker is not None for c in pool.contexts.values()):
            raise RuntimeError('State updates require the active actual-table/no-nominal collision policy')
        self.models = {role: deepcopy(view['cfg']) for role, view in pool.views.items()}
        self.model_files = {}
        for cfg in self.models.values():
            urdf = Path(cfg['kinematics']['urdf_path'])
            for path in [urdf, urdf.with_name('arx.yml')]:
                self.model_files[path] = sha256(path)
        profiles = {r: native.require_current_profile({'robot_cfg': c}) for r, c in self.models.items()}
        self.builder = InputBuilder(self.models, profiles, native.frozen.FINGERS, native.frozen.pose7)
        self.templates = {role: ModelTemplate(cfg, native.frozen.FINGERS) for role, cfg in self.models.items()}
        self.scenes = UnionSceneUpdates(native.SceneCfg.create, Pose.from_list)
        self.writer = None
        self.last_inputs = {}
        self.events = []
        self.attempts = 0
        self.admit_static_models()

    def admit_static_models(self):
        import torch
        for role, parameters in self.pool.parameters.items():
            reference = self.pool.views[role]['zero']
            for target in parameters:
                if target.num_dof != reference.num_dof or target.joint_names != reference.joint_names:
                    raise RuntimeError('Frozen native joint layout changed')
                for field in STATIC_FIELDS:
                    old, new = getattr(target, field), getattr(reference, field)
                    if (old is None) != (new is None) or old is not None and not torch.equal(old, new):
                        raise RuntimeError('Static kinematics changed: ' + field)
                for field in ['position', 'velocity', 'acceleration', 'jerk']:
                    old, new = getattr(target.joint_limits, field, None), getattr(reference.joint_limits, field, None)
                    if (old is None) != (new is None) or old is not None and not torch.equal(old, new):
                        raise RuntimeError('Frozen joint limits changed: ' + field)

    def begin(self):
        if self.writer is not None:
            raise RuntimeError('Memory update requests cannot overlap')
        self.files = GeometryFiles()
        for path, digest in self.model_files.items():
            self.files.digest(path, digest)
        self.admit_static_models()
        self.scenes.reset()
        self.last_inputs, self.events, self.attempts = {}, [], 0
        self.writer = ReceiptWriter()

    def finish(self):
        if self.writer is None:
            raise RuntimeError('No memory update request is active')
        writer, self.writer = self.writer, None
        report = writer.finish()
        self.files.verify()
        self.admit_static_models()
        return {'binding_calls': self.attempts, 'completed_bindings': len(self.events),
                'scenes': self.scenes.report(), 'receipts': report,
                'events': self.events, 'models_and_meshes_verified_after_request': True,
                'original_live_FK_and_start_collision_retained': True}

    def require_scope(self):
        if self.writer is None:
            raise RuntimeError('Memory updates require an active request scope')

    def update_robot(self, role, measured, targets):
        import torch
        from modules.curobo.gripper_sweep.lock_state import copy_lock_dependent_fields
        view = self.pool.views[role]
        scratch = view['scratch']
        values = [float(measured[n]) for n in view['names']]
        for field, slopes in view['slopes'].items():
            getattr(scratch, field).copy_(getattr(view['zero'], field) + values[0] * slopes[0] + values[1] * slopes[1])
        locked = scratch.lock_jointstate
        position = torch.tensor([measured[n] for n in locked.joint_names],
                                device=locked.position.device, dtype=locked.position.dtype)
        locked.position.copy_(position.reshape_as(locked.position))
        for target in targets:
            copy_lock_dependent_fields(target, scratch)
        return scratch

    def bind(self, directory, state, closed=False, include_target=False):
        import torch
        self.require_scope()
        self.attempts += 1
        began = time.perf_counter()
        pool, native = self.pool, self.native
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with pool.stage_timing.stage('binding_input_snapshot'):
            item = self.builder.build(state, closed=closed, include_target=include_target, files=self.files)
        role = item['role']
        context = pool.contexts[role]
        context.out, context.model = directory, directory / 'arx.yml'
        context.base, context.invbase = item['base'], item['invbase']
        self.last_inputs[id(context)] = item
        measured = dict(zip(state['joint_names'], state['qpos']))
        packet = {'arx.yml': self.templates[role].render(item['cfg']['robot_cfg']['kinematics']['lock_joints']),
                  'scene.json': item['scene'],
                  'collision_profile.json': item['profile']}
        changes = []
        try:
            with pool.stage_timing.stage('binding_dynamic_robot_update'):
                self.update_robot(role, measured, pool.parameters[role])
            with pool.stage_timing.stage('binding_scene_update'):
                scene = collision_inputs(item['scene'], item['mesh_digests'])
                changes.append(self.scenes.update(context.planner, scene))
                changes.append(self.scenes.update(context.checker, scene))
                context.planner.reset_seed()
            with pool.stage_timing.stage('binding_live_validation'):
                context.limits = torch.tensor([state['joint_limits'][state['joint_names'].index(n)]
                                               for n in native.frozen.NAMES], device='cuda')
                q = torch.tensor([measured[n] for n in native.frozen.NAMES], device='cuda', dtype=torch.float32)
                kin = context.planner.compute_kinematics(context.js(q)).tool_poses
                xyz = kin.position.detach().cpu().numpy().reshape(-1, 3)[0]
                quat = kin.quaternion.detach().cpu().numpy().reshape(-1, 4)[0]
                expected = context.invbase @ np.asarray(state['links']['right_eef_link'])
                rotation = native.frozen.Rotation.from_quat([*quat[1:], quat[0]]).as_matrix()
                error = {'position_error_m': float(np.linalg.norm(xyz - expected[:3, 3])),
                         'rotation_error_rad': float(native.frozen.Rotation.from_matrix(rotation @ expected[:3, :3].T).magnitude())}
                packet['fk_validation.json'] = error
                if not (error['position_error_m'] < 2e-5 and error['rotation_error_rad'] < 2e-4):
                    raise AssertionError(error)
                current = context.measured(state)
                if not pool.validate_start_state(context, state, current, closed, include_target):
                    raise MeasuredStartCollision()
                packet['start_state.json'] = {'actual': q.tolist(), 'planning_start': current.tolist(),
                    'max_clamp_rad': float((current - q).abs().max()), 'policy': 'Original 1e-4 rad inset'}
                torch.cuda.synchronize()
            receipt = {'role': role, 'model_load_count': pool.loads[role],
                'measured_finger_joints': {n: measured[n] for n in native.frozen.FINGERS},
                'kinematics_tensor_copies_updated': len(pool.parameters[role]),
                'world_replaced': any(c['operation'] == 'replace_geometry' for c in changes),
                'scene_operations': changes, 'search_state_reset': True,
                'mesh_cache_key_includes_geometry_sha256': True, 'runtime_robot_configuration_rebuilds': 0,
                'in_memory_inputs': True, 'static_models_verified_at_request_boundaries': True,
                'bind_seconds': time.perf_counter() - began}
            packet['resident_rebind.json'] = receipt
            self.events.append(receipt)
            return context
        finally:
            with pool.stage_timing.stage('binding_receipt_enqueue'):
                self.writer.submit(directory, packet)

    def contact_view(self, context, state):
        self.require_scope()
        role = 'closed' if context is self.pool.contexts['closed'] else 'open'
        view = self.pool.contact_views[role]
        item = self.last_inputs[id(context)]
        scene = deepcopy(item['scene'])
        if 'table' not in scene.get('cuboid', {}):
            raise ValueError('Measured table geometry is required')
        del scene['cuboid']['table']
        with self.pool.stage_timing.stage('contact_dynamic_robot_update'):
            from modules.curobo.gripper_sweep.lock_state import copy_lock_dependent_fields
            params = context.checker.kinematics.kinematics_config
            for target in view['parameters']:
                copy_lock_dependent_fields(target, params)
        with self.pool.stage_timing.stage('contact_scene_update'):
            self.scenes.update(view['checker'], collision_inputs(scene, item['mesh_digests']))
        with self.pool.stage_timing.stage('contact_state_update'):
            view['validator'].update(context, state)
        view['key'] = None
        return view['validator']

    def config(self, context):
        self.require_scope()
        return deepcopy(self.last_inputs[id(context)]['cfg']['robot_cfg'])
