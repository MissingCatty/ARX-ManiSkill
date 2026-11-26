"""Request-scoped binding reuse with an isolated resident pool for each arm."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from arx_project.layout import read_json, sha256


def binding_key(state, *, closed, include_target, model_epoch):
    """Include the complete measured input and recheck actual mesh bytes."""
    meshes = {}
    for name, geometry in state.get('scene_meshes', {}).items():
        digest = sha256(geometry['file_path'])
        if geometry.get('file_sha256', digest) != digest:
            raise RuntimeError('Measured scene mesh changed: '+name)
        meshes[name] = digest
    value = {'state': state, 'closed': closed, 'include_target': include_target,
             'model_epoch': model_epoch, 'mesh_content': meshes}
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
        separators=(',', ':')).encode()).hexdigest()


def verify_cached_start(pool, context, state, closed, include_target):
    """Repeat the original FK and collision gates even when binding is reused."""
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    names = ['right_joint'+str(i) for i in range(21, 27)]
    q = torch.tensor([state['qpos'][state['joint_names'].index(n)] for n in names],
        device=context.limits.device, dtype=torch.float32)
    joint_state = context.js(q)
    if list(joint_state.joint_names) != names:
        raise RuntimeError('Unexpected native arm joint order during reuse')
    kin = context.planner.compute_kinematics(joint_state).tool_poses
    xyz = kin.position.detach().cpu().numpy().reshape(-1, 3)[0]
    quat = kin.quaternion.detach().cpu().numpy().reshape(-1, 4)[0]
    expected = context.invbase @ np.asarray(state['links']['right_eef_link'])
    pe = float(np.linalg.norm(xyz-expected[:3, 3]))
    re = float(Rotation.from_matrix(Rotation.from_quat([*quat[1:], quat[0]]).as_matrix()
        @ expected[:3, :3].T).magnitude())
    if not np.isfinite([pe, re]).all() or pe >= 2e-5 or re >= 2e-4:
        raise RuntimeError('Cached binding FK changed: '+str((pe, re)))
    if not pool.validate_start_state(context, state, context.measured(state), closed, include_target):
        raise RuntimeError('An identical cached binding changed its start collision result')
    torch.cuda.synchronize()
    return {'position_error_m': pe, 'rotation_error_rad': re, 'start_collision_clear': True}


def reusable_pool(base):
    class ReusablePool(base):
        def __init__(self, root, warmup_trial):
            self.reuse_enabled = False
            self.reuse_scope = None
            self.binding_cache = {}
            self.binding_records = []
            self.model_epoch = None
            super().__init__(root, warmup_trial)

        def bind(self, directory, state, closed=False, include_target=False):
            began = time.perf_counter()
            directory = Path(directory)
            role = 'closed' if closed else 'open'
            enabled = self.reuse_enabled and self.reuse_scope is not None
            previous = self.binding_cache.get(role) if enabled else None
            # A failed bind must never leave an earlier entry eligible for reuse.
            self.binding_cache.pop(role, None)
            try:
                key = binding_key(state, closed=closed, include_target=include_target,
                    model_epoch=self.model_epoch) if enabled else None
                hit = bool(previous and previous['key'] == key and previous['directory'] == directory
                    and previous['context'] is self.contexts[role])
                if hit:
                    for path, digest in previous['artifacts'].items():
                        if not path.is_file() or sha256(path) != digest:
                            raise RuntimeError('A cached binding receipt changed or disappeared: '+str(path))
                    context = previous['context']
                    if context.out != directory:
                        raise RuntimeError('Cached context output directory changed')
                    context.planner.reset_seed()
                    validation = verify_cached_start(self, context, state, closed, include_target)
                    entry = previous
                else:
                    context = super().bind(directory, state, closed=closed, include_target=include_target)
                    validation = read_json(directory/'fk_validation.json')
                    entry = {'key': key, 'directory': directory, 'context': context,
                        'artifacts': {directory/name: sha256(directory/name) for name in
                            ['arx.yml', 'scene.json', 'fk_validation.json', 'start_state.json',
                             'collision_profile.json', 'resident_rebind.json', 'nominal_closed.yml']
                            if (directory/name).is_file()}}
                if enabled:
                    self.binding_cache[role] = entry
                self.binding_records.append({'role': role, 'arm': state.get('planning_arm', 'right'),
                    'directory': str(directory), 'cache_hit': hit, 'binding_key': key,
                    'full_world_rebinds': 0 if hit else 1,
                    'seconds': time.perf_counter()-began, 'validation': validation,
                    'scope': self.reuse_scope})
                return context
            except BaseException:
                self.binding_cache.pop(role, None)
                raise

        def contact_view(self, context, state):
            validator = super().contact_view(context, state)
            # Each search window gets its own receipts, including on cache hits.
            validator.update(context, state)
            return validator

    return ReusablePool


def routed_pool(base):
    """Control shares one worker; C routes each arm to its resident worker."""
    child_type = reusable_pool(base)

    class RoutedPool:
        def __init__(self, root, warmup_trial):
            self.root = Path(root)
            self.workers = {}
            self.active_arm = 'right'
            self.reuse_enabled = True
            self.in_request = False
            self.memory_loads = []
            for side in ['right', 'left']:
                self.workers[side] = child_type(self.root/side, warmup_trial)
                import torch
                torch.cuda.synchronize()
                self.memory_loads.append({'arm_worker': side,
                    'allocated_mib': torch.cuda.memory_allocated()/1024**2,
                    'reserved_mib': torch.cuda.memory_reserved()/1024**2})
            # This proven canonical six-axis IK has no world or finger locks.
            # Share its RNG and call stream exactly as in the control group.
            self.workers['left'].pure_solver = self.workers['right'].pure_solver

        def __getattr__(self, name):
            return getattr(self.workers[self.active_arm], name)

        @property
        def contexts(self):
            return {side+'_'+role: context for side, worker in self.workers.items()
                    for role, context in worker.contexts.items()}

        @property
        def contact_views(self):
            return {side+'_'+role: view for side, worker in self.workers.items()
                    for role, view in worker.contact_views.items()}

        @property
        def loads(self):
            return {side: worker.loads for side, worker in self.workers.items()}

        def worker(self, state):
            side = state.get('planning_arm', 'right')
            if side not in self.workers:
                raise ValueError('Unknown physical arm: '+str(side))
            self.active_arm = side if self.reuse_enabled else 'right'
            return self.workers[self.active_arm]

        def bind(self, directory, state, closed=False, include_target=False):
            return self.worker(state).bind(directory, state, closed=closed, include_target=include_target)

        def contact_view(self, context, state):
            worker = self.worker(state)
            if not any(context is value for value in worker.contexts.values()):
                raise RuntimeError('Contact validator received the other arm context')
            return worker.contact_view(context, state)

        def request(self, directory, observe=False, approach=False, lift=False, selector='graspcl'):
            filename = 'closed_state.json' if lift else 'pregrasp_state.json' if approach else 'state.json'
            worker = self.worker(read_json(Path(directory)/filename))
            return worker.request(directory, observe=observe, approach=approach, lift=lift, selector=selector)

        @contextmanager
        def request_scope(self, scope, *, enabled, model_epoch):
            if self.in_request:
                raise RuntimeError('Resident arm requests cannot overlap')
            self.in_request = True
            self.reuse_enabled = enabled
            for worker in self.workers.values():
                worker.binding_cache.clear()
                worker.binding_records.clear()
                worker.reuse_enabled = enabled
                worker.model_epoch = model_epoch
                worker.reuse_scope = scope
            try:
                yield
            finally:
                for worker in self.workers.values():
                    worker.binding_cache.clear()
                    worker.reuse_scope = None
                self.in_request = False

        def reset_seeds(self):
            for context in self.contexts.values():
                context.planner.reset_seed()
            self.workers['right'].pure_solver.reset_seed()

        def binding_report(self):
            records = [deepcopy(row) for worker in self.workers.values() for row in worker.binding_records]
            return {'events': records, 'calls': len(records),
                'cache_hits': sum(row['cache_hit'] for row in records),
                'full_world_rebinds': sum(row['full_world_rebinds'] for row in records),
                'seconds': sum(row['seconds'] for row in records),
                'per_arm_resident_contexts': self.reuse_enabled,
                'request_scoped_only': True, 'start_FK_and_collision_rechecked_on_hits': True}

    return RoutedPool
