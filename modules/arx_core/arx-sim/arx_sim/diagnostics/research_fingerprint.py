"""Read-only research evidence; never author USD or call a physics setter."""
import hashlib
import json
import time
import math

import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def plain(value):
    if isinstance(value, float) and not math.isfinite(value):
        return '+Inf' if value > 0 else '-Inf' if value < 0 else 'NaN'
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if hasattr(value, 'numpy'):
        return plain(value.numpy())
    if hasattr(value, 'tolist'):
        return plain(value.tolist())
    try:
        return [plain(v) for v in value]
    except TypeError:
        return str(value)


def capture(bridge, *, object_reader=None, progress_interval_s=0.):
    from pxr import UsdPhysics, UsdGeom
    c = bridge.context
    if bridge.active is not None or c.controller.active is not None:
        raise RuntimeError('Fingerprint requires idle controllers')
    config = {}
    began = time.monotonic()
    last_progress = [None, float('-inf')]
    def progress(phase, **details):
        now = time.monotonic()
        if (phase != 'complete' and phase == last_progress[0] and
                now-last_progress[1] < progress_interval_s):
            return
        (bridge.root / 'fingerprint_progress.json').write_text(json.dumps(
            {'phase': phase, 'elapsed_s': now - began, **details}))
        last_progress[:] = [phase, now]
    progress('usd')
    dynamic = ('physics:velocity', 'physics:angularVelocity')
    geometry = {'points', 'faceVertexIndices', 'faceVertexCounts', 'extent',
                'size', 'radius', 'height', 'axis', 'subdivisionScheme'}
    for number, prim in enumerate(c.stage.Traverse()):
        if number % 500 == 0:
            progress('usd', prim_count=number, path=str(prim.GetPath()))
        # The user's display camera may move; calibrated sensor transforms remain frozen.
        if str(prim.GetPath()).startswith('/World/SceneCameras'):
            continue
        values = {}
        rigid = prim.HasAPI(UsdPhysics.RigidBodyAPI)
        for attr in prim.GetAttributes():
            name = attr.GetName()
            if name in dynamic or name.startswith('state:'):
                continue
            physical = name.startswith(('physics:', 'physx', 'drive:'))
            transform = name.startswith('xformOp') and not rigid
            shape = name in geometry and prim.IsA(UsdGeom.Gprim) and prim.HasAPI(UsdPhysics.CollisionAPI)
            if physical or transform or shape:
                raw = attr.Get()
                # Keep large collision meshes auditable without enormous replies.
                if raw is not None and not isinstance(raw, (str, dict)) and hasattr(raw, '__len__') and len(raw) > 64:
                    array = np.asarray(raw)
                    if array.dtype.kind in 'biuf':
                        values[name] = {'sha256': hashlib.sha256(array.tobytes()).hexdigest(),
                                        'shape': list(array.shape), 'dtype': str(array.dtype)}
                        continue
                values[name] = plain(raw)
        relations = {r.GetName(): [str(p) for p in r.GetTargets()]
                     for r in prim.GetRelationships()
                     if r.GetName().startswith(('physics:', 'material:binding', 'filtered'))}
        if values or relations or rigid:
            config[str(prim.GetPath())] = {'type': prim.GetTypeName(),
                'schemas': list(prim.GetAppliedSchemas()), 'attributes': values,
                'relationships': relations}
    runtime = {}; errors = {}
    for name in ('get_dof_gains', 'get_dof_max_efforts', 'get_dof_limits',
                 'get_dof_max_velocities', 'get_link_masses', 'get_link_inertias',
                 'get_dof_friction_properties', 'get_dof_armatures',
                 'get_dof_drive_types', 'get_dof_drive_model_properties',
                 'get_solver_iteration_counts', 'get_enabled_self_collisions'):
        try:
            progress('runtime_robot', method=name)
            runtime[name] = plain(getattr(c.robot, name)())
        except Exception as exc:
            errors[name] = str(exc)
    for name in ('get_dof_gains', 'get_dof_max_efforts', 'get_dof_limits'):
        if name not in runtime:
            raise RuntimeError('Required runtime evidence unavailable: ' + name)
    objects = dict(bridge.objects)
    objects.update({'asset:' + n: p for n, p in bridge.tabletop_objects.items()})
    if object_reader is None:
        poses = {name: {'path': p.paths[0], 'world': c.controller._world_pose(p).tolist()}
                 for name, p in objects.items()}
        asset_runtime = {}
        for name, obj in objects.items():
            progress('runtime_objects', name=name)
            values = {}
            for method in ('get_masses', 'get_inertias'):
                values[method] = plain(getattr(obj, method)())
            asset_runtime[obj.paths[0]] = values
    else:
        progress('runtime_objects', mode='batched_readonly')
        poses, asset_runtime = object_reader(objects)
    # USD is composed state; all-DOF getters additionally cover live drive overrides.
    invariant = {'usd': config, 'runtime_robot': runtime, 'runtime_objects': asset_runtime,
                 'configured': plain(c.config),
                 'physics_dt': c.world.get_physics_dt(),
                 'rendering_dt': c.world.get_rendering_dt(),
                 'runtime_api_errors': errors}
    progress('complete', prim_count=len(config))
    return {'invariant_sha256': digest(invariant), 'invariant': invariant,
            'scene_state': {'joint_names': list(c.robot.dof_names),
                'joint_positions': c.controller._positions().tolist(), 'objects': poses,
                'robot_base_world': bridge.snapshot()['link_world']['body4']},
            'used_layers': [layer.realPath for layer in c.stage.GetUsedLayers()
                            if layer.realPath],
            'scope': 'Composed USD physics/geometry/static transforms and available live robot getters; unsupported APIs explicitly recorded. No setters.',
            'sim_time': c.world.current_time}
