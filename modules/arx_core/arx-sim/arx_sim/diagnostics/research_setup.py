"""Atomic historical state reset and capture after a fixed number of physics steps.

Only positions, velocities and position targets are reset. No robot/world
configuration, material, gravity, collision or controller property is authored.
"""
import json
import os
from pathlib import Path
import numpy as np


def begin(bridge, request):
    if bridge.active is not None or bridge.context.controller.active is not None:
        raise RuntimeError('Atomic research setup requires idle controllers')
    if getattr(bridge, 'research_pending_capture', None):
        raise RuntimeError('Research setup already pending')
    root = bridge.root.parents[2]
    setup_path = Path(os.environ.get('ARX_HISTORICAL_SETUP') or root / 'runs/tabletop_trials/initial_setup.json')
    catalog_path = Path(os.environ.get('ARX_TABLETOP_CATALOG') or root / 'runs/arx_grasp/tabletop_catalog.json')
    setup = json.loads(setup_path.read_text())
    catalog = {a['name']: a for a in json.loads(catalog_path.read_text())}
    scenario = request['scenario']
    target = scenario.get('target', 'test_object')
    asset = catalog[scenario.get('asset', 'bbq_sauce_bottle')]
    position = np.asarray(scenario['placement']['position'], dtype=float)
    orientation = np.asarray(scenario['placement'].get('orientation', [1., 0., 0., 0.]), dtype=float)
    if position.shape != (3,) or orientation.shape != (4,) or not np.isfinite(position).all() or not np.isfinite(orientation).all():
        raise ValueError('Invalid declared historical pose')
    if target not in ('test_object', 'green_sphere', 'blue_cylinder', 'red_cube'):
        raise ValueError('Unknown historical target')
    c = bridge.context
    limits = np.stack([v.numpy()[0] for v in c.robot.get_dof_limits()], axis=-1)
    q = np.clip(np.asarray(setup['joint_positions']), limits[:, 0], limits[:, 1])
    c.robot.set_dof_positions(q)
    c.robot.set_dof_position_targets(q)
    c.robot.set_dof_velocities(np.zeros_like(q))
    initial = {}

    def pose(name, obj, xyz, quat):
        obj.set_world_poses(positions=np.asarray(xyz)[None], orientations=np.asarray(quat)[None])
        obj.set_velocities(linear_velocities=np.zeros((1, 3)), angular_velocities=np.zeros((1, 3)))
        initial[name] = {'path': obj.paths[0], 'position': list(map(float, xyz)),
                         'quaternion_wxyz': list(map(float, quat))}

    for name, xyz in [('blue_cylinder', [4.33, 1.79, .76]),
                      ('green_sphere', [4.46, 1.96, .76]), ('red_cube', [4.22, 1.75, .73])]:
        pose(name, bridge.objects[name], xyz, [1., 0., 0., 0.])
    for i, (name, obj) in enumerate(bridge.tabletop_objects.items()):
        pose('asset:' + name, obj, [10 + i * .4, 10, .4], [1., 0., 0., 0.])
    obj = bridge.tabletop_objects[asset['name']]
    xyz, quat = (position, orientation) if target == 'test_object' else ([10, 10, .4], [1., 0., 0., 0.])
    pose('asset:' + asset['name'], obj, xyz, quat)
    bridge.objects['test_object'] = obj
    bridge.test_object_size = asset['size_m']
    if target != 'test_object':
        pose(target, bridge.objects[target], position, orientation)
    # after_step runs once per rendered world.step, which advances 24 physics
    # substeps in this frozen scene. Fifteen renders = 360 physics steps = 3s.
    if abs(c.world.get_physics_dt() - .0083333333) > 1e-10:
        raise RuntimeError('Unexpected physics timestep; do not change configuration')
    if abs(c.world.get_rendering_dt() - .2) > 1e-10:
        raise RuntimeError('Unexpected rendering timestep; do not change configuration')
    bridge.research_pending_capture = {'request': request, 'remaining': 15,
        'reset_sim_time': c.world.current_time,
        'initial_conditions': {'joint_names': list(c.robot.dof_names),
            'joint_positions': q.tolist(), 'objects': initial,
            'linear_and_angular_velocities': 'All reset objects zero; all robot DOF velocities zero'},
        'asset': asset}
    return None


def tick(bridge):
    pending = getattr(bridge, 'research_pending_capture', None)
    if not pending:
        return
    pending['remaining'] -= 1
    if pending['remaining'] > 0:
        return
    bridge.research_pending_capture = None
    from arx_sim.diagnostics import research_fingerprint
    from arx_sim.diagnostics.curobo_runner import write_json
    try:
        evidence = research_fingerprint.capture(bridge)
        result = {'snapshot': bridge.snapshot(), 'fingerprint': evidence,
                  'initial_conditions': pending['initial_conditions'],
                  'reset_sim_time': pending['reset_sim_time'], 'captured_sim_time': bridge.context.world.current_time,
                  'physics_steps': 360, 'render_steps': 15, 'asset': pending['asset']}
        if abs(result['captured_sim_time'] - result['reset_sim_time'] - 3.) > .001:
            raise RuntimeError('Atomic setup capture did not cover exactly 3 simulation seconds')
        response = {'id': pending['request']['id'], 'ok': True, 'result': result}
    except Exception as exc:
        response = {'id': pending['request']['id'], 'ok': False, 'error': str(exc)}
    write_json(bridge.root / 'response.json', response)
