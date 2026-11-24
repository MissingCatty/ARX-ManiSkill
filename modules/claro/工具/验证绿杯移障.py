"""在冻结的 I0069 场景中用实际仿真验证绿杯夹取与放置。"""

import argparse
from collections import deque
import hashlib
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import os
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from 挡物规划联调 import ROOT, MS, POLICY, WARMUP, read, sha256, validate_input

sys.path[:0] = [str(MS), str(MS / 'arx_pipeline')]
from resident_backend import rpc, save
from short_lift_protocol import HOLD_SECONDS, MIN_HOLD_LIFT_M, validate_selected

RESOURCES = ROOT / 'docs/03_夹走并放下挡物/resources'
ARCHIVE = RESOURCES / 'I0069_规划联调'
ARM = [f'right_joint{i}' for i in range(21, 27)]
FINGERS = ['right_joint27', 'right_joint28']
POSITION_MATCH = .002
ROTATION_MATCH = .03
CARRY_CLEARANCE = .003
TARGET_DRIFT = .005
RELEASE_HEIGHT = .006
PLACE_OFFSETS = (.16, -.16, .20, -.20, .24, -.24)
STABLE_SAMPLES = 120
STABLE_SHIFT = .001
STABLE_ANGLE = np.deg2rad(1.)
MAX_LOWER_SHIFT = .005
MAX_LOWER_ANGLE = .08
BLOCKER_FINGERS = frozenset(('right_link27', 'right_link28'))
RETREAT_HEIGHTS = (.07, .05, .03)


class StageStopped(Exception):
    """The evidence or physical checks did not admit the next action."""


def arm_q(state):
    return np.asarray([state['qpos'][state['joint_names'].index(n)] for n in ARM])


def pose_errors(first, second):
    a, b = np.asarray(first), np.asarray(second)
    return (float(np.linalg.norm(a[:3, 3] - b[:3, 3])),
            float(Rotation.from_matrix(a[:3, :3] @ b[:3, :3].T).magnitude()))


def dense_path(path):
    q = np.asarray(path['q'])
    times = np.asarray(path['times'])
    if (q.shape != (len(times), 6) or not np.isfinite(q).all() or not np.isfinite(times).all()
            or len(q) < 2 or times[0] != 0 or not np.all(np.diff(times) > 0)):
        raise ValueError('Invalid joint trajectory')
    u = np.arange(5)[:, None] / 5
    return np.vstack([(q[:-1, None] * (1 - u) + q[1:, None] * u).reshape(-1, 6), q[-1]])


def model_matches(env, state):
    """Verify loaded native colliders before reusing their archived exports."""
    geometry = state['scene_meshes']
    if set(geometry) != set(env.objects):
        raise ValueError('Rendered scene actor set differs')
    for name, actor in env.objects.items():
        item = geometry[name]
        vertices, faces, offset = [], [], 0
        shapes = actor._bodies[0].get_collision_shapes()
        for shape in shapes:
            v = np.asarray(shape.vertices, dtype=np.float64) * np.asarray(shape.scale)
            t = shape.local_pose.to_transformation_matrix()
            vertices.append(v @ t[:3, :3].T + t[:3, 3])
            faces.append(np.asarray(shape.triangles, dtype=np.int64).reshape(-1, 3) + offset)
            offset += len(v)
        digest = hashlib.sha256(np.concatenate(vertices).tobytes() + np.concatenate(faces).tobytes()).hexdigest()
        if (item['geometry_sha256'] != digest or item['model_id'] != env.object_models[name]
                or sha256(Path(item['file_path'])) != item['file_sha256']
                or abs(item['mass_kg'] - actor._bodies[0].mass) > 1e-8):
            raise ValueError(f'Collider or mass mismatch: {name}')
    env.exported_geometry = {n: {k: v for k, v in g.items() if k != 'world'} for n, g in geometry.items()}


def mesh_vertices(info):
    with Path(info['file_path']).open() as mesh:
        if mesh.readline().strip() != 'ply' or mesh.readline().strip() != 'format ascii 1.0':
            raise ValueError('Expected verified native ASCII PLY')
        elem = mesh.readline().split()
        if elem[:2] != ['element', 'vertex']:
            raise ValueError('Missing PLY vertex count')
        while (line := mesh.readline().strip()) != 'end_header':
            if not line:
                raise ValueError('Truncated PLY header')
        return np.array([mesh.readline().split() for _ in range(int(elem[2]))], dtype=float)


def transformed(points, pose):
    matrix = np.asarray(pose)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def bounds(points):
    return points.min(axis=0), points.max(axis=0)


def separate(first, second, margin):
    return bool(np.any(first[0] >= second[1] + margin) or np.any(second[0] >= first[1] + margin))


def held_check(state, vertices, obstacle_boxes, samples):
    """Check exact held vertices' world AABB against conservative obstacle AABBs."""
    import torch
    config = samples['context']
    q = torch.as_tensor(samples['q'], dtype=torch.float32, device='cuda')
    eef = np.asarray(state['links']['right_eef_link'])
    local_vertices = transformed(vertices, np.linalg.inv(eef) @ state['object_world'])
    tabletop = state['scene_cuboids']['table']
    center = np.asarray(tabletop['center'])
    half = np.asarray(tabletop['dims']) / 2
    minimum = np.inf
    points = 0
    for batch in q.split(64):
        kin = config.planner.compute_kinematics(config.js(batch[0])).tool_poses if len(batch) == 1 else None
        if kin is None:
            # The resident model is warmed on fixed shapes. Pad the final batch.
            from curobo.types import JointState
            if len(batch) < 64:
                padded = torch.cat((batch, batch[-1:].expand(64-len(batch), -1)))
            else:
                padded = batch
            kin = config.planner.compute_kinematics(JointState.from_position(padded.clone(), joint_names=ARM)).tool_poses
        pos = kin.position.detach().cpu().numpy().reshape(-1, 3)[:len(batch)]
        wxyz = kin.quaternion.detach().cpu().numpy().reshape(-1, 4)[:len(batch)]
        rot = Rotation.from_quat(wxyz[:, [1, 2, 3, 0]]).as_matrix()
        for xyz, rotation in zip(pos, rot):
            cloud = (local_vertices @ rotation.T + xyz) @ config.base[:3, :3].T + config.base[:3, 3]
            box = bounds(cloud)
            clearance = float(box[0][2] - (center + half)[2])
            minimum = min(minimum, clearance)
            points += 1
            if clearance < CARRY_CLEARANCE:
                return {'passed': False, 'obstacle': 'table', 'points': points, 'minimum_table_clearance_m': minimum}
            for name, other in obstacle_boxes.items():
                if not separate(box, other, CARRY_CLEARANCE):
                    return {'passed': False, 'obstacle': name, 'points': points, 'minimum_table_clearance_m': minimum}
    return {'passed': True, 'points': points, 'minimum_table_clearance_m': minimum}


def placement(worker, directory):
    state = read(directory / 'held_state.json')
    cup_name, = (n for n, g in state['scene_meshes'].items() if g['target'])
    cup = mesh_vertices(state['scene_meshes'][cup_name])
    others = {n: bounds(transformed(mesh_vertices(g), g['world']))
              for n, g in state['scene_meshes'].items() if n != cup_name}
    ctx = worker.bind(directory / '放置规划', state, closed=True)
    current = ctx.measured(state)
    table = state['scene_cuboids']['table']
    t_min = np.asarray(table['center']) - np.asarray(table['dims']) / 2
    t_max = np.asarray(table['center']) + np.asarray(table['dims']) / 2
    held = np.asarray(state['object_world'])
    pos = np.asarray(state['links']['right_eef_link'])
    cup_bounds = bounds(transformed(cup, held))
    trials = []
    for offset in PLACE_OFFSETS:
        delta = np.array([0., offset, t_max[2] + RELEASE_HEIGHT - cup_bounds[0][2]])
        landing = tuple(p + delta for p in cup_bounds)
        row = {'world_offset_y_m': offset}
        trials.append(row)
        if (np.any(landing[0][:2] <= t_min[:2] + .02) or np.any(landing[1][:2] >= t_max[:2] - .02)
                or not all(separate(landing, other, .025) for other in others.values())):
            row['reason'] = 'landing_area_not_clear'
            continue
        carry_goal = pos.copy()
        carry_goal[1, 3] += offset
        carry_goal[2, 3] += .015
        place_goal = pos.copy()
        place_goal[:3, 3] += delta
        start = current
        segments = {}
        for label, goal, duration in [('carry', carry_goal, 6.), ('lower', place_goal, 5.)]:
            dest = ctx.ik(goal, start)
            raw = ctx.path(start, dest) if dest is not None else None
            if raw is None:
                row['reason'] = f'{label}_ik' if dest is None else f'{label}_path'
                break
            path = ctx.optimize(f'{label}_{len(trials)}', raw, duration)
            check = held_check(state, cup, others, {'context': ctx, 'q': dense_path(path)})
            row[label] = check
            if not check['passed']:
                row['reason'] = f'{label}_held_object_clearance'
                break
            segments[label] = {**path, 'goal': goal.tolist()}
            start = dest.new_tensor(path['q'][-1])
        if len(segments) == 2:
            row['passed'] = True
            return {'success': True, 'candidates': trials, 'paths': segments,
                    'planned_object_xy_m': (held[:2, 3] + delta[:2]).tolist()}
    return {'success': False, 'reason': 'no_valid_carry_and_placement', 'candidates': trials}


def plan_measured_lower(worker, directory):
    state = read(directory / 'carry_settled_state.json')
    cup_name, = (name for name, geometry in state['scene_meshes'].items() if geometry['target'])
    cup = mesh_vertices(state['scene_meshes'][cup_name])
    cup_world = np.asarray(state['object_world'])
    tabletop = state['scene_cuboids']['table']
    center = np.asarray(tabletop['center'])
    half = np.asarray(tabletop['dims']) / 2
    cup_box = bounds(transformed(cup, cup_world))
    obstacles = {name: bounds(transformed(mesh_vertices(g), g['world']))
                 for name, g in state['scene_meshes'].items() if name != cup_name}
    # Keep the measured XY and orientation, and set the mesh bottom's ground clearance.
    dz = (center + half)[2] + RELEASE_HEIGHT - cup_box[0][2]
    if dz >= 0:
        return {'success': False, 'reason': 'measured_cup_already_below_release_height'}
    landing = tuple(point + [0., 0., dz] for point in cup_box)
    if (np.any(landing[0][:2] <= (center - half)[:2] + .02)
            or np.any(landing[1][:2] >= (center + half)[:2] - .02)
            or not all(separate(landing, other, .025) for other in obstacles.values())):
        return {'success': False, 'reason': 'measured_landing_area_not_clear'}
    goal = np.asarray(state['links']['right_eef_link'])
    goal[2, 3] += dz
    ctx = worker.bind(directory / '实测下降规划', state, closed=True)
    start = ctx.measured(state)
    end = ctx.ik(goal, start)
    raw = None if end is None else ctx.path(start, end)
    if raw is None:
        return {'success': False, 'reason': 'measured_lower_ik' if end is None else 'measured_lower_path'}
    path = ctx.optimize('lower', raw, 5.)
    check = held_check(state, cup, obstacles, {'context': ctx, 'q': dense_path(path)})
    if not check['passed']:
        return {'success': False, 'reason': 'measured_lower_held_collision', 'held_check': check}
    return {'success': True, 'path': path, 'goal': goal.tolist(), 'planned_vertical_motion_m': float(dz),
            'planned_object_xy_m': cup_world[:2, 3].tolist(), 'held_check': check}


def plan_retreat(worker, directory):
    import torch

    state = read(directory / 'released_state.json')
    ctx = worker.bind(directory / '撤离规划', state, include_target=True)
    current = ctx.measured(state)
    trials = []
    for distance in RETREAT_HEIGHTS:
        goal = np.array(state['links']['right_eef_link'])
        goal[2, 3] += distance
        dest = ctx.ik(goal, current)
        row = {'vertical_distance_m': distance, 'ik_and_endpoint_collision_passed': dest is not None}
        trials.append(row)
        if dest is None:
            continue
        row['q_goal'] = dest.tolist()
        raw = ctx.path(current, dest)
        row['path_found'] = raw is not None
        if raw is None:
            continue
        path = ctx.optimize('retreat', raw, 5.)
        q = torch.as_tensor(dense_path(path), device='cuda', dtype=torch.float32)
        passed = all(bool(ctx.checker.validate(chunk[:, None]).all()) for chunk in q.split(64))
        row.update(points=len(q), path_collision_check_passed=passed)
        if passed:
            return {'success': True, 'path': path, 'goal': goal.tolist(), 'trials': trials,
                    'selected_vertical_distance_m': distance, 'includes_released_blocker': True}
    return {'success': False, 'reason': 'retreat_ik_or_path', 'trials': trials}


def serve(directory, address):
    import torch
    import resident_planner
    import resident_backend
    from checked_native_services import checked_pool
    from fast_resident_planner import FastPlannerPool
    from short_lift_planner import ShortLiftPlanner

    spec = importlib.util.spec_from_file_location('frozen_table_policy', POLICY)
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    policy.install_collision_policy()
    resident_planner.GraspCLPlanner = ShortLiftPlanner

    class RemovalPool(checked_pool(FastPlannerPool)):
        def request(self, directory, **request):
            action = request.pop('action', None)
            if action == 'validate':
                label = request['label']
                state = read(directory / request['state'])
                path = request['path']
                q = dense_path(path)
                if np.max(np.abs(arm_q(state) - q[0])) >= .005:
                    return {'success': False, 'reason': 'trajectory_start_mismatch'}
                ctx = self.bind(directory / f'{label}_复核', state, include_target=True)
                q = torch.as_tensor(q, device='cuda', dtype=torch.float32)
                accepted = all(bool(ctx.checker.validate(batch[:, None]).all()) for batch in q.split(64))
                return {'success': accepted, 'points': len(q), 'scene_includes_blocker': True}
            if action == 'placement':
                return placement(self, directory)
            if action == 'measured_lower':
                return plan_measured_lower(self, directory)
            if action == 'retreat':
                return plan_retreat(self, directory)
            return super().request(directory, **request)

    resident_planner.PlannerPool = RemovalPool
    resident_backend.serve(SimpleNamespace(kind='curobo', root=directory, socket=address, warmup_trial=WARMUP))


def call(address, directory, name, **request):
    response = rpc(address, {'directory': str(directory), **request}, timeout=180)
    save(directory / (name + '_回执.json'), response)
    if response['status'] != 'ok':
        error = response.get('error', '')
        if any(msg in error for msg in ('Measured state collides in unchanged ARX model',
                                       'Initial ARX state collides in cuRobo model')):
            raise StageStopped(name + '_state_collision')
        raise RuntimeError(error or str(response))
    if not response['result']['success']:
        raise StageStopped(name + ': ' + response['result'].get('reason', 'collision_check_failed'))
    return response['result']


def require_view_match(directory, actual, expected):
    camera = pose_errors(actual['camera_world_optical'], expected['camera_world_optical'])
    actors = {name: pose_errors(g['world'], expected['scene_meshes'][name]['world'])
              for name, g in actual['scene_meshes'].items()}
    intrinsics = bool(np.allclose(actual['intrinsics'], expected['intrinsics'], rtol=0, atol=1e-6))
    passed = intrinsics and all(p <= POSITION_MATCH and r <= ROTATION_MATCH for p, r in [camera, *actors.values()])
    save(directory / '观测配准核验.json', {'passed': passed, 'camera_error_m_rad': camera,
         'actor_errors_m_rad': actors, 'intrinsics_match': intrinsics,
         'position_limit_m': POSITION_MATCH, 'rotation_limit_rad': ROTATION_MATCH})
    if not passed:
        raise StageStopped('archived_view_mismatch')


def initial_match(env, state, seed, trial):
    from environment import BASE_POSE

    expected = read(trial / 'initial_conditions.json')
    if (not np.array_equal(env.initial_q, np.asarray(expected['initial_q'], dtype=env.initial_q.dtype))
            or BASE_POSE != expected['base_pose'] or env.spawn != expected['spawn']):
        raise ValueError('Frozen reset metadata differs')
    save(state, {'seed': seed, 'initial_q': expected['initial_q'],
                 'base_pose': BASE_POSE, 'spawn': env.spawn})


def stability(samples, pose_key):
    if len(samples) < STABLE_SAMPLES:
        raise StageStopped('insufficient_stability_samples')
    errors = [pose_errors(row[pose_key], samples[-1][pose_key]) for row in samples]
    shift, angle = np.max(errors, axis=0)
    return {'samples': len(samples), 'max_shift_m': float(shift), 'max_rotation_rad': float(angle),
            'stable': bool(shift <= STABLE_SHIFT and angle <= STABLE_ANGLE)}


class PlacementObservation:
    def __init__(self, env, execution, vertices):
        self.env, self.execution = env, execution
        self.vertices = vertices
        self.cup = env.obj._objs[0]
        self.table = env.table._objs[0]
        self.other_objects = [actor._objs[0] for actor in env.objects.values() if actor._objs[0] != self.cup]
        self.eef = env.agent.robot.links_map['right_eef_link']
        self.samples = deque(maxlen=STABLE_SAMPLES)
        self.log = None

    def record(self):
        import json

        self.execution.record()
        cup = self.env.obj.pose.to_transformation_matrix().cpu().numpy()[0]
        eef = self.eef.pose.to_transformation_matrix().cpu().numpy()[0]
        fingers, support, touches_other = set(), False, False
        for contact in self.env.scene.get_contacts():
            a, b = (body.entity for body in contact.bodies)
            if a != self.cup and b != self.cup:
                continue
            if sum(float(np.linalg.norm(point.impulse)) for point in contact.points) <= 1e-8:
                continue
            other = b if a == self.cup else a
            fingers.update(name for name in BLOCKER_FINGERS if name in other.name)
            support |= other == self.table
            touches_other |= other in self.other_objects
        row = {'world': cup.tolist(), 'in_gripper': (np.linalg.inv(eef) @ cup).tolist(),
               'cup_bottom_m': float(transformed(self.vertices, cup)[:, 2].min()),
               'fingers': sorted(fingers), 'table_contact': support,
               'other_object_contact': touches_other, 'tick': self.execution.tick,
               'stage': self.execution.label}
        self.samples.append(row)
        self.log.write(json.dumps(row) + '\n')

    def segment(self, label, path, **kwargs):
        self.samples = deque(maxlen=None if label == 'lower' else STABLE_SAMPLES)
        with (self.execution.out / '放置观测.jsonl').open('a') as log:
            self.log = log
            self.env.trace_callback = self.record
            try:
                self.execution.segment(label, path['q'], path['times'], **kwargs)
            finally:
                self.env.trace_callback = self.execution.record
                self.log = None
        return list(self.samples)


def grip_confirmed(samples):
    return all(set(row['fingers']) == BLOCKER_FINGERS and not row['other_object_contact']
               and not row['table_contact'] for row in samples)


def until_still(q, seconds):
    return {'q': [q, q], 'times': [0, seconds]}


def physical(directory, address, trial, blocker):
    import torch
    from PIL import Image
    import imageio.v2 as imageio
    from run import Execution, make, snapshot
    from scene_tools import visibility

    source = read(trial / 'result.json')
    historical = read(ARCHIVE / 'state.json')
    plan = read(ARCHIVE / 'plan_result.json')
    selected = plan['selected']
    validate_selected(selected)
    env = None
    execution = None
    result = {'scene': source['id'], 'blocker': blocker, 'physical_attempt': False,
              'close_started': False, 'grasp_passed': False, 'placement_passed': False, 'success': False}
    started = time.monotonic()
    try:
        env = make(source['model_id'], perception_camera=source['camera'], scene_spec=source['scene'])
        env.reset(seed=source['seed'])
        model_matches(env, historical)
        initial_match(env, directory / '初始条件.json', source['seed'], trial)
        for _ in range(80):
            env._step_action(torch.tensor(env.initial_q[None], device=env.device))
        save(directory / 'reset_state.json', snapshot(env))
        observe = read(trial / 'observation_plan_result.json')['paths']['observe']
        call(address, directory, '观察轨迹', action='validate', label='observe', state='reset_state.json', path=observe)
        execution = Execution(env, directory)
        execution.segment('observe', observe['q'], observe['times'], goal_world=source['observation']['goal_world'])
        view = visibility(env)
        save(directory / '原目标可见性.json', view)
        if not view['admitted_view'] or not .20 <= view['occlusion_fraction'] <= .40:
            raise StageStopped('initial_occlusion_outside_registered_range')

        # Change the recorder's contact/height reference after the original target's visibility check.
        env.obj = env.objects[blocker]
        info = env.object_info[blocker]
        env.object_bbox, env.object_scale = info['bbox'], info.get('scales', [1.])[0]
        execution.initial_z = float(env.obj.pose.p[0, 2])
        save(directory / '挡物高度基准.json', {'z_m': execution.initial_z, 'stage': 'after_observe'})
        obs = env.get_obs()
        measured = snapshot(env, obs)
        save(directory / 'state.json', measured)
        data = obs['sensor_data'][env.perception_camera]
        Image.fromarray(data['rgb'][0].cpu().numpy()).save(directory / '进场腕部.png')
        Image.fromarray(data['depth'][0, :, :, 0].cpu().numpy().astype(np.uint16)).save(directory / '进场深度.png')
        imageio.imwrite(directory / '移障前.png', env.render()[0].cpu().numpy())
        require_view_match(directory, measured, historical)

        shutil.copy2(ARCHIVE / 'plan_result.json', directory / 'plan_result.json')
        path = plan['paths']['pregrasp']
        call(address, directory, '预抓取轨迹', action='validate', label='pregrasp', state='state.json', path=path)
        result['physical_attempt'] = True
        execution.segment('pregrasp', path['q'], path['times'], goal_world=selected['pregrasp_world'])
        save(directory / 'pregrasp_state.json', snapshot(env))
        path = call(address, directory, '实测接近规划', approach=True)['paths']['approach']
        execution.segment('approach', path['q'], path['times'], goal_world=selected['grasp_world'])
        print('I0069 进入闭爪', flush=True)
        result['close_started'] = True
        q = selected['q_grasp']
        execution.segment('close', [q, q], [0, 3], finger=0., goal_world=selected['grasp_world'])
        result['bilateral_fingers'] = sorted(execution.bilateral)
        closed = snapshot(env)
        save(directory / 'closed_state.json', closed)
        result['closed_joint_positions_m'] = {n: closed['qpos'][closed['joint_names'].index(n)] for n in FINGERS}
        if len(execution.bilateral) != 2:
            raise StageStopped('no_bilateral_blocker_contact')
        path = call(address, directory, '实测短抬规划', lift=True)['paths']['lift']
        execution.segment('lift', path['q'], path['times'], finger=0., goal_world=selected['lift_world'])
        q = path['q'][-1]
        execution.segment('hold', [q, q], [0, HOLD_SECONDS], finger=0., goal_world=selected['lift_world'])
        held = snapshot(env)
        save(directory / 'held_state.json', held)
        result['hold'] = {'minimum_lift_m': min(execution.heights) - execution.initial_z,
                          'seconds': len(execution.heights) / 120, 'samples': len(execution.heights)}
        if result['hold']['minimum_lift_m'] < MIN_HOLD_LIFT_M or result['hold']['seconds'] < HOLD_SECONDS:
            raise StageStopped('insufficient_hold_height')
        result['grasp_passed'] = True
        print('I0069 短抬通过，开始实测搬运规划', flush=True)

        place = call(address, directory, '实测搬运规划', action='placement')
        vertices = mesh_vertices(held['scene_meshes'][blocker])
        reference = np.linalg.inv(held['links']['right_eef_link']) @ held['object_world']
        path = place['paths']['carry']
        execution.segment('carry', path['q'], path['times'], finger=0., goal_world=path['goal'])
        observer = PlacementObservation(env, execution, vertices)
        settle_samples = observer.segment('carry_settle', until_still(path['q'][-1], 2.), finger=0.)
        settled = snapshot(env)
        save(directory / 'carry_settled_state.json', settled)
        relative = np.linalg.inv(settled['links']['right_eef_link']) @ settled['object_world']
        result['carry_grip_drift_m_rad'] = pose_errors(reference, relative)
        result['carry_stability'] = stability(settle_samples, 'in_gripper')
        result['carry_stability']['bilateral_throughout'] = grip_confirmed(settle_samples)
        result['carry_stability']['minimum_bottom_m'] = min(row['cup_bottom_m'] for row in settle_samples)
        if (not result['carry_stability']['stable'] or not result['carry_stability']['bilateral_throughout']
                or result['carry_stability']['minimum_bottom_m'] < RELEASE_HEIGHT):
            raise StageStopped('cup_still_moving_or_unsupported_after_transport')
        if np.linalg.norm(np.asarray(settled['object_world'])[:2, 3] - place['planned_object_xy_m']) > .02:
            raise StageStopped('cup_outside_planned_landing_area')
        print('I0069 侧移后持杯稳定，用实测夹持姿态规划下降', flush=True)
        lower = call(address, directory, '下降轨迹', action='measured_lower')
        path = lower['path']
        lower_samples = observer.segment('lower', path, finger=0., goal_world=lower['goal'])
        errors = [pose_errors(relative, row['in_gripper']) for row in lower_samples]
        shift, angle = np.max(errors, axis=0)
        result['lower_grip_drift_m_rad'] = [float(shift), float(angle)]
        result['lower_minimum_bottom_m'] = min(row['cup_bottom_m'] for row in lower_samples)
        if (not grip_confirmed(lower_samples) or shift > MAX_LOWER_SHIFT or angle > MAX_LOWER_ANGLE
                or result['lower_minimum_bottom_m'] < CARRY_CLEARANCE):
            raise StageStopped('cup_shifted_or_touched_during_lowering')
        print('I0069 下降到位，开始松爪和落稳确认', flush=True)
        q = path['q'][-1]
        release_samples = observer.segment('release', until_still(q, 3.), finger=.044)
        released = snapshot(env)
        save(directory / 'released_state.json', released)
        result['release_xy_error_m'] = float(np.linalg.norm(
            np.asarray(released['object_world'])[:2, 3] - lower['planned_object_xy_m']))
        result['placed_bottom_z_m'] = float(transformed(vertices, released['object_world'])[:, 2].min())
        result['displacement_m'] = pose_errors(released['object_world'], measured['object_world'])[0]
        result['original_target_drift_m'] = pose_errors(released['scene_meshes']['target']['world'],
                                                       measured['scene_meshes']['target']['world'])[0]
        result['original_target_drift_rad'] = pose_errors(released['scene_meshes']['target']['world'],
                                                         measured['scene_meshes']['target']['world'])[1]
        result['release_stability'] = stability(release_samples, 'world')
        result['release_stability']['supported_and_free'] = all(
            row['table_contact'] and not row['fingers'] and not row['other_object_contact'] for row in release_samples)
        result['placed_tilt_deg'] = float(np.degrees(np.arccos(np.clip(released['object_world'][2][2], -1., 1.))))
        if (not result['release_stability']['stable'] or not result['release_stability']['supported_and_free']
                or result['placed_tilt_deg'] > 10. or result['release_xy_error_m'] > .02
                or abs(result['placed_bottom_z_m']) > .004
                or result['original_target_drift_m'] > TARGET_DRIFT or result['original_target_drift_rad'] > .03):
            raise StageStopped('placement_not_stable_or_original_target_disturbed')
        result['placement_passed'] = True
        imageio.imwrite(directory / '杯子已落稳.png', env.render()[0].cpu().numpy())
        retreat = call(address, directory, '实测撤离规划', action='retreat')
        path = retreat['path']
        execution.segment('retreat', path['q'], path['times'], goal_world=retreat['goal'])
        final = snapshot(env)
        result['after_retreat_cup_drift_m'] = pose_errors(final['object_world'], released['object_world'])[0]
        result['after_retreat_target_drift_m'], result['after_retreat_target_drift_rad'] = pose_errors(
            final['scene_meshes']['target']['world'], measured['scene_meshes']['target']['world'])
        if (result['after_retreat_cup_drift_m'] > .005 or result['after_retreat_target_drift_m'] > TARGET_DRIFT
                or result['after_retreat_target_drift_rad'] > .03):
            raise StageStopped('scene_shift_after_retreat')
        imageio.imwrite(directory / '移障后.png', env.render()[0].cpu().numpy())
        result.update(success=True, reason='placed_and_retreated')
    except StageStopped as error:
        result['reason'] = str(error)
    except Exception as error:
        result.update(reason=str(error), error=traceback.format_exc())
    finally:
        try:
            if execution:
                result.update(stages=execution.stages, forbidden_contact_events=len(execution.forbidden),
                              sim_samples=execution.tick, control_ticks=execution.control_ticks)
                execution.finish()
            if env is not None and getattr(env, 'exported_geometry', None):
                save(directory / 'final_state.json', snapshot(env))
        finally:
            if env is not None:
                env.close()
            result['wall_seconds'] = time.monotonic() - started
            save(directory / '验证结果.json', result)
    print({k: v for k, v in result.items() if k != 'stages'}, flush=True)


def planner_env(gpu):
    cache = ROOT / '缓存'
    return {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, 'ARX_PLANNER_GPU': gpu,
            'WARP_CACHE_PATH': str(cache / 'warp'), 'CUDA_CACHE_PATH': str(cache / 'cuda'),
            'TRITON_CACHE_DIR': str(cache / 'triton'), 'MPLCONFIGDIR': str(cache / 'matplotlib'),
            'XDG_CACHE_HOME': str(cache), 'PYTHONDONTWRITEBYTECODE': '1'}


def run(directory, planner_gpu, released_state=None):
    from benchmark_lifecycle import mark_ready, require_ready

    sim_gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not planner_gpu.startswith('GPU-') or (released_state is None and not sim_gpu.startswith('GPU-')):
        raise ValueError('Identify each active device by physical GPU UUID')
    if directory.parent.resolve() != RESOURCES.resolve():
        raise ValueError('Output must be in this step’s resources')
    reg = validate_input(ARCHIVE)
    trial = Path(reg['source_trial'])
    original = read(trial / 'result.json')
    if original['id'].split('_')[0] != 'I0069' or original['physical_attempt'] or original['success']:
        raise ValueError('Expected the frozen I0069 failed direct-grasp scene')
    verified = read(ARCHIVE / '挡物规划联调结果.json')
    if not verified['planning_passed'] or not verified['transit_all_scene_check']['passed']:
        raise ValueError('Archive was not cleared for physical validation')
    assets = MS / 'assets'
    ycb = assets / 'data/assets/mani_skill2_ycb'
    asset_files = [ycb / 'info_pick_v0.json']
    for actor in original['scene']['actors']:
        model = ycb / 'models' / actor['model_id']
        asset_files.extend(model / name for name in ('collision.ply', 'textured.obj'))
    missing = [str(file) for file in asset_files if not file.is_file()]
    if missing:
        raise FileNotFoundError('Frozen scene assets missing: ' + ', '.join(missing))
    os.environ['MS_ASSET_DIR'] = str(assets)
    os.environ.setdefault('SAPIEN_CACHE_DIR', str(ROOT / '缓存/sapien'))
    directory.mkdir(exist_ok=False)
    files = [Path(__file__), POLICY, *(trial / name for name in
             ('result.json', 'initial_conditions.json', 'observation_plan_result.json')),
             *(ARCHIVE / name for name in ('state.json', 'plan_result.json', '移障输入登记.json')),
             *asset_files]
    if released_state is not None:
        release_result = released_state.parent / '验证结果.json'
        if released_state.name != 'released_state.json' or not read(release_result)['placement_passed']:
            raise ValueError('Expected a release state from a verified placement')
        files.extend((released_state, release_result))
        shutil.copy2(released_state, directory / 'released_state.json')
    save(directory / '验证登记.json', {'scene': original['id'], 'budget': int(released_state is None),
         'sim_gpu': sim_gpu if released_state is None else None,
         'planner_gpu': planner_gpu, 'planner': 'frozen cuRobo TCP43 actual table, measured gripper',
         'required_components': ['curobo'], 'frozen_source_hashes': {str(f): sha256(f) for f in files},
         'cup_model': reg['model_id'], 'held_clearance_m': CARRY_CLEARANCE,
         'placement_y_offsets_m': PLACE_OFFSETS, 'release_height_m': RELEASE_HEIGHT,
         'settle_sim_seconds': 2, 'stability_samples': STABLE_SAMPLES, 'stability_shift_m': STABLE_SHIFT,
         'stability_angle_rad': STABLE_ANGLE, 'maximum_lower_grip_shift_m': MAX_LOWER_SHIFT,
         'maximum_lower_grip_rotation_rad': MAX_LOWER_ANGLE, 'source_trial': str(trial),
         'retreat_heights_m': RETREAT_HEIGHTS,
         'method': '侧移后实测持物位姿并重新规划下降',
         'scope': '一次物理移障验证' if released_state is None else '已释放场景状态的离线撤离规划'})
    resident = directory / 'resident'
    resident.mkdir()
    env_script = MS.parent / 'arx_core/scripts/curobo_arx/environment.sh'
    with tempfile.TemporaryDirectory(prefix='claro_') as tmp, (resident / 'curobo.log').open('x') as log:
        address = Path(tmp) / 'service.sock'
        command = ['bash', '-c', 'source "$1"; shift; export TORCH_EXTENSIONS_DIR="$CLARO_TORCH_CACHE"; exec "$@"',
                   'claro-worker', str(env_script), str(MS.parent / 'curobo/.venv/bin/python'),
                   str(Path(__file__).resolve()), 'worker', str(resident), str(address)]
        proc = subprocess.Popen(command, env={**planner_env(planner_gpu),
                                'CLARO_TORCH_CACHE': str(ROOT / '缓存/torch_extensions')},
                                stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        try:
            end = time.monotonic() + 420
            while not (resident / 'curobo_ready.json').exists():
                if proc.poll() is not None or time.monotonic() >= end:
                    raise RuntimeError('Planner preload did not complete')
                time.sleep(2)
            mark_ready(resident, ['curobo'], {'sockets': {'curobo': str(address)}}, {'curobo': proc})
            require_ready(resident, ['curobo'])
            for file, digest in read(directory / '验证登记.json')['frozen_source_hashes'].items():
                if sha256(Path(file)) != digest:
                    raise RuntimeError('Frozen replay inputs changed during preheat')
            if released_state is None:
                print('I0069 规划器已预热，开始复位和观测', flush=True)
                physical(directory, address, trial, reg['selected_scene_actor'])
            else:
                response = rpc(address, {'directory': str(directory), 'action': 'retreat'}, timeout=180)
                save(directory / '撤离规划_回执.json', response)
                if response['status'] != 'ok':
                    raise RuntimeError(response.get('error', str(response)))
                print({key: value for key, value in response['result'].items() if key != 'path'}, flush=True)
        except Exception:
            save(directory / '流程故障.json', {'error': traceback.format_exc()})
            raise
        finally:
            if proc.poll() is None:
                try:
                    rpc(address, {'command': 'shutdown'}, timeout=5)
                    proc.wait(timeout=10)
                except (OSError, subprocess.TimeoutExpired):
                    proc.terminate()
                    try:
                        proc.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('mode', choices=('run', 'worker'))
    cli.add_argument('directory', type=Path)
    cli.add_argument('gpu_or_socket')
    cli.add_argument('--released-state', type=Path)
    args = cli.parse_args()
    if args.mode == 'worker':
        serve(args.directory.resolve(), Path(args.gpu_or_socket))
    else:
        run(args.directory.resolve(), args.gpu_or_socket,
            args.released_state.resolve() if args.released_state is not None else None)


if __name__ == '__main__':
    main()
