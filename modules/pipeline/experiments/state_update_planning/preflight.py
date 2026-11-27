"""Compare original and memory-bound native geometry before timed requests."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np

from arx_project.layout import now, read_json, write_json
from modules.graspgenx.native_postprocessing.width_planning import target_state
from modules.curobo.gripper_sweep.adapter import narrowing_canonical


def run(planner, warmup, root):
    import torch
    pool = planner.pool
    source = read_json(warmup / 'state.json')
    reports, sweeps = [], []
    for side in ['right', 'left']:
        canonical = planner.model.bind(source, side, root / side / 'physical_arm')
        for width in [.088, .060, .040]:
            state = target_state(canonical, width)
            work = root / side / str(width)
            values = {}
            for mode in ['control', 'memory']:
                pool.memory_updates_enabled = mode == 'memory'
                if mode == 'memory':
                    pool.memory_bindings.begin()
                try:
                    context = pool.bind(work / mode, state, include_target=False)
                    validator = pool.contact_view(context, state)
                    q = context.measured(state)
                    offsets = torch.linspace(-.01, .01, 32, device=q.device)[:, None]
                    qs = (q[None] + offsets * torch.tensor([1., -.4, .6, -.3, .2, -.5], device=q.device)[None])
                    qs = qs.clamp(context.limits[:, 0] + 1e-4, context.limits[:, 1] - 1e-4)
                    spheres = context.checker.get_kinematics(qs[:, None]).robot_spheres.detach().cpu().numpy().copy()
                    valid = context.checker.validate(qs[:, None]).detach().cpu().numpy().copy()
                    local = validator.validate(qs, stage='final_approach').copy()
                    locks = context.checker.kinematics.kinematics_config.lock_jointstate.position.detach().cpu().numpy().copy()
                    values[mode] = {'spheres': spheres, 'valid': valid, 'local': local, 'locks': locks}
                finally:
                    if mode == 'memory':
                        pool.memory_bindings.finish()
                    pool.memory_updates_enabled = False
            for filename in ['arx.yml', 'scene.json', 'collision_profile.json', 'fk_validation.json', 'start_state.json']:
                import yaml
                a = yaml.load((work / 'control' / filename).read_text(), Loader=yaml.CSafeLoader)
                b = yaml.load((work / 'memory' / filename).read_text(), Loader=yaml.CSafeLoader)
                if a != b:
                    raise RuntimeError('In-memory binding changed its original values: ' + filename)
            if any((work / m / 'nominal_closed.yml').exists() for m in ['control', 'memory']):
                raise RuntimeError('An obsolete nominal-closed model was restored')
            a, b = values['control'], values['memory']
            error = float(np.max(np.abs(a['spheres'] - b['spheres'])))
            if (error > 2e-7 or not np.array_equal(a['spheres'][..., 3], b['spheres'][..., 3])
                    or not all(np.array_equal(a[k], b[k]) for k in ['valid', 'local', 'locks'])):
                raise RuntimeError('Memory binding changed native sphere/FK/collision state')
            reports.append({'arm': side, 'width_m': width, 'states': 32,
                'maximum_sphere_error_m': error, 'radii_masks_locks_equal': True,
                'original_receipt_values_equal': True})
            write_json(root / 'progress.json', {'at': now(), 'completed_width_pairs': len(reports), 'total_width_pairs': 6})
    from modules.curobo.dual_arm import accelerated_adapter as native
    for side, width in [('right', .088), ('left', .088), ('right', .060), ('left', .040)]:
        work = root / 'sweeps' / (side + '_' + str(width))
        canonical = planner.model.bind(source, side, work / 'physical_arm')
        if width == .088:
            observed = {n: canonical['qpos'][canonical['joint_names'].index(n)] for n in ['right_joint27', 'right_joint28']}
            canonical = target_state(canonical, width)
            canonical['planned_opening'] = {'target_per_finger_m': .044, 'observed_solver_finger_positions_m': observed}
        else:
            canonical = narrowing_canonical(canonical, width)
        values = {}
        original = native.validate_sphere_batch
        for mode in ['control', 'memory']:
            captured = []
            def capture(checker, spheres, q):
                captured.append(spheres.detach().cpu().numpy().copy())
                return original(checker, spheres, q)
            pool.memory_updates_enabled = mode == 'memory'
            if mode == 'memory':
                pool.memory_bindings.begin()
            try:
                with patch.object(native, 'validate_sphere_batch', capture):
                    result = planner.opening_sweep(work / mode, canonical)
                values[mode] = (result, captured)
            finally:
                if mode == 'memory':
                    pool.memory_bindings.finish()
                pool.memory_updates_enabled = False
        (a, ac), (b, bc) = values['control'], values['memory']
        if (a['passed'] != b['passed'] or a.get('first_invalid_sample') != b.get('first_invalid_sample')
                or a.get('valid_per_sample') != b.get('valid_per_sample') or len(ac) != len(bc)):
            raise RuntimeError('Memory-bound opening changed its collision verdict')
        error = max((float(np.max(np.abs(x - y))) for x, y in zip(ac, bc)), default=0.)
        if error > 2e-7:
            raise RuntimeError('Memory-bound opening changed its native FK samples')
        sweeps.append({'arm': side, 'width_m': width, 'samples': a['samples_checked'],
                       'maximum_sphere_error_m': error, 'masks_equal': True})
    mutation = scene_mutation(planner, source, root / 'scene_mutation')
    result = {'at': now(), 'passed': True, 'width_pairs': reports, 'sweep_pairs': sweeps,
              'scene_mutation': mutation, 'complete_planning_requests': 0, 'physical_grasps': 0}
    write_json(root / 'acceptance.json', result)
    return result


def scene_mutation(planner, source, root):
    pool = planner.pool
    state = planner.model.bind(source, 'right', root / 'physical_arm')
    state = target_state(state, .088)
    state['scene_cuboids']['update_probe'] = {'center': [10., 10., 10.], 'dims': [.001] * 3}
    pool.memory_updates_enabled = True
    pool.memory_bindings.begin()
    try:
        context = pool.bind(root / 'far', state, include_target=False)
        q = context.measured(state)
        spheres = context.checker.get_kinematics(q[None, None]).robot_spheres.detach().cpu().numpy().reshape(-1, 4)
        choices = np.flatnonzero(spheres[:, 3] > .002)
        point = spheres[choices[len(choices) // 2], :3]
        world = point @ np.asarray(state['links']['body4'])[:3, :3].T + np.asarray(state['links']['body4'])[:3, 3]
        changed = deepcopy(state)
        changed['scene_cuboids']['update_probe']['center'] = world.tolist()
        try:
            pool.bind(root / 'colliding', changed, include_target=False)
        except AssertionError as error:
            if str(error) != 'Measured state collides in unchanged ARX model':
                raise
        else:
            raise RuntimeError('A moved obstacle failed to invalidate the start collision gate')
        removed = deepcopy(state)
        del removed['scene_cuboids']['update_probe']
        pool.bind(root / 'disabled', removed, include_target=False)
        try:
            pool.bind(root / 'reenabled', changed, include_target=False)
        except AssertionError as error:
            if str(error) != 'Measured state collides in unchanged ARX model':
                raise
        else:
            raise RuntimeError('A reenabled obstacle did not restore its collision')
        pool.bind(root / 'restored', state, include_target=False)
    finally:
        summary = pool.memory_bindings.finish()
        pool.memory_updates_enabled = False
    if not summary['scenes']['pose_updates']:
        raise RuntimeError('Scene fixture did not exercise fixed-storage pose updates')
    if not summary['scenes']['enable_flag_updates']:
        raise RuntimeError('Scene fixture did not exercise obstacle enable changes')
    return {'passed': True, 'moved_obstacle_blocked': True, 'restored_scene_clear': True,
            'disabled_obstacle_clear': True, 'reenabled_obstacle_blocked': True,
            'scene_updates': summary['scenes']}
