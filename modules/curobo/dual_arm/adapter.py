"""Run the same native candidate pool against each physical ARX arm."""
from copy import deepcopy
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from arx_project.layout import ROOT, read_json, write_json
from modules.arx_core.arms import get_arm
from modules.curobo.dual_arm.model import ArmModel


def choose_plan(plans, state):
    """Keep native score order, then use the smaller joint displacement."""
    feasible = []
    for arm, result in plans.items():
        if not result['success']:
            continue
        spec = get_arm(arm)
        start = np.asarray([state['qpos'][state['joint_names'].index(n)] for n in spec.joints])
        selected = result['selected']
        travel = float(np.abs(np.asarray(selected['q_pre'])-start).sum())
        feasible.append((int(selected['index']), travel, 0 if arm == 'right' else 1, arm))
    return min(feasible)[-1] if feasible else None


class DualArmPlanner:
    def __init__(self, pool, model, root):
        self.pool, self.model, self.root = pool, model, Path(root)

    def prepare_arm(self, directory, state, arm, filename, *, planned_open=False):
        work = directory/'arms'/arm
        canonical = self.model.bind(state, arm, work/filename.removesuffix('.json'))
        if planned_open:
            observed = {}
            for name in get_arm('right').finger_joints:
                i = canonical['joint_names'].index(name)
                observed[name] = canonical['qpos'][i]
                canonical['qpos'][i] = .044
            canonical['planned_opening'] = {'target_per_finger_m': .044,
                'observed_solver_finger_positions_m': observed,
                'execution_required_before_arm_motion': True}
        write_json(work/filename, canonical)
        candidates = directory/'candidates/candidate_pool.json'
        if not (work/'candidates/candidate_pool.json').exists():
            write_json(work/'candidates/candidate_pool.json', read_json(candidates))
        return work

    def opening_sweep(self, work, canonical):
        """Check the whole closed-to-open preparation at the measured arm pose."""
        target = canonical['planned_opening']['target_per_finger_m']
        observed = canonical['planned_opening']['observed_solver_finger_positions_m']
        count = max(2, int(np.ceil(max(abs(target-v) for v in observed.values())/.0005))+1)
        checks = []
        for fraction in np.linspace(0., 1., count):
            sample = deepcopy(canonical)
            for name, value in observed.items():
                sample['qpos'][sample['joint_names'].index(name)] = float(value+fraction*(target-value))
            try:
                self.pool.bind(work/'opening_check', sample, include_target=True)
            except AssertionError as error:
                if str(error) != 'Measured state collides in unchanged ARX model':
                    raise
                result = {'passed': False, 'reason': 'gripper_opening_sweep_in_collision',
                          'fraction': float(fraction), 'samples_checked': len(checks)+1}
                write_json(work/'opening_sweep.json', result)
                return result
            checks.append(float(fraction))
        result = {'passed': True, 'samples_checked': count, 'maximum_finger_step_m': .0005,
                  'target_per_finger_m': target, 'target_and_parked_arm_collision_checked': True}
        write_json(work/'opening_sweep.json', result)
        return result

    def request(self, directory, observe=False, approach=False, lift=False, selector='graspcl', arm=None):
        directory = Path(directory)
        if observe or int(approach)+int(lift) > 1:
            raise ValueError('V3 starts from the fixed head view and accepts one planning stage')
        started = time.perf_counter()
        filename = 'closed_state.json' if lift else 'pregrasp_state.json' if approach else 'state.json'
        state = read_json(directory/filename)
        if approach or lift:
            selection = read_json(directory/'plan_result.json')['selected']
            chosen = selection['arm']
            if arm is not None and arm != chosen:
                raise ValueError('Arm switching after selection is forbidden')
            if state.get('execution_arm') != chosen:
                raise ValueError('Measured execution arm does not match the selected arm')
            work = self.prepare_arm(directory, state, chosen, filename)
            result = self.pool.request(work, approach=approach, lift=lift, selector=selector)
            result = deepcopy(result)
            result['arm'] = chosen
            name = 'lift_plan' if lift else 'approach_plan'
        else:
            sides = ['right', 'left'] if arm is None else [get_arm(arm).name]
            plans = {}
            for side in sides:
                work = self.prepare_arm(directory, state, side, filename, planned_open=True)
                sweep = self.opening_sweep(work, read_json(work/filename))
                if not sweep['passed']:
                    plans[side] = {'success': False, 'reason': sweep['reason'], 'opening_sweep': sweep}
                    write_json(work/'plan_result.json', plans[side])
                else:
                    plans[side] = self.pool.request(work, selector=selector)
            chosen = choose_plan(plans, state)
            if chosen is None:
                result = {'success': False, 'reason': 'no_arm_has_executable_candidate'}
            else:
                result = deepcopy(plans[chosen])
                selected = result['selected']
                selected.update(arm=chosen, joint_names=get_arm(chosen).joints,
                                finger_joint_names=get_arm(chosen).finger_joints,
                                open_finger_m=.044)
                candidate = read_json(directory/'candidates/candidate_pool.json')['candidates'][selected['index']]
                if not np.allclose(candidate['pose'], selected['contact_pose_world'], atol=1e-10, rtol=0):
                    raise ValueError('Selected arm changed the native world grasp pose')
            result['arm_results'] = {side: {
                'success': p['success'], 'reason': p.get('reason', 'success'),
                'candidate_index': p.get('selected', {}).get('index'),
                'score': p.get('selected', {}).get('score'),
                'rejected': p.get('rejected', {}), 'wall_seconds': p.get('wall_seconds'),
                'receipt_directory': str((directory/'arms'/side).relative_to(ROOT))}
                for side, p in plans.items()}
            result['arm_selection_policy'] = 'native_candidate_order_then_joint_displacement'
            result['forced_arm'] = arm
            name = 'plan'
        result['wall_seconds'] = time.perf_counter()-started
        result['version'] = 'sam3_graspgenx_v3_dual_arm'
        write_json(directory/(name+'_result.json'), result)
        return result


def create(args):
    from modules.curobo.arx_adapter import create as create_single_arm
    model = ArmModel()
    root = Path(args.root)/'dual_arm'
    warm = root/'warmup'
    for filename in ['state.json', 'closed_state.json']:
        state = read_json(Path(args.warmup_trial)/filename)
        canonical = model.bind(state, 'right', warm/filename.removesuffix('.json'))
        write_json(warm/filename, canonical)
    pool = create_single_arm(SimpleNamespace(**{**vars(args), 'root': root, 'warmup_trial': warm}))
    # Rebind and warm each arm through the same solver without moving the robot.
    for side in ['right', 'left']:
        for closed, filename in [(False, 'state.json'), (True, 'closed_state.json')]:
            state = model.bind(read_json(Path(args.warmup_trial)/filename), side, root/'warmup'/side/filename.removesuffix('.json'))
            context = pool.bind(root/'warmup'/side/('closed' if closed else 'open'), state, closed=closed)
            pool.contact_view(context, state)
            if context.ik(np.asarray(state['links']['right_eef_link']), context.measured(state)) is None:
                raise RuntimeError('Dual-arm measured-pose warmup failed: '+side)
    write_json(root/'readiness.json', {'passed': True, 'arms': ['right', 'left'],
        'shared_solver_loads': pool.loads, 'kinematics': model.proof, 'collision_spheres': model.sphere_proof,
        'parked_arm_is_world_obstacle': True, 'physical_motions': 0})
    return DualArmPlanner(pool, model, root)
