"""Optional map placement gate: original endpoint collision rules and planner."""
from copy import deepcopy
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from arx_project.layout import ROOT, read_json, write_json
from modules.arx_core.arms import get_arm
from modules.curobo.dual_arm.accelerated_adapter import AcceleratedDualArmPlanner


def connection_gate(context, target, approach_axis, q):
    pre = target.copy(); pre[:3, 3] -= .08*np.asarray(approach_axis)
    lifted = target.copy(); lifted[2, 3] += .03
    qp, ql = context.ik(pre, q), context.ik(lifted, q)
    return {'passed': qp is not None and ql is not None,
            'pregrasp_endpoint_clear': qp is not None, 'lift_endpoint_clear': ql is not None,
            'new_IK_calls': 2}


class PlacementPlanner(AcceleratedDualArmPlanner):
    def prepare_arm(self, directory, state, arm, filename, *, planned_open=False):
        work = super().prepare_arm(directory, state, arm, filename, planned_open=planned_open)
        witness_file = Path(directory)/'placement_witness.json'
        if planned_open and witness_file.is_file():
            proposal = read_json(witness_file)
            if proposal['arm'] != arm:
                raise ValueError('Known endpoint witness belongs to another arm')
            candidates = read_json(Path(directory)/'candidates/candidate_pool.json')['candidates']
            if len(candidates) != 1:
                raise ValueError('Known endpoint witness requires one original grasp')
            target = np.asarray(candidates[0]['pose']).copy()
            target[:3, 3] -= .043*target[:3, 2]
            canonical = read_json(work/filename)
            canonical['kinematic_endpoint_witness'] = {'q': proposal['witness']['q'],
                'eef_target_world': target.tolist(), 'source_map_row': proposal['source_map_row'],
                'physical_arm': arm, 'original_candidate_index': proposal['candidate_index']}
            write_json(work/filename, canonical)
        return work

    def request(self, directory, *, station_probe=False, **flags):
        if not station_probe:
            return super().request(directory, **flags)
        if set(flags) - {'arm'}:
            raise ValueError('Endpoint probe cannot execute another planning stage')
        return self.endpoint(directory, flags.get('arm'))

    def endpoint(self, directory, side):
        import torch
        started = time.perf_counter()
        directory = Path(directory)
        state = read_json(directory/'state.json')
        proposal = read_json(directory/'placement_witness.json')
        arm = get_arm(side)
        witness = proposal['witness']
        if proposal['arm'] != side or witness['joint_names'] != list(arm.joints):
            raise ValueError('Endpoint witness belongs to another physical arm')
        candidates = read_json(directory/'candidates/candidate_pool.json')['candidates']
        if len(candidates) != 1:
            raise ValueError('Endpoint probe requires one unchanged world grasp')
        contact = np.asarray(candidates[0]['pose'])
        target = contact.copy(); target[:3, 3] -= .043*contact[:3, 2]
        q = np.asarray(witness['q'], dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError('Invalid witness joint vector')
        checked = deepcopy(state)
        checked['links'] = {'base_link': state['links']['base_link']}
        for name, value in zip(arm.joints, q):
            i = state['joint_names'].index(name)
            lower, upper = state['joint_limits'][i]
            if not lower <= value <= upper:
                raise ValueError('Endpoint witness violates joint limits')
            checked['qpos'][i] = float(value)
        actual = self.model.frames(checked)[arm.eef]
        pe = float(np.linalg.norm(actual[:3, 3]-target[:3, 3]))
        re = float(Rotation.from_matrix(actual[:3, :3].T@target[:3, :3]).magnitude())
        profile = self.pool.contact_profile
        if pe > profile.position_tolerance_m or re > profile.rotation_tolerance_rad:
            raise ValueError('Endpoint witness does not match the unchanged grasp')
        work = self.prepare_arm(directory, state, side, 'state.json', planned_open=True)
        canonical = read_json(work/'state.json')
        try:
            context = self.pool.bind(work/'placement_endpoint', canonical, include_target=False)
        except AssertionError as error:
            if str(error) != 'Measured state collides in unchanged ARX model':
                raise
            result = {'passed': False, 'reason': 'planned_open_start_collision'}
        else:
            validator = self.pool.contact_view(context, canonical)
            value = torch.as_tensor(q, dtype=torch.float32, device=context.limits.device)
            passed = bool(validator.validate(value[None], stage='final_approach').all())
            result = {'passed': passed, 'reason': 'endpoint_clear' if passed else 'endpoint_collision',
                      'native_full_world_clear': bool(context.checker.validate(value[None, None]).all()),
                      'collision_receipt': validator.receipts[-1], 'sphere_count': 871,
                      'contact_policy_id': profile.profile_id,
                      'maximum_tip_penetration_m': profile.maximum_penetration_m,
                      'distal_finger_length_m': profile.distal_length_m}
            if passed:
                # Eliminate an unusable pregrasp/lift before opening checks and
                # the full base transit sweep. These are the original goals.
                connections = connection_gate(context, target, contact[:3, 2], value)
                result.update(connections)
                if not connections['passed']:
                    result.update(passed=False, reason='pregrasp_or_lift_endpoint_unavailable')
        result.update(independent_position_error_m=pe, independent_orientation_error_rad=re,
                      wall_seconds=time.perf_counter()-started,
                      new_IK_calls=2 if result.get('pregrasp_endpoint_clear') is not None else 0)
        write_json(directory/'endpoint_gate.json', result)
        return result


def create(args):
    from modules.curobo.dual_arm.adapter import create as original_factory
    from modules.curobo.dual_arm.witness_pool import witness_pool
    original = original_factory(args, pool_extension=witness_pool)
    return PlacementPlanner(original.pool, original.model, original.root)
