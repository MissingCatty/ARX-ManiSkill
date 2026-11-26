"""Batch original fallback branches without changing their order or gates."""
import numpy as np
from scipy.spatial.transform import Rotation


def validate_rows(validator, joints):
    """Use the existing policy as authority and retain one failure per row.

    The additional non-table query recovers per-row diagnostics which the
    original validator only exposes as counts. It does not decide validity.
    """
    import torch
    q = torch.as_tensor(joints, dtype=torch.float32,
                        device=validator.context.limits.device).reshape(-1, 6)
    if not len(q) or not bool(torch.isfinite(q).all()):
        raise ValueError('Expected nonempty finite fallback joint branches')
    valid = np.asarray(validator.validate(q, stage='final_approach'), dtype=bool)
    receipt = validator.receipts[-1]
    if valid.shape != (len(q),) or receipt['states'] != len(q):
        raise RuntimeError('Local validator lost the branch-row correspondence')
    limits = validator.context.limits
    bounds = ((q >= limits[:, 0]-1e-6).all(-1)
              & (q <= limits[:, 1]+1e-6).all(-1)).detach().cpu().numpy()
    other = validator.other_checker.validate(q[:, None]).reshape(-1).detach().cpu().numpy()
    if other.shape != (len(q),) or int(bounds.sum()) != receipt['joint_bounds_clear'] \
            or int(other.sum()) != receipt['other_world_clear']:
        raise RuntimeError('Batched row diagnostics disagree with the original validator')
    details = {}
    for item in receipt['narrow_phase']:
        index = item['index']
        if type(index) is not int or not 0 <= index < len(q) or index in details:
            raise RuntimeError('Invalid or repeated narrow-phase branch index')
        details[index] = item
    reasons = []
    for index, passed in enumerate(valid):
        if passed:
            reasons.append(None)
        elif not bounds[index]:
            reasons.append('joint_limits')
        elif not other[index]:
            reasons.append('ground_or_other_object_collision')
        else:
            reasons.append(details[index]['reason'] if index in details else 'native_collision')
    return valid, reasons


def fallback_endpoint(pool, context, grasp, start, validator):
    """Return the first original passing branch, with the original evidence."""
    native = pool.native_grasp_ik(context, grasp, start)
    if native is not None:
        return native, {'kind': 'native_full_world_ik'}
    solutions = pool.unconstrained_ik(context, grasp, start)
    if not len(solutions):
        return None, {'kind': 'no_local_grasp_endpoint', 'kinematic_solutions': 0, 'reasons': []}
    values = solutions.detach().cpu().numpy()
    if values.ndim != 2 or values.shape[1] != 6 or not np.isfinite(values).all():
        raise ValueError('Invalid unconstrained IK branch array')
    actual = validator.geometry.frames(values)['right_eef_link']
    if actual.shape != (len(values), 4, 4) or not np.isfinite(actual).all():
        raise RuntimeError('Independent FK lost the branch-row correspondence')
    position = np.linalg.norm(actual[:, :3, 3]-grasp[:3, 3], axis=-1)
    rotation = Rotation.from_matrix(actual[:, :3, :3] @ grasp[:3, :3].T).magnitude()
    profile = pool.contact_profile
    indices = np.flatnonzero((position <= profile.position_tolerance_m)
                             & (rotation <= profile.rotation_tolerance_rad))
    reasons = ['independent_fk_error']*len(values)
    valid = np.zeros(len(values), dtype=bool)
    if len(indices):
        passed, details = validate_rows(validator, solutions[indices.tolist()])
        valid[indices] = passed
        for index, reason in zip(indices, details):
            reasons[index] = reason
    if valid.any():
        index = int(np.flatnonzero(valid)[0])
        q = solutions[index]
        return q.clone(), {'kind': 'bounded_ik_with_local_table_check',
            'independent_position_error_m': float(position[index]),
            'independent_rotation_error_rad': float(rotation[index]),
            'native_full_world_clear': bool(context.checker.validate(q[None, None]).all())}
    return None, {'kind': 'no_local_grasp_endpoint',
                  'kinematic_solutions': len(solutions), 'reasons': reasons}


def branch_pool(base):
    class BranchPool(base):
        vectorized_fallback = False

        def grasp_endpoint(self, context, state, grasp, start, validator):
            if not self.vectorized_fallback:
                return super().grasp_endpoint(context, state, grasp, start, validator)
            return fallback_endpoint(self, context, grasp, start, validator)
    return BranchPool
