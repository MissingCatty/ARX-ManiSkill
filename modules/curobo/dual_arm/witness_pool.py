"""Reuse independently verified endpoint joints; retain native path planning."""
import numpy as np
from scipy.spatial.transform import Rotation


def witness_pool(base):
    class WitnessPool(base):
        def grasp_endpoint(self, context, state, grasp, start, validator):
            import torch
            witness = state.get('kinematic_endpoint_witness')
            if witness is None or not np.allclose(witness['eef_target_world'], grasp, atol=1e-10, rtol=0):
                return super().grasp_endpoint(context, state, grasp, start, validator)
            q = np.asarray(witness['q'], dtype=float)
            if q.shape != (6,) or not np.isfinite(q).all():
                raise ValueError('Invalid supplied kinematic witness')
            limits = context.limits.detach().cpu().numpy()
            if not ((q >= limits[:, 0]) & (q <= limits[:, 1])).all():
                raise ValueError('Supplied kinematic witness exceeds current joint limits')
            actual = validator.geometry.frames(q[None])['right_eef_link'][0]
            pe = float(np.linalg.norm(actual[:3, 3]-grasp[:3, 3]))
            re = float(Rotation.from_matrix(actual[:3, :3].T@grasp[:3, :3]).magnitude())
            profile = self.contact_profile
            if pe > profile.position_tolerance_m or re > profile.rotation_tolerance_rad:
                raise ValueError('Supplied kinematic witness failed independent FK')
            value = torch.as_tensor(q, dtype=torch.float32, device=start.device).clone()
            if not bool(validator.validate(value[None], stage='final_approach').all()):
                # Another joint branch may still work; keep the original fallback.
                return super().grasp_endpoint(context, state, grasp, start, validator)
            return value, {'kind': 'verified_map_kinematic_witness',
                'source_map_row': witness['source_map_row'],
                'independent_position_error_m': pe, 'independent_rotation_error_rad': re,
                'native_full_world_clear': bool(context.checker.validate(value[None, None]).all()),
                'original_kinematic_tolerances': True, 'new_grasp_IK_calls': 0}
    return WitnessPool
