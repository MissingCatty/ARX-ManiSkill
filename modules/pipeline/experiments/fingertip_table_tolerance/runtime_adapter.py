"""Explicit local-path adapter; upstream cuRobo scene and models stay intact."""
import numpy as np

from geometry import Geometry
from narrow_phase import ConvexFingerPolicy


class LocalContactValidator:
    """Combine original native non-table checks with the exact local table gate."""
    def __init__(self, context, state, other_checker, asset, tolerance_m):
        self.context = context
        self.other_checker = other_checker
        self.geometry = Geometry(context.model, state)
        self.policy = ConvexFingerPolicy(self.geometry, state['scene_cuboids']['table'], tolerance_m, asset)
        self.receipts = []

    def validate(self, joints, *, stage):
        import torch
        q = torch.as_tensor(joints, dtype=torch.float32, device='cuda').reshape(-1, 6).clone()
        other = self.other_checker.validate(q[:, None]).reshape(-1).detach().cpu().numpy()
        bounds = ((q >= self.context.limits[:, 0]-1e-6) & (q <= self.context.limits[:, 1]+1e-6)).all(-1).cpu().numpy()
        native = self.context.checker.validate(q[:, None]).reshape(-1).detach().cpu().numpy()
        values = q.cpu().numpy()
        valid = native & bounds & other
        receipts = []
        for i in np.flatnonzero(~valid & bounds & other):
            record = self.policy.evaluate_one(values[i], stage=stage, other_world_clear=True)
            valid[i] = record['valid']
            receipts.append({'index': int(i), **record})
        self.receipts.append({'stage': stage, 'states': len(q), 'native_clear': int(native.sum()),
                              'local_clear': int(valid.sum()), 'narrow_phase': receipts})
        return valid

    def straight_approach(self, q_pre, q_grasp, *, duration_s=5., control_dt=.025):
        """Dense, timed joint-space connection; no unchecked optimizer output.

        This is a local connector, not a replacement for upstream PRM. The
        original checker handles the initial reach. Quintic timing gives zero
        endpoint velocity/acceleration. Intermediate joint limits and native
        non-table collisions are checked at <=0.5 mrad per joint.
        """
        start, end = np.asarray(q_pre), np.asarray(q_grasp)
        if start.shape != (6,) or end.shape != (6,): raise ValueError('Six-joint endpoints required')
        n = max(1001, int(np.ceil(np.max(np.abs(end-start))/.0005))+1)
        if n > 40001: raise ValueError('Unbounded local connection')
        alpha = np.linspace(0., 1., n)
        check_q = start[None]+alpha[:, None]*(end-start)
        # The entire 8 cm pregrasp-to-grasp connector is the final approach;
        # the local geometric exception still only concerns distal tip contact.
        for begin in range(0, len(check_q), 128):
            if not self.validate(check_q[begin:begin+128], stage='final_approach').all():
                return None
        # Conservative authored joint velocity limit, with 20% reserve.
        velocity = np.array([float(self.geometry.joints[n].find('limit').get('velocity')) for n in self.geometry.config['cspace']['joint_names'] if n in self.geometry.joints and n in [f'right_joint{i}' for i in range(21, 27)]])
        if velocity.shape != (6,): raise ValueError('Incomplete authored velocity limits')
        delta = np.abs(end-start)
        duration_s = max(float(duration_s), float(np.max(1.875*delta/(.8*velocity))),
                         float(np.max(np.sqrt((10/np.sqrt(3))*delta/5.))),
                         float(np.max(np.cbrt(60*delta/100.))))
        steps = int(np.ceil(duration_s/control_dt)); times = np.arange(steps+1)*control_dt
        u = times/times[-1]
        fraction = 10*u**3-15*u**4+6*u**5
        q = start[None]+fraction[:, None]*(end-start)
        # Execution samples themselves are also checked and retain receipts.
        for begin in range(0, len(q), 128):
            if not self.validate(q[begin:begin+128], stage='final_approach').all(): return None
        return {'q': q.tolist(), 'times': times.tolist(), 'connector': 'dense_validated_quintic_joint_connection',
                'validation_states': n, 'maximum_validation_joint_step_rad': float(np.max(np.abs(np.diff(check_q, axis=0)))),
                'time_parameterization': 'quintic_zero_endpoint_velocity_acceleration',
                'max_authored_velocity_fraction': float(np.max(1.875*delta/(times[-1]*velocity))),
                'max_analytic_acceleration_rad_s2': float(np.max((10/np.sqrt(3))*delta/times[-1]**2)),
                'max_analytic_jerk_rad_s3': float(np.max(60*delta/times[-1]**3))}
