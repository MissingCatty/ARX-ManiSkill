"""Explicit local-path adapter; upstream cuRobo scene and models stay intact."""
import numpy as np
from dataclasses import replace

from .geometry import Geometry
from .narrow_phase import ConvexFingerPolicy
from .profile import load_profile
from .physical_contact import ReleaseEnvelope


class LocalContactValidator:
    """Combine original native non-table checks with the exact local table gate."""
    def __init__(self, context, state, other_checker, asset, tolerance_m=None, profile=None):
        self.context = context
        self.other_checker = other_checker
        self.geometry = Geometry(context.model, state)
        self.profile = profile or load_profile()
        if tolerance_m is not None:
            self.profile = replace(self.profile, maximum_penetration_m=tolerance_m)
        self.policy = ConvexFingerPolicy(self.geometry, state['scene_cuboids']['table'],
                                        self.profile.maximum_penetration_m, asset, self.profile.distal_length_m)
        self.receipts = []

    def update(self, context, state):
        self.context = context
        self.geometry.base = np.asarray(state['links']['body4'], dtype=float)
        measured = dict(zip(state['joint_names'], state['qpos']))
        self.geometry.locked = {name: measured[name] for name in self.geometry.locked}
        self.policy.table = state['scene_cuboids']['table']
        self.policy.center = np.asarray(self.policy.table['center'], dtype=float)
        self.policy.half = np.asarray(self.policy.table['dims'], dtype=float)/2
        self.receipts = []

    def validate(self, joints, *, stage):
        import torch
        q = torch.as_tensor(joints, dtype=torch.float32, device=self.context.limits.device).reshape(-1, 6).clone()
        if not torch.isfinite(q).all():
            raise ValueError('Nonfinite contact-validation joint state')
        other = self.other_checker.validate(q[:, None]).reshape(-1).detach().cpu().numpy()
        bounds = (torch.isfinite(q).all(-1) & (q >= self.context.limits[:, 0]-1e-6).all(-1)
                  & (q <= self.context.limits[:, 1]+1e-6).all(-1)).cpu().numpy()
        native = self.context.checker.validate(q[:, None]).reshape(-1).detach().cpu().numpy()
        values = q.cpu().numpy()
        valid = native & bounds & other
        receipts = []
        selected = bounds & other if stage == 'lift_release' else ~valid & bounds & other
        for i in np.flatnonzero(selected):
            record = self.policy.evaluate_one(values[i], stage='closure' if stage == 'lift_release' else stage,
                other_world_clear=True, require_exact=stage == 'lift_release')
            record['stage'] = stage
            valid[i] = record['valid']
            receipts.append({'index': int(i), **record})
        self.receipts.append({'stage': stage, 'states': len(q), 'native_clear': int(native.sum()),
                              'local_clear': int(valid.sum()), 'other_world_clear': int(other.sum()),
                              'joint_bounds_clear': int(bounds.sum()), 'narrow_phase': receipts})
        return valid

    def validate_trajectory(self, path, *, stage, release_anchor_m=None):
        """Recheck optimized samples and their dense interpolants in time order."""
        q = np.asarray(path['q'], dtype=float)
        times = np.asarray(path['times'], dtype=float)
        if q.ndim != 2 or q.shape[1] != 6 or len(q) < 2 or times.shape != (len(q),):
            raise ValueError('Invalid timed contact trajectory')
        if not np.isfinite(q).all() or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError('Invalid trajectory values or timing')
        counts = np.maximum(1, np.ceil(np.max(np.abs(np.diff(q, axis=0)), axis=1)/.0005).astype(int))
        if counts.sum() > 40000:
            raise ValueError('Unbounded contact trajectory')
        dense = np.concatenate([q[i] + np.arange(n)[:, None]/n*(q[i+1]-q[i])
                                for i, n in enumerate(counts)] + [q[-1:]])
        return self.check_stream(dense, stage=stage, release_anchor_m=release_anchor_m) and self.check_stream(q, stage=stage, release_anchor_m=release_anchor_m)

    def check_stream(self, values, *, stage, release_anchor_m=None):
        envelope = ReleaseEnvelope(self.profile) if stage == 'lift_release' else None
        if envelope is not None:
            if release_anchor_m is None:
                raise ValueError('Measured closure contact anchor is required for release')
            envelope.anchor(release_anchor_m)
        for begin in range(0, len(values), 128):
            batch = values[begin:begin+128]
            if not self.validate(batch, stage=stage).all():
                return False
            if envelope is not None:
                records = {row['index']: row for row in self.receipts[-1]['narrow_phase']}
                for index in range(len(batch)):
                    depth = records.get(index, {}).get('maximum_real_finger_depth_m', 0.)
                    if not envelope.accept(depth):
                        self.receipts[-1]['release_failure'] = {'index': index, 'depth_m': depth,
                            'ceiling_m': envelope.ceiling, 'reason': 'contact_depth_increased_during_release'}
                        return False
        return True

    def straight_approach(self, q_pre, q_grasp, *, duration_s=5., control_dt=.025,
                          stage='final_approach', release_anchor_m=None):
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
        if not self.check_stream(check_q, stage=stage, release_anchor_m=release_anchor_m):
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
        if not self.check_stream(q, stage=stage, release_anchor_m=release_anchor_m):
            return None
        return {'q': q.tolist(), 'times': times.tolist(), 'connector': 'dense_validated_quintic_joint_connection',
                'validation_states': n, 'maximum_validation_joint_step_rad': float(np.max(np.abs(np.diff(check_q, axis=0)))),
                'time_parameterization': 'quintic_zero_endpoint_velocity_acceleration',
                'max_authored_velocity_fraction': float(np.max(1.875*delta/(times[-1]*velocity))),
                'max_analytic_acceleration_rad_s2': float(np.max((10/np.sqrt(3))*delta/times[-1]**2)),
                'max_analytic_jerk_rad_s3': float(np.max(60*delta/times[-1]**3))}
