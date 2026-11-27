"""Derive the three-parallel/two-intersecting decomposition from the real URDF."""
import numpy as np

from modules.curobo.indexed_planning.geometric.kinematics import ArmChain


class AnalyticModel:
    def __init__(self, arm='right'):
        self.chain = ArmChain(arm)
        self.signs = np.ones(6)
        for index in [2, 3]:
            axis, reference = self.chain.H[:, index], self.chain.H[:, 1]
            if np.linalg.norm(np.cross(axis, reference)) > 1e-9:
                raise ValueError('ARX no longer has three parallel inner axes')
            self.signs[index] = 1 if axis @ reference > 0 else -1
        self.H = self.chain.H*self.signs
        self.H[:, 2:4] = self.H[:, 1:2]
        self.P = self.chain.P.copy()
        a, b = self.H[:, 4], self.H[:, 5]
        shifts = np.linalg.lstsq(np.column_stack([a, -b]), self.P[:, 5], rcond=None)[0]
        self.intersection_residual_m = float(np.linalg.norm(a*shifts[0]-b*shifts[1]-self.P[:, 5]))
        if self.intersection_residual_m > 1e-9 or np.linalg.norm(np.cross(a, b)) < 1e-8:
            raise ValueError('Last two joint axes do not have a unique intersection')
        # Move each origin along its own axis; the physical axis line is unchanged.
        self.P[:, 4] += a*shifts[0]
        self.P[:, 5] += b*shifts[1]-a*shifts[0]
        self.P[:, 6] -= b*shifts[1]
        self.P[:, 5] = 0.
        if np.any(np.ptp(self.chain.limits, axis=1) >= 2*np.pi):
            raise ValueError('This fixed-capacity implementation requires limits narrower than one turn')
        self.origin_shifts = shifts

    def receipt(self):
        return {'arm': self.chain.arm.name, 'urdf_sha256': self.chain.urdf_sha256,
            'family': 'three_parallel_inner_axes_and_two_intersecting_wrist_axes',
            'axis_signs': self.signs.tolist(), 'axis_5_6_intersection_error_m': self.intersection_residual_m,
            'origin_shifts_along_own_axes_m': self.origin_shifts.tolist(),
            'H': self.H.tolist(), 'P': self.P.tolist(), 'R_tool': self.chain.R_tool.tolist(),
            'physical_limits': self.chain.limits.tolist(), 'regular_branch_capacity': 8,
            'singular_family_requires_fallback': True}
