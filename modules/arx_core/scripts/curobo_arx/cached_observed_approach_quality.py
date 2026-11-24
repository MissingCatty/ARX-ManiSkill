"""Exact union-of-spheres cache for the fixed ARX nominal approach feature.

This is supplementary observed-target risk evidence, never a planner model.
"""
import numpy as np
from scipy.spatial import cKDTree
from observed_early_approach_quality import ObservedEarlyApproachQuality, PROXY


class CachedObservedApproachQuality(ObservedEarlyApproachQuality):
    def __init__(self, folder):
        super().__init__(folder)
        remaining = .08 * (1 - np.linspace(0, 1, 17))
        self.remaining = remaining[remaining >= .010 - 1e-10]
        self.swept = {}
        with np.load(PROXY) as data:
            for link in ('right_link27', 'right_link28'):
                body = data[link] + np.asarray(self.meta['motion_axis_by_link'][link]) * self.open_state[link]
                cloud = np.concatenate([body - np.array([0., 0., distance]) for distance in self.remaining])
                self.swept[link] = cKDTree(cloud)

    def evaluate(self, grasp, pregrasp):
        grasp, pregrasp = np.asarray(grasp), np.asarray(pregrasp)
        normal = (np.allclose(grasp[:3, :3], pregrasp[:3, :3], atol=1e-10, rtol=0)
                  and np.allclose(pregrasp[:3, 3] - grasp[:3, 3], -.08 * grasp[:3, 2], atol=1e-10, rtol=0))
        if not normal:
            return super().evaluate(grasp, pregrasp)
        local = (self.points - grasp[:3, 3]) @ grasp[:3, :3]
        masks = [tree.query(local, workers=1)[0] <= self.meta['sample_cover_radius_m']
                 for tree in self.swept.values()]
        contact = masks[0] | masks[1]
        fraction = float(contact.mean())
        return dict(finger_sweep_point_fraction=fraction, finger_sweep_point_count=int(contact.sum()),
                    observed_points=len(self.points), potential_open_finger_contact=fraction >= .01,
                    terminal_contact_allowance_m=.010, nominal_approach_samples=17,
                    measured_open_finger_state_m=self.open_state,
                    proxy_data_sha256=self.meta['data_sha256'], cache='exact nominal translation union',
                    scope='Observed target and fixed sampled ARX CAD only; no hidden-surface, actual-path or force-closure certification.')
