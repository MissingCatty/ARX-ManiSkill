"""Exact broad-phase acceleration of the registered observed ARX feature.

Reuse the original Slerp, floating-point distances and cover radius. A point
outside the CAD bounding box expanded by the cover radius cannot overlap any
CAD sample sphere. This is a quality-feature calculation, not a model edit.
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from observed_early_approach_quality import ObservedEarlyApproachQuality


class BoundedObservedApproachQuality(ObservedEarlyApproachQuality):
    def evaluate(self, grasp, pregrasp):
        grasp, pregrasp = np.asarray(grasp), np.asarray(pregrasp)
        alpha = np.linspace(0, 1, 17)
        rotations = Slerp([0, 1], Rotation.from_matrix(np.stack([pregrasp[:3, :3], grasp[:3, :3]])))(alpha).as_matrix()
        translations = pregrasp[:3, 3][None] * (1 - alpha[:, None]) + grasp[:3, 3][None] * alpha[:, None]
        touched = np.zeros(len(self.points), dtype=bool)
        radius = self.meta['sample_cover_radius_m']
        for rotation, translation in zip(rotations, translations):
            if float(np.linalg.norm(translation - grasp[:3, 3])) < .010 - 1e-10:
                continue
            local = (self.points - translation) @ rotation
            for link in ('right_link27', 'right_link28'):
                tree = self.trees[link]
                query = local - np.asarray(self.meta['motion_axis_by_link'][link]) * self.open_state[link]
                possible = np.all((query >= tree.mins - radius) & (query <= tree.maxes + radius), axis=1)
                ids = np.flatnonzero(possible)
                if len(ids):
                    touched[ids] |= tree.query(query[ids], workers=1)[0] <= radius
        fraction = float(touched.mean())
        return dict(finger_sweep_point_fraction=fraction, finger_sweep_point_count=int(touched.sum()),
                    observed_points=len(self.points), potential_open_finger_contact=fraction >= .01,
                    terminal_contact_allowance_m=.010, nominal_approach_samples=17,
                    measured_open_finger_state_m=self.open_state,
                    proxy_data_sha256=self.meta['data_sha256'], acceleration='Exact expanded-CAD-bounding-box broad phase',
                    scope='Observed target and fixed sampled ARX CAD only; no hidden-surface, actual-path or force-closure certification.')
