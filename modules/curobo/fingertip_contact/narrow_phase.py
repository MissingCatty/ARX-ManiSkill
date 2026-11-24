"""Local exact convex-collider checks after a finger/table sphere rejection."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

from .contact_policy import LOCAL_STAGES
from .geometry import FINGERS, Geometry, box_depth, collision_meshes, tip_plane


def extreme_points(points):
    points = np.unique(np.asarray(points), axis=0)
    if not len(points) or not np.isfinite(points).all():
        raise ValueError('Empty or invalid original collider')
    centered = points-points.mean(0)
    rank = np.linalg.matrix_rank(centered)
    # The frozen export includes planar surface pieces. Keep those exact
    # polygons rather than dropping them or inventing a thickness.
    if rank == 3: return points[ConvexHull(points).vertices]
    if rank == 2:
        _, _, basis = np.linalg.svd(centered, full_matrices=False)
        return points[ConvexHull(centered @ basis[:2].T).vertices]
    if rank == 1:
        _, _, basis = np.linalg.svd(centered, full_matrices=False)
        coordinate = centered @ basis[0]
        return points[[coordinate.argmin(), coordinate.argmax()]]
    return points[:1]


def linear_extreme(vertices, objective, center, half, *, include_z=False, equality=None):
    axes = 3 if include_z else 2
    a = np.concatenate([vertices[:, :axes].T, -vertices[:, :axes].T])
    b = np.concatenate([center[:axes]+half[:axes], -center[:axes]+half[:axes]])
    eq_a = [np.ones(len(vertices))]; eq_b = [1.]
    if equality is not None:
        axis, value = equality
        eq_a.append(vertices[:, axis]); eq_b.append(value)
    result = linprog(vertices @ objective, A_ub=a, b_ub=b,
                     A_eq=np.asarray(eq_a), b_eq=eq_b, bounds=(0., None), method='highs')
    if result.status == 2: return None
    if not result.success: raise RuntimeError('Convex/table intersection failed: ' + result.message)
    return result.x @ vertices


@dataclass
class ConvexFingerPolicy:
    geometry: Geometry
    table: dict
    tolerance_m: float
    asset: object
    distal_length_m: float = .010

    def __post_init__(self):
        if not (0 <= self.tolerance_m <= .002 and 0 < self.distal_length_m <= .010):
            raise ValueError('Unregistered local tolerance or fingertip region')
        meshes, self.source_paths = collision_meshes(self.asset)
        self.parts = {name: [extreme_points(p) for p in meshes[name]] for name in FINGERS}
        self.tip, self.per_finger_tip = tip_plane(self.geometry, meshes)
        self.center = np.asarray(self.table['center'], dtype=float)
        self.half = np.asarray(self.table['dims'], dtype=float)/2
        if not np.isfinite([*self.center, *self.half]).all() or not (self.half > 0).all():
            raise ValueError('Invalid table')

    def evaluate_one(self, q, *, stage, other_world_clear, require_exact=False):
        q = np.asarray(q, dtype=float)
        result = {'valid': False, 'stage': stage, 'tolerance_m': self.tolerance_m,
                  'reason': None, 'native_table_clear': False, 'mesh_refinement_used': False,
                  'actual_tip_contact': False, 'maximum_real_finger_depth_m': 0., 'contacts': []}
        if q.shape != (6,) or not np.isfinite(q).all():
            result['reason'] = 'invalid_joint_state'; return result
        if not ((q >= self.geometry.lower-1e-6) & (q <= self.geometry.upper+1e-6)).all():
            result['reason'] = 'joint_limits'; return result
        if not other_world_clear:
            result['reason'] = 'other_world_collision'; return result
        frames = self.geometry.frames(q[None])
        spheres = self.geometry.spheres(frames)
        depths = box_depth(spheres, self.table)[0]
        result['maximum_native_sphere_depth_m'] = float(depths.max())
        if depths.max() <= 0 and not require_exact:
            result.update(valid=True, native_table_clear=True, reason='native_clear'); return result
        result['native_table_clear'] = bool(depths.max() <= 0)
        if stage not in LOCAL_STAGES:
            result['reason'] = 'nonlocal_stage_table_collision'; return result
        fingers = np.isin(np.asarray(self.geometry.names), FINGERS)
        if (depths[~fingers] > 0).any():
            result['reason'] = 'nonfinger_table_collision'; return result
        result['mesh_refinement_used'] = True
        eef = frames['right_eef_link'][0]
        top = self.center[2]+self.half[2]
        lower, upper = self.center-self.half, self.center+self.half
        for name, parts in self.parts.items():
            frame = frames[name][0]
            for index, part in enumerate(parts):
                v = part @ frame[:3, :3].T+frame[:3, 3]
                if (v.max(0) < lower).any() or (v.min(0) > upper).any(): continue
                inside_xy = ((v[:, :2] >= lower[:2]) & (v[:, :2] <= upper[:2])).all()
                point = v[v[:, 2].argmin()] if inside_xy else linear_extreme(v, [0., 0., 1.], self.center, self.half)
                if point is None: continue
                depth = float(top-point[2])
                result['maximum_real_finger_depth_m'] = max(result['maximum_real_finger_depth_m'], depth)
                if depth <= 0: continue
                result['actual_tip_contact'] = True
                contact = {'link': name, 'collider': index, 'depth_m': depth, 'minimum_world_point': point.tolist()}
                result['contacts'].append(contact)
                if depth > self.tolerance_m:
                    result['reason'] = 'real_finger_depth_exceeds_tolerance'; return result
                axial_point = linear_extreme(v, eef[:3, 2], self.center, self.half, include_z=True)
                if axial_point is None: raise RuntimeError('Inconsistent convex intersection')
                axial = float((axial_point-eef[:3, 3]) @ eef[:3, 2])
                contact['minimum_contact_eef_z_m'] = axial
                if axial < self.tip-self.distal_length_m:
                    result['reason'] = 'finger_body_contact'; return result
                # A top-face allowance must never release a side-face collision.
                if not inside_xy:
                    for axis in [0, 1]:
                        for boundary in [lower[axis], upper[axis]]:
                            hit = linear_extreme(v, [0., 0., 1.], self.center, self.half,
                                                 include_z=True, equality=(axis, boundary))
                            if hit is not None and hit[2] < top:
                                result['reason'] = 'table_side_contact'; return result
        result.update(valid=True, reason='bounded_real_tip_contact' if result['actual_tip_contact'] else 'sphere_only_false_positive')
        return result
