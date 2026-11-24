"""Stage- and surface-specific fingertip/table policy, without world edits."""
from dataclasses import dataclass

import numpy as np

from geometry import FINGERS, box_depth

LOCAL_STAGES = frozenset(('final_approach', 'closure'))


@dataclass(frozen=True)
class TableContactPolicy:
    tolerance_m: float
    tip_plane_eef_z_m: float
    distal_length_m: float = .010

    def __post_init__(self):
        if not (0 <= self.tolerance_m <= .002):
            raise ValueError('Registered tolerance range is 0..2 mm')
        if not (.020 < self.tip_plane_eef_z_m < .080 and 0 < self.distal_length_m <= .010):
            raise ValueError('Invalid authored fingertip region')

    def evaluate(self, spheres, eef_world, sphere_names, table, *, stage, other_world_clear):
        """Require every intersecting sphere cap to remain inside the distal region.

        Caller supplies bounds and all non-table world checks. No table lowering,
        sphere shrinking or generic per-finger exception is used.
        """
        spheres = np.asarray(spheres, dtype=float)
        eef = np.asarray(eef_world, dtype=float)
        other = np.asarray(other_world_clear, dtype=bool).reshape(-1)
        if spheres.ndim != 3 or spheres.shape[-1] != 4 or eef.shape != (len(spheres), 4, 4):
            raise ValueError('Invalid batched geometry shapes')
        if len(sphere_names) != spheres.shape[1] or len(other) != len(spheres):
            raise ValueError('Incomplete collision receipt')
        center, half = np.asarray(table['center']), np.asarray(table['dims']) / 2
        if center.shape != (3,) or half.shape != (3,) or not np.isfinite([*center, *half]).all() or not (half > 0).all():
            raise ValueError('Invalid table geometry')
        finite = np.isfinite(spheres).all(axis=(1, 2)) & np.isfinite(eef).all(axis=(1, 2))
        rotations = eef[:, :3, :3]
        rigid = (np.max(np.abs(rotations @ rotations.transpose(0, 2, 1) - np.eye(3)), axis=(1, 2)) < 1e-5)
        rigid &= np.abs(np.linalg.det(rotations) - 1.) < 1e-5
        rigid &= np.max(np.abs(eef[:, 3] - np.array([0., 0., 0., 1.])), axis=1) < 1e-8
        finite &= rigid
        penetration = box_depth(spheres, table)
        xyz, radius = spheres[..., :3], spheres[..., 3]
        height = xyz[..., 2] - (center[2] + half[2])
        inside_xy = (np.abs(xyz[..., :2] - center[:2]) <= half[:2]).all(axis=-1)
        finger = np.isin(np.asarray(sphere_names), FINGERS)[None]
        # Minimum EEF-Z of the spherical cap that lies at/below the table top.
        axial_z = eef[:, 2, 2:3]
        axial_xy_norm = np.linalg.norm(eef[:, :2, 2], axis=-1)[:, None]
        cap_radius = np.sqrt(np.maximum(radius ** 2 - height ** 2, 0))
        minimum_delta = np.where(radius * axial_z >= height, -radius,
                                 -axial_z * height - axial_xy_norm * cap_radius)
        axial_center = np.einsum('bsi,bi->bs', xyz - eef[:, None, :3, 3], eef[:, :3, 2])
        cap_min_z = axial_center + minimum_delta
        allowed = (finger & inside_xy & (height >= 0) & (radius > 0)
                   & (cap_min_z >= self.tip_plane_eef_z_m - self.distal_length_m))
        if stage not in LOCAL_STAGES:
            allowed[:] = False
        strict_max = np.max(np.where(allowed, -np.inf, penetration), axis=1)
        tip_max = np.max(np.where(allowed, penetration, -np.inf), axis=1)
        all_finger_max = np.max(np.where(finger, penetration, -np.inf), axis=1)
        nonfinger_max = np.max(np.where(~finger, penetration, -np.inf), axis=1)
        valid = finite & other & (strict_max <= 0) & (tip_max <= self.tolerance_m)
        return {'valid': valid, 'table_depth_m': penetration.max(axis=1),
                'strict_depth_m': strict_max, 'tip_depth_m': tip_max,
                'all_finger_depth_m': all_finger_max, 'nonfinger_depth_m': nonfinger_max,
                'allowed_contact_spheres': allowed, 'penetration_m': penetration,
                'cap_min_eef_z_m': cap_min_z, 'finite': finite}
