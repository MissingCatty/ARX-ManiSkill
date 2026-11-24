"""The same finite contact-preserving proposal policy for every pending object."""
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from .advisor36_geometry import pose


def distinct(a, b):
    return (np.linalg.norm(a[:3, 3] - b[:3, 3]) >= .001 or
            Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude() >= np.deg2rad(.5))


def generate(item, original, tried, count=60):
    # Surface families and their frozen 3/5-degree normal thresholds originate
    # in the geometry inventory; this expansion has no object-name branches.
    parents = []
    for candidate in sorted(original['candidates'], key=lambda c: (c['cost'], c['index'])):
        if not candidate['pass_local_geometry']:
            continue
        g = np.array(candidate['Gcanonical_object'])
        if all(distinct(g, np.array(p['Gcanonical_object'])) for p in parents):
            parents.append(candidate)
        if len(parents) == 64:
            break
    candidates = []
    old = np.array(item['historical_G_object'])
    for parent in parents:
        base = np.array(parent['Gcanonical_object'])
        for degrees in (0., 6., -6., 12., -12., 20., -20., 30., -30., 45., -45., 60., -60., 75., -75.):
            frame = base.copy()
            frame[:3, :3] = base[:3, :3] @ Rotation.from_rotvec([0., np.deg2rad(degrees), 0.]).as_matrix()
            g = pose(np.array(parent['pad_center_object']), base[:3, 1], frame)
            if not all(distinct(g, previous) for previous in tried):
                continue
            if not all(distinct(g, np.array(c['Gcanonical_object'])) for c in candidates):
                continue
            row = copy.deepcopy(parent)
            angle = np.rad2deg(Rotation.from_matrix(old[:3, :3].T @ g[:3, :3]).magnitude())
            distance = np.linalg.norm(g[:3, 3] - old[:3, 3])
            row.update(index=380000 + len(candidates), parent_index=parent['index'],
                       Gcanonical_object=g.tolist(), closing_axis_roll_degrees=degrees,
                       goal_position_change_m=float(distance), goal_rotation_change_degrees=float(angle),
                       cost=float(distance * 1000 + angle * .3),
                       method='uniform_contact_preserving_roll_and_cross_section_pool')
            candidates.append(row)
    # A finite pool spans poses, instead of using all 60 slots near one pose.
    remaining = sorted(candidates, key=lambda c: (c['cost'], c['index']))
    selected = []
    if remaining:
        matrices = np.array([c['Gcanonical_object'] for c in remaining])
        nearest, used, index = np.full(len(remaining), np.inf), set(), 0
        for _ in range(min(count, len(remaining))):
            selected.append(remaining[index])
            used.add(index)
            trace = np.einsum('nij,ij->n', matrices[:, :3, :3], matrices[index, :3, :3])
            angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
            distance = np.linalg.norm(matrices[:, :3, 3] - matrices[index, :3, 3], axis=1)
            nearest = np.minimum(nearest, distance / .010 + angle / np.deg2rad(15.))
            nearest[list(used)] = -np.inf
            index = int(np.argmax(nearest))
    return {'object_id': item['object_id'], 'candidates': selected,
            'expanded_distinct_candidates': len(candidates), 'registered_pool_count': len(selected),
            'maximum_pool_count': count, 'normal_thresholds_unchanged': True,
            'contact_points_and_normals_preserved_under_closing_axis_roll': True,
            'physical_trials': 0, 'old_candidates_excluded': len(tried)}
