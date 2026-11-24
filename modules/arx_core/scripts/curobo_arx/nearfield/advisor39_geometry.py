"""Progressively finer real-surface candidates, independent of object names."""
import copy
import itertools
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .advisor36_geometry import Surface, pose
from .advisor38_geometry import distinct


def triggered(initial_records, pool, physical_candidates, released):
    if released or physical_candidates >= 15:
        return False
    expected = {r['index'] for r in pool}
    rows = [r for r in initial_records if r['candidate_index'] in expected]
    if len({r['candidate_index'] for r in rows}) != len(rows):
        raise ValueError('Repeated initial screening identity')
    return len(expected) == 60 and {r['candidate_index'] for r in rows} == expected and not any(r['pass'] for r in rows)


def dense_candidates(item, original, mesh, previous, level, cap=4096, deadline=float('inf')):
    if not 1 <= level <= 8:
        raise ValueError('Unregistered geometry density level')
    surface = Surface(mesh)
    anchors = []
    for row in sorted(original['candidates'], key=lambda r: (r['cost'], r['index'])):
        if row['pass_local_geometry'] and all(distinct(np.array(row['Gcanonical_object']), np.array(a['Gcanonical_object'])) for a in anchors):
            anchors.append(row)
        if len(anchors) == 64:
            break
    # All valid contact sections are eligible; offset/angle grids are
    # interleaved across anchors, rather than exhausting one anchor first.
    scale = 2 ** ((level - 1) // 2)
    step, degrees = .0015 / scale, 2.5 / scale
    radius = 1 if level % 2 else 2
    grid = sorted(itertools.product(*([range(-radius, radius + 1)] * 4), range(-30*scale, 30*scale + 1)),
                  key=lambda r: (sum(x*x for x in r[:4]) + .01*r[4]*r[4], r))
    existing = [np.asarray(g) for g in previous]
    old = np.asarray(item['historical_G_object'])
    count = 0
    for dx, dz, tilt_x, tilt_z, roll in grid:
        if not any((dx, dz, tilt_x, tilt_z, roll)):
            continue
        for anchor in anchors:
            if count >= cap or time.time() >= deadline:
                return
            frame = np.asarray(anchor['Gcanonical_object'])
            # A diametric sphere grasp retains its fitted center. Latitude
            # changes come from tilting the closing diameter, not from moving
            # the object or inventing off-surface contacts.
            sphere = anchor['method'].startswith('sphere_')
            if sphere and (dx or dz):
                continue
            center = np.asarray(anchor['pad_center_object']) + (dx * frame[:3, 0] + dz * frame[:3, 2]) * step
            rotated = frame.copy()
            rotated[:3, :3] = frame[:3, :3] @ Rotation.from_rotvec(
                np.deg2rad([tilt_x * degrees, roll * degrees, tilt_z * degrees])).as_matrix()
            pair = surface.pair(center, rotated[:3, 1])
            if pair is None or max(pair['angles']) > anchor['normal_limit_degrees'] or not .004 < pair['width_m'] < .1:
                continue
            goal = pose(pair['center'], pair['closing'], rotated)
            if any(not distinct(goal, g) for g in existing):
                continue
            existing.append(goal)
            angle = float(np.rad2deg(Rotation.from_matrix(old[:3, :3].T @ goal[:3, :3]).magnitude()))
            distance = float(np.linalg.norm(goal[:3, 3] - old[:3, 3]))
            row = copy.deepcopy(anchor)
            row.update(index=39000000 + level * 100000 + count, parent_index=anchor['index'],
                method='dense_actual_surface_reintersection', parent_method=anchor['method'],
                density_level=level, section_grid_m=step, orientation_grid_degrees=degrees,
                grid_offset=[dx, dz, tilt_x, tilt_z, roll], pass_local_geometry=True,
                contact_normal_errors_degrees=pair['angles'],
                normal_opposition_error_degrees=pair['normal_opposition_error_degrees'],
                contact_width_m=pair['width_m'], Gcanonical_object=goal.tolist(),
                pad_center_object=pair['center'].tolist(),
                hits=[{'point': h['point'].tolist(), 'normal': h['normal'].tolist(), 'face': h['face']} for h in pair['hits']],
                goal_position_change_m=distance, goal_rotation_change_degrees=angle,
                cost=distance*1000 + angle*.3)
            count += 1
            yield row
