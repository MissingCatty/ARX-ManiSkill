"""Fixed tabletop sampling geometry, independent of inverse kinematics."""
import numpy as np
from scipy.spatial.transform import Rotation

from workspace_geometry import reach_bound, world_corners


def define_region(state, urdf):
    table = state['scene_cuboids']['table']
    center, dims = np.asarray(table['center']), np.asarray(table['dims'])
    lo, hi = center[:2] - dims[:2] / 2, center[:2] + dims[:2] / 2
    shoulder, radius = reach_bound(state, urdf)
    return {'center_roi_min_xy_m': [float(lo[0]), float(center[1] - .40)],
            'center_roi_max_xy_m': [float(center[0]), float(center[1] + .40)],
            'table_min_xy_m': lo.tolist(), 'table_max_xy_m': hi.tolist(),
            'region_rule': 'Robot-side half of the physical table, central 80 cm lateral strip',
            'shoulder_world_m': shoulder.tolist(), 'eef_length_upper_bound_m': radius,
            'contact_offset_m': .043, 'contact_length_upper_bound_m': radius + .043,
            'table_clearance_m': .012, 'head_border_px': 12,
            'near_edge_gap_m': [.09, .11], 'numeric_tolerance_m': .002}


def check_placement(spec, db, state, region):
    points = world_corners(spec, db)
    lo, hi = points.min(0), points.max(0)
    center = (lo + hi) / 2
    tolerance = region['numeric_tolerance_m']
    inside_roi = bool((center[:2] >= np.asarray(region['center_roi_min_xy_m']) - tolerance).all()
                      and (center[:2] <= np.asarray(region['center_roi_max_xy_m']) + tolerance).all())
    on_table = bool((lo[:2] >= np.asarray(region['table_min_xy_m']) + region['table_clearance_m']).all()
                    and (hi[:2] <= np.asarray(region['table_max_xy_m']) - region['table_clearance_m']).all())
    p = np.asarray(spec['pose'])
    rotation = Rotation.from_quat(p[[4, 5, 6, 3]]).as_matrix()
    shoulder = np.asarray(region['shoulder_world_m'])
    local_shoulder = (shoulder - p[:3]) @ rotation
    bbox = db[spec['model_id']]['bbox']
    closest_local = np.clip(local_shoulder, bbox['min'], bbox['max'])
    closest_world = closest_local @ rotation.T + p[:3]
    nearest_distance = float(np.linalg.norm(closest_world - shoulder))
    not_clearly_outside = nearest_distance <= region['contact_length_upper_bound_m'] + tolerance
    camera, k = np.asarray(state['camera_world_optical']), np.asarray(state['intrinsics'])
    xyz = (points - camera[:3, 3]) @ camera[:3, :3]
    in_view = False
    if xyz[:, 2].min() > .10 and xyz[:, 2].max() < 2.:
        uv = xyz @ k.T
        uv = uv[:, :2] / uv[:, 2:]
        in_view = bool((uv.min(0) >= [12, 12]).all() and (uv.max(0) <= [628, 468]).all())
    return {'passed': bool(inside_roi and on_table and not_clearly_outside and in_view),
            'inside_registered_roi': inside_roi, 'inside_table': on_table,
            'fully_in_head_frame': in_view, 'not_clearly_beyond_link_length': bool(not_clearly_outside),
            'center_world_m': center.tolist(), 'center_distance_m': float(np.linalg.norm(center - shoulder)),
            'nearest_bbox_distance_m': nearest_distance,
            'bbox_world_min_m': lo.tolist(), 'bbox_world_max_m': hi.tolist(),
            'near_table_edge_gap_m': float(lo[0] - region['table_min_xy_m'][0])}


def sample_target(entry, baseline, region, rng):
    """The only input scene fields are layout and object identity."""
    yaw = float(rng.uniform(-np.pi, np.pi))
    center = rng.uniform(region['center_roi_min_xy_m'], region['center_roi_max_xy_m'])
    if entry['layout'] == 'S2_edge':
        temporary = baseline.actor_spec('target', entry['model_id'], [0., 0.], yaw)
        lo, hi = baseline.bounds(temporary)
        center[0] = region['table_min_xy_m'][0] + (hi[0] - lo[0]) / 2 + rng.uniform(*region['near_edge_gap_m'])
    return baseline.actor_spec('target', entry['model_id'], center, yaw)


def edge_passed(spec, db, region):
    gap = float(world_corners(spec, db)[:, 0].min() - region['table_min_xy_m'][0])
    return region['near_edge_gap_m'][0] - .003 <= gap <= region['near_edge_gap_m'][1] + .003
