"""CPU geometry checks for placing scenes inside the fixed ARX work area."""
from itertools import product
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

TABLE_MIN = np.array([-.7245, -1.209])
TABLE_MAX = np.array([.4845, 1.209])
MAX_CENTER_DISTANCE_M = .70
EDGE_GAP_M = (.09, .11)


def reach_bound(state, urdf):
    """Triangle-inequality upper bound; inside the sphere still needs IK."""
    joints = ET.parse(Path(urdf)).getroot().findall('joint')
    parents = {j.find('child').get('link'): j for j in joints}
    chain, link = [], 'right_eef_link'
    while link != 'body4':
        joint = parents[link]
        chain.append(joint)
        link = joint.find('parent').get('link')
    transform = np.asarray(state['links']['body4']).copy()
    pivot, radius = None, 0.
    for joint in reversed(chain):
        if joint.get('type') == 'prismatic':
            raise ValueError('Expected fixed-lift six-axis arm, got prismatic joint')
        origin = joint.find('origin')
        xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
        rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')
        if pivot is None:
            local = np.eye(4)
            local[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
            local[:3, 3] = xyz
            transform = transform @ local
            if joint.get('type') in ('revolute', 'continuous'):
                pivot = transform[:3, 3].copy()
        else:
            radius += float(np.linalg.norm(xyz))
    return pivot, radius


def corners(record):
    bounds = record['bbox']
    return np.array(list(product(*zip(bounds['min'], bounds['max']))))


def world_corners(spec, db):
    p = np.asarray(spec['pose'])
    rotation = Rotation.from_quat(p[[4, 5, 6, 3]]).as_matrix()
    return corners(db[spec['model_id']]) @ rotation.T + p[:3]


def placement_check(spec, db, state, pivot, radius):
    points = world_corners(spec, db)
    lo, hi = points.min(0), points.max(0)
    center = (lo + hi) / 2
    distance = float(np.linalg.norm(center - pivot))
    camera, k = np.asarray(state['camera_world_optical']), np.asarray(state['intrinsics'])
    xyz = (points - camera[:3, 3]) @ camera[:3, :3]
    if (xyz[:, 2] <= .10).any() or (xyz[:, 2] >= 2.).any():
        visible = False
    else:
        uv = xyz @ k.T
        uv = uv[:, :2] / uv[:, 2:]
        visible = bool((uv.min(0) >= [12, 12]).all() and (uv.max(0) <= [628, 468]).all())
    on_table = bool((lo[:2] >= TABLE_MIN + .012).all() and (hi[:2] <= TABLE_MAX - .012).all())
    near_edge_gap = float(lo[0] - TABLE_MIN[0])
    return {'passed': bool(visible and on_table and distance <= MAX_CENTER_DISTANCE_M),
            'fully_in_head_frame': visible, 'inside_table': on_table,
            'center_distance_m': distance, 'center_margin_to_upper_bound_m': radius - distance,
            'maximum_center_distance_m': MAX_CENTER_DISTANCE_M,
            'center_world_m': center.tolist(), 'bbox_world_min_m': lo.tolist(),
            'bbox_world_max_m': hi.tolist(), 'near_table_edge_gap_m': near_edge_gap}


def edge_check(spec, db):
    gap = float(world_corners(spec, db)[:, 0].min() - TABLE_MIN[0])
    return EDGE_GAP_M[0] - .003 <= gap <= EDGE_GAP_M[1] + .003
