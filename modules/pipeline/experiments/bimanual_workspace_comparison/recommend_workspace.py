"""Choose a tabletop TCP region using atlas coverage and measured camera geometry."""
from itertools import product
from pathlib import Path

import numpy as np

from arx_project.layout import ROOT, now, read_json, sha256, write_json
from arx_project.scenes import current_dataset
from analyze import encode, lookup

OUTPUT = ROOT/'tmp/reachability/recommended_workspace_20261125_01'
DATA = ROOT/'tmp/reachability/shared_region_comparison_20261125_01'
STATE = current_dataset()/'cases/D0008_S1_open_teapot/state.json'
HALF_WIDTH = 0.15
TCP_HEIGHT = (0.02, 0.30)
MIN_DOWN_BINS = 4
PIXEL_MARGIN = 30
TABLE_INSET = 0.08


def corners(lower, upper):
    return np.asarray(list(product(*zip(lower, upper))), dtype=float)


def world_points(points, frame):
    return points@frame[:3, :3].T+frame[:3, 3]


def camera_check(state, x_range):
    body = np.asarray(state['links']['body4'])
    camera = np.asarray(state['camera_world_optical'])
    k = np.asarray(state['intrinsics'])
    top = state['scene_cuboids']['table']['center'][2]+state['scene_cuboids']['table']['dims'][2]/2
    # Include the object support plane as well as the upper TCP range.
    local = corners([x_range[0], -HALF_WIDTH, top-body[2, 3]],
                    [x_range[1], HALF_WIDTH, top-body[2, 3]+TCP_HEIGHT[1]])
    optical = (world_points(local, body)-camera[:3, 3])@camera[:3, :3]
    uvz = optical@k.T
    uv = uvz[:, :2]/uvz[:, 2, None]
    valid = (optical[:, 2] > 0).all() and (uv >= PIXEL_MARGIN).all()
    valid = bool(valid and (uv[:, 0] <= 640-PIXEL_MARGIN).all() and (uv[:, 1] <= 480-PIXEL_MARGIN).all())
    return {'passed': valid, 'uv_corners': uv.tolist(), 'minimum_pixel_margin': float(np.min(
        np.c_[uv[:, 0], 640-uv[:, 0], uv[:, 1], 480-uv[:, 1]])),
        'optical_depth_range_m': [float(optical[:, 2].min()), float(optical[:, 2].max())]}


def region_metrics(data, lower, upper):
    # Select centers in a half-open box using integer grid indices.
    first = np.ceil(np.asarray(lower)/0.02-0.5-1e-9).astype(int)
    last = np.ceil(np.asarray(upper)/0.02-0.5-1e-9).astype(int)
    cells = np.asarray(list(product(*(range(a, b) for a, b in zip(first, last)))), dtype=np.int64)
    keys = encode(cells)
    right, ri = lookup(data['voxel_id'], keys)
    left, li = lookup(data['voxel_id'], keys-25*4096)
    common = right & left
    down_r = np.where(right, data['down_orientation_count'][ri], 0)
    down_l = np.where(left, data['down_orientation_count'][li], 0)
    return {'position_cells': len(cells), 'both_covered_cells': int(common.sum()),
        'both_covered_fraction': float(common.mean()),
        'minimum_observed_down_bins_per_arm': int(np.minimum(down_r, down_l).min()),
        'median_min_arm_down_bins': float(np.median(np.minimum(down_r, down_l))),
        'right_samples_mean': float(np.where(right, data['sample_count'][ri], 0).mean()),
        'left_samples_mean': float(np.where(left, data['sample_count'][li], 0).mean())}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    if not read_json(DATA/'completion.json')['passed']:
        raise ValueError('Shared-region comparison is incomplete')
    state = read_json(STATE)
    body = np.asarray(state['links']['body4'])
    base = np.asarray(state['links']['base_link'])
    table = state['scene_cuboids']['table']
    top = table['center'][2]+table['dims'][2]/2
    near_edge = table['center'][0]-table['dims'][0]/2-body[0, 3]
    height = np.asarray(TCP_HEIGHT)+top-body[2, 3]
    with np.load(DATA/'statistics_2cm.npz', allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    near_candidates = []
    for x in np.arange(0.24, 0.46, 0.02):
        check = camera_check(state, [x, x+0.20])
        eligible = bool(x-near_edge >= TABLE_INSET and check['passed'])
        near_candidates.append({'x_body4_m': float(x), 'eligible': eligible, 'camera': check})
    near = next(row['x_body4_m'] for row in near_candidates if row['eligible'])
    far_candidates = []
    for far in np.arange(near+0.02, 0.76, 0.02):
        lower = [near, -HALF_WIDTH, height[0]]
        upper = [far, HALF_WIDTH, height[1]]
        metrics = region_metrics(data, lower, upper)
        check = camera_check(state, [near, far])
        eligible = bool(metrics['both_covered_fraction'] == 1. and
                        metrics['minimum_observed_down_bins_per_arm'] >= MIN_DOWN_BINS and check['passed'])
        far_candidates.append({'far_body4_m': float(far), 'eligible': eligible, 'metrics': metrics, 'camera': check})
    selected = max((row for row in far_candidates if row['eligible']), key=lambda row: row['far_body4_m'])
    lower = np.array([near, -HALF_WIDTH, height[0]])
    upper = np.array([selected['far_body4_m'], HALF_WIDTH, height[1]])
    world = world_points(corners(lower, upper), body)
    relative_base = (world-base[:3, 3])@base[:3, :3]
    result = {'at': now(), 'frame': 'body4/action_tcp43', 'lift_joint_m': float(state['qpos'][state['joint_names'].index('joint4')]),
        'lower_body4_m': lower.tolist(), 'upper_body4_m': upper.tolist(),
        'dimensions_m': (upper-lower).tolist(), 'tcp_height_above_table_m': TCP_HEIGHT,
        'world_xyz_bounds_m': [world.min(0).tolist(), world.max(0).tolist()],
        'base_link_xyz_bounds_m': [relative_base.min(0).tolist(), relative_base.max(0).tolist()],
        'table_inset_range_m': [near-near_edge, selected['far_body4_m']-near_edge],
        'table_height_world_m': top, 'table_geometry_world': table,
        'body4_world': body.tolist(), 'base_link_world': base.tolist(),
        'recommended_metrics': selected['metrics'], 'camera_check': selected['camera'],
        'engineering_choices': {'lateral_half_width_m': HALF_WIDTH, 'pixel_margin': PIXEL_MARGIN,
            'minimum_table_inset_m': TABLE_INSET, 'minimum_observed_down_bins_per_arm': MIN_DOWN_BINS,
            'down_direction_definition': 'TCP approach axis within 60 degrees of downward in body4',
            'selection': 'Nearest camera-visible row, then farthest passing 2cm boundary'},
        'near_candidates': near_candidates, 'far_candidates': far_candidates,
        'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in [STATE, DATA/'statistics_2cm.npz',
            DATA/'completion.json', Path(__file__)]},
        'physical_attempts': 0, 'new_IK_queries': 0,
        'interpretation': 'Recommended TCP position region from observed kinematic coverage, table placement and camera projection'}
    write_json(OUTPUT/'workspace.json', result)
    print('Recommended body4 bounds:', result['lower_body4_m'], result['upper_body4_m'], flush=True)
    print('From near table edge:', result['table_inset_range_m'], flush=True)
    print('Coverage:', result['recommended_metrics'], flush=True)


if __name__ == '__main__':
    main()
