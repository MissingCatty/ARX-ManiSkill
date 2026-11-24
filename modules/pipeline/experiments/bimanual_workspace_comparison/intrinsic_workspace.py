"""Choose body-relative arm workspaces without table or camera constraints."""
from pathlib import Path

import numpy as np

from arx_project.assets import resource_path
from arx_project.layout import ROOT, now, read_json, sha256, write_json
from arx_project.scenes import current_dataset
from analyze import encode, lookup
from compare_regions import verify_arm_relation

OUTPUT = ROOT/'tmp/reachability/intrinsic_workspace_20261125_01'
ATLAS = ROOT/'tmp/reachability/products/前方精细_20261121_01'
STATS = ROOT/'tmp/reachability/shared_region_comparison_20261125_01/statistics_2cm.npz'
STATE = current_dataset()/'cases/D0008_S1_open_teapot/state.json'
STEP = .02
MIN_FORWARD = .20
MIN_DOWN_BINS = 4
QUALITY_QUANTILE = .25


def largest_rectangle(mask):
    """Largest all-true X/Z rectangle; return area and half-open indices."""
    heights = np.zeros(mask.shape[1], dtype=int)
    best = (0, 0, 0, 0, 0)
    for row_index, row in enumerate(mask):
        heights = np.where(row, heights+1, 0)
        stack = []
        for column in range(len(heights)+1):
            height = heights[column] if column < len(heights) else 0
            start = column
            while stack and stack[-1][1] > height:
                start, previous = stack.pop()
                item = (int(previous*(column-start)), row_index-int(previous)+1,
                        row_index+1, start, column)
                if item[0] > best[0]:
                    best = item
            if not stack or stack[-1][1] < height:
                stack.append((start, int(height)))
    return best


def values_at(data, cells, side):
    keys = encode(cells.reshape(-1, 3))
    shift = 25*4096 if side == 'left' else 0
    present, index = lookup(data['voxel_id'], keys-shift)
    shape = cells.shape[:-1]
    return {name: np.where(present, data[name][index], 0).reshape(shape)
            for name in ['sample_count', 'orientation_count', 'down_orientation_count']}


def select_box(data, scope, thresholds):
    """Selection depends only on the intrinsic FK statistics, never a scene."""
    xs = np.arange(int(round(MIN_FORWARD/STEP)), int(data['cells'][:, 0].max())+1)
    zs = np.arange(int(data['cells'][:, 2].min()), int(data['cells'][:, 2].max())+1)
    ys = np.arange(-30, 31) if scope == 'right' else np.arange(-15, 15)
    cells = np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), axis=-1)
    sides = ['right', 'left'] if scope == 'shared' else ['right']
    values = {side: values_at(data, cells, side) for side in sides}
    eligible = np.ones(cells.shape[:-1], dtype=bool)
    for side in sides:
        for name, threshold in thresholds.items():
            eligible &= values[side][name] >= threshold
    # Shared rectangles are centered on the body midline. Single-arm rectangles
    # are centered on that arm's shoulder Y (-25cm for the reference right arm).
    center_y = 0. if scope == 'shared' else -.25
    widths = range(10, 31, 2) if scope == 'shared' else range(11, 32, 2)
    best = None
    search = []
    for width in widths:
        lo = int(round(center_y/STEP-width/2))
        hi = lo+width
        selected_y = (ys >= lo) & (ys < hi)
        if selected_y.sum() != width:
            continue
        area, x0, x1, z0, z1 = largest_rectangle(eligible[:, selected_y, :].all(axis=1))
        count = area*width
        search.append({'width_cells': width, 'maximum_rectangle_area_cells': area, 'volume_cells': count})
        if count and (best is None or count > best['volume_cells']):
            best = {'volume_cells': count, 'lower_cell': [int(xs[x0]), lo, int(zs[z0])],
                    'upper_cell': [int(xs[x1-1])+1, hi, int(zs[z1-1])+1]}
    if best is None:
        raise ValueError('No workspace satisfies the registered criteria')
    lower = np.asarray(best['lower_cell']); upper = np.asarray(best['upper_cell'])
    chosen_cells = np.stack(np.meshgrid(*(np.arange(a, b) for a, b in zip(lower, upper)), indexing='ij'), axis=-1)
    metrics = {}
    for side in sides:
        selected = values_at(data, chosen_cells, side)
        metrics[side] = {}
        for name, threshold in thresholds.items():
            if np.any(selected[name] < threshold):
                raise ValueError('Selected box contains a below-threshold cell')
            metrics[side][name] = {'minimum': int(selected[name].min()),
                                  'median': float(np.median(selected[name])),
                                  'mean': float(selected[name].mean())}
    return {'lower_body4_m': (lower*STEP).tolist(), 'upper_body4_m': (upper*STEP).tolist(),
            'center_body4_m': ((lower+upper)*STEP/2).tolist(),
            'dimensions_m': ((upper-lower)*STEP).tolist(),
            'position_cells': best['volume_cells'], 'volume_m3': best['volume_cells']*STEP**3,
            'metrics': metrics, 'width_search': search}


def lift_reference(state, region):
    """Measure the body/lift relation from the real URDF, with no table input."""
    from modules.curobo.dual_arm.model import ArmModel
    model = ArmModel()
    index = state['joint_names'].index('joint4')
    low, high = state['joint_limits'][index]
    inv_base = np.linalg.inv(np.asarray(state['links']['base_link']))
    probes = []
    for value in [low, (low+high)/2, high]:
        sample = {'joint_names': state['joint_names'], 'qpos': list(state['qpos']),
                  'links': {'base_link': state['links']['base_link']}}
        sample['qpos'][index] = value
        frame = inv_base@model.frames(sample)['body4']
        probes.append({'lift_m': value, 'body4_in_base_link': frame.tolist()})
    frames = np.asarray([row['body4_in_base_link'] for row in probes])
    mount = frames[0, :3, 3].copy(); mount[2] -= low
    if not np.allclose(frames[:, :3, :3], frames[0, :3, :3], atol=1e-7):
        raise ValueError('Lift changed the body rotation')
    for frame, row in zip(frames, probes):
        expected = mount+np.array([0., 0., row['lift_m']])
        if not np.allclose(frame[:3, 3], expected, atol=1e-6):
            raise ValueError('Lift translation is not the expected vertical relation')
    zlo, zhi = region['lower_body4_m'][2], region['upper_body4_m'][2]
    center_z = region['center_body4_m'][2]
    examples = []
    for target_height in [.5, .6, .75, .9]:
        interval = [max(low, target_height-mount[2]-zhi), min(high, target_height-mount[2]-zlo)]
        examples.append({'target_height_above_base_link_m': target_height,
                         'lift_interval_m': interval if interval[0] <= interval[1] else None,
                         'center_aligned_lift_m': float(target_height-mount[2]-center_z)})
    return {'mount_xyz_at_zero_lift_m': mount.tolist(), 'lift_limits_m': [low, high],
            'probes': probes, 'examples': examples,
            'formula': 'target_z_base = mount_z + joint4 + desired_tcp_z_body4',
            'scope': 'upright base, other chassis joint coordinates held at their recorded values'}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    protected = [ROOT/'modules/reachability/docs/recommended_workspace/推荐操作空间.md',
                 ROOT/'modules/pipeline/experiments/bimanual_workspace_comparison/recommend_workspace.py',
                 ROOT/'modules/pipeline/experiments/bimanual_workspace_comparison/draw_workspace.py',
                 *sorted((ROOT/'modules/reachability/docs/recommended_workspace/resources').glob('*.png')),
                 *sorted((ROOT/'tmp/reachability/recommended_workspace_20261125_01').glob('*.png')),
                 ROOT/'tmp/reachability/recommended_workspace_20261125_01/workspace.json']
    protected_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in protected}
    inputs = [STATS, ATLAS/'地图元数据.json', ATLAS/'概览体素统计.npz', resource_path('arx_robot')/'robot.urdf']
    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in inputs}
    write_json(OUTPUT/'registration.json', {'at': now(), 'source_sha256': sha256(Path(__file__)),
        'selection_inputs': source_hashes, 'protected_original_files': protected_hashes,
        'minimum_forward_body4_m': MIN_FORWARD, 'grid_m': STEP,
        'quality_quantile': QUALITY_QUANTILE, 'minimum_down_bins': MIN_DOWN_BINS,
        'environment_constraints': [], 'camera_constraints': [],
        'optimization': 'largest complete cuboid on the 2cm grid, symmetric about the body or shoulder Y center',
        'front_operation_choice': '20cm minimum forward offset is an engineering work-area preference; the atlas itself stays complete',
        'physical_attempts': 0, 'new_IK_queries': 0})
    model = verify_arm_relation(read_json(ATLAS/'地图元数据.json'))
    with np.load(STATS, allow_pickle=False) as stored:
        data = {name: stored[name] for name in stored.files}
    thresholds = {name: int(np.ceil(np.quantile(data[name], QUALITY_QUANTILE)))
                  for name in ['sample_count', 'orientation_count']}
    thresholds['down_orientation_count'] = MIN_DOWN_BINS
    shared = select_box(data, 'shared', thresholds)
    right = select_box(data, 'right', thresholds)
    left = {**right, 'metrics': {'left': right['metrics']['right']}}
    for key in ['lower_body4_m', 'upper_body4_m', 'center_body4_m']:
        left[key] = (np.asarray(right[key])+[0., .5, 0.]).tolist()
    state = read_json(STATE)
    result = {'at': now(), 'frame': 'body4/action_tcp43',
        'selection_uses_table_or_camera': False, 'source_sha256': source_hashes,
        'thresholds_per_arm_per_2cm_cell': thresholds,
        'thresholds_interpretation': 'Configuration and full-orientation counts each meet the full-atlas lower quartile, with at least four downward bins',
        'downward_definition': 'approach axis within 60 degrees of body4 -Z',
        'engineering_min_forward_m': MIN_FORWARD, 'regions': {'shared': shared, 'right': right, 'left': left},
        'arm_model_verification': model, 'height_and_base_reference': lift_reference(state, shared),
        'physical_attempts': 0, 'new_IK_queries': 0,
        'interpretation': 'Body-relative kinematic preferred regions for future base and lift initialization; each specific grasp continues through full-pose and collision checks'}
    write_json(OUTPUT/'workspace.json', result)
    for name, digest in {**source_hashes, **protected_hashes}.items():
        if sha256(ROOT/name) != digest:
            raise ValueError('Input or protected old figure changed: '+name)
    write_json(OUTPUT/'selection_verification.json', {'passed': True, 'original_files_unchanged': True,
        'model_alignment_passed': model['passed'], 'environment_filter_calls': 0,
        'camera_filter_calls': 0, 'physical_attempts': 0, 'new_IK_queries': 0})
    for side, region in result['regions'].items():
        print(side, region['lower_body4_m'], region['upper_body4_m'], region['position_cells'], flush=True)


if __name__ == '__main__':
    main()
