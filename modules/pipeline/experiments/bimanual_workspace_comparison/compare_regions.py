"""Compare shared and exclusive arm regions in the frozen forward FK atlas."""
import csv
from pathlib import Path

import numpy as np

from arx_project.assets import resource_path
from arx_project.layout import ROOT, now, read_json, sha256, write_json
from analyze import Arm, encode, lookup

ATLAS = ROOT/'tmp/reachability/products/前方精细_20261121_01'
OUTPUT = ROOT/'tmp/reachability/shared_region_comparison_20261125_01'
BAND = (-0.25, -0.10)
ORIENTATIONS = 1152


def decode(ids):
    return np.column_stack([ids//4096//4096-2048, ids//4096 % 4096-2048, ids % 4096-2048])


def regrid(fine, pose_keys, factor):
    ids = encode(np.floor_divide(fine['cells'], factor))
    unique, inverse = np.unique(ids, return_inverse=True)
    counts = np.bincount(inverse, weights=fine['sample_count']).astype(np.int64)
    fine_ids = pose_keys//ORIENTATIONS
    target_ids = encode(np.floor_divide(decode(fine_ids), factor))
    keys = np.unique(target_ids*ORIENTATIONS + pose_keys % ORIENTATIONS)
    orientation_ids, orientations = np.unique(keys//ORIENTATIONS, return_counts=True)
    if not np.array_equal(unique, orientation_ids):
        raise ValueError('Position and pose populations disagree after regridding')
    down_ids, down_counts = np.unique(
        (keys//ORIENTATIONS)[(keys % ORIENTATIONS)//12 % 8 < 2], return_counts=True)
    down = np.zeros(len(unique), dtype=np.int64)
    down[np.searchsorted(unique, down_ids)] = down_counts
    return {'voxel_id': unique, 'cells': decode(unique), 'sample_count': counts,
            'orientation_count': orientations, 'down_orientation_count': down}


def summarize(data, resolution):
    ids = data['voxel_id']
    shift = int(round(0.5/resolution))*4096
    membership = {'right': lookup(ids, ids-shift)[0], 'left': lookup(ids, ids+shift)[0]}
    z = (data['cells'][:, 2]+0.5)*resolution
    populations = {'all_heights': np.ones(len(ids), dtype=bool),
                   'table_height_band': (z >= BAND[0]) & (z <= BAND[1])}
    rows = []
    for scope, population in populations.items():
        threshold = float(np.quantile(data['sample_count'][population], 0.75))
        for arm, shared in membership.items():
            for region, selected in [('shared', shared), ('exclusive', ~shared)]:
                mask = population & selected
                if not mask.any():
                    raise ValueError('Empty region in comparison')
                row = {'resolution_m': resolution, 'scope': scope, 'arm': arm,
                       'region': region, 'position_cells': int(mask.sum()),
                       'volume_m3': float(mask.sum()*resolution**3),
                       'high_density_threshold': threshold,
                       'high_density_fraction': float((data['sample_count'][mask] >= threshold).mean())}
                for name in ['sample_count', 'orientation_count', 'down_orientation_count']:
                    values = data[name][mask]
                    row[name+'_mean'] = float(values.mean())
                    row[name+'_median'] = float(np.median(values))
                rows.append(row)
    return rows


def verify_arm_relation(meta):
    left = Arm(resource_path('arx_robot')/'robot.urdf', 'left')
    right = Arm(resource_path('arx_robot')/'robot.urdf', 'right')
    if not np.array_equal(left.limits, right.limits) or not np.allclose(right.limits, meta['joint_limits'], atol=1e-5, rtol=0):
        raise ValueError('Atlas and actual arm limits disagree')
    if not np.array_equal(left.offsets[0][:3, :3], right.offsets[0][:3, :3]):
        raise ValueError('Shoulder orientations differ')
    if not np.allclose(left.offsets[0][:3, 3]-right.offsets[0][:3, 3], [0., 0.5, 0.], atol=1e-12):
        raise ValueError('The left arm is not a 50cm translation')
    if not all(np.array_equal(a, b) for a, b in zip(left.offsets[1:], right.offsets[1:])):
        raise ValueError('Arm link transforms differ')
    if not all(np.array_equal(a, b) for a, b in zip(left.axes, right.axes)) or not np.array_equal(left.tail, right.tail):
        raise ValueError('Arm axes or end frames differ')
    bank = np.memmap(ATLAS/meta['witness_file'], dtype=meta['dtype'], mode='r',
                     shape=(meta['witnesses'], len(meta['fields'])))
    rng = np.random.default_rng(20261125)
    tcp = np.asarray(meta['T_eef_action'])
    errors = []
    for index in rng.choice(len(bank), 512, replace=False):
        q = np.asarray(bank[index, :6], dtype=float)
        if not ((q >= right.limits[:, 0]) & (q <= right.limits[:, 1])).all():
            raise ValueError('Stored witness exceeds actual joint limits')
        r, l = right.fk(q)@tcp, left.fk(q)@tcp
        errors.append(float(np.linalg.norm(r[:3, 3]-bank[index, 6:9])))
        if not np.allclose(l[:3, 3]-r[:3, 3], [0., 0.5, 0.], atol=1e-12):
            raise ValueError('Independent paired FK disagrees')
        if not np.allclose(r[:3, :3], l[:3, :3], atol=1e-12):
            raise ValueError('Paired FK orientations disagree')
    if max(errors) > 2e-5:
        raise ValueError('Atlas position differs from actual robot FK')
    return {'passed': True, 'paired_fk_samples': 512, 'maximum_position_error_m': max(errors),
            'translation_body4_m': [0., 0.5, 0.], 'joint_limits_equal': True}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    meta = read_json(ATLAS/'地图元数据.json')
    completed = read_json(ATLAS/'完成.json')
    for name, expected in completed['file_sha256'].items():
        if sha256(ATLAS/name) != expected:
            raise ValueError('Frozen atlas changed: '+name)
    model = verify_arm_relation(meta)
    inputs = {str((ATLAS/name).relative_to(ROOT)): digest for name, digest in completed['file_sha256'].items()}
    inputs[str((resource_path('arx_robot')/'robot.urdf').relative_to(ROOT))] = sha256(resource_path('arx_robot')/'robot.urdf')
    write_json(OUTPUT/'registration.json', {'at': now(), 'input_sha256': inputs,
        'source_sha256': sha256(Path(__file__)), 'question': 'Does each arm have more configurations and orientations in the shared region?',
        'frame': 'body4/action_tcp43', 'height_band_m': BAND, 'resolutions_m': [0.05, 0.02],
        'weighting': 'Equal weight per occupied position cell, each arm evaluated separately',
        'orientations': 'Observed bins in the fixed 12x8x12 orientation partition',
        'new_IK_queries': 0, 'physical_attempts': 0})
    with np.load(ATLAS/meta['coarse']['file'], allow_pickle=False) as saved:
        coarse = {name: saved[name] for name in saved.files}
    with np.load(ATLAS/meta['fine']['file'], allow_pickle=False) as saved:
        fine = {name: saved[name] for name in ['voxel_id', 'cells', 'sample_count']}
    pose_keys = np.load(ATLAS/'已覆盖位姿格.npy', mmap_mode='r', allow_pickle=False)
    rebuilt = regrid(fine, pose_keys, 10)
    for name in rebuilt:
        if not np.array_equal(rebuilt[name], coarse[name]):
            raise ValueError('Reaggregation does not reproduce frozen 5cm statistics: '+name)
    fine2 = regrid(fine, pose_keys, 4)
    if fine2['sample_count'].sum() != meta['witnesses']:
        raise ValueError('Reaggregation lost witness samples')
    rows = summarize(coarse, 0.05) + summarize(fine2, 0.02)
    np.savez_compressed(OUTPUT/'statistics_2cm.npz', **fine2)
    write_json(OUTPUT/'summary.json', {'at': now(), 'rows': rows, 'model_alignment': model,
        'witnesses_per_arm': meta['witnesses'], 'source_atlas': str(ATLAS),
        'coarse_reaggregation_exact': True, 'new_IK_queries': 0, 'physical_attempts': 0})
    with (OUTPUT/'region_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    for row in rows:
        if row['scope'] == 'table_height_band':
            print(row['resolution_m'], row['arm'], row['region'], row['position_cells'],
                  round(row['sample_count_mean'], 2), round(row['orientation_count_mean'], 2), flush=True)
    for name, expected in inputs.items():
        if sha256(ROOT/name) != expected:
            raise ValueError('Source atlas or model changed during analysis')
    write_json(OUTPUT/'completion.json', {'passed': True, 'input_hashes_unchanged': True,
        'model_alignment_passed': True, 'coarse_reaggregation_exact': True, 'physical_attempts': 0})


if __name__ == '__main__':
    main()
