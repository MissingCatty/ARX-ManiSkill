"""Independent front-half kinematic atlas; no scene checker or robot I/O."""
import argparse
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

from 公共 import ROOT, MS, REFERENCE, URDF, TCP, JOINTS, FIELDS
from 公共 import read, save, sha, now, voxel_ids, orientation_ids, ORIENTATIONS
from 机器人几何 import RobotGeometry
from 构建地图 import aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--输出', type=Path, required=True)
    parser.add_argument('--显卡', required=True)
    parser.add_argument('--采样幂', type=int, default=24)
    parser.add_argument('--批大小', type=int, default=16384)
    args = parser.parse_args()
    assert 10 <= args.采样幂 <= 26 and args.批大小 > 0
    os.environ['CUDA_VISIBLE_DEVICES'] = args.显卡
    out = args.输出.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__).resolve(), Path(__file__).with_name('公共.py'),
               Path(__file__).with_name('机器人几何.py'), Path(__file__).with_name('构建地图.py'),
               URDF, TCP, REFERENCE/'state.json', MS/'arx_pipeline/scene_tools.py']
    hashes = {str(p): sha(p) for p in sources}
    snapshot = out/'配置快照'
    snapshot.mkdir()
    for p in sources[:6]:
        shutil.copy2(p, snapshot/p.name)
    state = read(REFERENCE/'state.json')
    tcp = np.asarray(read(TCP)['T_eef_action'])
    total = 2**args.采样幂
    save(out/'登记.json', {
        'created_at': now(), 'source_hashes': hashes, 'gpu_uuid': args.显卡,
        'sampled_joint_states': total, 'seed': 20261120,
        'front_halfspace': 'action TCP position in body4: x >= 0; azimuth -90 to +90 degrees',
        'position_statistics_resolution_m': .005,
        'pose_matching': 'Continuous stored xyz/quaternion, no orientation-bin matching',
        'table_checked': False, 'ground_checked': False, 'self_collision_checked': False,
        'scene_collision_checked': False, 'physical_attempts': 0,
        'query_grasp_solutions_added_to_map': False,
        'scope': 'Six arm joints in moving body4 frame; lift is represented by the query transform, not frozen world height.'})
    import torch
    began = time.monotonic()
    geometry = RobotGeometry(tcp, device='cuda')
    measured_limits = np.array([state['joint_limits'][state['joint_names'].index(n)] for n in JOINTS])
    bounds = np.stack([geometry.lower, geometry.upper], axis=1)
    assert np.max(np.abs(measured_limits-bounds)) < 1e-5
    assert np.allclose(tcp[:3, 3], [0, 0, .043])
    engine = qmc.Sobol(d=6, scramble=True, seed=20261120)
    lower, upper = geometry.lower+1e-4, geometry.upper-1e-4
    accepted = rear = violations = 0
    minimum_margin = np.inf
    path = out/'关节见证.bin'
    with path.open('xb') as stream, torch.no_grad():
        for offset in range(0, total, args.批大小):
            count = min(args.批大小, total-offset)
            qnp = (lower+engine.random(count)*(upper-lower)).astype(np.float32)
            q = torch.as_tensor(qnp, device='cuda')
            p, r, condition, _, _ = geometry.fk(q)
            positions = p.cpu().numpy()
            valid = ((qnp >= geometry.lower) & (qnp <= geometry.upper)).all(axis=1)
            violations += int((~valid).sum())
            assert valid.all(), 'Generated joint limit violation'
            front = positions[:, 0] >= 0.
            rear += int((~front).sum())
            selected = np.flatnonzero(front)
            if len(selected):
                qs = qnp[selected]
                rotations = r[selected].cpu().numpy()
                quat = Rotation.from_matrix(rotations).as_quat()[:, [3, 0, 1, 2]]
                margins = np.minimum(qs-geometry.lower, geometry.upper-qs)
                minimum_margin = min(minimum_margin, float(margins.min()))
                normalized = (margins/(geometry.upper-geometry.lower)).min(axis=1)
                data = np.column_stack([qs, positions[selected], quat, margins.min(axis=1),
                                        normalized, condition[selected].cpu().numpy()]).astype('<f4')
                assert data.shape[1] == len(FIELDS) and np.isfinite(data).all()
                stream.write(data.tobytes())
                accepted += len(data)
            if (offset//args.批大小) % 32 == 0 or offset+count == total:
                progress = {'phase': 'front_half_fk_sampling', 'processed': offset+count,
                            'requested': total, 'accepted': accepted, 'rear_rejected': rear,
                            'joint_limit_violations': violations, 'wall_seconds': time.monotonic()-began,
                            'physical_attempts': 0}
                save(out/'当前进度.json', progress)
                print('SAMPLING', offset+count, '/', total, 'front', accepted, flush=True)
        stream.flush()
        os.fsync(stream.fileno())
    assert accepted > 0 and accepted+rear == total
    assert path.stat().st_size == accepted*len(FIELDS)*4
    bank = np.memmap(path, dtype='<f4', mode='r', shape=(accepted, len(FIELDS)))
    save(out/'当前进度.json', {'phase': 'aggregating_5mm_cells', 'accepted': accepted,
                              'physical_attempts': 0})
    fine = aggregate(out, bank, .005, '精细体素统计')
    coarse = aggregate(out, bank, .05, '概览体素统计')
    ids, _ = voxel_ids(bank[:, 6:9], .005)
    # The coarse orientation bins support old display readers only. Matching uses raw rotations.
    seen = []
    for start in range(0, len(bank), 131072):
        rows = np.asarray(bank[start:start+131072])
        rotations = Rotation.from_quat(rows[:, 9:13][:, [1, 2, 3, 0]]).as_matrix()
        orientation, _ = orientation_ids(rotations)
        seen.append(ids[start:start+len(rows)]*ORIENTATIONS+orientation)
    np.save(out/'已覆盖位姿格.npy', np.unique(np.concatenate(seen)), allow_pickle=False)
    del ids, seen
    sys.path.insert(0, str(MS/'arx_pipeline'))
    from scene_tools import ArmFK
    fk = ArmFK()
    rng = np.random.default_rng(1121)
    pe, re = [], []
    for i in rng.choice(len(bank), min(512, len(bank)), replace=False):
        row = np.asarray(bank[i], dtype=float)
        t = fk.fk(row[:6])@tcp
        reference_rotation = Rotation.from_quat(row[9:13][[1, 2, 3, 0]]).as_matrix()
        pe.append(float(np.linalg.norm(t[:3, 3]-row[6:9])))
        re.append(float(Rotation.from_matrix(t[:3, :3].T@reference_rotation).magnitude()))
    assert max(pe) < 2e-5 and max(re) < 2e-4
    verification = {'passed': True, 'independent_fk_samples': len(pe),
                    'maximum_position_error_m': max(pe), 'maximum_orientation_error_rad': max(re),
                    'all_generated_joint_limits_checked': True, 'joint_limit_violations': violations,
                    'minimum_joint_margin_rad': minimum_margin,
                    'all_saved_witnesses_front': True, 'physical_attempts': 0}
    save(out/'独立模型核验.json', verification)
    meta = {'schema_version': 2, 'created_at': now(), 'frame': 'body4', 'pose_frame': 'action_tcp43',
            'T_eef_action': tcp.tolist(), 'reference_base_world': state['links']['body4'],
            'lift_joint_m': None, 'lift_semantics': 'Use the actual body4 transform for each lift height.',
            'front_halfspace': {'axis': 'body4 +X', 'minimum_x_m': 0.},
            'joint_names': JOINTS, 'joint_limits': bounds.tolist(),
            'fields': FIELDS, 'dtype': '<f4', 'witness_file': path.name, 'witnesses': accepted,
            'sampled_joint_states': total, 'rear_rejected': rear, 'bounds_rejected': violations,
            'table_checked': False, 'ground_checked': False, 'self_collision_checked': False,
            'scene_collision_checked': False, 'orientation_matching': 'Unquantized xyz/quaternion pose distances',
            'orientation_grid': {'azimuth_bins': 12, 'cos_polar_bins': 8, 'roll_bins': 12,
                                 'total_bins': ORIENTATIONS, 'display_statistics_only': True},
            'uncovered_semantics': 'Unknown; a strict matcher may skip this correction, not prove unreachable.',
            'fine': fine, 'coarse': coarse, 'wall_seconds': time.monotonic()-began,
            'physical_attempts': 0}
    save(out/'地图元数据.json', meta)
    assert all(sha(Path(p)) == h for p, h in hashes.items())
    names = ['关节见证.bin', '精细体素统计.npz', '概览体素统计.npz',
             '已覆盖位姿格.npy', '地图元数据.json', '独立模型核验.json', '登记.json']
    save(out/'完成.json', {'passed': True, 'file_sha256': {name: sha(out/name) for name in names},
                          'source_hashes_unchanged': True, 'physical_attempts': 0, 'completed_at': now()})
    save(out/'当前进度.json', {'phase': 'complete', 'processed': total, 'accepted': accepted,
                              'wall_seconds': time.monotonic()-began, 'physical_attempts': 0})
    print('COMPLETE', accepted, '5mm_cells', fine['voxels'], flush=True)


if __name__ == '__main__':
    main()
