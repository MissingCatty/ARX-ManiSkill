"""Offline comparison only. Never changes production grasp candidates or executes motion."""
from 公共 import EXPERIMENTS
import argparse
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from 公共 import ROOT, MS, JOINTS, read, save, sha, now
from 能力索引 import CapabilityMap


def valid_pose(pose):
    pose = np.asarray(pose, dtype=float)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(pose[:3, :3].T@pose[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1., atol=1e-6)):
        raise ValueError('Expected a rigid 4x4 pose')
    return pose


class EvaluationMatcher:
    """Exact thresholded SE(3) lookup; optional analytic vertical lift, no solver."""
    def __init__(self, bank, tree, bounds, xy_tree=None):
        self.bank, self.tree, self.bounds = bank, tree, np.asarray(bounds)
        self.xy_tree = xy_tree

    def query(self, pose, position_m=.005, rotation_deg=3., minimum_similarity=.6, lift=None):
        began = time.monotonic()
        pose = valid_pose(pose)
        if position_m <= 0 or rotation_deg <= 0 or not 0 < minimum_similarity <= 1:
            raise ValueError('Invalid comparison thresholds')
        if pose[0, 3] < 0:
            return {'accepted': False, 'reason': 'outside_front_scope',
                    'original_candidate_retained': True, 'unreachable': False}
        if len(self.bank) == 0:
            raise ValueError('Empty atlas')
        lifted = lift is not None
        tree = self.xy_tree if lifted else self.tree
        if tree is None:
            raise ValueError('Missing planar index for lift comparison')
        target = pose[:3, 3]
        target_quat = Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
        angle_scale = np.deg2rad(rotation_deg)
        if lifted:
            height, lower, upper = map(float, lift)
            if not lower <= height <= upper:
                raise ValueError('Measured lift outside limits')
        search_point = target[:2] if lifted else target

        def distances(indices):
            rows = np.asarray(self.bank[indices], dtype=float)
            positions = rows[:, 6:9].copy()
            heights = None
            if lifted:
                heights = np.clip(height+target[2]-positions[:, 2], lower, upper)
                positions[:, 2] += heights-height
            dp = np.linalg.norm(positions-target, axis=1)
            quats = rows[:, 9:13]
            quats = quats/np.linalg.norm(quats, axis=1, keepdims=True)
            angles = 2*np.arccos(np.clip(np.abs(quats@target_quat), 0., 1.))
            squared = (dp/position_m)**2+(angles/angle_scale)**2
            return rows, positions, heights, dp, angles, squared

        # An initial upper bound gives a positional search radius containing every possible
        # improvement. No fixed-k truncation is allowed to drop a better-oriented witness.
        _, initial = tree.query(search_point, k=min(256, len(self.bank)), workers=1)
        initial = np.atleast_1d(initial)
        initial_distance = distances(initial)[-1]
        radius = position_m*np.sqrt(initial_distance.min())+1e-9
        indices = np.asarray(tree.query_ball_point(search_point, radius, workers=1), dtype=np.int64)
        indices = np.unique(np.r_[indices, initial])
        rows, positions, heights, dp, angles, squared = distances(indices)
        best = int(np.argmin(squared))
        q = rows[best, :6]
        bounded = bool(((q >= self.bounds[:, 0]-1e-7) & (q <= self.bounds[:, 1]+1e-7)).all())
        if not bounded:
            raise RuntimeError('Map witness violates joint limits')
        similarity = float(np.exp(-.5*squared[best]))
        accepted = bool(dp[best] <= position_m+1e-9 and angles[best] <= angle_scale+1e-9
                        and similarity >= minimum_similarity-1e-12)
        actual = np.eye(4)
        actual[:3, 3] = positions[best]
        actual[:3, :3] = Rotation.from_quat(rows[best, 9:13][[1, 2, 3, 0]]).as_matrix()
        return {'accepted': accepted, 'reason': 'close_witness' if accepted else 'insufficient_similarity',
                'similarity': similarity, 'position_difference_m': float(dp[best]),
                'orientation_difference_deg': float(np.rad2deg(angles[best])),
                'bank_row': int(indices[best]), 'q_arm': q.tolist(),
                'lift_m': None if heights is None else float(heights[best]),
                'map_pose_in_current_body4': actual.tolist(), 'joint_limits_valid': bounded,
                'query_seconds': time.monotonic()-began, 'examined_witnesses': len(indices),
                'position_limit_m': position_m, 'rotation_limit_deg': rotation_deg,
                'minimum_similarity': minimum_similarity, 'nearest_search_is_exact_for_metric': True,
                'geometry_revalidated': False, 'original_candidate_retained': True, 'unreachable': False}


def self_test():
    bounds = np.tile([-2., 2.], (6, 1))
    bank = np.zeros((3002, 16))
    bank[:, 6] = .5
    bank[:, 9:13] = Rotation.from_euler('z', 180, degrees=True).as_quat()[[3, 0, 1, 2]]
    bank[-2, 6] += .001
    bank[-2, 9:13] = [-1, 0, 0, 0]  # Antipodal quaternion, same identity orientation.
    bank[-1, 6] += .05
    bank[-1, 9:13] = [1, 0, 0, 0]
    matcher = EvaluationMatcher(bank, cKDTree(bank[:, 6:9]), bounds, cKDTree(bank[:, 6:8]))
    target = np.eye(4)
    target[0, 3] = .5
    result = matcher.query(target)
    assert result['accepted'] and result['bank_row'] == 3000
    assert result['orientation_difference_deg'] < 1e-5
    target[0, 3] = .52
    assert not matcher.query(target)['accepted']
    target[0, 3] = -.1
    assert matcher.query(target)['reason'] == 'outside_front_scope'
    target[0, 3] = .5
    target[2, 3] = -.1
    assert not matcher.query(target)['accepted']
    assert matcher.query(target, lift=(.2, 0., .46))['accepted']
    invalid = target.copy()
    invalid[0, 0] = -1
    try:
        matcher.query(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError('Reflection accepted as a rotation')
    bank[3000, 0] = 10
    target[2, 3] = 0
    try:
        matcher.query(target)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Out-of-limit witness accepted')
    return ['full orientation and quaternion sign', 'no fixed-k false miss', 'threshold skip',
            'front scope', 'lift adaptation', 'invalid transform', 'joint-limit corruption']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--地图', type=Path, default=ROOT/'产物/前方精细_20261121_01')
    p.add_argument('--输出', type=Path)
    p.add_argument('--自检', action='store_true')
    args = p.parse_args()
    tests = self_test()
    if args.自检:
        print('SELF_TEST_PASSED', tests)
        return
    if args.输出 is None:
        p.error('--输出 is required')
    out = args.输出.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = EXPERIMENTS / '抓取位姿局部修正_20261121_02'
    registration = read(source/'登记.json')
    sources = [Path(__file__).resolve(), ROOT/'工具/能力索引.py', source/'登记.json',
               args.地图/'完成.json', MS/'assets/arx/robot.urdf', MS/'七轴升降规划.py']
    for case in registration['cases']:
        sources += [Path(case['source'])/'state.json', source/'候选'/f"{case['case']}.json",
                    source/'比较'/case['case']/'map_local_search/结果.json']
    hashes = {str(f): sha(f) for f in sources}
    save(out/'登记.json', {'created_at': now(), 'scope': 'Offline workflow assessment only',
                         'map': str(args.地图), 'source_hashes': hashes, 'checks': tests,
                         'physical_attempts': 0, 'production_pipeline_modified': False,
                         'thresholds': {'position_m': .005, 'rotation_deg': 3., 'similarity': .6},
                         'note': 'Provisional assessment thresholds, not a proven grasp-quality bound.'})
    atlas = CapabilityMap(args.地图)
    matcher = EvaluationMatcher(atlas.bank, atlas.tree, atlas.bounds, cKDTree(atlas.bank[:, 6:8]))
    sys.path.insert(0, str(MS))
    from 七轴升降规划 import BodySweep, NAMES
    fk = BodySweep.__new__(BodySweep)
    fk.joints = ET.parse(MS/'assets/arx/robot.urdf').getroot().findall('joint')
    rows, comparisons = [], []
    for case in registration['cases']:
        state = read(Path(case['source'])/'state.json')
        body = np.asarray(state['links']['body4'])
        inv_body = np.linalg.inv(body)
        lift_index = state['joint_names'].index('joint4')
        lift = (state['qpos'][lift_index], *state['joint_limits'][lift_index])
        actual_body = fk.fk(state)['body4']
        assert np.linalg.norm(actual_body-body) < 2e-5
        higher = fk.fk(state, {'joint4': lift[0]+.001})['body4']
        axis_body = body[:3, :3].T@(higher[:3, 3]-actual_body[:3, 3])/.001
        assert np.linalg.norm(axis_body-[0, 0, 1]) < 1e-5
        variants = read(source/'候选'/f"{case['case']}.json")['variants']
        originals = [v for v in variants if v['angle_deg'] == 0.]
        actual_old = read(source/'比较'/case['case']/'map_local_search/结果.json')
        eligible = [v for v in variants if v['geometry_admitted']]
        soft = []
        current = [state['qpos'][state['joint_names'].index(n)] for n in JOINTS]
        for v in eligible:
            action = inv_body@np.asarray(v['eef_goal_world'])@atlas.tcp
            nearest = atlas.query_pose(action, current_q=current, num_seeds=1, neighbors=2048)['best_pose_sample']
            score = nearest['translation_error_m']/.02+nearest['orientation_error_rad']/.35
            soft.append({'variant': v['id'], 'angle_deg': v['angle_deg'], 'score': score})
        soft.sort(key=lambda v: (v['score'], abs(v['angle_deg']), int(v['variant'].split('_')[0])))
        for v in originals:
            action = inv_body@np.asarray(v['eef_goal_world'])@atlas.tcp
            item = {'case': case['case'], 'role': case['role'], 'source_index': v['source_index'],
                    'original_geometry_admitted': v['geometry_admitted'], 'source_variant': v['id']}
            for mode, lift_args in [('fixed_lift', None), ('joint7_lift', lift)]:
                result = matcher.query(action, lift=lift_args)
                if result.get('bank_row') is not None:
                    height = lift[0] if result['lift_m'] is None else result['lift_m']
                    changes = dict(zip(NAMES, [height, *result['q_arm']]))
                    independent = fk.fk(state, changes)['right_eef_link']@atlas.tcp
                    expected = body@np.asarray(result['map_pose_in_current_body4'])
                    pe = float(np.linalg.norm(independent[:3, 3]-expected[:3, 3]))
                    re = float(Rotation.from_matrix(independent[:3, :3].T@expected[:3, :3]).magnitude())
                    assert pe < 2e-5 and re < 2e-4
                    limits = np.array([state['joint_limits'][state['joint_names'].index(n)] for n in NAMES])
                    q = np.array([height, *result['q_arm']])
                    assert ((q >= limits[:, 0]-1e-6) & (q <= limits[:, 1]+1e-6)).all()
                    result.update(independent_fk_position_error_m=pe, independent_fk_rotation_error_rad=re,
                                  all_seven_joint_limits_valid=True, q_witness_7=q.tolist())
                    # Convert the candidate convention back from the action TCP axes.
                    contact = expected.copy()
                    contact[:3, :3] = expected[:3, :3]@atlas.tcp[:3, :3].T
                    p1 = contact[:3, 3]-v['width']/2*contact[:3, 1]
                    p2 = contact[:3, 3]+v['width']/2*contact[:3, 1]
                    result['nominal_contact_max_displacement_m'] = float(max(
                        np.linalg.norm(p1-v['point1']), np.linalg.norm(p2-v['point2'])))
                item[mode] = result
            rows.append(item)
        selected = actual_old.get('selected')
        comparisons.append({'case': case['case'], 'role': case['role'], 'original_candidates': len(originals),
                            'historical_selected_angle_deg': None if not selected else selected['variant']['angle_deg'],
                            'historical_complete_offline_plan_passed': actual_old['success'],
                            'same_new_map_old_soft_first_choice': soft[0] if soft else None,
                            'strict_fixed_lift_matches': sum(r['case']==case['case'] and r['fixed_lift']['accepted'] for r in rows),
                            'strict_joint7_matches': sum(r['case']==case['case'] and r['joint7_lift']['accepted'] for r in rows)})
        save(out/'当前进度.json', {'case': case['case'], 'evaluated_candidates': len(rows), 'physical_attempts': 0})
        print('ASSESSED', case['case'], comparisons[-1], flush=True)
    assert all(sha(Path(f)) == h for f, h in hashes.items())
    save(out/'结果.json', {'passed': True, 'created_at': now(), 'rows': rows, 'cases': comparisons,
                         'source_hashes_unchanged': True, 'map_load_seconds': atlas.load_seconds,
                         'self_tests': tests, 'physical_attempts': 0, 'production_pipeline_modified': False,
                         'interpretation': 'Match rate and bounded nominal changes only; no path or grasp quality inference.'})
    print('COMPLETE_ASSESSMENT', len(rows), flush=True)


if __name__ == '__main__':
    main()
