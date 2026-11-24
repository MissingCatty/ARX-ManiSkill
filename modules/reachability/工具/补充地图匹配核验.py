"""Check exact lookup and threshold sensitivity without changing the assessed workflow."""
from 公共 import EXPERIMENTS
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from 公共 import ROOT, read, save, sha
from 能力索引 import CapabilityMap
from 评估地图匹配流程 import EvaluationMatcher


def main():
    out = EXPERIMENTS / '地图流程比较_20261121_01'
    target = out/'补充检索核验.json'
    assert not target.exists()
    atlas = CapabilityMap(ROOT/'产物/前方精细_20261121_01')
    matcher = EvaluationMatcher(atlas.bank, atlas.tree, atlas.bounds, cKDTree(atlas.bank[:, 6:8]))
    rng = np.random.default_rng(5729)
    checks = []
    for index in rng.choice(len(atlas.bank), 32, replace=False):
        row = np.asarray(atlas.bank[index], dtype=float)
        pose = np.eye(4)
        pose[:3, 3] = row[6:9]
        pose[:3, :3] = Rotation.from_quat(row[9:13][[1, 2, 3, 0]]).as_matrix()
        for lift in [None, (.2, 0., .46)]:
            result = matcher.query(pose, lift=lift)
            assert result['accepted'] and result['similarity'] > .999999
        checks.append(int(index))
    subset = np.array(atlas.bank[rng.choice(len(atlas.bank), 5000, replace=False)])
    small = EvaluationMatcher(subset, cKDTree(subset[:, 6:9]), atlas.bounds, cKDTree(subset[:, 6:8]))
    for row in subset[:8]:
        pose = np.eye(4)
        pose[:3, 3] = row[6:9]+rng.normal(0, .006, 3)
        pose[0, 3] = abs(pose[0, 3])
        rotation = Rotation.from_quat(row[9:13][[1, 2, 3, 0]])*Rotation.from_rotvec(rng.normal(0, .05, 3))
        pose[:3, :3] = rotation.as_matrix()
        for lift in [None, (.2, 0., .46)]:
            result = small.query(pose, lift=lift)
            positions = subset[:, 6:9].astype(float)
            if lift is not None:
                positions[:, 2] += np.clip(.2+pose[2, 3]-positions[:, 2], 0., .46)-.2
            dp = np.linalg.norm(positions-pose[:3, 3], axis=1)
            q = subset[:, 9:13].astype(float)
            q /= np.linalg.norm(q, axis=1, keepdims=True)
            rq = rotation.as_quat()[[3, 0, 1, 2]]
            dr = 2*np.arccos(np.clip(np.abs(q@rq), 0, 1))
            expected = np.min((dp/.005)**2+(dr/np.deg2rad(3))**2)
            got = (result['position_difference_m']/.005)**2+(result['orientation_difference_deg']/3)**2
            assert abs(expected-got) < 1e-7
    original = read(EXPERIMENTS / '抓取位姿局部修正_20261121_02/登记.json')
    sensitivity = []
    for position, angle in [(.005, 3.), (.005, 5.), (.010, 5.), (.010, 10.), (.020, 10.)]:
        failures = controls = 0
        candidate_matches = 0
        for case in original['cases']:
            state = read(Path(case['source'])/'state.json')
            inverse = np.linalg.inv(np.asarray(state['links']['body4']))
            idx = state['joint_names'].index('joint4')
            lift = (state['qpos'][idx], *state['joint_limits'][idx])
            variants = read(EXPERIMENTS / '抓取位姿局部修正_20261121_02/候选'/f"{case['case']}.json")['variants']
            matched = 0
            for v in variants:
                if v['angle_deg'] != 0:
                    continue
                result = matcher.query(inverse@np.asarray(v['eef_goal_world'])@atlas.tcp,
                                       position_m=position, rotation_deg=angle, lift=lift)
                matched += result['accepted']
            candidate_matches += matched
            if case['role'] == 'diagnostic_failure':
                failures += matched > 0
            else:
                controls += matched > 0
        sensitivity.append({'position_limit_m': position, 'rotation_limit_deg': angle,
                            'minimum_similarity': .6, 'lift_policy': 'joint7_legal_vertical_lift',
                            'matched_candidates': candidate_matches, 'total_candidates': 22,
                            'matched_problem_cases': failures, 'problem_cases': 3,
                            'matched_control_cases': controls, 'control_cases': 2,
                            'operational_threshold_changed': False})
        print('SENSITIVITY', sensitivity[-1], flush=True)
    save(target, {'passed': True, 'stored_witness_retrieval_tests': 64,
                  'stored_bank_rows': checks, 'brute_force_comparisons': 16,
                  'brute_force_subset_size': 5000, 'threshold_sensitivity': sensitivity,
                  'code_sha256': sha(Path(__file__)), 'physical_attempts': 0,
                  'production_pipeline_modified': False})


if __name__ == '__main__':
    main()
