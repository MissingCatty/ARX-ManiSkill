"""Independent FK and ledger checks for the completed projection pilot."""
from 公共 import EXPERIMENTS
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation

from 公共 import ROOT, MS, JOINTS, TCP, read, save, sha, now


def main():
    first = EXPERIMENTS / '关节空间投影_20261122_01'
    second = EXPERIMENTS / '关节空间硬约束投影_20261122_01'
    target = first/'独立验收.json'
    assert not target.exists()
    original = read(first/'结果汇总.json')
    followup = read(second/'结果汇总.json')
    registration = read(first/'登记.json')
    for folder in [first, second]:
        assert read(folder/'进度.json')['phase'] == 'complete'
        for path, digest in read(folder/'登记.json')['source_hashes'].items():
            assert sha(Path(path)) == digest, path
    assert original['physical_attempts'] == followup['physical_attempts'] == 0
    assert not original['production_pipeline_modified']
    sys.path.insert(0, str(MS/'arx_pipeline'))
    from scene_tools import ArmFK
    fk = ArmFK()
    tcp = np.asarray(read(TCP)['T_eef_action'])
    names = ['joint4']+JOINTS
    cases = {c['case']: c for c in registration['cases']}
    evidence, per_case, keys = [], [], set()
    for case in registration['cases']:
        rows = [r for r in original['rows'] if r['case'] == case['case']]
        assert len(rows) == case['candidates']
        state = read(Path(case['source'])/'state.json')
        pool = read(Path(case['source'])/'graspcl/candidate_pool.json')['candidates']
        bounds = np.array([state['joint_limits'][state['joint_names'].index(n)] for n in names])
        current_height = state['qpos'][state['joint_names'].index('joint4')]
        axis = np.asarray(state['links']['base_link'])[:3, :3]@np.array([0., 0., 1.])
        result_case = {'case': case['case'], 'role': case['role'], 'candidates': len(rows),
                      'original_ik_passed': sum(not r['projection_triggered'] for r in rows), 'modes': {}}
        for row in rows:
            key = (row['case'], row['index'])
            assert key not in keys
            keys.add(key)
            c = pool[row['index']]
            goal = np.asarray(c['pose']).copy()
            goal[:3, 3] -= .043*goal[:3, 2]
            assert np.array_equal(goal, np.asarray(row['original_eef_goal_world']))
            for mode, output in row['modes'].items():
                if not row['projection_triggered']:
                    assert output['status'] == 'original_pose_retained' and not output['pose_changed']
                    assert not output['map_query_performed']
                    assert np.array_equal(goal, output['eef_goal_world'])
                    details = [{'q_witness_7': output['q_witness_7'], 'eef_goal_world': output['eef_goal_world']}]
                else:
                    details = output['attempts']
                    assert len(details) <= 8
                for detail in details:
                    q = np.asarray(detail['q_witness_7'])
                    assert ((q >= bounds[:, 0]-1e-7) & (q <= bounds[:, 1]+1e-7)).all()
                    body = np.asarray(state['links']['body4']).copy()
                    body[:3, 3] += axis*(q[0]-current_height)
                    actual = body@fk.fk(q[1:])
                    expected = np.asarray(detail['eef_goal_world'])
                    pe = float(np.linalg.norm(actual[:3, 3]-expected[:3, 3]))
                    re = float(Rotation.from_matrix(actual[:3, :3]@expected[:3, :3].T).magnitude())
                    if not row['projection_triggered']:
                        assert pe <= .002001 and re <= .030001
                    else:
                        assert pe < 2e-5 and re < 2e-4
                if row['projection_triggered'] and output['accepted']:
                    selected = output['selected']
                    q = np.asarray(selected['q_witness_7'])
                    body = np.asarray(state['links']['body4']).copy()
                    body[:3, 3] += axis*(q[0]-current_height)
                    actual = body@fk.fk(q[1:])
                    action, target_action = actual@tcp, goal@tcp
                    dp = float(np.linalg.norm(action[:3, 3]-target_action[:3, 3]))
                    dr = float(Rotation.from_matrix(actual[:3, :3]@goal[:3, :3].T).magnitude())
                    p1 = action[:3, 3]-c['width']/2*actual[:3, 1]
                    p2 = action[:3, 3]+c['width']/2*actual[:3, 1]
                    contact = float(max(np.linalg.norm(p1-c['point1']), np.linalg.norm(p2-c['point2'])))
                    assert dp <= .005001 and dr <= np.deg2rad(3)+1e-6 and contact <= .003001
                    assert selected['observed_pad_support'] > 0 and row['observed_pad_support_original'] > 0
                    eef_dp = float(np.linalg.norm(actual[:3, 3]-goal[:3, 3]))
                    evidence.append({'case': case['case'], 'index': row['index'], 'mode': mode,
                         'tcp_position_change_m': dp, 'orientation_change_deg': float(np.rad2deg(dr)),
                         'eef_position_change_m': eef_dp, 'nominal_contact_change_m': contact,
                         'within_original_goal_tolerances': bool(eef_dp <= .002001 and dr <= .030001),
                         'q_witness_7': q.tolist(), 'joint_limits_valid': True,
                         'projection_seconds': output['wall_seconds'], 'selected_start': selected['start'],
                         'independent_fk_world': actual.tolist(), 'physical_grasp_quality_verified': False})
        for mode in registration['modes']:
            result_case['modes'][mode] = {'total_pose_witnesses': sum(r['modes'][mode]['accepted'] for r in rows),
                'new_admissible_projections': sum(r['projection_triggered'] and r['modes'][mode]['accepted'] for r in rows)}
        per_case.append(result_case)
    assert len(keys) == 22 and len(evidence) == 4
    for row in followup['rows']:
        prior = next(r for r in original['rows'] if (r['case'], r['index']) == (row['case'], row['index']))
        assert prior['projection_triggered']
        for mode, output in row['modes'].items():
            if output['carried_from_previous']:
                assert prior['modes'][mode]['accepted'] and output['new_optimization_attempts'] == 0
                assert output['selected'] == prior['modes'][mode]['selected']
            else:
                assert not output['accepted'] and output['new_optimization_attempts'] <= 8
    save(target, {'passed': True, 'created_at': now(), 'original_candidates': len(keys),
         'original_ik_passed': original['summary']['original_ik_passed'],
         'projection_inputs': original['summary']['projection_inputs'], 'cases': per_case,
         'accepted_projection_evidence': evidence, 'original_passes_retained_without_map': True,
         'original_goal_tolerance_was_not_relaxed': True,
         'map_additional_recoveries_over_direct_projection': 0,
         'hard_constraint_followup_additional_recoveries': followup['summary'],
         'physical_attempts': 0, 'source_hashes_unchanged': True,
         'code_sha256': sha(Path(__file__)),
         'source_result_sha256': {str(p): sha(p) for p in [first/'结果汇总.json', second/'结果汇总.json']}})
    print('INDEPENDENT_ACCEPTANCE_PASSED', per_case)


if __name__ == '__main__':
    main()
