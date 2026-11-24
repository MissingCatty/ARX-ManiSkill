"""Check explicit pose/contact constraints on prior rejected projections, CPU only."""
from 公共 import EXPERIMENTS
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from 公共 import ROOT, MS, TCP, read, save, sha, now
from 关节空间位姿投影 import JointProjection
from 验证关节空间投影 import observed_points, pad_support, PADS, NAMES


def main():
    previous = EXPERIMENTS / '关节空间投影_20261122_01'
    out = EXPERIMENTS / '关节空间硬约束投影_20261122_01'
    out.mkdir(parents=True, exist_ok=False)
    baseline = read(previous/'结果汇总.json')
    registration = read(previous/'登记.json')
    hashes = dict(registration['source_hashes'])
    hashes.update({str(p): sha(p) for p in [Path(__file__).resolve(), previous/'结果汇总.json']})
    save(out/'登记.json', {'created_at': now(), 'source_hashes': hashes,
        'source_run': str(previous), 'physical_attempts': 0, 'modes': registration['modes'],
        'purpose': 'Check whether post-filtering a weighted optimum missed feasible small modifications.',
        'total_starts': 8, 'max_iterations': 250,
        'hard_constraints': {'tcp_position_change_m': .005, 'rotation_change_deg': 3.,
                             'each_nominal_contact_change_m': .003, 'joint_limits': 'unchanged'},
        'inheritance': 'Carry original IK and already accepted projections; explicitly recheck the rejected branches only.'})
    sys.path.insert(0, str(MS))
    from 七轴升降规划 import BodySweep
    independent = BodySweep.__new__(BodySweep)
    independent.joints = ET.parse(MS/'assets/arx/robot.urdf').getroot().findall('joint')
    tcp = np.asarray(read(TCP)['T_eef_action'])
    pads = read(PADS)['parts']
    cases = {c['case']: c for c in registration['cases']}
    rows = []
    began = time.monotonic()
    for row in baseline['rows']:
        if not row['projection_triggered']:
            continue
        case = cases[row['case']]
        source = Path(case['source'])
        state = read(source/'state.json')
        c = read(source/'graspcl/candidate_pool.json')['candidates'][row['index']]
        points = observed_points(source)
        projector = JointProjection(state, tcp)
        goal = np.asarray(row['original_eef_goal_world'])
        p1, p2 = np.asarray(c['point1']), np.asarray(c['point2'])
        fun, jac = projector.objective(goal, c['width'], p1, p2)

        def inequalities(q):
            r = fun(q)
            return np.array([1-r[:3]@r[:3], 1-r[3:6]@r[3:6],
                             1-2*r[6:9]@r[6:9], 1-2*r[9:12]@r[9:12]])

        def inequalities_jac(q):
            r, j = fun(q), jac(q)
            return np.vstack([-2*r[:3]@j[:3], -2*r[3:6]@j[3:6],
                              -4*r[6:9]@j[6:9], -4*r[9:12]@j[9:12]])

        result_row = {'case': row['case'], 'index': row['index'], 'role': row['role'], 'modes': {}}
        for mode, before in row['modes'].items():
            if before['accepted']:
                result_row['modes'][mode] = {'accepted': True, 'carried_from_previous': True,
                    'selected': before['selected'], 'new_optimization_attempts': 0}
                continue
            starts = [before['best']['q_witness_7']]+[a['q_seed'] for a in before['attempts'][1:]]
            assert len(starts) == 8
            selected, attempts, best = None, [], None
            for start_index, seed in enumerate(starts):
                t = time.monotonic()
                result = minimize(lambda q: .5*fun(q)@fun(q), seed,
                    jac=lambda q: jac(q).T@fun(q), method='SLSQP', bounds=projector.bounds,
                    constraints=[{'type': 'ineq', 'fun': inequalities, 'jac': inequalities_jac}],
                    options={'maxiter': 250, 'ftol': 1e-10})
                q = result.x
                actual = independent.fk(state, dict(zip(NAMES, q)))['right_eef_link']
                own = projector.fk(q)[0]
                fk_position = float(np.linalg.norm(actual[:3, 3]-own[:3, 3]))
                fk_rotation = float(Rotation.from_matrix(actual[:3, :3]@own[:3, :3].T).magnitude())
                assert fk_position < 2e-5 and fk_rotation < 2e-4
                action, target = actual@tcp, goal@tcp
                dp = float(np.linalg.norm(action[:3, 3]-target[:3, 3]))
                dr = float(Rotation.from_matrix(actual[:3, :3]@goal[:3, :3].T).magnitude())
                contacts = [action[:3, 3]-c['width']/2*actual[:3, 1], action[:3, 3]+c['width']/2*actual[:3, 1]]
                contact = float(max(np.linalg.norm(contacts[0]-p1), np.linalg.norm(contacts[1]-p2)))
                support = pad_support(points, actual, c['width'], pads)
                bounded = bool(((q >= projector.bounds[:, 0]-1e-7) & (q <= projector.bounds[:, 1]+1e-7)).all())
                front = bool((np.linalg.inv(state['links']['body4'])@action)[0, 3] >= 0)
                accepted = bool(bounded and front and dp <= .005+1e-8 and dr <= np.deg2rad(3)+1e-8
                                and contact <= .003+1e-8 and support > 0 and row['observed_pad_support_original'] > 0)
                detail = {'start': start_index, 'q_witness_7': q.tolist(), 'eef_goal_world': actual.tolist(),
                          'position_change_m': dp, 'orientation_change_deg': float(np.rad2deg(dr)),
                          'nominal_contact_change_m': contact, 'observed_pad_support': support,
                          'joint_limits_valid': bounded, 'accepted': accepted,
                          'independent_fk_position_error_m': fk_position,
                          'independent_fk_rotation_error_rad': fk_rotation,
                          'constraint_minimum': float(inequalities(q).min()), 'cost': float(result.fun),
                          'iterations': result.nit, 'optimizer_success': bool(result.success),
                          'optimizer_message': result.message, 'wall_seconds': time.monotonic()-t}
                attempts.append(detail)
                if best is None or detail['constraint_minimum'] > best['constraint_minimum']:
                    best = detail
                if accepted:
                    selected = detail
                    break
            result_row['modes'][mode] = {'accepted': selected is not None, 'carried_from_previous': False,
                'attempts': attempts, 'best_constraint_attempt': best, 'selected': selected,
                'new_optimization_attempts': len(attempts)}
            print('CONSTRAINED', row['case'], row['index'], mode, selected is not None,
                  'best_constraint', best['constraint_minimum'], flush=True)
        rows.append(result_row)
        save(out/'进度.json', {'phase': 'hard_constraint_check', 'completed_projection_inputs': len(rows),
                              'total_projection_inputs': 9, 'physical_attempts': 0})
    assert all(sha(Path(p)) == digest for p, digest in hashes.items())
    summary = {mode: {'accepted_projection_inputs': sum(r['modes'][mode]['accepted'] for r in rows),
                     'additional_over_previous': sum(r['modes'][mode]['accepted'] and not r['modes'][mode]['carried_from_previous'] for r in rows)}
               for mode in registration['modes']}
    save(out/'结果汇总.json', {'passed': True, 'rows': rows, 'summary': summary, 'physical_attempts': 0,
         'source_hashes_unchanged': True, 'wall_seconds': time.monotonic()-began,
         'scope': 'Explicit bounded projection feasibility follow-up; not new physical attempts or independent repetitions.'})
    save(out/'进度.json', {'phase': 'complete', 'completed_projection_inputs': len(rows),
                          'summary': summary, 'physical_attempts': 0})
    print('COMPLETE_HARD_CONSTRAINTS', summary, flush=True)


if __name__ == '__main__':
    main()
