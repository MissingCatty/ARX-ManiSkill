"""Independently verify frozen inputs, grasp identities, FK and recorded paths."""
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from arx_project.layout import ROOT, ensure_run_path, now, read_json, sha256, write_json
from modules.arx_core.arms import get_arm
from modules.curobo.dual_arm.model import ArmModel


def audit(root):
    reg = read_json(root/'registration.json')
    acceptance = read_json(root/'acceptance.json')
    if (root/'STOP').exists() or not acceptance.get('complete') or len(acceptance['paired_cases']) != len(reg['cases']):
        raise ValueError('Replay not complete or stopped')
    for name, digest in {**reg['source_hashes'], **reg['input_hashes']}.items():
        if sha256(ROOT/name) != digest or ('source_registered' not in name and
                name in reg['source_hashes'] and sha256(root/'source_registered'/name) != digest):
            raise ValueError('Replay source/input mismatch: '+name)
    model = ArmModel()
    policy = reg['options'].get('trajopt_policy')
    if policy:
        ready = read_json(root/'readiness.json')
        if ready['runtime_environment']['GRASP_TRAJOPT_POLICY'] != policy:
            raise ValueError('Planner runtime policy differs from registration')
        for optimization in (root/'searches').rglob('*_optimization.json'):
            if read_json(optimization)['policy'] != policy:
                raise ValueError('Actual trajectory optimization used a different policy')
    rows = []
    for pair in acceptance['paired_cases']:
        case = pair['case']
        original = ROOT/reg['source_run']/'trials'/case/'stationing/coarse'
        source_pool = read_json(original/'candidates/candidate_pool.json')['candidates']
        source_state = read_json(original/'state.json')
        for mode in ['grid', 'map']:
            report = read_json(root/'searches'/case/mode/'search_result.json')
            if report['full_planning_calls'] > reg['settings']['maximum_full_planning_stations']:
                raise ValueError('Search exceeded full planning budget')
            selected = read_json(root/'searches'/case/mode/'selected.json') if report['success'] else None
            record = {'case': case, 'mode': mode, 'planning_success': report['success'], 'fk_checks': []}
            if selected:
                result_path = ensure_run_path(selected['coarse_plan'])
                result = read_json(result_path)
                planned = result['selected']
                index = planned['index']
                if not np.array_equal(planned['contact_pose_world'], source_pool[index]['pose']):
                    raise ValueError('Selected grasp differs from the original candidate')
                record.update(selected_index=index, selected_arm=planned['arm'], grasp_world_pose_identical=True)
                state = read_json(result_path.parent/'state.json')
                # Robot station may change; all original scene/object fields remain identical.
                for key in ['object_world', 'object_center', 'scene_meshes', 'scene_cuboids', 'objects']:
                    if state.get(key) != source_state.get(key):
                        raise ValueError('Station search changed scene state: '+key)
                arm = get_arm(planned['arm'])
                for label, qkey, targetkey in [('pregrasp', 'q_pre', 'pregrasp_world'),
                                               ('grasp', 'q_grasp', 'grasp_world'),
                                               ('lift_goal', 'q_lift', 'lift_world')]:
                    current = deepcopy(state)
                    current['links'] = {'base_link': current['links']['base_link']}
                    q = np.asarray(planned[qkey])
                    if q.shape != (6,) or not np.isfinite(q).all():
                        raise ValueError('Invalid selected endpoint joint vector')
                    for name, value in zip(arm.joints, q):
                        i = current['joint_names'].index(name)
                        lo, hi = current['joint_limits'][i]
                        if not lo <= value <= hi:
                            raise ValueError('Selected joint witness exceeds physical limits')
                        current['qpos'][i] = float(value)
                    actual = model.frames(current)[arm.eef]
                    target = np.asarray(planned[targetkey])
                    pe = float(np.linalg.norm(actual[:3, 3]-target[:3, 3]))
                    re = float(Rotation.from_matrix(actual[:3, :3].T@target[:3, :3]).magnitude())
                    if pe > .002 or re > .03:
                        raise ValueError('Selected endpoint failed independent FK')
                    record['fk_checks'].append({'stage': label, 'position_error_m': pe, 'orientation_error_rad': re})
                path_counts = {}
                for label in ['pregrasp', 'approach']:
                    path = result['paths'][label]
                    q, times = np.asarray(path['q']), np.asarray(path['times'])
                    if (q.ndim != 2 or q.shape[1] != 6 or len(q) != len(times) or len(q) < 2 or
                            not np.isfinite(q).all() or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
                        raise ValueError('Invalid native timed path')
                    path_counts[label] = len(q)
                record['recorded_path_samples'] = path_counts
                if selected['source'] == 'inverse_reachability':
                    gate = read_json(result_path.parent/'endpoint_gate.json')
                    if not gate['passed'] or gate['sphere_count'] != 871:
                        raise ValueError('Map plan lacks native endpoint collision clearance')
                    record['map_endpoint_collision_passed'] = True
                if not selected['transit']['passed']:
                    raise ValueError('Missing full robot transit clearance')
            rows.append(record)
    report = {'at': now(), 'passed': True, 'recorded_searches': len(rows),
        'physical_grasps': 0, 'same_gpu': acceptance['same_gpu'], 'source_and_input_hashes_unchanged': True,
        'case_results': rows, 'auditor_sha256': sha256(__file__),
        'observed_planning_regressions': acceptance['observed_regressions'],
        'warmup_excluded_from_search_timing': True}
    write_json(root/'verification.json', report)
    print({key: report[key] for key in ['passed', 'recorded_searches', 'observed_planning_regressions']})
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    audit(ensure_run_path(args.root))
