"""H027 selective lower-risk candidate ordering with the frozen pose retained.

Inspired by GraspNet observed approach occupancy; not a learned model or an
implementation of its rectangular-gripper collision volume.
"""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from cached_observed_approach_quality import CachedObservedApproachQuality
from grasp_geometry_candidates import construct


def select(original, route, folder, start, cube, ik, path, closed_checker, tracking_valid, enabled):
    folder = Path(folder)
    began = time.monotonic()
    quality = CachedObservedApproachQuality(folder)
    baseline_risk = quality.evaluate(original['grasp_world'], original['pregrasp_world'])
    diagnostic = dict(hypothesis='H027', applied=False, enabled=enabled, original_risk=baseline_risk,
                      baseline_fallback_retained=True, candidate_rows_cap=32, pose_cap=320,
                      alternate_model_pose_cap=24, alternate_IK_calls=0, alternate_graph_calls=0,
                      assessed_alternates=[], physical_configuration_changed=False)

    def save():
        diagnostic['elapsed_s'] = time.monotonic() - began
        (folder / 'approach_rerank_policy.json').write_text(json.dumps(diagnostic, indent=2))

    (folder / 'selected_grasp_before_rerank.json').write_text(json.dumps(original, indent=2))
    save()
    if not enabled or not baseline_risk['potential_open_finger_contact']:
        diagnostic['reason'] = 'Control arm' if not enabled else 'Original pose below fixed risk trigger'
        save()
        return original, route
    rows = np.load(folder / 'zerograsp/object_1.grasp.npy')
    cam = np.asarray(json.loads((folder / 'camera_world_optical.json').read_text()))
    axes = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)
    pool, seen = [], set()
    for index, row in enumerate(rows[:32]):
        if time.monotonic() - began > 60:
            diagnostic['pool_time_cap_reached'] = True
            break
        rg = cam[:3, :3] @ row[4:13].reshape(3, 3)
        surface = cam[:3, :3] @ row[13:16] + cam[:3, 3]
        contact = surface + row[3] * rg[:, 0]
        if rg[2, 0] > -.35 or np.linalg.norm(contact - cube) > .065:
            continue
        clearance_raise = max(0., float(cube[2] + .015 - contact[2]))
        if clearance_raise > .030:
            continue
        contact[2] += clearance_raise
        for flip_index, flip in enumerate((np.eye(3), np.diag([1, -1, -1]))):
            raw = np.eye(4)
            raw[:3, :3] = rg @ flip @ axes
            raw[:3, 3] = contact - .040 * rg[:, 0]
            for shift in (0., .003, .006, .009, .012):
                grasp, pre, lift, meta = construct(raw, contact, {'accepted': False}, shift)
                key = hashlib.sha256(np.round(grasp, 10).tobytes()).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                risk = quality.evaluate(grasp, pre)
                fraction = risk['finger_sweep_point_fraction']
                candidate = dict(index=index, score=float(row[0]), width_m=float(row[1]), depth_m=float(row[3]),
                                 table_clearance_raise_m=clearance_raise, network_row=row.tolist(),
                                 target_contact_world=contact.tolist(), grasp_world=grasp.tolist(),
                                 pregrasp_world=pre.tolist(), lift_world=lift.tolist(), research_change=meta)
                pool.append(dict(candidate=candidate, risk=risk, fraction=fraction,
                                 index=index, flip_index=flip_index, shift_m=shift))
    diagnostic['pool_poses'] = len(pool)
    # This threshold gates only trying an ALTERNATIVE. The baseline is retained.
    lower = [item for item in pool if item['fraction'] < .01
             and item['fraction'] < baseline_risk['finger_sweep_point_fraction']]
    lower.sort(key=lambda x: (x['fraction'], x['index'], x['flip_index'], x['shift_m']))
    diagnostic['eligible_alternates'] = len(lower)
    (folder / 'approach_rerank_pool.json').write_text(json.dumps(pool, indent=2))
    search = time.monotonic()
    for item in lower[:24]:
        if time.monotonic() - search > 180:
            diagnostic['model_search_time_cap_reached'] = True
            break
        candidate = item['candidate']
        record = {k: item[k] for k in ('fraction', 'index', 'flip_index', 'shift_m')}
        diagnostic['assessed_alternates'].append(record)

        def solve(pose, seed):
            diagnostic['alternate_IK_calls'] += 1
            return ik(np.asarray(pose), seed)

        def graph(a, b):
            diagnostic['alternate_graph_calls'] += 1
            return path(a, b)

        qg = solve(candidate['grasp_world'], start)
        if qg is None:
            record['rejected'] = 'grasp_ik'; save(); continue
        if not closed_checker.validate(qg[None, None]).all():
            record['rejected'] = 'grasp_closed'; save(); continue
        if not tracking_valid(qg, 'rerank_grasp_goal'):
            record['rejected'] = 'grasp_65_state'; save(); continue
        qp = solve(candidate['pregrasp_world'], qg)
        ql = solve(candidate['lift_world'], qg)
        if qp is None or ql is None:
            record['rejected'] = 'pre_or_lift_ik'; save(); continue
        if not closed_checker.validate(ql[None, None]).all():
            record['rejected'] = 'lift_closed'; save(); continue
        if not tracking_valid(ql, 'rerank_lift_goal'):
            record['rejected'] = 'lift_65_state'; save(); continue
        alternate_route = graph(start, qp)
        if alternate_route is None or graph(qp, qg) is None:
            record['rejected'] = 'route'; save(); continue
        candidate.update(q_pre=qp.tolist(), q_grasp=qg.tolist(), q_lift=ql.tolist())
        candidate['research_change'].update(hypothesis='H027', applied=True,
                                            sampled_tracking_delta_rad=.003,
                                            original_pose_retained_as_fallback=True,
                                            original_index=original['index'], observed_approach_risk=item['risk'])
        record['admitted'] = True
        diagnostic.update(applied=True, selected_risk=item['risk'],
                          reason='First lower-risk alternative passing unchanged original model/path checks')
        save()
        return candidate, alternate_route
    diagnostic['reason'] = 'No admitted lower-risk alternative; retain original pose and route'
    save()
    return original, route
