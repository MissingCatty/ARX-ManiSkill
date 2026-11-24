"""UNREGISTERED RGB-D thin-body pose study. No IK, model updates or commands.

The original pad calibration is used verbatim. These algebraic poses are not
evidence of reachability, planning validity, table clearance or physical grasp.
"""
import numpy as np
from sphere_refinement import PAD_CENTER_EEF


def proposals(rows, camera, fit):
    if not fit.get('accepted'):
        return []
    center = np.asarray(fit['center_world'])
    axis = np.asarray(fit['axis_world'])
    normal = np.asarray(fit['closing_axis_world'])
    seed = None
    for index, row in enumerate(rows):
        if not np.isfinite(row).all():
            continue
        rotation = camera[:3, :3] @ row[4:13].reshape(3, 3)
        contact = camera[:3, :3] @ row[13:16] + camera[:3, 3] + row[3] * rotation[:, 0]
        if rotation[2, 0] < -.35 and np.linalg.norm(contact-center) < .10:
            seed = index, row, rotation[:, 0]
            break
    if seed is None:
        return []
    index, row, original_approach = seed
    seed_pitch = np.rad2deg(np.arcsin(np.clip(-original_approach[2], 0, 1)))
    pitches = sorted([90., 75., 60., 45., 30.], key=lambda x: abs(x-seed_pitch))
    directions = sorted([1., -1.], key=lambda sign: -float(sign*axis @ original_approach))
    candidates = []
    # Stay within the observed central body; no object identity or simulator pose.
    for axial_fraction in (0., -.125, .125):
        for height in (0., .003, .006, .009, .012):
            pad = center + axis * (axial_fraction*fit['length_m']) + [0., 0., height]
            for sign in (1., -1.):
                closing = sign*normal
                for pitch in pitches:
                    for direction in directions:
                        approach = np.cos(np.deg2rad(pitch))*direction*axis + [0., 0., -np.sin(np.deg2rad(pitch))]
                        rotation = np.column_stack([np.cross(closing, approach), closing, approach])
                        grasp = np.eye(4)
                        grasp[:3, :3] = rotation
                        grasp[:3, 3] = pad - rotation @ PAD_CENTER_EEF
                        pre = grasp.copy(); pre[2, 3] += .08
                        lift = grasp.copy(); lift[2, 3] += .12
                        candidates.append({
                            'index': index, 'network_row': row.tolist(), 'score': float(row[0]),
                            'width_m': float(row[1]), 'depth_m': float(row[3]),
                            'grasp_world': grasp.tolist(), 'pregrasp_world': pre.tolist(),
                            'lift_world': lift.tolist(), 'target_contact_world': pad.tolist(),
                            'research_change': {
                                'hypothesis': 'UNREGISTERED_thin_body_pose',
                                'method': 'Observed central-body pinch, tilted vertical entry',
                                'approach_down_pitch_deg': pitch, 'approach_axis_sign': direction,
                                'closing_sign': sign, 'axial_fraction': axial_fraction,
                                'world_z_pose_search_offset_m': height,
                                'pad_center_eef_m_frozen': PAD_CENTER_EEF.tolist(),
                                'pad_target_world': pad.tolist(),
                                'network_score_scope': 'Original seed only, not a learned score for the refined pose',
                                'robot_configuration_changed': False,
                                'scope': 'Pose construction only; original checks required before any future execution',
                            },
                        })
    return candidates


def select(rows, camera, fit, start, ik, path, closed_checker,
           tracking_valid, out, started):
    """UNREGISTERED adapter; requires the unchanged original checking functions.

    No standalone model query or execution has been performed for this draft.
    """
    import json
    import time
    candidates = proposals(rows, camera, fit)
    report = {'method': 'UNREGISTERED thin central-body poses',
              'proposals': len(candidates), 'examined': 0,
              'rejected': {}, 'selected': False}
    def save():
        (out/'thin_candidate_diagnostics.json').write_text(json.dumps(report, indent=2))
    def reject(key):
        report['rejected'][key] = report['rejected'].get(key, 0)+1
        save()
    for index, candidate in enumerate(candidates):
        if time.monotonic()-started > 180:
            report['timeout'] = True
            break
        report['examined'] = index+1
        qg = ik(np.asarray(candidate['grasp_world']), start)
        if qg is None:
            reject('grasp_ik'); continue
        if not closed_checker.validate(qg[None, None]).all() or not tracking_valid(qg, 'thin_grasp_goal'):
            reject('closed_grasp_goal'); continue
        qp = ik(np.asarray(candidate['pregrasp_world']), qg)
        ql = ik(np.asarray(candidate['lift_world']), qg)
        if qp is None or ql is None:
            reject('pre_or_lift_ik'); continue
        if not closed_checker.validate(ql[None, None]).all() or not tracking_valid(ql, 'thin_lift_goal'):
            reject('closed_lift_goal'); continue
        route = path(start, qp)
        if route is None:
            reject('pregrasp_route'); continue
        if path(qp, qg) is None:
            reject('approach_route'); continue
        candidate.update(q_pre=qp.tolist(), q_grasp=qg.tolist(),
                         q_lift=ql.tolist(), table_clearance_raise_m=None)
        report.update(selected=True, selected_proposal=index)
        save()
        return candidate, route
    save()
    return None, None
