"""Experimental neural-orientation/body-centre proposals; preregister before use.

Pose construction is pure CPU. No model checks/commands occur at import.
The original checking callbacks must validate every eventual selected pose.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from grasp_geometry_candidates import construct
from sphere_refinement import PAD_CENTER_EEF


def proposals(rows, camera, fit):
    if not fit.get('accepted'):
        return []
    center = np.asarray(fit['center_world'])
    axis = np.asarray(fit['axis_world'])
    convention = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)
    seeds = []
    for index, row in enumerate(rows):
        if not np.isfinite(row).all():
            continue
        rg = camera[:3, :3] @ row[4:13].reshape(3, 3)
        contact = camera[:3, :3] @ row[13:16] + camera[:3, 3] + row[3] * rg[:, 0]
        if rg[2, 0] >= -.35 or np.linalg.norm(contact-center) >= .10:
            continue
        rotations = []
        for flip in (np.eye(3), np.diag([1., -1., -1.])):
            original = np.eye(4)
            original[:3, :3] = rg @ flip @ convention
            refined, _, _, _ = construct(original, center, {'accepted': False})
            rotation = refined[:3, :3]
            # Oblique cross-body pinches up to45degrees from the normal are
            # allowed; pinches directed along the long axis are excluded.
            if abs(rotation[:, 1] @ axis) <= np.sqrt(.5):
                rotations.append(rotation)
        if rotations:
            seeds.append((index, row, contact, rotations))
        if len(seeds) == 32:
            break
    candidates = []
    for axial in (0., -.125, .125):
        for yaw in (0., 15., -15., 30., -30.):
            yaw_rotation = Rotation.from_rotvec([0., 0., np.deg2rad(yaw)]).as_matrix()
            for height in (0., .003, .006, .009, .012):
                pad = center + axial*fit['length_m']*axis + [0., 0., height]
                for index, row, contact, rotations in seeds:
                    for seed_rotation in rotations:
                        rotation = yaw_rotation @ seed_rotation
                        if abs(rotation[:, 1] @ axis) > np.sqrt(.5):
                            continue
                        grasp = np.eye(4)
                        grasp[:3, :3] = rotation
                        grasp[:3, 3] = pad - rotation @ PAD_CENTER_EEF
                        pre = grasp.copy(); pre[2, 3] += .08
                        lift = grasp.copy(); lift[2, 3] += .12
                        candidates.append({
                            'index': index, 'network_row': row.tolist(),
                            'score': float(row[0]), 'width_m': float(row[1]), 'depth_m': float(row[3]),
                            'grasp_world': grasp.tolist(), 'pregrasp_world': pre.tolist(),
                            'lift_world': lift.tolist(), 'target_contact_world': pad.tolist(),
                            'research_change': {
                                'hypothesis': 'H016_thin_neural_orientation',
                                'method': 'Observed central-body targets with original continuous neural approach angles',
                                'seed_grasp_index': index,
                                'world_yaw_search_deg': yaw,
                                'approach_down_pitch_deg': float(np.rad2deg(np.arcsin(np.clip(-rotation[2, 2], -1, 1)))),
                                'closing_dot_body_axis': float(abs(rotation[:, 1] @ axis)),
                                'axial_fraction': axial, 'world_z_pose_search_offset_m': height,
                                'pad_center_eef_m_frozen': PAD_CENTER_EEF.tolist(),
                                'pad_target_world': pad.tolist(), 'seed_contact_world': contact.tolist(),
                                'network_score_scope': 'Original seed only, not learned score for refined pose',
                                'robot_configuration_changed': False,
                                'scope': 'Candidate construction only; all original model/trajectory checks mandatory before execution',
                            },
                        })
    return candidates

def select(rows, camera, fit, start, ik, path, closed_checker,
           tracking_valid, out, started):
    """Experimental adapter; requires the unchanged original checking functions.

    No standalone model query or execution has been performed for this draft.
    """
    import json
    import time
    candidates = proposals(rows, camera, fit)
    report = {'method': 'H016 neural-orientation central-body poses',
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
