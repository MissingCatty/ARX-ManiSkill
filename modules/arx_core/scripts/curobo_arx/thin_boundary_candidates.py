"""Experimental local height search under the unchanged ARX goal checks.

This searches target poses, never robot/model/physics parameters. Local binary
refinement does not assume/prove global monotonicity or a global minimum.
"""
import copy
import json
import time
import numpy as np

from thin_neural_candidates import proposals as neural_proposals
from sphere_refinement import PAD_CENTER_EEF


def profiles(rows, camera, fit):
    if not fit.get('accepted'):
        return []
    seeds = [c for c in neural_proposals(rows, camera, fit)
             if c['research_change']['world_yaw_search_deg'] == 0
             and c['research_change']['axial_fraction'] == 0
             and c['research_change']['world_z_pose_search_offset_m'] == 0]
    if not seeds:
        return []
    first = seeds[0]
    lowest = min(seeds, key=lambda c: c['research_change']['approach_down_pitch_deg'])
    indices = {first['index'], lowest['index']}
    rotations = [s for s in seeds if s['index'] in indices]
    center = np.asarray(fit['center_world'])
    axis = np.asarray(fit['axis_world'])
    contact = np.asarray(first['research_change']['seed_contact_world'])
    direction = 1. if float((contact-center) @ axis) >= 0 else -1.
    distance = abs(float((contact-center) @ axis))/fit['length_m']
    fractions = sorted([0., .125, .25, .325], key=lambda f: abs(f-distance))
    result = []
    for fraction in fractions:
        origin = center+direction*fraction*fit['length_m']*axis
        for seed in rotations:
            result.append({'index': seed['index'], 'network_row': seed['network_row'],
                           'score': seed['score'], 'width_m': seed['width_m'],
                           'depth_m': seed['depth_m'],
                           'rotation_world': np.asarray(seed['grasp_world'])[:3, :3].tolist(),
                           'axis_origin_world': origin.tolist(),
                           'research_change': {
                               'hypothesis': 'H017_thin_boundary',
                               'method': 'Local lower-feasible-height search with all original robust goal checks',
                               'axial_fraction_toward_original_neural_contact': fraction,
                               'approach_down_pitch_deg': seed['research_change']['approach_down_pitch_deg'],
                               'seed_contact_world': contact.tolist(),
                               'pad_center_eef_m_frozen': PAD_CENTER_EEF.tolist(),
                               'network_score_scope': 'Original seed only, not learned refined-pose score',
                               'robot_configuration_changed': False,
                               'scope': 'Finite local height heuristic, not global monotonicity/minimum or contact guarantee',
                           }})
    return result


def at_height(profile, height):
    if not .012-1e-12 <= height <= .018+1e-12:
        raise ValueError('Height outside preregistered local target search')
    rotation = np.asarray(profile['rotation_world'])
    pad = np.asarray(profile['axis_origin_world'])+[0., 0., height]
    grasp = np.eye(4)
    grasp[:3, :3] = rotation
    grasp[:3, 3] = pad-rotation @ PAD_CENTER_EEF
    pre = grasp.copy(); pre[2, 3] += .08
    lift = grasp.copy(); lift[2, 3] += .12
    result = {k: copy.deepcopy(v) for k, v in profile.items()
              if k not in ('rotation_world', 'axis_origin_world')}
    result.update(grasp_world=grasp.tolist(), pregrasp_world=pre.tolist(),
                  lift_world=lift.tolist(), target_contact_world=pad.tolist())
    result['research_change'].update(height_above_fit_axis_m=float(height),
                                     sampled_tracking_delta_rad=.003)
    return result


def select(rows, camera, fit, start, ik, path, closed_checker,
           tracking_valid, out, started):
    candidates = profiles(rows, camera, fit)
    report = {'method': 'H017 local target-height refinement',
              'proposals': len(candidates), 'examined': 0, 'goal_queries': 0,
              'rejected': {}, 'selected': False, 'queries': [],
              'goal_queries_scope': 'Grasp-target/initial-guess IK calls; original pre/lift IK calls separate',
              'scope': 'Profiles, not individual goals. Original model/checkers/3mrad65state test unchanged.'}

    class SearchBudgetExceeded(Exception):
        pass

    def save():
        (out/'thin_candidate_diagnostics.json').write_text(json.dumps(report, indent=2))

    def budget():
        if time.monotonic()-started > 180:
            report['timeout'] = True
            save()
            raise SearchBudgetExceeded

    def reject(key):
        report['rejected'][key] = report['rejected'].get(key, 0)+1

    def goal(profile, height, seed, stage):
        budget()
        candidate = at_height(profile, height)
        q = ik(np.asarray(candidate['grasp_world']), seed)
        entry = {'profile': report['examined']-1, 'height_m': height,
                 'stage': stage, 'ik_accepted': q is not None}
        report['goal_queries'] += 1
        report['queries'].append(entry)
        if q is None:
            reject('grasp_ik'); save(); return None
        entry['q'] = q.tolist()
        entry['closed_nominal_valid'] = bool(closed_checker.validate(q[None, None]).all())
        if not entry['closed_nominal_valid']:
            reject('closed_grasp_goal'); save(); return None
        entry['robust_goal_valid'] = bool(tracking_valid(q, 'boundary_grasp_goal'))
        save()
        if not entry['robust_goal_valid']:
            reject('tracking_grasp_goal'); return None
        return candidate, q

    try:
        for index, profile in enumerate(candidates):
            budget()
            report['examined'] = index+1
            # A higher pose is only a computational initial guess, never an
            # intermediate robot command or an unvalidated selected target.
            high = at_height(profile, .018)
            qhigh = ik(np.asarray(high['grasp_world']), start)
            report['goal_queries'] += 1
            report['queries'].append({'profile': index, 'height_m': .018,
                                      'stage': 'initial_guess_only',
                                      'ik_accepted': qhigh is not None,
                                      'q': qhigh.tolist() if qhigh is not None else None})
            save()
            seed = qhigh if qhigh is not None else start
            best = None
            for height in (.012, .013, .014, .015, .016, .017, .018):
                trial = goal(profile, height, seed, 'coarse')
                if trial is not None:
                    best = trial
                    break
            if best is None:
                continue
            upper = best[0]['research_change']['height_above_fit_axis_m']
            lower = max(.012, upper-.001)
            # Nonmonotonic constraints are possible. Retain a validated best
            # even when an unsuccessful midpoint discards other local branches.
            for _ in range(5):
                if upper-lower < 1e-9:
                    break
                midpoint = .5*(lower+upper)
                trial = goal(profile, midpoint, best[1], 'local_refinement')
                if trial is not None:
                    best = trial; upper = midpoint
                else:
                    lower = midpoint
            offset = min(.018, upper+.0001)
            trial = goal(profile, offset, best[1], 'positive_0p1mm_target_offset')
            if trial is not None:
                best = trial
            selected, qg = best
            qp = ik(np.asarray(selected['pregrasp_world']), qg)
            ql = ik(np.asarray(selected['lift_world']), qg)
            budget()
            if qp is None or ql is None:
                reject('pre_or_lift_ik'); save(); continue
            if not closed_checker.validate(ql[None, None]).all() or not tracking_valid(ql, 'boundary_lift_goal'):
                reject('closed_lift_goal'); save(); continue
            route = path(start, qp)
            budget()
            if route is None:
                reject('pregrasp_route'); save(); continue
            if path(qp, qg) is None:
                reject('approach_route'); save(); continue
            budget()
            selected.update(q_pre=qp.tolist(), q_grasp=qg.tolist(), q_lift=ql.tolist(),
                            table_clearance_raise_m=None)
            selected['research_change'].update(local_refinement_iterations=5,
                                                original_height_interval_m=[.012, .018])
            report.update(selected=True, selected_proposal=index,
                          selected_height_m=selected['research_change']['height_above_fit_axis_m'])
            save()
            return selected, route
    except SearchBudgetExceeded:
        pass
    save()
    return None, None
