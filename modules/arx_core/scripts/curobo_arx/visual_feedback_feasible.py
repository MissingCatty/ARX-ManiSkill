"""Feasibility-aware sampling adapter for the ARX VFAS-inspired experiment.

Preserves the geometric proxy and all physical/planning gates. Unlike the first
adapter, a geometrically attractive but unreachable pose cannot move the next
sampling center. Small local probes complement the nominal VFAS-sized samples.
"""
import json
import time

import numpy as np
from scipy.spatial.transform import Rotation

from visual_feedback import VisualFeedbackRefiner as BaseRefiner, contact_quality


class VisualFeedbackRefiner(BaseRefiner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.report['policy'] = 'vfas_inspired_feasibility_aware_geometric_feedback'
        self.report['sampling_policy'] = 'Update center only after ARX IK/collision/path admission; add 3 mm / 3 degree probes; permit rotation about the closing axis to adjust ARX tabletop approach tilt'
        self.persist()

    def refine(self, selected, start, ik, route, open_checker, closed_checker,
               ik_diagnostics=None):
        started = time.monotonic()
        points, normals = self.observe()
        seed = np.asarray(selected['grasp_world']).copy()
        center = seed.copy()
        seed_quality = float(contact_quality(seed[None], points, normals)[0][0])
        best_score = seed_quality
        best_candidate, best_joints = None, None
        checks, rounds, cache = [], [], {}

        def admissible(candidate):
            grasp = np.asarray(candidate['pose_world'])
            key = tuple(np.round(grasp[:3].flatten(), 6))
            if key in cache:
                return cache[key]
            entry = {'score': candidate['score'], 'reason': None}
            before = dict(ik_diagnostics or {})
            qg = ik(grasp, start)
            ql = None
            if qg is None:
                entry['reason'] = 'grasp_ik'
            elif not closed_checker.validate(qg[None, None]).all():
                entry['reason'] = 'nominal_closed_collision'
            else:
                lift = grasp.copy(); lift[2, 3] += .12
                ql = ik(lift, qg)
                if ql is None or not closed_checker.validate(ql[None, None]).all():
                    entry['reason'] = 'lift_ik_or_collision'
                elif route(start, qg) is None:
                    entry['reason'] = 'approach_route'
            entry['ik_diagnostic_increments'] = {k: v - before.get(k, 0)
                                                  for k, v in (ik_diagnostics or {}).items()}
            checks.append(entry)
            cache[key] = None if entry['reason'] else (qg, ql)
            return cache[key]

        for round_index in range(3):
            count = int(16 * self.scale)
            perturb = np.repeat(np.eye(4)[None], count + 28, axis=0)
            perturb[:count, :3, 3] = self.rng.uniform(-.02 * self.scale, .02 * self.scale, (count, 3))
            angles = self.rng.uniform(-5 * self.scale, 5 * self.scale, (count, 3))
            perturb[:count, :3, :3] = Rotation.from_euler('xyz', angles, degrees=True).as_matrix()
            # Dense small probes avoid ranking only table-penetrating or
            # unreachable candidates around a seed near a workspace boundary.
            perturb[count:count + 16, :3, 3] = self.rng.uniform(-.005, .005, (16, 3))
            for axis in range(3):
                perturb[count + 16 + 2 * axis, axis, 3] = .003
                perturb[count + 17 + 2 * axis, axis, 3] = -.003
            for i, (axis, sign) in enumerate(((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1))):
                angle = np.zeros(3); angle[axis] = sign * 3
                perturb[count + 22 + i, :3, :3] = Rotation.from_euler('xyz', angle, degrees=True).as_matrix()
            poses = np.concatenate([center[None], center @ perturb], axis=0)
            noisy = [points, points + self.rng.normal(0, .001, points.shape),
                     points + self.rng.normal(0, .001, points.shape)]
            qualities = np.stack([contact_quality(poses, cloud, normals)[0] for cloud in noisy])
            mean, spread = qualities.mean(axis=0), np.ptp(qualities, axis=0)
            translation = np.linalg.norm(poses[:, :3, 3] - seed[:3, 3], axis=1)
            angle = Rotation.from_matrix(poses[:, :3, :3] @ seed[:3, :3].T).magnitude()
            scores = mean - 2. * translation - .1 * angle - .6 * spread
            details = contact_quality(poses, points, normals)[1]
            samples = []
            for i, pose in enumerate(poses):
                samples.append({'pose_world': pose.tolist(), 'score': float(scores[i]),
                                'quality_mean': float(mean[i]), 'quality_noise_range': float(spread[i]),
                                'seed_distance_m': float(translation[i]), 'seed_angle_rad': float(angle[i]),
                                **{k: float(v[i]) for k, v in details.items()}})
            admitted_this_round = False
            for index in np.argsort(scores)[::-1]:
                candidate = samples[int(index)]
                if candidate['score'] <= best_score + .001:
                    break
                if np.allclose(candidate['pose_world'], seed, atol=1e-7):
                    continue
                joints = admissible(candidate)
                if joints is not None:
                    best_score, best_candidate, best_joints = candidate['score'], candidate, joints
                    center = np.asarray(candidate['pose_world'])
                    admitted_this_round = True
                    break
            rounds.append({'round': round_index, 'scale': self.scale, 'samples': samples,
                           'feasible_center_updated': admitted_this_round,
                           'best_feasible_score': best_score})
            self.scale = min(3., self.scale * 1.3) if best_score < .5 else 1.
        if best_candidate is None:
            chosen = selected
        else:
            grasp = np.asarray(best_candidate['pose_world'])
            lift = grasp.copy(); lift[2, 3] += .12
            qg, ql = best_joints
            chosen = {**selected, 'grasp_world': grasp.tolist(), 'lift_world': lift.tolist(),
                      'q_grasp': qg.tolist(), 'q_lift': ql.tolist(),
                      'target_contact_world': (grasp[:3, 3] + grasp[:3, :3] @ [-.006, 0, .0375]).tolist(),
                      'visual_feedback': best_candidate}
        update = {'iteration': self.iteration, 'initial_pose': seed.tolist(),
                  'seed_geometric_quality': seed_quality, 'rounds': rounds,
                  'checked_candidates': checks, 'applied': chosen is not selected,
                  'pose_changed': not np.allclose(chosen['grasp_world'], seed, atol=1e-7),
                  'chosen_pose': chosen['grasp_world'], 'wall_seconds': time.monotonic() - started}
        self.report['refinements'].append(update)
        self.iteration += 1
        self.persist()
        print('VISUAL_FEEDBACK', json.dumps({k: update[k] for k in
              ('iteration', 'seed_geometric_quality', 'applied', 'pose_changed', 'wall_seconds')}), flush=True)
        return chosen
