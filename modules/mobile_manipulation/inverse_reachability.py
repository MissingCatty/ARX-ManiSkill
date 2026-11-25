"""Keep world grasps fixed and solve base/lift placement from FK witnesses."""
from copy import deepcopy
import time

import numpy as np
from scipy.spatial.transform import Rotation

from modules.arx_core.arms import get_arm
from .placement import pose_matrix


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


class InverseReachability:
    def __init__(self, index, model, action_transform, options):
        self.index, self.model, self.options = index, model, dict(options)
        self.action = np.asarray(action_transform, dtype=float)
        if not np.allclose(self.action, index.meta['T_eef_action'], atol=1e-12):
            raise ValueError('Current TCP and map TCP differ')
        if self.options['position_tolerance_m'] != .002 or self.options['orientation_tolerance_rad'] != .03:
            raise ValueError('Placement must retain the original full-pose tolerances')

    def generate(self, state, candidates, settings, *, start=0):
        started = time.perf_counter()
        base = np.asarray(state['links']['base_link'], dtype=float)
        yaw0 = float(Rotation.from_matrix(base[:3, :3]).as_euler('xyz')[2])
        if np.max(np.abs(base[2, :3]-[0., 0., 1.])) > 1e-5:
            raise ValueError('Inverse placement requires a level SE2 base')
        mount_state = deepcopy(state)
        mount_state['links'] = {'base_link': np.eye(4).tolist()}
        lift_idx = state['joint_names'].index('joint4')
        mount_state['qpos'][lift_idx] = 0.
        mount = self.model.frames(mount_state)['body4']
        mount_state['qpos'][lift_idx] = .1
        up = self.model.frames(mount_state)['body4']
        if not np.allclose(up[:3, 3]-mount[:3, 3], [0, 0, .1], atol=1e-8):
            raise ValueError('Lift does not follow the registered world vertical')
        if not np.allclose(mount[:3, :3], np.eye(3), atol=1e-8):
            raise ValueError('Unsupported base/body4 mounting orientation')
        tolerance = self.options['orientation_tolerance_rad']
        yaw_limit = np.deg2rad(max(abs(a) for a in settings['yaw_offsets_degrees']))
        lower_lift, upper_lift = state['joint_limits'][lift_idx]
        results, statistics = [], []
        stop = min(len(candidates), start+self.options['candidate_query_limit'])
        for rank, candidate in enumerate(candidates[start:stop], start=start):
            contact = np.asarray(candidate['pose'], dtype=float)
            if contact.shape != (4, 4) or not np.isfinite(contact).all():
                raise ValueError('Invalid grasp pose')
            # Candidate positions are TCP43, but candidate rotations are EEF axes.
            eef = contact.copy()
            eef[:3, 3] -= self.action[2, 3]*contact[:3, 2]
            goal = eef@self.action
            indices = self.index.query(goal[:3, :3], tolerance)
            statistic = {'candidate_index': rank, 'orientation_neighborhood': len(indices),
                         'feasible_witnesses': {}, 'map_miss_is_unreachable': False}
            if not len(indices):
                statistics.append(statistic)
                continue
            rows = np.asarray(self.index.bank[indices], dtype=float)
            rotations = Rotation.from_quat(rows[:, [10, 11, 12, 9]]).as_matrix()
            delta = goal[:3, :3][None]@rotations.transpose(0, 2, 1)
            a, b = delta[:, 0, 0]+delta[:, 1, 1], delta[:, 1, 0]-delta[:, 0, 1]
            yaw = yaw0+np.clip(wrap(np.arctan2(b, a)-yaw0), -yaw_limit, yaw_limit)
            error = np.arccos(np.clip((np.cos(yaw)*a+np.sin(yaw)*b+delta[:, 2, 2]-1)/2, -1., 1.))
            for side in ['right', 'left']:
                arm = get_arm(side)
                points = rows[:, 6:9].copy()
                if side == 'left':
                    points[:, 1] += self.model.left_from_right[1, 3]
                points += mount[:3, 3]
                h = goal[2, 3]-base[2, 3]-points[:, 2]
                px = np.cos(yaw)*points[:, 0]-np.sin(yaw)*points[:, 1]
                py = np.sin(yaw)*points[:, 0]+np.cos(yaw)*points[:, 1]
                xy = goal[:2, 3]-np.column_stack((px, py))
                distance = np.linalg.norm(xy-base[:2, 3], axis=1)
                bounds = np.array([state['joint_limits'][state['joint_names'].index(n)] for n in arm.joints])
                good = ((error <= tolerance) & (h >= lower_lift+.002) & (h <= upper_lift-.002) &
                        (distance <= settings['maximum_base_travel_m']) &
                        ((rows[:, :6] >= bounds[:, 0]) & (rows[:, :6] <= bounds[:, 1])).all(axis=1))
                valid = np.flatnonzero(good)
                statistic['feasible_witnesses'][side] = len(valid)
                current_q = np.array([state['qpos'][state['joint_names'].index(n)] for n in arm.joints])
                cost = (distance + .12*np.abs(wrap(yaw-yaw0)) + .3*np.abs(h-state['qpos'][lift_idx]) +
                        .01*np.linalg.norm((rows[:, :6]-current_q)/(bounds[:, 1]-bounds[:, 0]), axis=1))
                kept = []
                for i in valid[np.argsort(cost[valid], kind='stable')]:
                    # Diversify station locations without moving or changing a grasp.
                    if any(np.linalg.norm(xy[i]-xy[j]) < .03 and abs(h[i]-h[j]) < .02
                           and abs(wrap(yaw[i]-yaw[j])) < np.deg2rad(3.) for j in kept):
                        continue
                    new_base = pose_matrix(*xy[i], yaw[i], base[2, 3])
                    witness = self.verify(state, new_base, h[i], rows[i, :6], side, goal)
                    results.append({'base_world': new_base.tolist(), 'lift_m': float(h[i]),
                        'cost': float(cost[i]), 'source': 'inverse_reachability', 'arm': side,
                        'candidate_index': rank, 'source_map_row': int(indices[i]),
                        'witness': witness, 'base_travel_m': float(distance[i]),
                        'grasp_world_pose_unchanged': True})
                    kept.append(i)
                    if len(kept) >= self.options['witnesses_per_candidate_arm']:
                        break
            statistics.append(statistic)
        results.sort(key=lambda p: (p['candidate_index'], p['cost'], p['arm'] != 'right', p['source_map_row']))
        return results, {'query_seconds': time.perf_counter()-started, 'queried_candidates': stop-start,
                         'first_candidate_index': start,
                         'full_candidate_count': len(candidates), 'proposals': len(results),
                         'candidate_queries': statistics, 'grasp_world_poses_changed': False,
                         'unqueried_or_uncovered_candidates_retained_for_fallback': True}

    def verify(self, state, base, lift, q, side, goal):
        arm = get_arm(side)
        checked = deepcopy(state)
        checked['links'] = {'base_link': np.asarray(base).tolist()}
        for name, value in zip(arm.joints+('joint4',), [*q, lift]):
            i = checked['joint_names'].index(name)
            lower, upper = checked['joint_limits'][i]
            if not np.isfinite(value) or not lower <= value <= upper:
                raise ValueError('Map witness violates physical joint limits')
            checked['qpos'][i] = float(value)
        achieved_eef = self.model.frames(checked)[arm.eef]
        actual = achieved_eef@self.action
        pe = float(np.linalg.norm(actual[:3, 3]-goal[:3, 3]))
        re = float(Rotation.from_matrix(actual[:3, :3].T@goal[:3, :3]).magnitude())
        if pe > self.options['position_tolerance_m'] or re > self.options['orientation_tolerance_rad']:
            raise ValueError('Map witness failed independent full-robot FK')
        return {'q': np.asarray(q).tolist(), 'joint_names': list(arm.joints), 'joint_limits_clear': True,
                'position_error_m': pe, 'orientation_error_rad': re,
                'action_target_world': goal.tolist(),
                'achieved_eef_world': achieved_eef.tolist(), 'fk_passed': True}
