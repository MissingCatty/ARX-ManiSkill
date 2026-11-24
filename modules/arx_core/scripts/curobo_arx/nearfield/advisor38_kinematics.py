"""Finite CPU ARX IK screening. No simulator, dispatcher or robot edits."""
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .advisor37_motion import FrozenURDF, prepose
from .advisor36_contract import boundary_poses
from training.nearfield_arx.contract import pose_error


class ScreenRejected(Exception):
    pass


class CPUArm:
    def __init__(self, urdf, snapshot):
        self.model = FrozenURDF(urdf)
        if not self.model.verify(snapshot)['pass']:
            raise RuntimeError('Archived ARX link frames disagree with original URDF')
        self.snapshot = snapshot
        self.names = snapshot['right_arm']
        self.measured = dict(zip(snapshot['joint_names'], snapshot['joint_positions']))
        self.start = np.array([self.measured[n] for n in self.names])
        self.limits = np.array([snapshot['joint_limits'][snapshot['joint_names'].index(n)] for n in self.names])
        parents = {j[3]: j for j in self.model.joints}
        self.chain, link = [], 'right_eef_link'
        while link != 'body4':
            joint = parents[link]
            self.chain.append(joint)
            link = joint[2]
        self.chain.reverse()
        self.base = np.array(snapshot['link_world']['body4'])
        self.count = 0

    def fk(self, q):
        values = {**self.measured, **dict(zip(self.names, q))}
        matrix = self.base.copy()
        for name, kind, parent, child, origin, axis in self.chain:
            motion = np.eye(4)
            if kind == 'revolute':
                motion[:3, :3] = Rotation.from_rotvec(axis * values[name]).as_matrix()
            elif kind == 'prismatic':
                motion[:3, 3] = axis * values[name]
            elif kind != 'fixed':
                raise RuntimeError('Unsupported original joint type')
            matrix = matrix @ origin @ motion
        return matrix

    def solve(self, goal, seed, deadline, *, multistart=False):
        lo, hi = self.limits.T
        rng = np.random.default_rng(3812026)
        starts = [np.clip(seed, lo + 1e-8, hi - 1e-8)]
        if multistart:
            starts += [np.clip(self.start, lo + 1e-8, hi - 1e-8), (lo + hi) / 2]
            starts += list(rng.uniform(lo + 1e-8, hi - 1e-8, (9, len(lo))))

        def residual(q):
            if time.monotonic() > deadline:
                raise ScreenRejected('cpu_kinematic_time_budget_exhausted')
            actual = self.fk(q)
            return np.r_[actual[:3, 3] - goal[:3, 3],
                         .10 * Rotation.from_matrix(goal[:3, :3].T @ actual[:3, :3]).as_rotvec()]

        for start in starts:
            self.count += 1
            result = least_squares(residual, start, bounds=(lo, hi), max_nfev=60 if multistart else 12,
                                   ftol=1e-9, xtol=1e-9, gtol=1e-9)
            error = pose_error(self.fk(result.x), goal)
            if error['translation_m'] <= .0005 and error['rotation_rad'] <= np.deg2rad(.2):
                return result.x
        raise ScreenRejected('finite_cpu_IK_search_found_no_solution')

    def screen(self, canonical, wall_seconds=180.):
        started = time.monotonic()
        deadline = started + wall_seconds
        cases = []
        try:
            for case in boundary_poses(canonical, 'coupled_xyz'):
                goal = np.array(case['goal_world'])
                row = {'case': case['case'], 'target_world': goal.tolist(), 'pass': False}
                cases.append(row)
                pre = prepose(goal)
                qpre = self.solve(pre, self.start, deadline, multistart=True)
                qgoal = self.solve(goal, qpre, deadline, multistart=True)
                top = goal.copy()
                top[2, 3] += .120
                self.solve(top, qgoal, deadline, multistart=True)
                insertion = [qpre]
                for distance in np.linspace(.001, .060, 60):
                    target = pre.copy()
                    target[:3, 3] += pre[:3, 2] * distance
                    insertion.append(self.solve(target, insertion[-1], deadline))
                lift = [insertion[-1]]
                for distance in np.linspace(.001, .120, 120):
                    target = goal.copy()
                    target[2, 3] += distance
                    lift.append(self.solve(target, lift[-1], deadline))
                row.update(pass_=True, pregrasp_q=qpre.tolist(), insertion_q=np.array(insertion).tolist(),
                           lift_q=np.array(lift).tolist(), nominal_open_gripper_only=True)
                row['pass'] = True
            return {'pass': True, 'cases': cases, 'seconds': time.monotonic() - started,
                    'IK_solves': self.count, 'physical_commands': 0, 'GPU_collision_check_pending': True}
        except ScreenRejected as error:
            return {'pass': False, 'cases': cases, 'rejected': str(error),
                    'seconds': time.monotonic() - started, 'IK_solves': self.count,
                    'physical_commands': 0, 'global_unreachability_proven': False}
