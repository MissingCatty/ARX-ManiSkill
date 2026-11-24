"""Bounded seven-joint pose projection with analytic FK derivatives; no robot I/O."""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from 公共 import URDF, JOINTS


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def rotation_log_jacobian_inverse(v):
    theta = np.linalg.norm(v)
    k = skew(v)
    coefficient = 1/12+theta**2/720 if theta < 1e-5 else (1-.5*theta/np.tan(.5*theta))/theta**2
    return np.eye(3)-.5*k+coefficient*k@k


class JointProjection:
    def __init__(self, state, tcp, position_limit=.005, rotation_limit_deg=3., contact_limit=.003):
        self.state = state
        self.tcp = np.asarray(tcp, dtype=float)
        self.body = np.asarray(state['links']['body4'], dtype=float)
        self.names = ['joint4']+JOINTS
        self.current = np.array([state['qpos'][state['joint_names'].index(n)] for n in self.names])
        self.bounds = np.array([state['joint_limits'][state['joint_names'].index(n)] for n in self.names])
        self.position_limit, self.rotation_limit, self.contact_limit = position_limit, np.deg2rad(rotation_limit_deg), contact_limit
        self.lift_axis = np.asarray(state['links']['base_link'])[:3, :3]@np.array([0., 0., 1.])
        root = ET.parse(URDF).getroot()
        parents = {j.find('child').get('link'): j for j in root.findall('joint')}
        chain, name = [], 'right_eef_link'
        while name != 'body4':
            j = parents[name]
            chain.append(j)
            name = j.find('parent').get('link')
        self.chain = []
        for j in reversed(chain):
            node = j.find('origin')
            position = np.fromstring(node.get('xyz', '0 0 0'), sep=' ')
            rotation = Rotation.from_euler('xyz', np.fromstring(node.get('rpy', '0 0 0'), sep=' ')).as_matrix()
            index = 1+JOINTS.index(j.get('name')) if j.get('name') in JOINTS else None
            axis = None if index is None else np.fromstring(j.find('axis').get('xyz'), sep=' ')
            if index is not None:
                limits = [float(j.find('limit').get(k)) for k in ['lower', 'upper']]
                assert np.max(np.abs(limits-self.bounds[index])) < 1e-5
            self.chain.append((position, rotation, index, axis))

    def fk(self, q):
        q = np.asarray(q, dtype=float)
        rotation = self.body[:3, :3].copy()
        position = self.body[:3, 3]+self.lift_axis*(q[0]-self.current[0])
        axes = np.zeros((3, 7))
        origins = np.zeros((3, 7))
        for local_position, local_rotation, index, local_axis in self.chain:
            position = position+rotation@local_position
            rotation = rotation@local_rotation
            if index is not None:
                origins[:, index] = position
                axes[:, index] = rotation@local_axis
                rotation = rotation@Rotation.from_rotvec(local_axis*q[index]).as_matrix()
        eef = np.eye(4)
        eef[:3, :3], eef[:3, 3] = rotation, position
        action = eef@self.tcp
        return eef, action, axes, origins

    def point_jacobian(self, point, axes, origins):
        jac = np.cross(axes.T, (point[:, None]-origins).T).T
        jac[:, 0] = self.lift_axis
        return jac

    def objective(self, goal_eef, width, point1, point2):
        goal = np.asarray(goal_eef)@self.tcp
        cache = {}

        def compute(q):
            if 'q' in cache and np.array_equal(q, cache['q']):
                return cache['residual'], cache['jacobian']
            eef, action, axes, origins = self.fk(q)
            p1 = action[:3, 3]-width/2*eef[:3, 1]
            p2 = action[:3, 3]+width/2*eef[:3, 1]
            error_rotation = Rotation.from_matrix(action[:3, :3]@goal[:3, :3].T).as_rotvec()
            residual = np.r_[(action[:3, 3]-goal[:3, 3])/self.position_limit,
                             error_rotation/self.rotation_limit,
                             (p1-point1)/(self.contact_limit*np.sqrt(2)),
                             (p2-point2)/(self.contact_limit*np.sqrt(2))]
            jacobian = np.vstack([self.point_jacobian(action[:3, 3], axes, origins)/self.position_limit,
                                  rotation_log_jacobian_inverse(error_rotation)@axes/self.rotation_limit,
                                  self.point_jacobian(p1, axes, origins)/(self.contact_limit*np.sqrt(2)),
                                  self.point_jacobian(p2, axes, origins)/(self.contact_limit*np.sqrt(2))])
            cache.update(q=q.copy(), residual=residual, jacobian=jacobian)
            return residual, jacobian

        return lambda q: compute(q)[0], lambda q: compute(q)[1]

    def solve(self, seed, goal_eef, width, point1, point2, max_nfev=250):
        fun, jac = self.objective(goal_eef, width, point1, point2)
        seed = np.clip(np.asarray(seed), self.bounds[:, 0]+1e-8, self.bounds[:, 1]-1e-8)
        result = least_squares(fun, seed, jac=jac, bounds=self.bounds.T, method='trf',
                               x_scale='jac', ftol=1e-9, xtol=1e-9, gtol=1e-8, max_nfev=max_nfev)
        return result

    def check_derivatives(self, goal_eef, width, point1, point2):
        fun, jac = self.objective(goal_eef, width, point1, point2)
        rng = np.random.default_rng(1122)
        worst = 0.
        for _ in range(4):
            q = self.bounds[:, 0]+rng.uniform(.15, .85, 7)*np.ptp(self.bounds, axis=1)
            actual = jac(q).copy()
            expected = np.empty_like(actual)
            for j in range(7):
                a, b = q.copy(), q.copy()
                a[j] += 1e-7
                b[j] -= 1e-7
                expected[:, j] = (fun(a)-fun(b))/2e-7
            worst = max(worst, float(np.abs(actual-expected).max()))
        assert worst < 2e-4, ('Analytic derivative mismatch', worst)
        return worst
