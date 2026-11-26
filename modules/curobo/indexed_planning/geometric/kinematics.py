"""Build the physical six-axis chain and verify every geometric IK branch."""
from itertools import product
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from arx_project.assets import resource_path
from arx_project.layout import sha256
from modules.arx_core.arms import get_arm


def origin(node):
    value = np.eye(4)
    if node is not None:
        value[:3, 3] = np.fromstring(node.get('xyz', '0 0 0'), sep=' ')
        value[:3, :3] = Rotation.from_euler('xyz',
            np.fromstring(node.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    return value


def periodic_branches(values, limits):
    """Enumerate actual 2*pi equivalents; only snap 1e-12 boundary roundoff."""
    values, limits = np.asarray(values, dtype=float), np.asarray(limits, dtype=float)
    if values.shape != (6,) or limits.shape != (6, 2) or not np.isfinite([*values, *limits.ravel()]).all():
        raise ValueError('Invalid six-axis geometric branch or limits')
    if np.any(limits[:, 1] < limits[:, 0]):
        raise ValueError('Reversed joint limits')
    choices = []
    for angle, (lower, upper) in zip(values, limits):
        first = int(np.ceil((lower-angle-1e-12)/(2*np.pi)))
        last = int(np.floor((upper-angle+1e-12)/(2*np.pi)))
        options = []
        for turns in range(first, last+1):
            q = float(angle+2*np.pi*turns)
            if lower-1e-12 <= q < lower: q = float(lower)
            if upper < q <= upper+1e-12: q = float(upper)
            if lower <= q <= upper: options.append(q)
        if not options: return np.empty((0, 6))
        choices.append(options)
    if np.prod([len(c) for c in choices]) > 4096:
        raise ValueError('Unexpectedly wide periodic joint bounds')
    return np.asarray(list(product(*choices)), dtype=float).reshape(-1, 6)


class ArmChain:
    def __init__(self, arm='right', urdf=None):
        self.arm = get_arm(arm)
        self.urdf = Path(urdf) if urdf is not None else resource_path('arx_robot')/'robot.urdf'
        self.urdf_sha256 = sha256(self.urdf)
        tree = ET.parse(self.urdf).getroot()
        parents = {j.find('child').get('link'): j for j in tree.findall('joint')}
        self.chain = []
        link = self.arm.eef
        while link != 'body4':
            if link not in parents: raise ValueError('EEF is not descended from body4')
            joint = parents[link]
            self.chain.append(joint)
            link = joint.find('parent').get('link')
        self.chain.reverse()
        actual = [j.get('name') for j in self.chain if j.get('type') != 'fixed']
        if actual != list(self.arm.joints):
            raise ValueError('Unexpected movable joint in physical EEF chain: '+str(actual))
        self.lower, self.upper = [], []
        H, positions = [], []
        frame = np.eye(4)
        for joint in self.chain:
            frame = frame@origin(joint.find('origin'))
            if joint.get('type') == 'fixed': continue
            if joint.get('type') != 'revolute': raise ValueError('D requires the frozen six revolute joints')
            axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
            if axis.shape != (3,) or not np.isclose(np.linalg.norm(axis), 1., atol=1e-10):
                raise ValueError('Nonunit authored joint axis')
            H.append(frame[:3, :3]@axis)
            positions.append(frame[:3, 3].copy())
            self.lower.append(float(joint.find('limit').get('lower')))
            self.upper.append(float(joint.find('limit').get('upper')))
        positions = np.asarray(positions)
        self.H = np.asarray(H).T.copy()
        self.P = np.vstack([positions[0], np.diff(positions, axis=0), frame[:3, 3]-positions[-1]]).T.copy()
        self.R_tool = frame[:3, :3].copy()
        self.limits = np.asarray([self.lower, self.upper]).T

    def fk(self, values):
        q = np.asarray(values, dtype=float).reshape(-1, 6)
        if not np.isfinite(q).all(): raise ValueError('Nonfinite joint state')
        result = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
        for joint in self.chain:
            result = result@origin(joint.find('origin'))
            if joint.get('type') == 'fixed': continue
            index = self.arm.joints.index(joint.get('name'))
            axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
            move = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
            move[:, :3, :3] = Rotation.from_rotvec(q[:, index, None]*axis).as_matrix()
            result = result@move
        return result


class GeometricSolver:
    def __init__(self, arm='right'):
        from eaik.pybindings import EAIK
        self.chain = ArmChain(arm)
        # Express the three parallel inner axes in the same direction. The
        # corresponding angle signs are mapped back before physical limit/FK
        # checks, so this is the identical URDF chain in different coordinates.
        self.axis_signs = np.ones(6)
        for index in [2, 3]:
            reference, axis = self.chain.H[:, 1], self.chain.H[:, index]
            if np.linalg.norm(np.cross(reference, axis)) > 1e-9:
                raise ValueError('The frozen ARX inner axes are no longer parallel')
            self.axis_signs[index] = 1. if np.dot(reference, axis) >= 0 else -1.
        self.robot = EAIK.Robot(self.chain.H*self.axis_signs, self.chain.P, self.chain.R_tool, [], True)
        if not self.robot.has_known_decomposition():
            raise ValueError('EAIK did not recognize the frozen ARX geometry')
        self.family = self.robot.get_kinematic_family()

    def solve(self, target, start, position_tolerance=.002, rotation_tolerance=.03):
        began = time.perf_counter()
        target = np.asarray(target, dtype=float)
        start = np.asarray(start, dtype=float)
        if target.shape != (4, 4) or start.shape != (6,) or not np.isfinite([*target.ravel(), *start]).all():
            raise ValueError('Invalid geometric IK target or starting configuration')
        if (not np.allclose(target[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
                or np.linalg.norm(target[:3, :3].T@target[:3, :3]-np.eye(3)) > 2e-5
                or np.linalg.det(target[:3, :3]) < .99998):
            raise ValueError('Invalid geometric target transform')
        if not 0 < position_tolerance <= .002 or not 0 < rotation_tolerance <= .03:
            raise ValueError('The original endpoint tolerances must not be relaxed')
        # Match the native matrix-to-quaternion conversion for float32 rotations.
        query = target.copy()
        query[:3, :3] = Rotation.from_matrix(target[:3, :3]).as_matrix()
        solve_start = time.perf_counter()
        answer = self.robot.calculate_IK(query)
        native_seconds = time.perf_counter()-solve_start
        raw = np.asarray(answer.Q, dtype=float).reshape(-1, 6)*self.axis_signs
        if not np.isfinite(raw).all(): raise RuntimeError('EAIK returned a nonfinite branch')
        expanded = [periodic_branches(row, self.chain.limits) for row in raw]
        candidates = np.concatenate(expanded) if expanded else np.empty((0, 6))
        if len(candidates):
            _, ids = np.unique(np.round(candidates, 10), axis=0, return_index=True)
            candidates = candidates[np.sort(ids)]
            actual = self.chain.fk(candidates)
            pe = np.linalg.norm(actual[:, :3, 3]-target[:3, 3], axis=1)
            re = Rotation.from_matrix(actual[:, :3, :3]@query[:3, :3].T).magnitude()
            mask = (pe <= position_tolerance) & (re <= rotation_tolerance)
            candidates, pe, re = candidates[mask], pe[mask], re[mask]
            order = np.argsort(np.sum((candidates-start)**2, axis=1), kind='stable')
            candidates, pe, re = candidates[order], pe[order], re[order]
        else:
            pe = re = np.empty(0)
        return {'q': candidates.tolist(), 'position_errors_m': pe.tolist(),
            'rotation_errors_rad': re.tolist(), 'raw_branches': len(raw),
            'joint_limit_branches': sum(len(e) for e in expanded),
            'admitted_branches': len(candidates), 'native_solver_seconds': native_seconds,
            'least_squares_branches': int(np.asarray(answer.is_LS).sum()),
            'wall_seconds': time.perf_counter()-began, 'no_admitted_branch_means': 'fallback_to_original_IK'}
