"""Frozen-URDF forward checks and bounded Cartesian command construction."""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from training.nearfield_arx.contract import pose_error


def prepose(target):
    result = np.asarray(target, dtype=float).copy()
    result[:3, 3] -= result[:3, 2] * .060
    return result


def seed_arrival(error, object_drift):
    return bool(error['translation_m'] <= .003 and error['rotation_rad'] <= np.deg2rad(1.5)
                and object_drift['translation_m'] <= .001 and object_drift['rotation_rad'] <= .01)


def cartesian_knots(current, target, *, insertion):
    current, target = np.asarray(current), np.asarray(target)
    if insertion:
        start = prepose(target)
        error = pose_error(current, start)
        if error['translation_m'] > .004 or error['rotation_rad'] > np.deg2rad(1.5):
            raise ValueError('Measured prepose outside registered 4mm/1.5deg transit envelope')
        # Correct the measured controller residual at the 60 mm standoff first.
        # The subsequent insertion has constant orientation and pure local +Z.
        alignment = []
        rotation = Slerp([0., 1.], Rotation.from_matrix([current[:3, :3], start[:3, :3]]))
        for t in np.linspace(0, 1, 9):
            matrix = np.eye(4)
            matrix[:3, 3] = current[:3, 3] * (1 - t) + start[:3, 3] * t
            matrix[:3, :3] = rotation(t).as_matrix()
            alignment.append(matrix)
        insertion_knots = []
        for distance in np.linspace(.001, .060, 60):
            matrix = start.copy()
            matrix[:3, 3] += start[:3, 2] * distance
            insertion_knots.append(matrix)
        return alignment + insertion_knots, 8
    delta = target[:3, 3] - current[:3, 3]
    if (np.linalg.norm(delta[:2]) > 1e-10 or not 0 < delta[2] <= .12000001
        or pose_error(current, target)['rotation_rad'] > 1e-10):
        raise ValueError('Fallback must be unchanged-orientation world +Z only')
    count = int(np.ceil(delta[2] / .001))
    result = []
    for z in np.linspace(0, delta[2], count + 1):
        matrix = current.copy()
        matrix[2, 3] += z
        result.append(matrix)
    return result, 0


def command_times(positions, minimum_duration=8.):
    q = np.asarray(positions, dtype=float)
    if q.ndim != 2 or len(q) < 2 or not np.isfinite(q).all():
        raise ValueError('Invalid ARX joint trajectory')
    dt = max(.02, minimum_duration / (len(q) - 1))
    scale, values, fd = 1., {}, q
    for order, bound in ((1, 1.5), (2, 5.), (3, 100.)):
        fd = np.diff(fd, axis=0) / dt
        peak = float(np.max(np.abs(fd))) if fd.size else 0.
        values[str(order)] = peak
        scale = max(scale, (peak / bound) ** (1 / order))
    times = np.arange(len(q)) * dt * scale
    if times[-1] > 120:
        raise ValueError('Microstep command exceeds original 120second duration cap')
    return times, {key: value / scale ** int(key) for key, value in values.items()}


class FrozenURDF:
    """Read-only FK of the already pinned ARX URDF; no reconfiguration."""
    def __init__(self, path):
        self.joints = []
        for node in ET.parse(path).getroot().findall('joint'):
            origin = node.find('origin')
            matrix = np.eye(4)
            if origin is not None:
                matrix[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
                matrix[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
            axis = node.find('axis')
            self.joints.append((node.get('name'), node.get('type'), node.find('parent').get('link'),
                                node.find('child').get('link'), matrix,
                                np.fromstring(axis.get('xyz'), sep=' ') if axis is not None else None))

    def poses(self, measured, base):
        poses = {'body4': np.asarray(base, dtype=float)}
        pending = list(self.joints)
        while pending:
            remaining = []
            for name, kind, parent, child, origin, axis in pending:
                if parent not in poses:
                    remaining.append((name, kind, parent, child, origin, axis))
                    continue
                motion = np.eye(4)
                if kind == 'revolute':
                    motion[:3, :3] = Rotation.from_rotvec(axis * measured[name]).as_matrix()
                elif kind == 'prismatic':
                    motion[:3, 3] = axis * measured[name]
                elif kind != 'fixed':
                    raise ValueError('Unsupported original ARX joint type')
                poses[child] = poses[parent] @ origin @ motion
            if len(remaining) == len(pending):
                raise ValueError('Disconnected original ARX URDF')
            pending = remaining
        return poses

    def verify(self, snapshot):
        measured = dict(zip(snapshot['joint_names'], snapshot['joint_positions']))
        poses = self.poses(measured, snapshot['link_world']['body4'])
        errors = {name: pose_error(value, snapshot['link_world'][name])
                  for name, value in poses.items() if name in snapshot['link_world']}
        return {'pass': all(e['translation_m'] < .002 and e['rotation_rad'] < .01 for e in errors.values()),
                'link_errors': errors, 'URDF_or_state_modified': False}
