"""Independent URDF FK and measured collision meshes for saved V1 states."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

JOINTS = [f'right_joint{i}' for i in range(21, 27)]
FINGERS = ('right_link27', 'right_link28')


def origin(node):
    t = np.eye(4)
    if node is not None:
        t[:3, 3] = np.fromstring(node.get('xyz', '0 0 0'), sep=' ')
        t[:3, :3] = Rotation.from_euler('xyz', np.fromstring(node.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    return t


class Geometry:
    def __init__(self, model, state):
        self.config = yaml.safe_load(Path(model).read_text())['robot_cfg']['kinematics']
        self.urdf = Path(self.config['urdf_path'])
        tree = ET.parse(self.urdf).getroot()
        self.parents = {j.find('child').get('link'): j for j in tree.findall('joint')}
        self.joints = {j.get('name'): j for j in tree.findall('joint')}
        self.base = np.asarray(state['links']['body4'], dtype=float)
        self.locked = self.config['lock_joints']
        self.lower = np.array([float(self.joints[n].find('limit').get('lower')) for n in JOINTS])
        self.upper = np.array([float(self.joints[n].find('limit').get('upper')) for n in JOINTS])
        self.names = [name for name, spheres in self.config['collision_spheres'].items() for _ in spheres]
        self.spec = {n: np.asarray([[*s['center'], s['radius']] for s in v])
                     for n, v in self.config['collision_spheres'].items()}
        assert self.config['collision_sphere_buffer'] == 0.

    def frames(self, q):
        q = np.asarray(q, dtype=float).reshape(-1, 6)
        cache = {'body4': np.broadcast_to(self.base, (len(q), 4, 4)).copy()}

        def world(name):
            if name in cache:
                return cache[name]
            j = self.parents[name]
            t = world(j.find('parent').get('link')) @ origin(j.find('origin'))
            kind, joint_name = j.get('type'), j.get('name')
            if kind != 'fixed':
                axis = np.fromstring(j.find('axis').get('xyz'), sep=' ')
                if joint_name in JOINTS:
                    value = q[:, JOINTS.index(joint_name)]
                else:
                    assert joint_name in self.locked, joint_name
                    value = np.full(len(q), self.locked[joint_name])
                move = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
                if kind == 'revolute':
                    move[:, :3, :3] = Rotation.from_rotvec(value[:, None] * axis).as_matrix()
                elif kind == 'prismatic':
                    move[:, :3, 3] = value[:, None] * axis
                else:
                    raise ValueError(kind)
                t = t @ move
            cache[name] = t
            return t

        for name in [*self.spec, 'right_eef_link']:
            world(name)
        return cache

    def spheres(self, frames):
        values = []
        for name, local in self.spec.items():
            frame = frames[name]
            xyz = np.einsum('bij,sj->bsi', frame[:, :3, :3], local[:, :3]) + frame[:, None, :3, 3]
            values.append(np.concatenate([xyz, np.broadcast_to(local[None, :, 3:], (*xyz.shape[:2], 1))], axis=-1))
        return np.concatenate(values, axis=1)


def box_depth(spheres, box):
    delta = np.abs(spheres[..., :3] - np.asarray(box['center'])) - np.asarray(box['dims']) / 2
    signed_distance = np.linalg.norm(np.maximum(delta, 0), axis=-1) + np.minimum(delta.max(axis=-1), 0)
    return np.where(spheres[..., 3] > 0, spheres[..., 3] - signed_distance, -np.inf)


def collision_meshes(asset, arm='right'):
    """Actual exported convex parts, kept separate to avoid closing mesh gaps."""
    import trimesh
    asset = Path(asset)
    tree = ET.parse(asset / 'robot.urdf').getroot()
    meshes, paths = {}, [asset / 'robot.urdf']
    for link in tree.findall('link'):
        name = link.get('name')
        prefix = 'right_link2' if arm == 'right' else 'left_link1' if arm == 'left' else None
        if prefix is None:
            raise ValueError('Unknown collision arm: '+str(arm))
        if not name.startswith(prefix) or not name.removeprefix('right_link' if arm == 'right' else 'left_link').isdigit():
            continue
        # Solver names remain right_*; use the selected physical arm's actual
        # collider parts. Left fingers have a different convex decomposition.
        canonical = name if arm == 'right' else 'right_link'+str(int(name.removeprefix('left_link'))+10)
        parts = []
        for collision in link.findall('collision'):
            mesh = collision.find('geometry/mesh')
            if mesh is None:
                raise ValueError('Expected original mesh collision geometry')
            path = asset / mesh.get('filename'); paths.append(path)
            vertices = np.asarray(trimesh.load_mesh(path, process=False).vertices, dtype=float)
            vertices *= np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
            t = origin(collision.find('origin'))
            parts.append(vertices @ t[:3, :3].T + t[:3, 3])
        if parts:
            meshes[canonical] = parts
    return meshes, paths


def tip_plane(geometry, meshes):
    frames = geometry.frames(np.zeros((1, 6)))
    eef = frames['right_eef_link'][0]
    tips = {}
    for name in FINGERS:
        frame = frames[name][0]
        points = np.concatenate(meshes[name]) @ frame[:3, :3].T + frame[:3, 3]
        local = (points - eef[:3, 3]) @ eef[:3, :3]
        tips[name] = float(local[:, 2].max())
    assert .02 < min(tips.values()) < .08
    return min(tips.values()), tips
