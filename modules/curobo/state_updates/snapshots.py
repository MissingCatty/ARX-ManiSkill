"""Immutable model templates and request-local, content-verified scene inputs."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np


class GeometryFiles:
    def __init__(self):
        self.digests = {}

    def digest(self, filename, expected=None):
        path = Path(filename).resolve()
        if path not in self.digests:
            self.digests[path] = hashlib.sha256(path.read_bytes()).hexdigest()
        value = self.digests[path]
        if expected is not None and value != expected:
            raise RuntimeError('Measured geometry differs from its registered content: ' + str(path))
        return value

    def verify(self):
        for path, expected in self.digests.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError('A geometry snapshot changed during its request: ' + str(path))


class InputBuilder:
    def __init__(self, models, profiles, finger_names, pose7):
        self.models = deepcopy(models)
        self.profiles = deepcopy(profiles)
        self.finger_names = tuple(finger_names)
        self.pose7 = pose7

    def build(self, state, *, closed=False, include_target=False, files):
        role = 'closed' if closed else 'open'
        base = np.asarray(state['links']['body4'])
        invbase = np.linalg.inv(base)
        measured = dict(zip(state['joint_names'], state['qpos']))
        cfg = self.models[role]
        locks = {name: measured[name] for name in self.finger_names}
        # Shared nested model data is immutable. Only the lock dictionary varies.
        robot = {**cfg, 'kinematics': {**cfg['kinematics'], 'lock_joints': locks}}
        scene = {'cuboid': {}}
        for name, geometry in state['scene_cuboids'].items():
            transform = np.eye(4)
            transform[:3, 3] = geometry['center']
            scene['cuboid'][name] = {'pose': self.pose7(invbase @ transform),
                                     'dims': list(geometry['dims'])}
        meshes = {n: g for n, g in state.get('scene_meshes', {}).items()
                  if include_target or not g['target']}
        if len(meshes) > 8 or len(scene['cuboid']) > 8:
            raise RuntimeError('Registered resident collision-cache capacity exceeded')
        digests = {}
        for name, geometry in meshes.items():
            digests[name] = files.digest(geometry['file_path'], geometry.get('file_sha256'))
        if meshes:
            scene['mesh'] = {n: {'pose': self.pose7(invbase @ np.asarray(g['world'])),
                                 'file_path': g['file_path']} for n, g in meshes.items()}
        # Reject nonfinite scene inputs before any cache/solver mutation.
        json.dumps(scene, allow_nan=False)
        return {'cfg': {'robot_cfg': robot}, 'scene': scene, 'profile': self.profiles[role],
                'base': base, 'invbase': invbase, 'mesh_digests': digests, 'role': role}


def collision_inputs(scene, digests):
    """Use precisely the original content-qualified mesh names."""
    content = deepcopy(scene)
    if 'mesh' in content:
        content['mesh'] = {name + '_' + digests[name][:20]: geometry
                           for name, geometry in content['mesh'].items()}
    return content
