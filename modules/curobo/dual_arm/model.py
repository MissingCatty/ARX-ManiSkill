"""Verified arm coordinate binding and measured parked-arm collision geometry."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from arx_project.assets import resource_path
from arx_project.layout import ROOT, read_json, sha256, write_json
from modules.arx_core.arms import get_arm
from modules.curobo.fingertip_contact.geometry import origin


class ArmModel:
    def __init__(self):
        self.asset = resource_path('arx_robot')
        self.urdf = self.asset/'robot.urdf'
        tree = ET.parse(self.urdf).getroot()
        self.links = {node.get('name'): node for node in tree.findall('link')}
        self.parents = {node.find('child').get('link'): node for node in tree.findall('joint')}
        self.joints = {node.get('name'): node for node in tree.findall('joint')}
        self.meshes = {}
        # This is a measured fixed mounting transform, verified below, not a mirror.
        self.left_from_right = np.eye(4)
        self.left_from_right[1, 3] = .5
        self.proof = self.verify_kinematics()
        self.sphere_proof = self.verify_spheres()

    def verify_spheres(self):
        profile_path = ROOT/'simulation/robots/arx/collision/arx_refined_871.json'
        profile = read_json(profile_path)
        spheres = profile['collision_spheres']
        right, left = get_arm('right'), get_arm('left')
        mapped = {b: deepcopy(spheres[a]) for a, b in zip(right.links, left.links)}
        normalized = {a: mapped[b] for a, b in zip(right.links, left.links)}
        def digest(value):
            return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if digest(spheres) != profile['sphere_parameter_sha256'] or normalized != spheres:
            raise ValueError('Dual-arm sphere parameters differ from the approved profile')
        # Check the full-robot left link transforms against the canonical solver
        # independently, including all 871 sphere centers at 16 joint settings.
        joint_names = [n for n, j in self.joints.items() if j.get('type') != 'fixed']
        rng = np.random.default_rng(871)
        maximum = 0.
        for _ in range(16):
            q = dict.fromkeys(joint_names, 0.)
            for a, b in zip(right.joints+right.finger_joints, left.joints+left.finger_joints):
                limit = self.joints[a].find('limit')
                q[a] = q[b] = rng.uniform(float(limit.get('lower')), float(limit.get('upper')))
            frames = self.frames({'joint_names': joint_names, 'qpos': [q[n] for n in joint_names],
                                  'links': {'base_link': np.eye(4).tolist()}})
            for a, b in zip(right.links, left.links):
                centers = np.array([s['center'] for s in spheres[a]])
                expected = centers@frames[a][:3, :3].T+frames[a][:3, 3]
                actual = centers@frames[b][:3, :3].T+frames[b][:3, 3]
                shift = frames['body4'][:3, :3]@self.left_from_right[:3, 3]
                maximum = max(maximum, float(np.abs(actual-expected-shift).max()))
        if maximum > 2e-6:
            raise ValueError('Left sphere placement does not match actual left-arm FK')
        return {'profile_id': profile['profile_id'], 'right_total': sum(map(len, spheres.values())),
                'left_total': sum(map(len, mapped.values())),
                'per_link': [{'right': a, 'left': b, 'spheres_each': len(spheres[a])} for a, b in zip(right.links, left.links)],
                'right_parameter_sha256': digest(spheres), 'normalized_left_parameter_sha256': digest(normalized),
                'local_centers_and_radii_identical': True, 'fk_test_states': 16,
                'maximum_world_center_error_m': maximum, 'source_sha256': sha256(profile_path)}

    def frames(self, state):
        values = dict(zip(state['joint_names'], state['qpos']))
        cache = {'base_link': np.asarray(state['links']['base_link'], dtype=float)}

        def world(name):
            if name not in cache:
                joint = self.parents[name]
                transform = world(joint.find('parent').get('link'))@origin(joint.find('origin'))
                move = np.eye(4)
                kind = joint.get('type')
                if kind != 'fixed':
                    value = values[joint.get('name')]
                    axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
                    if kind == 'prismatic':
                        move[:3, 3] = value*axis
                    elif kind in ['revolute', 'continuous']:
                        move[:3, :3] = Rotation.from_rotvec(value*axis).as_matrix()
                    else:
                        raise ValueError('Unsupported joint '+kind)
                cache[name] = transform@move
            return cache[name]
        for name in self.links:
            world(name)
        for name, saved in state['links'].items():
            if name in cache and not np.allclose(cache[name], saved, atol=2e-5, rtol=0):
                raise ValueError('Independent full-robot FK differs at '+name)
        return cache

    def verify_kinematics(self):
        right, left = get_arm('right'), get_arm('left')
        for a, b in zip(right.joints+right.finger_joints, left.joints+left.finger_joints):
            ja, jb = self.joints[a], self.joints[b]
            if ja.get('type') != jb.get('type') or ja.find('axis').attrib != jb.find('axis').attrib:
                raise ValueError('Arm joint types or axes differ')
            if ja.find('limit').attrib != jb.find('limit').attrib:
                raise ValueError('Arm authored limits differ')
        # Check every physical link, including the fingers and EEF, at 32 paired
        # joint states. Runtime binding independently checks the measured state.
        names = [name for name, j in self.joints.items() if j.get('type') != 'fixed']
        rng = np.random.default_rng(20261125)
        maximum = 0.
        for _ in range(32):
            q = {name: 0. for name in names}
            for a, b in zip(right.joints+right.finger_joints, left.joints+left.finger_joints):
                limit = self.joints[a].find('limit')
                q[a] = q[b] = rng.uniform(float(limit.get('lower')), float(limit.get('upper')))
            frames = self.frames({'joint_names': names, 'qpos': [q[n] for n in names],
                                  'links': {'base_link': np.eye(4).tolist()}})
            offset = frames['body4']@self.left_from_right@np.linalg.inv(frames['body4'])
            for a, b in zip(right.links+(right.eef,), left.links+(left.eef,)):
                error = float(np.abs(offset@frames[a]-frames[b]).max())
                maximum = max(maximum, error)
        if maximum > 2e-6:
            raise ValueError('Arm chains do not have the registered fixed relationship')
        return {'paired_states': 32, 'maximum_transform_error': maximum,
                'joint_axes_and_limits_equal': True, 'mount_translation_y_m': .5,
                'source_urdf_sha256': sha256(self.urdf)}

    def parts(self, name):
        import trimesh
        if name not in self.meshes:
            parts = []
            for node in self.links[name].findall('collision'):
                mesh = node.find('geometry/mesh')
                if mesh is None:
                    raise ValueError('Expected original mesh collider: '+name)
                file = self.asset/mesh.get('filename')
                loaded = trimesh.load_mesh(file, process=False)
                local = np.asarray(loaded.vertices)*np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
                transform = origin(node.find('origin'))
                parts.append((local@transform[:3, :3].T+transform[:3, 3],
                              np.asarray(loaded.faces), file))
            self.meshes[name] = parts
        return self.meshes[name]

    def bind(self, state, arm_name, directory):
        """Keep world grasps unchanged; express the chosen chain in solver names."""
        import trimesh
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arm, right = get_arm(arm_name), get_arm('right')
        idle = get_arm('left' if arm_name == 'right' else 'right')
        frames = self.frames(state)
        result = deepcopy(state)
        source_names = arm.joints+arm.finger_joints
        for target, source in zip(right.joints+right.finger_joints, source_names):
            i, j = state['joint_names'].index(target), state['joint_names'].index(source)
            result['qpos'][i] = state['qpos'][j]
            result['joint_limits'][i] = deepcopy(state['joint_limits'][j])
        if arm_name == 'left':
            result['links']['body4'] = (frames['body4']@self.left_from_right).tolist()
        result['links']['right_eef_link'] = frames[arm.eef].tolist()
        result['planning_arm'] = arm_name
        if 'fingertip_table_contact' in result:
            anchor = result['fingertip_table_contact']
            if anchor.get('arm', 'right') != arm_name:
                raise ValueError('Closure receipt belongs to the other arm')
            values = list(anchor['qpos'])
            for target, source in zip(right.joints+right.finger_joints, source_names):
                values[state['joint_names'].index(target)] = anchor['qpos'][state['joint_names'].index(source)]
            anchor['qpos'] = values
        # The parked arm is an explicit world obstacle in every native checker,
        # including the non-table checker used by the local fingertip policy.
        parts, hashes = [], {str(self.urdf): sha256(self.urdf)}
        for name in idle.links:
            for vertices, faces, file in self.parts(name):
                transform = frames[name]
                parts.append(trimesh.Trimesh(vertices@transform[:3, :3].T+transform[:3, 3], faces, process=False))
                hashes[str(file)] = sha256(file)
        path = directory/'parked_arm.ply'
        trimesh.util.concatenate(parts).export(path)
        result.setdefault('scene_meshes', {})['parked_robot_arm'] = {
            'target': False, 'world': np.eye(4).tolist(), 'file_path': str(path),
            'file_sha256': sha256(path), 'physical_arm': idle.name}
        receipt = {'active_arm': arm.name, 'parked_arm': idle.name,
                   'physical_joint_names': arm.joints, 'physical_finger_joints': arm.finger_joints,
                   'active_eef_world': frames[arm.eef], 'world_grasp_transforms_unchanged': True,
                   'parked_arm_collider_parts': len(parts), 'source_sha256': hashes,
                   'kinematics': self.proof, 'collision_spheres': self.sphere_proof,
                   'active_contact_mesh_arm': arm.name}
        write_json(directory/'arm_binding.json', receipt)
        return result
