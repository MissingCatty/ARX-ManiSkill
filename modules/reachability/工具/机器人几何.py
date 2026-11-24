"""Batched URDF FK and analytic spatial Jacobian; no model changes."""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

from 公共 import URDF, JOINTS


class RobotGeometry:
    def __init__(self, tcp, device='cpu'):
        import torch
        self.torch = torch;self.device = device
        root = ET.parse(URDF).getroot()
        parent = {j.find('child').get('link'): j for j in root.findall('joint')}
        chain = [];name = 'right_eef_link'
        while name != 'body4':
            joint = parent[name];chain.append(joint);name = joint.find('parent').get('link')
        self.chain = [];self.lower = [];self.upper = []
        for joint in reversed(chain):
            node = joint.find('origin')
            xyz = np.fromstring(node.get('xyz', '0 0 0'), sep=' ') if node is not None else np.zeros(3)
            rpy = np.fromstring(node.get('rpy', '0 0 0'), sep=' ') if node is not None else np.zeros(3)
            rotation = Rotation.from_euler('xyz', rpy).as_matrix()
            index = JOINTS.index(joint.get('name')) if joint.get('name') in JOINTS else None
            axis = None
            if index is not None:
                assert joint.get('type') == 'revolute'
                assert index == len(self.lower)
                axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
                assert abs(np.linalg.norm(axis)-1) < 1e-6
                self.lower.append(float(joint.find('limit').get('lower')))
                self.upper.append(float(joint.find('limit').get('upper')))
            else:assert joint.get('type') == 'fixed'
            self.chain.append((torch.as_tensor(rotation, dtype=torch.float32, device=device),
                               torch.as_tensor(xyz, dtype=torch.float32, device=device), index,
                               None if axis is None else torch.as_tensor(axis, dtype=torch.float32, device=device)))
        assert len(self.lower) == 6
        self.lower = np.array(self.lower);self.upper = np.array(self.upper)
        self.tcp_r = torch.as_tensor(tcp[:3, :3], dtype=torch.float32, device=device)
        self.tcp_t = torch.as_tensor(tcp[:3, 3], dtype=torch.float32, device=device)

    def fk(self, q, jacobian=True):
        t = self.torch;n = len(q)
        rotation = t.eye(3, device=self.device).expand(n, 3, 3).clone()
        position = t.zeros((n, 3), device=self.device)
        origins = [];axes = []
        for r, p, index, axis in self.chain:
            position = position + (rotation @ p).reshape(n, 3)
            rotation = rotation @ r
            if index is not None:
                origins.append(position);axes.append((rotation @ axis).reshape(n, 3))
                x, y, z = axis.unbind()
                zero = t.zeros_like(x)
                skew = t.stack([zero,-z,y,z,zero,-x,-y,x,zero]).reshape(3, 3)
                angle = q[:, index, None, None]
                turn = t.eye(3, device=self.device) + t.sin(angle)*skew + (1-t.cos(angle))*(skew@skew)
                rotation = rotation @ turn
        position = position + (rotation @ self.tcp_t).reshape(n, 3)
        rotation = rotation @ self.tcp_r
        if not jacobian:return position, rotation
        axis = t.stack(axes, dim=-1);origin = t.stack(origins, dim=-1)
        delta = position[:, :, None]-origin
        linear = t.cross(axis, delta, dim=1)
        jac = t.cat([linear, .1*axis], dim=1)
        singular = t.linalg.svdvals(jac)
        condition = singular[:, -1]/t.clamp(singular[:, 0], min=1e-12)
        return position, rotation, condition, linear, axis
