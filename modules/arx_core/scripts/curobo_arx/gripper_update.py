"""Precompute exact locked-prismatic-joint updates for the ARX gripper."""
import copy
import xml.etree.ElementTree as ET

import torch
import yaml
from curobo.kinematics import KinematicsCfg


class WarmedGripperUpdate:
    def __init__(self, path, nominal_model):
        self.data = yaml.safe_load(path.read_text())
        spec = self.data['robot_cfg']['kinematics']
        self.names = ['right_joint27', 'right_joint28']
        joints = {j.attrib['name']: j for j in ET.parse(spec['urdf_path']).getroot().findall('joint')}
        assert all(joints[n].attrib['type'] == 'prismatic' for n in self.names)
        self.limits = {n: (float(joints[n].find('limit').attrib['lower']),
                           float(joints[n].find('limit').attrib['upper'])) for n in self.names}
        self.nominal = {n: spec['lock_joints'][n] for n in self.names}
        params = nominal_model.config.kinematics_config
        self.base = {k: v.clone() for k, v in vars(params).items() if isinstance(v, torch.Tensor)}
        self.derivatives = {}
        for joint in self.names:
            values = dict(self.nominal)
            values[joint] += .01
            shifted = self.load(values)
            self.derivatives[joint] = {}
            for name, baseline in self.base.items():
                value = getattr(shifted, name)
                assert value.shape == baseline.shape, 'Gripper update changed model topology'
                if baseline.is_floating_point():
                    self.derivatives[joint][name] = (value-baseline)/.01
                else:
                    assert torch.equal(value, baseline)
        # Validate the interpolation against an independently rebuilt ARX model
        # before motion; both fingers move by different amounts in this check.
        probe = dict(zip(self.names, [.008, .036]))
        expected = self.load(probe)
        for name, value in self.evaluate(probe).items():
            assert torch.allclose(value, getattr(expected, name), atol=1e-6, rtol=1e-6), name

    def load(self, values):
        data = copy.deepcopy(self.data)
        data['robot_cfg']['kinematics']['lock_joints'].update(values)
        return KinematicsCfg.from_data_dict(data).kinematics_config

    def evaluate(self, values):
        result = {}
        for name, baseline in self.base.items():
            if not baseline.is_floating_point():
                continue
            value = baseline.clone()
            for joint in self.names:
                value.add_(self.derivatives[joint][name], alpha=values[joint]-self.nominal[joint])
            result[name] = value
        return result

    def apply(self, models, values):
        values = {n: float(values[n]) for n in self.names}
        for name, value in values.items():
            lo, hi = self.limits[name]
            if not lo-1e-5 <= value <= hi+1e-5:
                raise ValueError('Measured gripper position is outside the validated model range')
        tensors = self.evaluate(values)
        for model in {id(m): m for m in models}.values():
            params = model.config.kinematics_config
            for name, value in tensors.items():
                getattr(params, name).copy_(value)
            locked = params.lock_jointstate
            positions = torch.tensor([values[n] for n in locked.joint_names],
                                     device=locked.position.device, dtype=locked.position.dtype)
            locked.position.copy_(positions.reshape_as(locked.position))
        return values
