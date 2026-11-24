"""H012 transient goals and measured joint states; frozen robot properties."""
import xml.etree.ElementTree as ET
import yaml
import torch
from gripper_update import WarmedGripperUpdate


class MeasuredOpenGripperUpdate(WarmedGripperUpdate):
    """Same affine state update, using only in-limit derivative probes."""
    def __init__(self,path,nominal_model):
        self.data=yaml.safe_load(path.read_text());spec=self.data['robot_cfg']['kinematics']
        self.names=['right_joint27','right_joint28']
        joints={j.attrib['name']:j for j in ET.parse(spec['urdf_path']).getroot().findall('joint')}
        assert all(joints[n].attrib['type']=='prismatic' for n in self.names)
        self.limits={n:(float(joints[n].find('limit').attrib['lower']),float(joints[n].find('limit').attrib['upper'])) for n in self.names}
        self.nominal={n:spec['lock_joints'][n] for n in self.names}
        params=nominal_model.config.kinematics_config
        self.base={k:v.clone() for k,v in vars(params).items() if isinstance(v,torch.Tensor)}
        self.derivatives={}
        for joint in self.names:
            step=.01 if self.nominal[joint]+.01<=self.limits[joint][1] else -.01
            values=dict(self.nominal);values[joint]+=step
            assert self.limits[joint][0]<=values[joint]<=self.limits[joint][1]
            shifted=self.load(values);self.derivatives[joint]={}
            for name,baseline in self.base.items():
                value=getattr(shifted,name);assert value.shape==baseline.shape
                if baseline.is_floating_point():self.derivatives[joint][name]=(value-baseline)/step
                else:assert torch.equal(value,baseline)
        probe=dict(zip(self.names,[.008,.036]));expected=self.load(probe)
        for name,value in self.evaluate(probe).items():
            assert torch.allclose(value,getattr(expected,name),atol=1e-6,rtol=1e-6),name


def neighborhood(checker, q):
    q=q.reshape(-1)
    if q.numel()!=6:raise ValueError('ARX six-joint arm required')
    bits=((torch.arange(64,device=q.device)[:,None]>>torch.arange(6,device=q.device))&1)*2-1
    states=torch.cat([q[None],q[None]+bits.to(q.dtype)*.001])
    valid=torch.cat([checker.validate(x[:,None]).flatten() for x in states.split(64)])
    return {'pass':bool(valid.all()),'valid_states':int(valid.sum()),'states':65,'half_width_rad':.001}
