"""Inspect small Cartesian commands in the unchanged ARX kinematic model."""
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from curobo.types import JointState


def forward(kinematics, positions, names):
    positions=np.asarray(positions,dtype=np.float32).reshape(-1,len(names))
    state=kinematics.compute_kinematics(JointState.from_position(
        torch.as_tensor(positions,device='cuda'),joint_names=names))
    p=state.tool_poses.position.detach().cpu().numpy().reshape(-1,3)
    q=state.tool_poses.quaternion.detach().cpu().numpy().reshape(-1,4)
    poses=np.broadcast_to(np.eye(4),(len(p),4,4)).copy()
    poses[:,:3,3]=p
    poses[:,:3,:3]=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
    return poses


def residual(kinematics, positions, goal, names):
    actual=forward(kinematics,positions,names)[0]
    return {'position_error_m':float(np.linalg.norm(goal[:3,3]-actual[:3,3])),
        'orientation_error_rad':float(Rotation.from_matrix(goal[:3,:3]@actual[:3,:3].T).magnitude())}
