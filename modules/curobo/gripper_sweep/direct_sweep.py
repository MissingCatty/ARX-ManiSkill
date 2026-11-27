"""Retain native FK at every width; validate static model data once per sweep."""
from pathlib import Path

from .lock_state import copy_lock_dependent_fields


def native_opening_spheres(pool,context,config,q,fractions,target,observed):
    import torch
    # The original guard checks the complete robot config and prepares its
    # unchanged URDF-derived affine buffers. The local config only changes
    # finger-lock scalars inside this serial sweep.
    initial=pool._measured_robot_view(config,context.planner.device_cfg)
    parameters=context.checker.kinematics.kinematics_config
    parameters.copy_(initial.kinematics.kinematics_config)
    role='closed' if Path(config['kinematics']['urdf_path']).parent.name=='model_closed' else 'open'
    view=pool.views[role];scratch=view['scratch'];spheres=[]
    for fraction in fractions:
        for name,value in observed.items():
            config['kinematics']['lock_joints'][name]=float(value+fraction*(target-value))
        values=[float(config['kinematics']['lock_joints'][name]) for name in view['names']]
        for field,slopes in view['slopes'].items():
            getattr(scratch,field).copy_(getattr(view['zero'],field)+values[0]*slopes[0]+values[1]*slopes[1])
        locked=scratch.lock_jointstate
        position=torch.tensor([config['kinematics']['lock_joints'][name] for name in locked.joint_names],
                              device=locked.position.device,dtype=locked.position.dtype)
        locked.position.copy_(position.reshape_as(locked.position))
        copy_lock_dependent_fields(parameters,scratch)
        spheres.append(context.checker.get_kinematics(q).robot_spheres.detach().clone())
    return torch.cat(spheres,dim=0)
