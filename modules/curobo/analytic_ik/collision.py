"""Native cuRobo FK and world/self checks for all analytic branches at once."""
from copy import deepcopy

import torch

from modules.arx_core.scripts.curobo_arx.collision_profile import require_current_profile


FINGER_JOINTS = ['right_joint27','right_joint28']
NAMES = ['right_joint'+str(i) for i in range(21,29)]


def unlocked_collision_model(source):
    result = deepcopy(source)
    require_current_profile(result)
    kin = result['robot_cfg']['kinematics']
    if set(kin['lock_joints']) != set(FINGER_JOINTS) or kin['cspace']['joint_names'] != NAMES:
        raise ValueError('Unexpected active kinematic chain for per-candidate gripper widths')
    kin['lock_joints'] = {}
    return result


class BranchCollisionChecker:
    def __init__(self, robot_config, scene):
        from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
        self.config = unlocked_collision_model(robot_config)
        self.checker = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
            robot_config=self.config, scene_model=scene, n_meshes=8, n_cuboids=8,
            collision_activation_distance=0., self_collision_activation_distance=0.))
        if self.checker.kinematics.joint_names != NAMES or self.checker.kinematics.total_spheres != 871:
            raise ValueError('Native collision FK did not retain eight measured coordinates and 871 spheres')
        if self.checker.self_collision_cost is None:
            raise ValueError('Standalone world/self diagnostic needs the native self-collision term')

    def joint_rows(self, q, widths):
        if q.ndim != 3 or q.shape[1:] != (8,6) or widths.shape != q.shape[:1]:
            raise ValueError('Expected analytic q[N,8,6] and opening widths[N]')
        opening = widths[:,None,None].expand(-1,8,2)/2
        safe = torch.where(torch.isfinite(opening),opening,torch.zeros_like(opening))
        return torch.cat([q,safe],dim=-1).reshape(-1,1,8).contiguous()

    def check(self, q, analytic_valid, widths, *, include_self=True):
        q8 = self.joint_rows(q,widths)
        checker = self.checker
        checker.setup_batch_tensors(q8.shape[0],1)
        state = checker.get_kinematics(q8)
        shape = q.shape[:2]
        bounded = (checker.get_bound(q8).sum(-1)==0).reshape(shape)
        world_clear = (checker.get_collision_constraint(state).sum(-1)==0).reshape(shape)
        self_clear = ((checker.get_self_collision(state.robot_spheres).sum(-1)==0).reshape(shape)
                      if include_self else torch.ones_like(world_clear))
        width_valid = torch.isfinite(widths)&(widths>0)&(widths<=.088)
        base = analytic_valid&bounded&width_valid[:,None]
        return {'world_clear':world_clear,'self_clear':self_clear,'joint_bounds_clear':bounded,
            'valid_world_only':base&world_clear,'valid_world_and_self':base&world_clear&self_clear,
            'q_with_measured_fingers':q8, 'native_spheres':state.robot_spheres}
