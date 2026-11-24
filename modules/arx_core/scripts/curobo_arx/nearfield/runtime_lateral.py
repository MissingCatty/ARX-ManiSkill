"""Real one/two-step recovery; all original ARX path and closure gates remain."""
import argparse
from training.nearfield_arx.contract import encode, decode
from training.nearfield_arx.lateral_recovery import TEACHER_STEPS, teacher_target
from .runtime_physical import PhysicalEnvironment, PhysicalGroup


class LateralEnvironment(PhysicalEnvironment):
    teacher_step_limit = TEACHER_STEPS
    teacher_grasp_method = 'D'

    def teacher_action(self, current, target):
        prior = self.group.active_trial['G0_world']
        goal, final = teacher_target(current, target, self.local_index, prior)
        token = encode('D', current, goal, prior)
        return 'D', token, decode('D', current, token, prior), final


class LateralGroup(PhysicalGroup):
    def __init__(self, campaign, launch):
        super().__init__(campaign, launch)
        self.envs = [LateralEnvironment(i, e.bridge, self) for i, e in enumerate(self.envs)]


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--campaign', required=True);p.add_argument('--launch', required=True)
    a = p.parse_args()
    from scripts.curobo_arx.data_factory import visual_geometry
    from scripts.curobo_arx.data_factory.common import read
    policy = read(a.campaign+'/registration.json')['visual_geometry_policy']
    if policy != {**visual_geometry.POLICY, 'minimum_surface_distance_m':.008}:
        raise ValueError('Only the inherited advisor26 visual policy is allowed')
    visual_geometry.POLICY.update(policy)
    LateralGroup(a.campaign, a.launch).run()
