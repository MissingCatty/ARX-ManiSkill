"""Physical ARX arm names shared by planning, execution and recording."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Arm:
    name: str
    first_joint: int

    @property
    def joints(self):
        return tuple(f'{self.name}_joint{i}' for i in range(self.first_joint, self.first_joint+6))

    @property
    def finger_joints(self):
        return tuple(f'{self.name}_joint{i}' for i in range(self.first_joint+6, self.first_joint+8))

    @property
    def finger_links(self):
        return tuple(f'{self.name}_link{i}' for i in range(self.first_joint+6, self.first_joint+8))

    @property
    def links(self):
        return tuple(f'{self.name}_link{i}' for i in range(self.first_joint, self.first_joint+8))

    @property
    def eef(self):
        return f'{self.name}_eef_link'


ARMS = {'right': Arm('right', 21), 'left': Arm('left', 11)}


def get_arm(name):
    try:
        return ARMS[name]
    except KeyError:
        raise ValueError('Unknown ARX arm: '+str(name)) from None


def select_execution_arm(env, name):
    arm = get_arm(name)
    env.grasp_arm = arm.name
    env.agent.tcp = env.agent.robot.links_map[arm.eef]
    env.check_inter_arm_contacts = True
    return arm
