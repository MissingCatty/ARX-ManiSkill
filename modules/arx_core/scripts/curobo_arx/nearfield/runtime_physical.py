"""Fresh perturbed ARX entries followed by a bounded, explicit teacher."""
import argparse
import numpy as np
from scripts.curobo_arx.data_factory.common import write
from training.nearfield_arx.contract import pose_error
from .runtime_v2 import ActionEnvironment, ActionGroup
from .physical_spec import bounded_teacher_goal, MAX_TEACHER_STEPS


class PhysicalEnvironment(ActionEnvironment):
    teacher_step_limit = MAX_TEACHER_STEPS
    teacher_grasp_method = 'C'

    def teacher_action(self, current, target):
        token, goal, final = bounded_teacher_goal(current, target)
        return 'C', token, goal, final

    def plan(self, phase, world):
        if phase == 'approach':
            self.physical_entry_ready = False
            self.teacher_final_issued = False
        return super().plan(phase, world)

    def tick(self):
        generating = (self.state == 'executing' and
                      (self.last_command or {}).get('label') == 'state_generation')
        result = super().tick()
        if generating and self.state == 'local_decision':
            # The next synchronized render is the actual new entry. The old
            # seed anchor and the path generating this state are not positives.
            self.stage = 'entry'
            self.entry_stopped_sim_time = self.group.world.current_time
        return result

    def decide(self):
        snapshot = self.bridge.snapshot()
        if self.bridge.active or self.bridge.fault:
            raise RuntimeError('Fresh physical observation requires confirmed idle')
        row = self.last_observation
        completed = (self.entry_stopped_sim_time if not self.physical_entry_ready
                     else self.previous_execution['final']['sim_time'])
        if row['frame_time'] <= completed + 1e-6:
            return  # Wait for a genuinely new synchronized sample.
        # A positional arrival receipt is not a zero-velocity receipt. Continue
        # normal physics/capture ticks until both live and rendered ARX states
        # pass the unchanged speed bound. The original 900s attempt deadline
        # still applies in ActionEnvironment.tick; no command is retried.
        for state in (snapshot, row['snapshot']):
            speeds = dict(zip(state['joint_names'], state['joint_velocities']))
            if max(abs(speeds[n]) for n in state['right_arm']) > .01:
                return
        if not self.physical_entry_ready:
            self.actual_entry = row
            write(self.episode/'physical_entry_observation.json', row)
            write(self.episode/'actual_entry.json', row)
            self.log_event({'kind':'physical_entry_stopped',
                'source_anchor_not_positive':True,
                'registered_G0_world':self.group.active_trial['G0_world'],
                'registered_H_world':self.group.active_trial['entry_world'],
                'actual_entry_error':pose_error(snapshot['link_world']['right_eef_link'],
                                                self.group.active_trial['entry_world']),
                'measured_to_source_Gstar':pose_error(snapshot['link_world']['right_eef_link'],
                                                     self.group.active_trial['source_Gstar_world'])})
            self.physical_entry_ready = True
        if self.teacher_final_issued:
            return self.apply_action(self.teacher_grasp_method, 'GRASP')
        if self.local_index >= self.teacher_step_limit:
            return self.finish('teacher_step_budget_exhausted')
        ee = self.bridge.snapshot()['link_world']['right_eef_link']
        method, token, goal, final = self.teacher_action(ee, self.group.active_trial['source_Gstar_world'])
        self.teacher_provenance = {'driver':'offline_source_Gstar_bounded_teacher',
            'source_Gstar_world':self.group.active_trial['source_Gstar_world'],
            'next_goal_world':goal.tolist(), 'final_goal':final,
            'policy_input_contains_teacher_target':False}
        self.teacher_final_issued = final
        return self.apply_action(method, token)


class PhysicalGroup(ActionGroup):
    def __init__(self, campaign, launch):
        super().__init__(campaign, launch)
        self.envs = [PhysicalEnvironment(i, e.bridge, self) for i, e in enumerate(self.envs)]

    def sample_handoff(self, task):
        trial = self.active_trial
        return {'world':trial['entry_world'],
            'distance_mm':trial['perturbation']['backoff_m'] * 1000,
            'sample':0, 'seed':self.registration['seed'],
            'source_seed_id':self.active_seed['seed_id'], 'policy':trial['policy'],
            'perturbation':trial['perturbation'],
            'formula':trial['perturbation']['formula'],
            'source_anchor_is_transit_only':True}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--campaign', required=True); p.add_argument('--launch', required=True)
    a = p.parse_args(); PhysicalGroup(a.campaign, a.launch).run()
