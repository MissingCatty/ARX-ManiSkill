"""Execute the exact opening already validated by the width-aware planner."""
import numpy as np

from arx_project.layout import read_json, sha256, write_json


def validate_opening(selected, opening):
    proof = selected['width_binding']
    values = [opening, selected['open_finger_m'], proof['open_finger_m'], proof['candidate_width_m'] / 2]
    if not np.isfinite(values).all() or not 0 < opening <= .044 or max(abs(v - opening) for v in values) > 1e-12:
        raise ValueError('Execution opening differs from the validated candidate width')
    for name in ['actual_collision_finger_locks_m', 'native_GPU_finger_locks_m']:
        locks = proof[name]
        if len(locks) != 2 or any(abs(v - opening) > 2e-7 for v in locks.values()):
            raise ValueError('Execution opening differs from actual collision model locks')
    if opening < .044:
        samples = proof.get('narrowing_samples', [])
        if (not samples or abs(samples[0] - .044) > 1e-12 or abs(samples[-1] - opening) > 1e-12
                or max(abs(np.diff(samples)), default=0) > .0005 + 1e-12
                or not proof.get('narrowing_target_collision_checked')):
            raise ValueError('Candidate narrowing sweep has not been checked')
    return {'candidate_width_m': 2 * opening, 'open_finger_m': opening,
        'actual_GPU_locks_verified': True, 'unchanged_narrowing_collision_gate': True}


def install(executor):
    parent = executor.Execution
    class WidthExecution(parent):
        def segment(self, label, q, times, finger=None, goal_world=None):
            if label in ['pregrasp', 'approach']:
                expected = self._planned_opening
                if finger is not None and abs(finger - expected) > 1e-12:
                    raise ValueError('Motion opening differs from the collision-checked candidate width')
                finger = expected
            elif finger is None:
                finger = .044
            commands = getattr(self, '_finger_commands', [])
            commands.append({'stage': label, 'finger_position_m': float(finger)})
            self._finger_commands = commands
            write_json(self.out / 'execution_finger_commands.json', commands)
            return super().segment(label, q, times, finger=finger, goal_world=goal_world)

        def open_gripper(self, opening):
            selected = read_json(self.out / 'plan_result.json')['selected']
            proof = validate_opening(selected, opening)
            self._planned_opening = opening
            proof.update(plan_sha256=sha256(self.out / 'plan_result.json'), arm=selected['arm'])
            write_json(self.out / 'execution_opening_binding.json', proof)
            q = self.env.agent.robot.qpos[0, self.arm_indices].cpu().numpy()
            self.segment('open', [q, q], [0., 2.], finger=opening)
            actual = self.env.agent.robot.qpos[0, self.finger_indices].cpu().numpy()
            error = float(np.max(np.abs(actual - opening)))
            self.stages['open'].update(finger_positions_m=actual.tolist(), finger_error_m=error,
                finger_opening_passed=error < .001)
            write_json(self.out / 'stages.json', self.stages)
            proof.update(measured_finger_positions_m=actual.tolist(), finger_error_m=error, passed=error < .001)
            write_json(self.out / 'execution_opening_binding.json', proof)
            if error >= .001: raise RuntimeError('gripper_open_not_converged')
    executor.Execution = WidthExecution
