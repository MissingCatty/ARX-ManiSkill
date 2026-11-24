"""Advisor29 C waypoint scheduling over the unchanged closure/lift interpreter."""
import argparse
from pathlib import Path
import time
from scripts.curobo_arx.data_factory.common import read, write, sha
from scripts.curobo_arx.settling import assess_settling
from training.nearfield_arx import learned_policy
from training.nearfield_arx.stage_verification import audit_policy
from . import runtime_learned, runtime_closure_recovery, waypoint_continuation

original_live_sample = learned_policy.live_sample


def live_sample(row, method, g0, previous_goal, request, calls, refines, previous_event):
    result = original_live_sample(row, method, g0, previous_goal, request, calls, refines, previous_event)
    if previous_event and previous_event['outcome'] == 'stationary_waypoint_continuation':
        result['inputs']['last_command_result'] = 'stationary_waypoint_not_arrived'
    return result


class StageEnvironment(runtime_closure_recovery.ClosureEnvironment):
    def tick(self):
        label = (self.last_command or {}).get('label', '')
        if self.state != 'executing' or not label.startswith('c_step_'):
            return super().tick()
        if self.group.active_trial['method'] != 'C':
            raise RuntimeError('C waypoint command in another method')
        if self.bridge.fault:
            raise RuntimeError(str(self.bridge.fault))
        if self.task and time.monotonic() - self.started > self.group.registration['attempt_seconds']:
            raise TimeoutError('Original per-attempt resource budget exceeded')
        result = self.bridge.result()
        if result is None:
            return
        write(self.episode / (label + '_execution.json'), result)
        assessment = assess_settling(result, self.desired)
        write(self.episode / (label + '_settling.json'), assessment)
        self.stages.append(label); self.task['state'] = 'executing'
        if waypoint_continuation.forbidden_contacts(result):
            return self.finish('forbidden_contact')
        if self.visual_issues:
            return self.finish('visual_geometry_rejected_before_close')
        outcome = 'accepted'
        if not assessment['accepted']:
            proof = waypoint_continuation.bind(self.episode, label, result, self.bridge.snapshot(), self.desired,
                active=self.bridge.active, fault=self.bridge.fault)
            path = self.episode / (label + '_continuation.json'); write(path, proof)
            self.log_event({'kind': 'C_waypoint_continuation', 'command_label': label,
                            'receipt': str(path), 'receipt_sha256': sha(path), 'pass': proof['pass'],
                            'original_arrival_accepted': False})
            if not proof['pass']:
                return self.finish('C_waypoint_not_stationary')
            outcome = 'stationary_waypoint_continuation'
        self.previous_execution = result; self.previous_goal = self.desired.copy()
        self.outcome(outcome, True)
        self.ready_time = self.group.world.current_time; self.state = 'local_decision'


def install(campaign):
    reg = read(Path(campaign) / 'registration.json')
    if reg['C_waypoint_continuation_policy'] != waypoint_continuation.POLICY:
        raise ValueError('Unregistered C waypoint scheduling')
    runtime_closure_recovery.install(campaign)
    runtime_learned.audit_policy = audit_policy
    runtime_learned.live_sample = live_sample
    runtime_learned.LearnedEnvironment = StageEnvironment


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--campaign', required=True); parser.add_argument('--launch', required=True)
    args = parser.parse_args(); install(args.campaign)
    runtime_learned.LearnedGroup(args.campaign, args.launch).run()
