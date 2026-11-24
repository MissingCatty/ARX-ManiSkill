"""Isolated measured-state C/D execution for the advisor30 registration."""
import argparse
from pathlib import Path
import time
import uuid
import numpy as np
from scripts.curobo_arx.data_factory.common import read, write, sha
from scripts.curobo_arx.settling import assess_settling
from training.nearfield_arx.contract import pose_error
from training.nearfield_arx import closure_recovery as policy
from training.nearfield_arx.learned_policy import read_bound_reply, dispatch_drift
from training.nearfield_arx.entry_guard import validate_receipt, rejection_reason
from . import runtime_learned, runtime_closure_recovery, runtime_stage_recovery, runtime_v2
from .runtime_physical import PhysicalEnvironment
from . import closedloop30_rules as rules


class ClosedLoopEnvironment(runtime_closure_recovery.ClosureEnvironment):
    def log_event(self, event):
        return super().log_event({**event, 'wall_unix': time.time()})

    def plan(self, phase, world):
        if phase == 'approach':
            self.axial_done = False
            write(self.episode/'closedloop_contract.json', {
                'policy': rules.POLICY, 'method': self.group.active_trial['method'],
                'G0_world': self.group.active_trial['G0_world'],
                'nominal_entry_world': self.group.active_trial['entry_world']})
        return super().plan(phase, world)

    def record(self):
        saved = self.stage
        if saved == 'd_entry_axis': self.stage = 'approach'
        try: return super().record()
        finally: self.stage = saved

    def submit(self, label, goal, phase='approach', action='plan'):
        trial = self.group.active_trial
        if label == 'state_generation' and len(rules.entry_candidates(trial)) > 1:
            action = 'adaptive_entry'
        return super().submit(label, goal, phase, action)

    def decide(self):
        snapshot = self.bridge.snapshot(); trial = self.group.active_trial
        if self.bridge.active or self.bridge.fault:
            raise RuntimeError('Policy requires a confirmed healthy idle robot')
        if self.pending_policy is not None:
            return self.consume_prediction(snapshot)
        row = self.last_observation
        completed = self.entry_stopped_sim_time if not self.physical_entry_ready else self.previous_execution['final']['sim_time']
        if row['frame_time'] <= completed+1e-6:
            return
        for state in (snapshot, row['snapshot']):
            velocities = dict(zip(state['joint_names'], state['joint_velocities']))
            if max(abs(velocities[n]) for n in state['right_arm']) > .01:
                return
        if not self.physical_entry_ready:
            self.actual_entry = row
            write(self.episode/'physical_entry_observation.json', row)
            write(self.episode/'actual_entry.json', row)
            selected = self.desired.tolist()
            self.log_event({'kind': 'physical_entry_stopped', 'source_anchor_not_positive': True,
                'registered_G0_world': trial['G0_world'], 'registered_H_world': trial['entry_world'],
                'selected_entry_world': selected,
                'actual_entry_error': pose_error(snapshot['link_world']['right_eef_link'], selected)})
            self.physical_entry_ready = True
        if trial['method'] == 'D' and not self.axial_done:
            proof = rules.axial_goal(snapshot['link_world']['right_eef_link'], trial['G0_world'])
            self.axial_done = True
            self.log_event({'kind': 'D_axial_entry_preparation', **proof,
                'G0_world': trial['G0_world'], 'command_label': 'd_entry_axis',
                'counts_as_visual_refinement': False})
            if proof['distance_m'] > 1e-6:
                return self.submit('d_entry_axis', proof['goal_world'], 'local')
        if trial['method'] == 'D' and self.calls == 0:
            error = pose_error(snapshot['link_world']['right_eef_link'], trial['G0_world'])
            if error['translation_m'] > rules.POLICY['D_nearfield_radius_m']:
                return self.finish('D_axial_preparation_outside_nearfield')
        return super().decide()

    def consume_prediction(self, snapshot):
        pending = self.pending_policy; path = pending['path']; request = pending['request']
        reply_path = path.with_name(request['request_id']+'_reply.json')
        if (self.group.campaign/'workers/policy/FAULT.json').exists():
            raise RuntimeError('Policy service fault')
        if not reply_path.exists():
            if time.monotonic()-pending['started'] > self.group.registration['learned_policy']['request_timeout_seconds']:
                raise TimeoutError('Policy request timed out; no retry')
            return
        _, reply = read_bound_reply(path, reply_path); self.pending_policy = None
        drift = dispatch_drift(pending['snapshot'], snapshot)
        write(path.with_name(request['request_id']+'_dispatch_check.json'), drift)
        if not drift['pass']: return self.finish('policy_observation_changed_before_dispatch')
        if not reply['projection']['target_visibility_pass']: return self.finish('policy_target_visibility_rejected')
        method = request['method']; token = reply['prediction'].strip()
        guard = validate_receipt(path, request, reply,
            self.group.registration['learned_policy']['visual_contract']['depth_scale']['near_m'])
        try:
            transform = policy.action_transform(method, token)
            trigger = rules.interpret(method, token, request['inputs'], self.local_index)
        except ValueError as exc:
            self.log_event({'kind': 'policy_invalid_action', 'policy_request_id': request['request_id'],
                            'raw_prediction': reply['prediction'], 'reason': str(exc)})
            return self.finish('policy_invalid_action')
        trace = self.trace_before(method, token)
        trace.update(policy_request_id=request['request_id'], policy_input_raw_index=request['raw_index'],
                     policy_input_eef_world=request['inputs']['eef_world'], policy_reply_sha256=sha(reply_path))
        write(self.episode/'decision_traces.json', self.traces)
        common = {'token': token, 'decision_id': trace['decision_id'], 'policy_request_id': request['request_id'],
                  'driver': rules.VERSION, 'policy_input_raw_index': request['raw_index']}
        self.log_event({**common, 'kind': 'policy_interpreter_decision',
                        'action_transform': transform, 'closure_trigger': trigger})
        if trigger['request_closure']:
            self.grasp_seen = True
            self.log_event({**common, 'kind': 'grasp_request', 'interpreter_action': 'GRASP',
                'closure_origin': trigger['origin'], 'model_explicit_GRASP': token == 'GRASP',
                'closure_trigger': trigger})
            return self.request_closure(trace['decision_id'])
        if trigger['motion_budget_exhausted']:
            return self.finish(method+'_budget_endpoint_outside_registered_closure_region')
        if not guard['pass']:
            self.log_event({**common, 'kind': 'policy_action_guard_rejected',
                            'reason': rejection_reason(method, guard), 'action_admission': guard})
            return self.finish('policy_action_guard_rejected')
        try:
            goal = policy.decode(method, request['inputs']['eef_world'], token, self.group.active_trial['G0_world'])
        except ValueError as exc:
            self.log_event({**common, 'kind': 'policy_target_rejected', 'reason': str(exc)})
            return self.finish('policy_target_rejected')
        if method == 'D':
            self.session.refines += 1; self.session.calls += 1
            trace['history_after_action'].update(refines_processed=self.session.refines)
            write(self.episode/'decision_traces.json', self.traces)
        self.local_index += 1
        label = f'c_step_{self.local_index:03d}' if method == 'C' else f'd_refine_{self.local_index:03d}'
        self.log_event({**common, 'kind': 'target_request', 'goal_world': goal.tolist(),
                        'action_transform': transform, 'execution_phase': 'local', 'command_label': label})
        return self.submit(label, goal, 'local')

    def tick(self):
        label = (self.last_command or {}).get('label', '')
        if self.state not in ('executing', 'waiting_waypoint_idle') or not rules.waypoint(label):
            return super().tick()
        if self.bridge.fault: raise RuntimeError(str(self.bridge.fault))
        if time.monotonic()-self.started > self.group.registration['attempt_seconds']:
            raise TimeoutError('Registered attempt limit exceeded while waiting for measured idle')
        if self.state == 'executing':
            result = self.bridge.result()
            if result is None: return
            write(self.episode/(label+'_execution.json'), result)
            assessment = assess_settling(result, self.desired)
            write(self.episode/(label+'_settling.json'), assessment)
            self.stages.append(label); self.task['state'] = 'executing'
            if rules.forbidden_contacts(result): return self.finish('forbidden_contact')
            if self.visual_issues: return self.finish('visual_geometry_rejected_before_close')
            self.completed_waypoint = result; self.state = 'waiting_waypoint_idle'
        snapshot = self.bridge.snapshot()
        proof = rules.bind_waypoint(self.episode, label, self.completed_waypoint, snapshot, self.desired,
                                   active=self.bridge.active, fault=self.bridge.fault)
        if not proof['pass']: return
        path = self.episode/(label+'_waypoint_stop.json'); write(path, proof)
        self.log_event({'kind': 'CD_waypoint_stop', 'command_label': label, 'receipt': str(path),
                        'receipt_sha256': sha(path), 'pass': True,
                        'original_arrival_accepted': proof['original_arrival_accepted']})
        self.previous_execution = self.completed_waypoint; self.previous_goal = self.desired.copy()
        outcome = 'accepted' if proof['original_arrival_accepted'] else 'stationary_waypoint_continuation'
        self.outcome(outcome, True)
        self.ready_time = self.group.world.current_time; self.state = 'local_decision'

    def dwell(self, label, desired):
        if label != 'close' or self.closure_binding.get('version') != rules.VERSION:
            return super().dwell(label, desired)
        q = self.closure_binding['hold_q']
        command = {'id': uuid.uuid4().hex, 'label': label,
            'joint_names': self.closure_binding['hold_joint_names'], 'position': [q, q],
            'time_from_start': [0., 3.], 'gripper_target': .1, 'position_tolerance_rad': .005,
            'planner': 'Registered measured stationary hold; independent original closed-model validation'}
        return self.dispatch(command, desired)

    def finish(self, reason=None):
        from .closedloop30_audit import audit_policy
        if self.recorder and hasattr(self, 'policy_requests'):
            self.group.writer.drain()
            write(self.episode/'policy_integrity.json', audit_policy(self.episode))
        return PhysicalEnvironment.finish(self, reason)


def install(campaign):
    from scripts.curobo_arx.data_factory import runtime as factory
    from .closedloop30_audit import physical
    reg = read(Path(campaign)/'registration.json')
    if reg['closedloop30_policy'] != rules.POLICY: raise ValueError('Unregistered advisor30 rules')
    runtime_closure_recovery.install(campaign)
    runtime_v2.bind = rules.bind_closure; runtime_v2.validate_reply = rules.validate_closure
    factory.physical = physical
    runtime_learned.live_sample = runtime_stage_recovery.live_sample
    runtime_learned.LearnedEnvironment = ClosedLoopEnvironment


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--campaign', required=True); p.add_argument('--launch', required=True)
    a = p.parse_args(); install(a.campaign); runtime_learned.LearnedGroup(a.campaign, a.launch).run()
