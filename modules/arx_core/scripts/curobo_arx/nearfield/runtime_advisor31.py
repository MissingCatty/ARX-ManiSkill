"""Prospectively registered ARX contact-aware continuation; frozen mechanics."""
import argparse
from pathlib import Path
import time
import numpy as np
from scripts.curobo_arx.data_factory.common import read, write, sha
from scripts.curobo_arx.settling import assess_settling
from training.nearfield_arx import closure_recovery as policy
from training.nearfield_arx.learned_policy import read_bound_reply, dispatch_drift
from training.nearfield_arx.entry_guard import validate_receipt, rejection_reason
from . import runtime_closedloop30, runtime_learned, runtime_v2, closedloop30_rules
from .runtime_physical import PhysicalEnvironment
from .advisor31_policy import verify_reply
from .advisor31_io import StatusWriter
from .attribution_spec import fixed_entry_candidates
from . import advisor31_rules as rules


class AdvisorEnvironment(runtime_closedloop30.ClosedLoopEnvironment):
    def consume_prediction(self, snapshot):
        pending = self.pending_policy; path = pending['path']; request = pending['request']
        reply_path = path.with_name(request['request_id']+'_reply.json')
        if (self.group.campaign/'workers/policy/FAULT.json').exists():
            raise RuntimeError('Policy service fault')
        if not reply_path.exists():
            if time.monotonic()-pending['started'] > self.group.registration['learned_policy']['request_timeout_seconds']:
                raise TimeoutError('Policy request timed out; no retry')
            return
        _, reply = read_bound_reply(path, reply_path)
        verify_reply(self.group.registration, request, reply)
        self.pending_policy = None
        if not reply['visual_control']['applicable']:
            self.log_event({'kind': 'visual_control_not_applicable', 'policy_request_id': request['request_id'],
                            'alignment': reply['visual_control']['alignment']})
            return self.finish('visual_control_pose_pair_not_comparable')
        drift = dispatch_drift(pending['snapshot'], snapshot)
        write(path.with_name(request['request_id']+'_dispatch_check.json'), drift)
        if not drift['pass']: return self.finish('policy_observation_changed_before_dispatch')
        if not reply['projection']['target_visibility_pass']: return self.finish('policy_target_visibility_rejected')
        method = request['method']; token = reply['prediction'].strip()
        guard = validate_receipt(path, request, reply,
            self.group.registration['learned_policy']['visual_contract']['depth_scale']['near_m'])
        try:
            transform = policy.action_transform(method, token)
            trigger = rules.interpret(method, token, request['inputs'], self.local_index,
                                      reply['advisor31_depth'], reply['contact_control'])
        except ValueError as exc:
            self.log_event({'kind': 'policy_invalid_action', 'policy_request_id': request['request_id'],
                            'raw_prediction': reply['prediction'], 'reason': str(exc)})
            return self.finish('policy_invalid_action')
        trace = self.trace_before(method, token)
        trace.update(policy_request_id=request['request_id'], policy_input_raw_index=request['raw_index'],
                     policy_input_eef_world=request['inputs']['eef_world'], policy_reply_sha256=sha(reply_path))
        write(self.episode/'decision_traces.json', self.traces)
        common = {'token': token, 'decision_id': trace['decision_id'], 'policy_request_id': request['request_id'],
                  'driver': rules.VERSION, 'policy_input_raw_index': request['raw_index'],
                  'model_inference': reply['model_inference']}
        self.log_event({**common, 'kind': 'policy_interpreter_decision',
                        'action_transform': transform, 'closure_trigger': trigger})
        if trigger['request_closure']:
            self.grasp_seen = True
            self.log_event({**common, 'kind': 'grasp_request', 'interpreter_action': 'GRASP',
                'closure_origin': trigger['origin'],
                'model_explicit_GRASP': bool(reply['model_inference'] and token == 'GRASP'),
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
        if label != 'close' or self.state not in ('executing', 'waiting_closed_idle'):
            return super().tick()
        if self.bridge.fault: raise RuntimeError(str(self.bridge.fault))
        if time.monotonic()-self.started > self.group.registration['attempt_seconds']:
            raise TimeoutError('Original per-attempt deadline exceeded after closure')
        if self.state == 'executing':
            result = self.bridge.result()
            if result is None: return
            write(self.episode/'close_execution.json', result)
            write(self.episode/'close_settling.json', assess_settling(result, self.desired))
            self.stages.append('close'); self.task['state'] = 'executing'
            self.completed_close = result; self.state = 'waiting_closed_idle'
        if self.bridge.active: return
        snapshot = self.bridge.snapshot()
        closed_m = float(self.bridge.context.controller._gripper_config('right')['closed_m'])
        proof = rules.post_close(self.completed_close, snapshot, self.last_command, closed_m,
                                 active=self.bridge.active, fault=self.bridge.fault)
        if not proof['stationary']: return
        proof['files'] = {str(self.episode/('close_'+s+'.json')): sha(self.episode/('close_'+s+'.json'))
                          for s in ('command', 'execution', 'settling')}
        path = self.episode/'close_contact_admission.json'; write(path, proof)
        self.log_event({'kind': 'post_close_contact_admission', 'pass': proof['pass'],
                        'receipt': str(path), 'receipt_sha256': sha(path),
                        'original_arrival_accepted': read(self.episode/'close_settling.json')['accepted']})
        if not proof['pass']:
            return self.finish('forbidden_contact' if proof['forbidden_contacts'] else 'close_without_grip_evidence')
        # ClosureEnvironment.plan selects the original independently validated lift.
        return self.plan('lift', np.asarray(self.closure_certificate['hold_world']))

    def finish(self, reason=None):
        from .advisor31_audit import audit_policy
        if self.recorder and hasattr(self, 'policy_requests'):
            self.group.writer.drain()
            write(self.episode/'policy_integrity.json', audit_policy(self.episode))
        write(self.group.out/'execution_timing.json', self.group.timing())
        return PhysicalEnvironment.finish(self, reason)


class AdvisorGroup(runtime_learned.LearnedGroup):
    def __init__(self, *args, **kwargs):
        self.step_timings = {}; super().__init__(*args, **kwargs)

    def step(self, record=True):
        stage = '|'.join(str(e.state) for e in getattr(self, 'envs', []))
        started = time.perf_counter()
        try: return super().step(record)
        finally:
            value = self.step_timings.setdefault(stage, {'calls': 0, 'wall_seconds': 0.})
            value['calls'] += 1; value['wall_seconds'] += time.perf_counter()-started

    def timing(self):
        return {'simulation_capture_by_state': self.step_timings, 'status_io': runtime_v2.write.metrics,
                'all_capture_checks_and_frames_retained': True}

    def run(self):
        try: return super().run()
        finally: write(self.out/'execution_timing.json', self.timing())


def install(campaign):
    from scripts.curobo_arx.data_factory import runtime as factory
    from .advisor31_audit import physical
    reg = read(Path(campaign)/'registration.json')
    if reg['advisor31_policy'] != rules.POLICY: raise ValueError('Unregistered advisor31 rules')
    runtime_closedloop30.install(campaign)
    closedloop30_rules.entry_candidates = fixed_entry_candidates
    runtime_learned.LearnedEnvironment = AdvisorEnvironment
    runtime_v2.write = StatusWriter(write)
    factory.physical = physical


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--campaign', required=True); p.add_argument('--launch', required=True)
    a = p.parse_args(); install(a.campaign); AdvisorGroup(a.campaign, a.launch).run()
