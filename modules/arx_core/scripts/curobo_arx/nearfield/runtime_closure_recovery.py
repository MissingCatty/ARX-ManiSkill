"""Advisor28 interpreter: bounded model motion and separately attributed closure."""
import argparse
from pathlib import Path
import time
import numpy as np
from scripts.curobo_arx.data_factory.common import read, write, sha
from scripts.curobo_arx.data_factory.runtime import Environment
from training.nearfield_arx import closure_recovery as policy
from training.nearfield_arx.learned_policy import read_bound_reply, dispatch_drift
from training.nearfield_arx.entry_guard import validate_receipt, rejection_reason
from . import runtime_learned


class ClosureEnvironment(runtime_learned.LearnedEnvironment):
    def plan(self, phase, world):
        if phase == 'lift':
            # The certificate is computed under the frozen model before closure.
            goal = self.closure_certificate['nominal_closed']['selected_lift_world']
            return Environment.plan(self, phase, np.asarray(goal))
        return super().plan(phase, world)

    def consume_prediction(self, snapshot):
        pending = self.pending_policy; path = pending['path']; request = pending['request']
        reply_path = path.with_name(request['request_id']+'_reply.json')
        if (self.group.campaign/'workers/policy/FAULT.json').exists():
            raise RuntimeError('Policy service fault')
        if not reply_path.exists():
            if time.monotonic()-pending['started'] > self.group.registration['learned_policy']['request_timeout_seconds']:
                raise TimeoutError('Policy request timed out; no retry')
            return
        _, reply = read_bound_reply(path, reply_path);self.pending_policy = None
        drift = dispatch_drift(pending['snapshot'], snapshot)
        write(path.with_name(request['request_id']+'_dispatch_check.json'), drift)
        if not drift['pass']:return self.finish('policy_observation_changed_before_dispatch')
        if not reply['projection']['target_visibility_pass']:return self.finish('policy_target_visibility_rejected')
        method = request['method'];token = reply['prediction'].strip()
        guard = validate_receipt(path, request, reply, self.group.registration['learned_policy']['visual_contract']['depth_scale']['near_m'])
        try:
            transform = policy.action_transform(method, token)
            trigger = policy.closure_trigger(method, token, request['inputs'], self.local_index)
        except ValueError as exc:
            self.log_event({'kind':'policy_invalid_action','policy_request_id':request['request_id'],
                            'raw_prediction':reply['prediction'],'reason':str(exc)})
            return self.finish('policy_invalid_action')
        trace = self.trace_before(method, token)
        trace.update(policy_request_id=request['request_id'], policy_input_raw_index=request['raw_index'],
                     policy_input_eef_world=request['inputs']['eef_world'], policy_reply_sha256=sha(reply_path))
        write(self.episode/'decision_traces.json', self.traces)
        common = {'token':token, 'decision_id':trace['decision_id'], 'policy_request_id':request['request_id'],
                  'driver':'learned_model_with_registered_closure_interpreter', 'policy_input_raw_index':request['raw_index']}
        self.log_event({**common, 'kind':'policy_interpreter_decision', 'action_transform':transform, 'closure_trigger':trigger})
        if trigger['request_closure']:
            self.grasp_seen = True
            self.log_event({**common, 'kind':'grasp_request', 'interpreter_action':'GRASP',
                            'closure_origin':trigger['origin'], 'model_explicit_GRASP':token == 'GRASP',
                            'closure_trigger':trigger})
            return self.request_closure(trace['decision_id'])
        if trigger['motion_budget_exhausted']:
            return self.finish(method+'_budget_endpoint_outside_registered_closure_region')
        if not guard['pass']:
            self.log_event({**common,'kind':'policy_action_guard_rejected','reason':rejection_reason(method,guard),'action_admission':guard})
            return self.finish('policy_action_guard_rejected')
        try:
            goal = policy.decode(method, request['inputs']['eef_world'], token, self.group.active_trial['G0_world'])
        except ValueError as exc:
            self.log_event({**common,'kind':'policy_target_rejected','reason':str(exc)})
            return self.finish('policy_target_rejected')
        if method == 'D':
            self.session.refines += 1;self.session.calls += 1
            trace['history_after_action'].update(refines_processed=self.session.refines)
            write(self.episode/'decision_traces.json',self.traces)
        self.local_index += 1
        label = f'c_step_{self.local_index:03d}' if method == 'C' else f'd_refine_{self.local_index:03d}'
        self.log_event({**common,'kind':'target_request','goal_world':goal.tolist(),
                        'action_transform':transform,'execution_phase':'local','command_label':label})
        return self.submit(label, goal, 'local')


def install(campaign):
    from scripts.curobo_arx.data_factory import visual_geometry
    from training.nearfield_arx.closure_verification import audit_policy
    reg=read(Path(campaign)/'registration.json')
    if reg['closure_recovery_policy'] != policy.POLICY:
        raise ValueError('Unregistered action or closure settings')
    if reg['visual_geometry_policy'] != {**visual_geometry.POLICY,'minimum_surface_distance_m':.008}:
        raise ValueError('Unexpected visual configuration')
    visual_geometry.POLICY.update(reg['visual_geometry_policy'])
    policy.install();runtime_learned.audit_policy=audit_policy
    runtime_learned.LearnedEnvironment=ClosureEnvironment


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--launch',required=True);a=p.parse_args()
    install(a.campaign);runtime_learned.LearnedGroup(a.campaign,a.launch).run()
