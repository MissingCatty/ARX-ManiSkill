"""Four-slot learned C/D loop; source Gstar is never an action generator."""
import argparse
import copy
from pathlib import Path
import time
import uuid
import numpy as np
from scripts.curobo_arx.data_factory.common import read, write, sha
from training.nearfield_arx.contract import parse_action, decode, pose_error
from training.nearfield_arx.learned_policy import live_sample, dispatch_drift, read_bound_reply, audit_policy
from .runtime_physical import PhysicalEnvironment, PhysicalGroup


class LearnedEnvironment(PhysicalEnvironment):
    def plan(self, phase, world):
        result = super().plan(phase, world)
        if phase == 'approach':
            self.pending_policy = None; self.policy_requests = []
            write(self.episode / 'policy_requests.json', [])
            self.submit('state_generation', self.group.active_trial['offset_goal_world'], 'local')
        return result

    def decide(self):
        snapshot = self.bridge.snapshot(); trial = self.group.active_trial; method = trial['method']
        if self.bridge.active or self.bridge.fault:raise RuntimeError('Policy requires confirmed healthy idle robot')
        if self.pending_policy is not None:return self.consume_prediction(snapshot)
        row = self.last_observation
        completed = self.entry_stopped_sim_time if not self.physical_entry_ready else self.previous_execution['final']['sim_time']
        if row['frame_time'] <= completed + 1e-6:return
        for state in (snapshot, row['snapshot']):
            speeds = dict(zip(state['joint_names'], state['joint_velocities']))
            if max(abs(speeds[n]) for n in state['right_arm']) > .01:return
        if not self.physical_entry_ready:
            self.actual_entry = row
            write(self.episode / 'physical_entry_observation.json', row);write(self.episode / 'actual_entry.json', row)
            self.log_event({'kind': 'physical_entry_stopped', 'source_anchor_not_positive': True,
                'registered_G0_world': trial['G0_world'], 'registered_H_world': trial['entry_world'],
                'actual_entry_error': pose_error(snapshot['link_world']['right_eef_link'], trial['entry_world'])})
            self.physical_entry_ready = True
        if self.calls >= self.group.registration['learned_policy']['max_policy_calls'][method]:
            return self.finish(method + '_policy_call_budget_exhausted')
        self.group.writer.drain()
        ident = uuid.uuid4().hex; index = self.recorder.rows - 1
        previous = next(r for r in self.requests if r['id'] == self.last_command['id'])
        sample = live_sample(row, method, trial['G0_world'], self.previous_goal, previous,
            self.calls, self.session.refines, self.prior_event)
        folder = self.episode / 'policy'; observation = folder / (ident + '_observation.json')
        write(observation, row)
        rgb, depth = self.episode / f'raw/{index:06d}.png', self.episode / f'raw/{index:06d}.npy'
        request = {'request_id': ident, 'method': method, 'trial_id': trial['trial_id'],
            'raw_index': index, 'observation': str(observation), 'rgb': str(rgb), 'depth': str(depth),
            'asset': self.asset['usd_path'], 'inputs': sample['inputs'],
            'registration_sha256': sha(self.group.campaign / 'registration.json'),
            'files': {str(p): sha(p) for p in (observation, rgb, depth)}}
        path = folder / (ident + '_request.json')
        self.policy_requests.append({'request_id': ident, 'raw_index': index, 'request': str(path)})
        write(self.episode / 'policy_requests.json', self.policy_requests);write(path, request)
        self.pending_policy = {'path': path, 'request': request, 'snapshot': copy.deepcopy(row['snapshot']),
            'started': time.monotonic()}
        write(self.group.campaign / 'workers/policy/queue' / (ident + '.json'), {'request': str(path)})

    def consume_prediction(self, snapshot):
        pending = self.pending_policy; path = pending['path']; request = pending['request']
        reply_path = path.with_name(request['request_id'] + '_reply.json')
        if (self.group.campaign / 'workers/policy/FAULT.json').exists():raise RuntimeError('Policy service fault')
        if not reply_path.exists():
            if time.monotonic() - pending['started'] > self.group.registration['learned_policy']['request_timeout_seconds']:
                raise TimeoutError('Policy request timed out; no retry')
            return
        _, reply = read_bound_reply(path, reply_path)
        self.pending_policy = None
        drift = dispatch_drift(pending['snapshot'], snapshot)
        write(path.with_name(request['request_id'] + '_dispatch_check.json'), drift)
        if not drift['pass']:return self.finish('policy_observation_changed_before_dispatch')
        if not reply['projection']['target_visibility_pass']:return self.finish('policy_target_visibility_rejected')
        method = request['method']; token = reply['prediction'].strip()
        try:
            parse_action(method, token)
            if method == 'D' and token != 'GRASP' and self.session.refines < 2:
                decode('D', request['inputs']['eef_world'], token, self.session._g0)
        except ValueError as exc:
            self.log_event({'kind': 'policy_invalid_action', 'policy_request_id': request['request_id'],
                'raw_prediction': reply['prediction'], 'reason': str(exc)})
            return self.finish('policy_invalid_action')
        trace = self.trace_before(method, token)
        trace.update(policy_request_id=request['request_id'], policy_input_raw_index=request['raw_index'],
            policy_input_eef_world=request['inputs']['eef_world'], policy_reply_sha256=sha(reply_path))
        write(self.episode / 'decision_traces.json', self.traces)
        common = {'token': token, 'decision_id': trace['decision_id'], 'policy_request_id': request['request_id'],
            'driver': 'learned_qwen_lora_and_projector', 'policy_input_raw_index': request['raw_index']}
        if token == 'GRASP':
            self.grasp_seen = True; self.log_event({**common, 'kind': 'grasp_request'})
            return self.request_closure(trace['decision_id'])
        ee = np.asarray(request['inputs']['eef_world'])
        if method == 'D':
            event = self.session.request(ee, token)
            trace['history_after_action'].update(refines_processed=self.session.refines)
            if event['kind'] == 'budget_exhausted':return self.finish('D_correction_budget_exhausted')
            goal = np.asarray(event['goal_world'])
        else:
            if self.local_index >= self.group.registration['learned_policy']['C_max_motion_requests']:
                return self.finish('C_motion_budget_exhausted')
            goal = decode('C', ee, token)
        self.local_index += 1
        label = f'c_step_{self.local_index:03d}' if method == 'C' else f'd_refine_{self.local_index:03d}'
        self.log_event({**common, 'kind': 'target_request', 'goal_world': goal.tolist(), 'execution_phase': 'local', 'command_label': label})
        return self.submit(label, goal, 'local')

    def finish(self, reason=None):
        if self.recorder and hasattr(self, 'policy_requests'):
            self.group.writer.drain(); write(self.episode / 'policy_integrity.json', audit_policy(self.episode))
        return super().finish(reason)


class LearnedGroup(PhysicalGroup):
    def __init__(self, campaign, launch):
        super().__init__(campaign, launch)
        ready = read(self.campaign / 'workers/policy/ready.json')
        if ready['registration_sha256'] != sha(self.campaign / 'registration.json'):
            raise RuntimeError('Policy service is not bound to this campaign')
        self.envs = [LearnedEnvironment(i, e.bridge, self) for i, e in enumerate(self.envs)]


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--campaign', required=True);p.add_argument('--launch', required=True)
    a = p.parse_args();LearnedGroup(a.campaign, a.launch).run()
