"""ARX measured execution of raw full6D REFINE and explicit GRASP only."""
import argparse
import copy
from pathlib import Path
import time
import uuid

import numpy as np

from scripts.curobo_arx.data_factory.common import read, write, sha, digest
from scripts.curobo_arx.data_factory import runtime as factory
from training.nearfield_arx.learned_policy import read_bound_reply
from training.nearfield_arx import native_mask
from .runtime_advisor34 import Environment as MeasuredEnvironment, Group as MeasuredGroup
from .advisor34_policy import stationary_dispatch
from .advisor36_contract import interpret
from .closure import observation_timing
from .advisor31_io import StatusWriter


class Environment(MeasuredEnvironment):
    def decide(self):
        self.sample_stop()
        if self.pending_policy is not None:
            return self.consume_prediction(self.bridge.snapshot())
        if self.last_observation['frame_time'] <= self.previous_execution['final']['sim_time'] + 1e-6:
            return
        if not self.proof(0.)['pass']:
            return
        if self.calls >= 3:
            return self.finish('pure_model_call_budget_exhausted_without_GRASP')
        row = self.last_observation; trial = self.group.active_trial
        if not self.entry_saved:
            self.actual_entry = row
            write(self.episode / 'physical_entry_observation.json', row)
            write(self.episode / 'actual_entry.json', row)
            self.entry_saved = True
            self.log_event({'kind': 'H_visual_takeover', 'G0_guidance_after_H': False,
                            'blind_axial_preparation': False})
        self.group.writer.drain(); ident = uuid.uuid4().hex; index = self.recorder.rows - 1
        folder = self.episode / 'policy'; observation = folder / (ident + '_observation.json')
        write(observation, row)
        rgb = self.episode / f'raw/{index:06d}.png'; depth = self.episode / f'raw/{index:06d}.npy'
        req = {'request_id': ident, 'method': 'D', 'trial_id': trial['trial_id'],
            'object_id': trial['object_id'], 'vision_arm': 'fresh', 'mode': 'decision',
            'motions_completed': self.local_index, 'raw_index': index,
            'observation': str(observation), 'rgb': str(rgb), 'depth': str(depth),
            'inputs': {'eef_world': row['snapshot']['link_world']['right_eef_link'], 'stopped': True},
            'registration_sha256': sha(self.group.campaign / 'registration.json'),
            'files': {str(p): sha(p) for p in (observation, rgb, depth)}}
        req['native_mask_binding'] = native_mask.bind(row, rgb)['native_mask_binding']
        req['files'].update(req['native_mask_binding']['files'])
        path = folder / (ident + '_request.json'); write(path, req)
        self.policy_requests.append({'request_id': ident, 'raw_index': index, 'request': str(path)})
        write(self.episode / 'policy_requests.json', self.policy_requests)
        self.pending_policy = {'path': path, 'request': req, 'snapshot': copy.deepcopy(row['snapshot']),
                               'started': time.monotonic()}
        write(self.group.campaign / 'workers/policy/queue' / (ident + '.json'), {'request': str(path)})

    def consume_prediction(self, snapshot):
        pending = self.pending_policy; path = pending['path']; req = pending['request']
        reply_path = path.with_name(req['request_id'] + '_reply.json')
        if (self.group.campaign / 'workers/policy/FAULT.json').exists():
            raise RuntimeError('Policy infrastructure fault')
        if not reply_path.exists():
            if time.monotonic() - pending['started'] > self.group.registration['learned_policy']['request_timeout_seconds']:
                raise TimeoutError('Policy request timeout')
            return
        _, reply = read_bound_reply(path, reply_path); self.pending_policy = None
        drift = stationary_dispatch(pending['snapshot'], snapshot, self.proof(0.))
        write(path.with_name(req['request_id'] + '_dispatch_check.json'), drift)
        if not drift['pass']:
            raise RuntimeError('Robot/observation stationary state unconfirmed before dispatch')
        token = reply['prediction'].strip(); self.calls += 1; self.neural_calls += 1
        trace = {'decision_id': uuid.uuid4().hex, 'raw_index': req['raw_index'],
            'observation_time_s': read(req['observation'])['frame_time'], 'method': 'D',
            'action': token, 'policy_request_id': req['request_id'],
            'history_after_action': {'calls_completed': self.calls}}
        self.traces.append(trace); write(self.episode / 'decision_traces.json', self.traces)
        common = {'token': token, 'policy_request_id': req['request_id'],
                  'decision_id': trace['decision_id'], 'model_inference': True}
        action = reply['action']
        try:
            expected = interpret(req['inputs']['eef_world'], token)
        except ValueError:
            expected = {'kind': 'reject'}
        if expected['kind'] != action['kind']:
            raise RuntimeError('Raw model interpretation changed in transit')
        self.log_event({**common, 'kind': 'policy_interpreter_decision', 'action': action})
        if action['kind'] == 'reject':
            return self.finish(action.get('reason', 'model_numeric_anomaly'))
        if action['kind'] == 'close':
            if token != 'GRASP':
                raise RuntimeError('Non-model closure request')
            self.grasp_seen = True
            self.log_event({**common, 'kind': 'grasp_request', 'closure_origin': 'explicit_model_GRASP',
                            'model_explicit_GRASP': True})
            return self.request_closure(trace['decision_id'])
        if self.local_index >= 2:
            return self.finish('pure_model_motion_budget_exhausted_without_GRASP')
        if not np.allclose(expected['goal_world'], action['goal_world'], atol=1e-12, rtol=0):
            raise RuntimeError('One or more raw REFINE coordinates were altered')
        self.local_index += 1; self.session.refines += 1
        label = f'd_refine_{self.local_index:03d}'
        self.log_event({**common, 'kind': 'target_request', 'goal_world': action['goal_world'],
            'command_label': label, 'controller_axial_only': False, 'all_six_values_preserved': True})
        return self.submit(label, action['goal_world'])

    def request_closure(self, ident):
        self.group.writer.drain(); snapshot = self.bridge.snapshot(); proof = self.proof(0.)
        if not proof['pass']:
            raise RuntimeError('Robot stationary state unconfirmed before explicit GRASP')
        names = snapshot['right_arm']
        q = [float(snapshot['joint_positions'][snapshot['joint_names'].index(n)]) for n in names]
        binding = {'pass': True, 'version': 'advisor40_explicit_only', 'grasp_request_id': ident,
            'closure_source': 'explicit_model_GRASP', 'hold_joint_names': names, 'hold_q': q,
            'snapshot_sha256': digest(snapshot), 'measured_hold_snapshot': snapshot,
            'files': proof['files'], 'arrival': proof,
            'previous_command_id': self.last_command['id'], 'previous_command_label': self.last_command['label'],
            'observation_timing': observation_timing(self.last_observation),
            'configuration_identity': self.group.registration['source_identity'],
            'target_geometry_closure_gate_applicable': False}
        write(self.episode / 'closure_request.json', binding)
        self.closure_binding = binding; self.closure_snapshot = snapshot
        scene = self.episode / 'closure_scene.json'; write(scene, self.bridge.export_scene())
        self.future = self.group.planner.submit('closure_validate', binding=binding, snapshot=snapshot,
                                               scene=str(scene), output=str(self.episode))
        self.state = 'validating_closure'

    def finish(self, reason=None):
        if reason in ('closure_not_stationary', 'closure_dispatch_not_stationary',
                      'stationary_wait_10sim_s_timeout', 'stationary_wall_safety_120s_timeout',
                      'observation_changed_before_dispatch'):
            raise RuntimeError('Confirmed idle is required; stop subsequent scheduling: ' + reason)
        return super().finish(reason)


class Group(MeasuredGroup):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.envs = [Environment(i, e.bridge, self) for i, e in enumerate(self.envs)]

    def run_seed_trial(self, asset):
        result = super().run_seed_trial(asset)
        env = self.envs[0]
        if env.bridge.active or env.bridge.fault:
            raise RuntimeError('Robot idle after slot is unconfirmed')
        if env.episode and Path(env.episode).is_relative_to(self.out) and Path(env.episode).exists():
            from .advisor40_audit import audit_policy, command_files
            command_files(env.episode); audit_policy(env.episode)
        if env.requires_world_isolation:
            raise RuntimeError('Measured state outside registered boundary; stop later slots')
        return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--campaign', required=True); p.add_argument('--launch', required=True)
    a = p.parse_args()
    from . import runtime_v2, advisor34_audit, advisor40_audit
    runtime_v2.write = StatusWriter(write)
    # Reuse unchanged raw-image checks while replacing its old controller audit.
    advisor34_audit.audit_policy = advisor40_audit.audit_policy
    factory.quality = advisor34_audit.record_quality
    factory.physical = advisor40_audit.physical
    Group(a.campaign, a.launch).run()
