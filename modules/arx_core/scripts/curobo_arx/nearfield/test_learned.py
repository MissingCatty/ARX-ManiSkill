"""Offline learned-loop faults and feature parity using real ARX recordings."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from scripts.curobo_arx.data_factory.common import ROOT, read, write, sha
from training.nearfield_arx.learned_policy import live_sample, dispatch_drift, read_bound_reply, audit_policy, inference_example
from training.nearfield_arx.visual_fit import model_example
from training.nearfield_arx.processor_check import messages
from training.nearfield_arx.contract import decode, DSession
from .runtime_learned import LearnedEnvironment


class LearnedARXTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = ROOT / 'runs/nearfield_dataset_20261108_01'
        manifest = read(data / 'manifest.json');new = set(manifest['new_physical_episodes'])
        samples = [json.loads(l) for l in (data / 'paired_samples.jsonl').read_text().splitlines()]
        cls.sample = next(s for s in samples if s['episode_id'] in new and s['group'] == 'entry_stopped')
        cls.episode = Path(cls.sample['inputs']['rgb']).parent.parent
        cls.raw = [json.loads(l) for l in (cls.episode / 'samples.jsonl').read_text().splitlines()][cls.sample['raw_index']]
        cls.trial = read(cls.episode.parent.parent / 'trial_result.json')
        cls.previous = next(r for r in read(cls.episode / 'requests.json') if r['label'] == 'state_generation')
        cls.visual = read(data / 'input_contract.json')
        cls.model_row = next(json.loads(l) for l in (ROOT / 'runs/nearfield_visual_fit_20261108_01/model_rows/C/train.jsonl').read_text().splitlines()
            if json.loads(l)['sample_id'] == cls.sample['id'])
        cls.asset = next(e['asset_path'] for e in manifest['episodes'] if e['id'] == cls.sample['episode_id'])

    def live(self, row=None, method='C'):
        return live_sample(row or self.raw, method, self.trial['G0_world'], self.trial['offset_goal_world'],
            self.previous, 0, 0, None)

    def test_actual_stopped_ARX_online_prompt_equals_registered_training_prompt(self):
        item = model_example(self.live(), 'C', self.model_row['images'], self.visual)
        self.assertEqual(item['instruction'], self.model_row['instruction'])
        chat = messages(item, 'C', answer=False)
        self.assertEqual([r['role'] for r in chat], ['user'])
        self.assertNotIn('assistant', json.dumps(chat))
        transferred = json.loads(json.dumps(self.live()['inputs'], sort_keys=True))
        self.assertEqual(inference_example(transferred, 'C', self.model_row['images'], self.visual)['instruction'], item['instruction'])

    def test_teacher_fields_and_future_labels_cannot_enter_live_features(self):
        row = copy.deepcopy(self.raw)
        row['state'] = {'teacher_goal_error': 'SHOULD_NOT_ENTER_PROMPT'}
        row['source_Gstar_world'] = 'SHOULD_NOT_ENTER_PROMPT';row['teacher_command'] = {'target': 'SHOULD_NOT_ENTER_PROMPT'}
        self.assertEqual(self.live(), self.live(row))
        self.assertNotIn('SHOULD_NOT_ENTER_PROMPT', model_example(self.live(row), 'C', self.model_row['images'], self.visual)['instruction'])

    def test_projected_live_ARX_views_equal_frozen_training_pixels(self):
        from PIL import Image
        from training.nearfield_arx.projection import Projector, model_views
        from training.nearfield_arx.publish import ROBOT
        depth = np.load(self.sample['inputs']['depth'], allow_pickle=False)
        masks, _, _ = Projector(ROBOT).project(self.asset, self.raw, depth)
        images = model_views(Image.open(self.sample['inputs']['rgb']), depth, masks, self.visual['depth_scale'])
        for image, frozen in zip(images, self.model_row['images']):
            np.testing.assert_array_equal(np.asarray(image), np.asarray(Image.open(frozen)))

    def test_motion_during_model_call_is_rejected(self):
        original = copy.deepcopy(self.raw['snapshot'])
        self.assertTrue(dispatch_drift(original, original)['pass'])
        for kind in ('joint', 'object', 'velocity'):
            changed = copy.deepcopy(original);index = changed['joint_names'].index(changed['right_arm'][0])
            if kind == 'joint':changed['joint_positions'][index] += .0006
            elif kind == 'velocity':changed['joint_velocities'][index] = .011
            else:changed['objects']['test_object']['world'][0][3] += .0011
            self.assertFalse(dispatch_drift(original, changed)['pass'])

    def fixture(self, root):
        root = Path(root);ident = 'recorded_arx_call'
        request = {'request_id': ident, 'method': 'C', 'registration_sha256': 'bound-registration', 'files': {}, 'raw_index': self.sample['raw_index']}
        path = root / 'policy' / (ident + '_request.json');write(path, request)
        reply = {'request_id': ident, 'method': 'C', 'registration_sha256': 'bound-registration', 'request_sha256': sha(path),
            'ok': True, 'files': {}, 'prediction': self.sample['labels']['C'], 'policy_seconds': 1.}
        reply_path = root / 'policy' / (ident + '_reply.json');write(reply_path, reply)
        write(root / 'policy_requests.json', [{'request_id': ident}])
        write(root / 'interpreter_events.json', [{'kind': 'target_request', 'policy_request_id': ident,
            'token': self.sample['labels']['C'], 'command_label': 'c_step_001'}])
        write(root / 'requests.json', [{'label': 'c_step_001'}])
        return path, reply_path

    def test_policy_chain_missing_request_reply_or_ledger_stops(self):
        for missing in ('request', 'reply', 'ledger'):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder);request, reply = self.fixture(root)
                self.assertTrue(audit_policy(root)['pass'])
                {'request': request, 'reply': reply, 'ledger': root / 'policy_requests.json'}[missing].unlink()
                with self.assertRaises(RuntimeError):audit_policy(root)

    def test_changed_reply_identity_or_request_hash_stops(self):
        for field in ('request_id', 'registration_sha256', 'method', 'request_sha256'):
            with tempfile.TemporaryDirectory() as folder:
                request, reply = self.fixture(folder);value = read(reply);value[field] = 'wrong';write(reply, value)
                with self.assertRaises(RuntimeError):read_bound_reply(request, reply)

    def test_executed_action_without_model_origin_stops(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);self.fixture(root);write(root / 'interpreter_events.json', [])
            with self.assertRaises(RuntimeError):audit_policy(root)

    def test_controller_cannot_substitute_teacher_action(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);self.fixture(root);events = read(root / 'interpreter_events.json')
            events[0]['token'] = 'GRASP';write(root / 'interpreter_events.json', events)
            with self.assertRaises(RuntimeError):audit_policy(root)

    def environment(self, root, method, token):
        e = LearnedEnvironment.__new__(LearnedEnvironment);e.episode = Path(root)
        request_path, reply_path = self.fixture(root)
        row = self.raw;request = {'request_id': 'recorded_arx_call', 'method': method, 'raw_index': self.sample['raw_index'],
            'inputs': self.live(method=method)['inputs']}
        e.pending_policy = {'path': request_path, 'request': request, 'snapshot': row['snapshot'], 'started': 0.}
        e.group = SimpleNamespace(campaign=Path(root), registration={'learned_policy': {'C_max_motion_requests': 12}})
        e.session = DSession(self.trial['G0_world']);e.local_index = 0;e.calls = 0;e.traces = [];e.events = []
        e.group.active_trial = {'source_Gstar_world': 'NO_TEACHER_TARGET_AVAILABLE'}
        def trace(method, token):
            item = {'decision_id': 'actual-decision', 'history_after_action': {}}
            e.traces.append(item);e.calls += 1;return item
        e.trace_before = trace;e.log_event = e.events.append
        e.finish = lambda reason: ('finished', reason)
        e.submit = lambda label, goal, phase: ('motion', label, goal, phase)
        e.request_closure = lambda ident: ('close_gate', ident)
        reply = {'prediction': token, 'projection': {'target_visibility_pass': True}}
        return e, request, reply

    def test_C_decodes_model_output_without_access_to_teacher(self):
        with tempfile.TemporaryDirectory() as folder:
            token = self.sample['labels']['C'];e, request, reply = self.environment(folder, 'C', token)
            with patch('scripts.curobo_arx.nearfield.runtime_learned.read_bound_reply', return_value=(request, reply)):
                result = e.consume_prediction(self.raw['snapshot'])
            self.assertEqual(result[0], 'motion')
            np.testing.assert_allclose(result[2], decode('C', request['inputs']['eef_world'], token))

    def test_D_third_REFINE_is_failure_and_never_forces_GRASP(self):
        with tempfile.TemporaryDirectory() as folder:
            token = self.sample['labels']['D'];e, request, reply = self.environment(folder, 'D', token)
            ee = request['inputs']['eef_world'];e.session.request(ee, token);e.session.request(ee, token)
            with patch('scripts.curobo_arx.nearfield.runtime_learned.read_bound_reply', return_value=(request, reply)):
                result = e.consume_prediction(self.raw['snapshot'])
            self.assertEqual(result, ('finished', 'D_correction_budget_exhausted'))
            self.assertFalse(any(x['kind'] == 'grasp_request' for x in e.events))

    def test_GRASP_always_enters_original_independent_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            e, request, reply = self.environment(folder, 'D', 'GRASP')
            with patch('scripts.curobo_arx.nearfield.runtime_learned.read_bound_reply', return_value=(request, reply)):
                result = e.consume_prediction(self.raw['snapshot'])
            self.assertEqual(result[0], 'close_gate');self.assertEqual(e.events[-1]['kind'], 'grasp_request')

    def test_invalid_model_text_fails_without_repair_or_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            e, request, reply = self.environment(folder, 'C', 'I would close now')
            with patch('scripts.curobo_arx.nearfield.runtime_learned.read_bound_reply', return_value=(request, reply)):
                result = e.consume_prediction(self.raw['snapshot'])
            self.assertEqual(result, ('finished', 'policy_invalid_action'))

    def test_advisor22_removes_performance_admission_for_exact_four_slots(self):
        from .learned_campaign import require_execution_authority
        actual = read(ROOT / 'runs/nearfield_visual_fit_20261108_01/final_verification.json')
        authority = require_execution_authority(actual, ROOT / 'research_plan/chat/gemini回复/gemini回复22.md')
        self.assertFalse(authority['performance_admission_required'])
        self.assertFalse(authority['old_admission_result_preserved'])
        with self.assertRaises(RuntimeError):require_execution_authority(actual)
        with self.assertRaises(RuntimeError):require_execution_authority(
            {**actual, 'record_integrity_pass': False}, ROOT / 'research_plan/chat/gemini回复/gemini回复22.md')

    def test_frozen_coordinator_mounts_original_registered_absolute_paths(self):
        from .learned_campaign import registered_launcher_project, launcher
        frozen = ROOT / 'runs/nearfield_learned_20261108_04/source_snapshot/COOLAS-ARX'
        reg = {'project_root': str(ROOT), 'source_snapshot': str(frozen)}
        with patch('scripts.curobo_arx.nearfield.campaign.ROOT', frozen):
            with registered_launcher_project(reg):
                self.assertEqual(launcher.ROOT, ROOT)
                self.assertTrue((launcher.ROOT/'runs/nearfield_learned_20261108_04/registration.json').is_file())
                self.assertEqual(reg['source_snapshot'], str(frozen))
            self.assertEqual(launcher.ROOT, frozen)


if __name__ == '__main__':unittest.main()
