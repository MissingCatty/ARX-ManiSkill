"""Offline fault checks based on archived ARX episodes; no simulated actions."""
import copy
from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np
from scripts.curobo_arx.data_factory.common import ROOT, read, write, sha, digest
from . import advisor31_rules as rules
from .advisor31_io import StatusWriter
from .advisor31_audit import verify_post_close, task_result
from .advisor31_spec import make_trials, validate_trials
from .closedloop30_audit import evaluate as old_evaluate
from training.nearfield_arx.entry_guard import residual_goal

DONOR = ROOT/'runs/nearfield_closedloop30_20261110_01'


class ARXRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg = read(DONOR/'registration.json')
        cls.bbq = next(DONOR.glob('groups/*/trials/r30_04_02_entry_offset_D/episodes/*'))
        cls.spam = next(DONOR.glob('groups/*/trials/r30_21_03_normal_C/episodes/*'))
        cls.execution = read(cls.bbq/'close_execution.json')
        cls.command = read(cls.bbq/'close_command.json')
        cls.req = [read(e['request']) for e in read(cls.spam/'policy_requests.json')]

    def idle(self):
        # Explicit offline counterfactual only. The archived endpoint was not
        # stationary; actual execution must obtain a new measured idle state.
        snap = copy.deepcopy(self.execution['final'])
        for n in snap['right_arm']: snap['joint_velocities'][snap['joint_names'].index(n)] = 0.
        return snap

    def inputs(self, distance=.024):
        t = next(t for t in self.reg['seed_trials'] if t['trial_id'] == 'r30_04_02_entry_offset_D')
        g0 = np.array(t['G0_world']); ee = g0.copy(); ee[:3, 3] -= distance*g0[:3, 2]
        return {'eef_world': ee.tolist(), 'G0_world': g0.tolist()}

    def test_historical_bbq_contacts_do_not_claim_success_or_idle(self):
        result = rules.post_close(self.execution, self.execution['final'], self.command, 0., active=False, fault=None)
        self.assertEqual(len(result['recent_fingers']), 2)
        self.assertTrue(result['nonempty_finger_opening'])
        self.assertFalse(result['stationary']); self.assertFalse(result['pass'])
        self.assertFalse(result['physical_success_claimed'])
        self.assertFalse(read(self.bbq/'close_settling.json')['accepted'])

    def test_idle_counterfactual_admits_contact_despite_4point81mm_error(self):
        proof = rules.post_close(self.execution, self.idle(), self.command, 0., active=False, fault=None)
        self.assertTrue(proof['pass']); self.assertFalse(proof['close_tracking_error_used_for_admission'])

    def test_nonempty_alternative_does_not_require_old_contact(self):
        ex = copy.deepcopy(self.execution); ex['contacts'] = []
        self.assertTrue(rules.post_close(ex, self.idle(), self.command, 0., active=False, fault=None)['pass'])

    def test_empty_without_contact_rejects(self):
        snap = self.idle(); ex = copy.deepcopy(self.execution); ex['contacts'] = []
        for n in ('right_joint27', 'right_joint28'): snap['joint_positions'][snap['joint_names'].index(n)] = 0.
        self.assertFalse(rules.post_close(ex, snap, self.command, 0., active=False, fault=None)['pass'])

    def test_forbidden_contact_not_waived(self):
        ex = copy.deepcopy(self.execution)
        ex['contacts'].append({'a': '/World/Robot/right_link26', 'b': '/World/Table',
                               'num_contacts': 1, 'sim_time': ex['final']['sim_time']})
        self.assertFalse(rules.post_close(ex, self.idle(), self.command, 0., active=False, fault=None)['pass'])

    def test_fault_or_active_controller_raises(self):
        for active, fault in ((True, None), (False, 'lost controller')):
            with self.assertRaises(RuntimeError):
                rules.post_close(self.execution, self.idle(), self.command, 0., active=active, fault=fault)

    def test_incomplete_close_raises(self):
        ex = copy.deepcopy(self.execution); ex['records'] = ex['records'][:1]
        with self.assertRaises(ValueError): rules.post_close(ex, self.idle(), self.command, 0., active=False, fault=None)

    def test_frozen_limit_exceed_raises(self):
        snap = self.idle(); i = snap['joint_names'].index(snap['right_arm'][0])
        snap['joint_positions'][i] = snap['joint_limits'][i][1]+.001
        with self.assertRaises(RuntimeError): rules.post_close(self.execution, snap, self.command, 0., active=False, fault=None)

    def test_nonfinite_state_raises(self):
        snap = self.idle(); i = snap['joint_names'].index(snap['right_arm'][0]); snap['joint_velocities'][i] = float('nan')
        with self.assertRaises(ValueError): rules.post_close(self.execution, snap, self.command, 0., active=False, fault=None)

    def test_second_5mm_translation_intercepts_rotation_too(self):
        r = rules.interpret('D', 'REFINE 0 0 5 0 0 5', self.inputs(), 1)
        self.assertTrue(r['request_closure']); self.assertEqual(r['origin'], 'D_second_step_nearfield_guard')
        self.assertTrue(r['advisor31']['reasons']['small_translation'])

    def test_6mm_inward_translation_not_intercepted(self):
        r = rules.interpret('D', 'REFINE 0 0 6 0 0 5', self.inputs(), 1)
        self.assertFalse(r['request_closure'])

    def test_outward_more_than3mm_intercepts(self):
        r = rules.interpret('D', 'REFINE 0 0 -6 0 0 0', self.inputs(), 1)
        self.assertTrue(r['request_closure']); self.assertTrue(r['advisor31']['reasons']['outward'])

    def test_first_motion_and_far_state_not_forced_to_close(self):
        for inputs, count in ((self.inputs(), 0), (self.inputs(.026), 1)):
            self.assertFalse(rules.interpret('D', 'REFINE 0 0 -6 0 0 0', inputs, count)['request_closure'])

    def test_predicted_depth_has_to_match_original_raw_target(self):
        inputs = self.inputs(); token = 'REFINE 0 0 6 0 0 5'
        proof = {'pass': False, 'goal_sha256': digest(residual_goal(inputs['eef_world'], token).tolist())}
        r = rules.interpret('D', token, inputs, 1, proof)
        self.assertTrue(r['request_closure']); self.assertTrue(r['advisor31']['reasons']['near_depth_saturation'])
        proof['goal_sha256'] = 'wrong'
        with self.assertRaises(ValueError): rules.interpret('D', token, inputs, 1, proof)

    def test_spam_real_fourth_step_contact_triggers(self):
        proof = rules.contact_from_request(self.req[4])
        self.assertTrue(proof['trigger']); self.assertTrue(proof['touching_fingers'])
        self.assertGreater(proof['target_displacement_m'], .003)
        result = rules.interpret('C', 'GRASP', self.req[4]['inputs'], 4, contact_proof=proof)
        self.assertEqual(result['origin'], 'C_contact_or_target_displacement')

    def test_spam_first_three_steps_do_not_trigger(self):
        for req in self.req[:4]: self.assertFalse(rules.contact_from_request(req)['trigger'])

    def test_spam_outside_grasp_region_does_not_close(self):
        req = copy.deepcopy(self.req[4]); req['inputs']['G0_world'][0][3] += .1
        self.assertFalse(rules.contact_from_request(req)['trigger'])

    def test_missing_close_command_stops_audit(self):
        with tempfile.TemporaryDirectory() as name:
            ep = Path(name)/'groups/test/episodes/ep'; ep.mkdir(parents=True)
            for suffix in ('execution', 'settling'):
                shutil.copy2(self.bbq/('close_'+suffix+'.json'), ep)
            with self.assertRaises(RuntimeError): verify_post_close(ep, [])

    def test_bound_close_receipt_cannot_hide_missing_command_or_false_state(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); ep = root/'groups/test/episodes/ep'; ep.mkdir(parents=True)
            write(root/'registration.json', {'render_configs': self.reg['render_configs']})
            for filename in ('close_command.json', 'close_execution.json', 'close_settling.json',
                             'closure_model_validation.json', 'requests.json'):
                shutil.copy2(self.bbq/filename, ep)
            proof = rules.post_close(self.execution, self.idle(), self.command, 0., active=False, fault=None)
            proof['files'] = {str(ep/('close_'+s+'.json')): sha(ep/('close_'+s+'.json')) for s in ('command','execution','settling')}
            path = ep/'close_contact_admission.json'; write(path, proof)
            event = {'kind': 'post_close_contact_admission', 'receipt': str(path), 'receipt_sha256': sha(path),
                     'snapshot': proof['snapshot'], 'pass': True, 'original_arrival_accepted': False}
            self.assertTrue(verify_post_close(ep, [event])['pass'])
            (ep/'close_command.json').unlink()
            with self.assertRaises((RuntimeError, FileNotFoundError)): verify_post_close(ep, [event])

    def test_old_bbq_without_lift_cannot_become_success(self):
        strict = old_evaluate(self.bbq)
        strict['pass'] = strict['strict_original_success']; strict['issues'] = strict['strict_original_issues']
        result = task_result(strict, read(self.bbq/'execution_roles.json')['sequence'], [], True, {'pass': True})
        self.assertFalse(result['pass']); self.assertFalse(result['strict_original_success'])

    def test_exact26_schedule_and_original24_targets(self):
        from .visual_grounding_spec import make_trials as previous
        old, _ = previous(self.reg); trials, seeds = make_trials(self.reg)
        self.assertEqual(len(trials), 26); self.assertEqual(len(seeds), 6)
        for a, b in zip(old, trials[2:]):
            for key in ('G0_world', 'entry_world', 'placement', 'renderer', 'vision_arm', 'repeat'):
                self.assertEqual(a[key], b[key])
        with self.assertRaises(ValueError): validate_trials(trials+[trials[0]])
        changed = copy.deepcopy(trials); changed[2]['G0_world'][0][3] += .001
        with self.assertRaises(ValueError): validate_trials(changed)


class StatusIO(unittest.TestCase):
    def test_only_identical_success_status_is_deduplicated(self):
        calls = []; save = StatusWriter(lambda p, value: calls.append((str(p), copy.deepcopy(value))))
        a = {'pass': True, 'failures': []}; b = {'pass': False, 'failures': ['camera']}
        for value in (a, a, b, b, a, a): save(Path('/tmp/arx-offline/capture_preflight.json'), value)
        self.assertEqual(len(calls), 4); self.assertEqual(save.metrics['safety_checks_skipped'], 0)

    def test_commands_changes_and_different_paths_are_never_skipped(self):
        calls = []; save = StatusWriter(lambda p, value: calls.append(str(p)))
        for _ in range(2): save(Path('/tmp/arx-offline/close_command.json'), {'pass': True})
        for name in ('a', 'b'): save(Path('/tmp/arx-offline')/name/'capture_preflight.json', {'pass': True})
        save(Path('/tmp/arx-offline/a/capture_preflight.json'), {'pass': True, 'changed': True})
        self.assertEqual(len(calls), 5)


if __name__ == '__main__': unittest.main()
