"""Offline ARX counterexamples; no GPU, service or simulated motion."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np
from scripts.curobo_arx.data_factory.common import ROOT, read, write, digest
from scripts.curobo_arx.settling import assess_settling
from . import closedloop30_rules as rules
from .closedloop30_audit import task_result, evaluate, same_trigger
from training.nearfield_arx import closure_recovery

PREVIOUS = ROOT/'runs/nearfield_main48_20261110_07'


class RulesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg = read(PREVIOUS/'registration.json')
        cls.final = read(PREVIOUS/'main48_verification.json')
        cls.spam = next(t for t in cls.reg['registered_trials'] if t['case_id'] == '03_normal' and t['method'] == 'D')
        ident = cls.spam['trial_id']
        cls.episode = next(Path(p).parent for p in cls.final['evidence']
                           if p.endswith('/physical_receipt.json') and '/'+ident+'/' in p)
        cls.execution = read(cls.episode/'d_refine_002_execution.json')
        cls.snapshot = cls.execution['final']

    def test_axial_translation_preserves_lateral_and_orientation(self):
        current = np.asarray(self.spam['entry_world']); g0 = np.asarray(self.spam['G0_world'])
        proof = rules.axial_goal(current, g0); goal = np.asarray(proof['goal_world'])
        np.testing.assert_allclose(goal[:3, :3], current[:3, :3], atol=1e-12)
        delta = goal[:3, 3]-current[:3, 3]
        np.testing.assert_allclose(delta, g0[:3, 2]*.055, atol=1e-12)
        self.assertFalse(proof['Gstar_used'])

    def test_entry_range_preserves_perturbation_and_limits(self):
        trial = next(t for t in self.reg['registered_trials'] if t['case_id'] == '03_entry_offset')
        rows = rules.entry_candidates(trial)
        self.assertEqual([r['backoff_m'] for r in rows], [.08, .07, .06, .05, .04])
        for row in rows:
            np.testing.assert_allclose(np.array(row['goal_world'])[:3, :3], np.array(trial['entry_world'])[:3, :3])
        self.assertEqual(len(rules.entry_candidates(self.spam)), 1)

    def inputs(self, distance):
        g0 = np.array(self.spam['G0_world']); ee = g0.copy(); ee[:3, 3] -= distance*g0[:3, 2]
        return {'eef_world': ee.tolist(), 'G0_world': g0.tolist()}

    def test_second_outward_motion_intercepted_without_modifying_model_text(self):
        result = rules.interpret('D', 'REFINE 0 0 -20 0 0 0', self.inputs(.01778), 1)
        self.assertTrue(result['request_closure'])
        self.assertEqual(result['origin'], 'D_second_step_outward_intercept')

    def test_outside_core_or_first_motion_is_not_forced_to_close(self):
        for distance, count in [(.025, 1), (.017, 0)]:
            self.assertFalse(rules.interpret('D', 'REFINE 0 0 -20 0 0 0', self.inputs(distance), count)['request_closure'])

    def test_small_residual_and_budget_trigger_preserved(self):
        for count, text in [(0, 'REFINE 0 0 2 0 0 0'), (2, 'REFINE 0 0 10 0 0 0')]:
            actual = rules.interpret('D', text, self.inputs(.02), count)
            expected = closure_recovery.closure_trigger('D', text, self.inputs(.02), count)
            for key in expected: self.assertEqual(actual[key], expected[key])

    def test_real_spam_endpoint_still_moving_requires_fresh_idle_observation(self):
        self.assertFalse(read(self.episode/'d_refine_002_settling.json')['accepted'])
        proof = rules.stationary(self.execution, self.snapshot, active=False, fault=None, label='d_refine_002')
        self.assertGreater(proof['maximum_arm_speed_rad_s'], .01)
        self.assertFalse(proof['pass'])

    def test_tracking_error_alone_does_not_reject_stationary_state(self):
        # Offline velocity counterfactual on an archived ARX state, not a claim
        # that the historical trajectory actually reached this idle state.
        snap = copy.deepcopy(self.snapshot); snap['joint_velocities'] = [0.]*len(snap['joint_names'])
        self.assertTrue(rules.stationary(self.execution, snap, active=False, fault=None, label='d_refine_002')['pass'])

    def test_unknown_idle_or_controller_fault_stops(self):
        for active, fault in [(True, None), (False, 'lost connection')]:
            with self.assertRaises(RuntimeError):
                rules.stationary(self.execution, self.snapshot, active=active, fault=fault, label='d_refine_002')

    def test_nonfinite_velocity_rejected(self):
        snap = copy.deepcopy(self.snapshot); snap['joint_velocities'][snap['joint_names'].index(snap['right_arm'][0])] = float('nan')
        with self.assertRaises(ValueError): rules.stationary(self.execution, snap, active=False, fault=None, label='d_refine_002')

    def test_moving_arm_not_released(self):
        snap = copy.deepcopy(self.snapshot); snap['joint_velocities'][snap['joint_names'].index(snap['right_arm'][0])] = .02
        self.assertFalse(rules.stationary(self.execution, snap, active=False, fault=None, label='d_refine_002')['pass'])

    def test_forbidden_contact_not_exempted(self):
        execution = copy.deepcopy(self.execution)
        execution['contacts'].append({'a': '/World/Robot/right_link26', 'b': '/World/Table', 'num_contacts': 1})
        self.assertFalse(rules.stationary(execution, self.snapshot, active=False, fault=None, label='d_refine_002')['pass'])

    def test_close_cannot_be_a_waypoint(self):
        with self.assertRaises(ValueError): rules.stationary(self.execution, self.snapshot, active=False, fault=None, label='close')

    def test_missing_command_with_execution_present_stops(self):
        with tempfile.TemporaryDirectory() as name:
            dest = Path(name)
            for suffix in ('execution', 'settling'):
                shutil.copy2(self.episode/f'd_refine_002_{suffix}.json', dest)
            shutil.copy2(self.episode/'requests.json', dest)
            with self.assertRaises(FileNotFoundError):
                rules.bind_waypoint(dest, 'd_refine_002', self.execution, self.snapshot,
                    np.eye(4), active=False, fault=None)

    def test_closure_holds_measured_joints_and_preserves_old_failure(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); label = 'd_refine_002'
            for suffix in ('command', 'execution', 'settling'):
                shutil.copy2(self.episode/f'{label}_{suffix}.json', root)
            shutil.copy2(self.episode/'requests.json', root)
            command = read(root/(label+'_command.json'))
            target = next(e['goal_world'] for e in read(self.episode/'interpreter_events.json')
                          if e['kind'] == 'target_request' and e['command_label'] == label)
            snap = copy.deepcopy(self.snapshot); snap['joint_velocities'] = [0.]*len(snap['joint_names'])
            observation = {'frame_time': snap['sim_time'], 'snapshot': snap,
                           'visual_geometry': {'pass': True}, 'attempt_id': 'offline_ARX_fault_fixture'}
            binding = rules.bind_closure(root, snap, observation, command, target,
                active=False, fault=None, request_id='offline_check', configuration_identity='offline')
            self.assertTrue(binding['pass'])
            self.assertNotEqual(binding['hold_q'], command['position'][-1])
            self.assertFalse(binding['arrival']['original_arrival_accepted'])
            self.assertFalse(read(root/(label+'_settling.json'))['accepted'])
            certificate = {'pass': True, 'binding_sha256': digest(binding), 'hold_q': binding['hold_q']}
            self.assertTrue(rules.validate_closure(binding, certificate, command))
            certificate['hold_q'] = command['position'][-1]
            with self.assertRaises(RuntimeError): rules.validate_closure(binding, certificate, command)

    def test_finite_budget_covers_all_original_CD_and_unknowns(self):
        from .closedloop30_campaign import make_trials
        trials = make_trials(self.reg['registered_trials'])
        self.assertEqual(len(trials), 24)
        self.assertEqual(len({t['historical_trial_id'] for t in trials}), 24)
        unknown = {t['trial_id'] for t in self.final['trials'] if t.get('task_result') == 'infrastructure_interrupted'}
        self.assertTrue(unknown <= {t['historical_trial_id'] for t in trials[:10]})
        self.assertTrue(all(t['no_retry'] for t in trials))

    def test_failed_real_grasp_does_not_become_success_from_tracking_waiver(self):
        strict = read(self.episode/'physical_receipt.json')
        sequence = read(self.episode/'execution_roles.json')['sequence']
        self.assertFalse(task_result(strict, sequence, ['d_refine_002'], False)['pass'])

    def test_metric_only_exempts_certified_open_tracking_error(self):
        success = copy.deepcopy(next(t['physical'] for t in self.final['trials']
            if t['case_id'] == '03_normal' and t['method'] == 'C'))
        label = 'c_step_001'; success['pass'] = False
        success['converged_stages'].remove(label)
        success['issues'] = [{'stage': label, 'error': 'Original ARX convergence gate rejected execution'}]
        sequence = success['executed_stages']
        self.assertFalse(task_result(success, sequence, [], True)['pass'])
        self.assertTrue(task_result(success, sequence, [label], True)['pass'])
        for key, value in [('bilateral_closure', False), ('all_recorded_hold_samples_above_6cm', False),
                           ('forbidden_contact_events', 1), ('missing_stages', ['hold'])]:
            damaged = copy.deepcopy(success); damaged[key] = value
            self.assertFalse(task_result(damaged, sequence, [label], True)['pass'])
        self.assertFalse(task_result(success, sequence, [label], False)['pass'])

    def test_rule_comparison_does_not_relax_gate_decisions(self):
        result = rules.interpret('D', 'REFINE 0 0 -20 0 0 0', self.inputs(.01778), 1)
        altered = copy.deepcopy(result); altered['request_closure'] = False
        self.assertFalse(same_trigger(result, altered))


if __name__ == '__main__': unittest.main()
