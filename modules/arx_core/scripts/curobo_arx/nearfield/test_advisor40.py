"""Offline actual ARX registration and fault-injection checks; zero simulator calls."""
import copy
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts.curobo_arx.data_factory.common import ROOT, read, write
from .advisor40_online import make_cases, validate_cases, choose_roles, summarize
from .advisor40_policy import decision
from .advisor40_audit import command_files
from .runtime_advisor40 import Environment
from .advisor36_contract import interpret


class ConditionalPhysicalTests(unittest.TestCase):
    def test_actual_three_objects_nine_slots_fixed_H(self):
        root = ROOT / 'runs/nearfield_advisor40_20261112_01'
        old = read(ROOT / 'runs/nearfield_advisor34_20261111_01/registration.json')
        trials, seeds = make_cases(read(root / 'predecessor_snapshot/advisor39/source_inventory.json'), old)
        self.assertEqual(len(trials), 9); self.assertEqual(len(seeds), 9)
        for obj in ('household_hope_milk_carton', 'raisin_box', 'cheez_it'):
            rows = [r for r in trials if r['object_id'] == obj]
            normal = next(r for r in rows if r['condition'] == 'normal')
            for r in rows:
                self.assertEqual(r['entry_world'], normal['entry_world'])
                self.assertAlmostEqual(r['placement']['placement']['position'][0] - normal['placement']['placement']['position'][0],
                                       r['target_delta_world_m'][0])
                self.assertEqual(r['placement']['placement']['position'][1:], normal['placement']['placement']['position'][1:])
                self.assertEqual(r['renderer'], normal['renderer'])
            changed = copy.deepcopy(trials); changed[0]['entry_world'][0][3] += .001
            with self.assertRaises(ValueError): validate_cases(changed)

    def test_model_input_cannot_include_goal_or_budget(self):
        req = {'method': 'D', 'mode': 'decision', 'vision_arm': 'fresh',
               'inputs': {'eef_world': np.eye(4).tolist(), 'stopped': True}}
        for bad in ('G0_world', 'remaining_corrections', 'target_world', 'canonical_goal'):
            changed = copy.deepcopy(req); changed['inputs'][bad] = 0
            with self.assertRaises(ValueError): decision(changed, 'GRASP')
        result = decision(req, 'REFINE 5 -7 83 1 2 3')
        np.testing.assert_allclose(np.array(result['goal_world'])[:3, 3], [.005, -.007, .083])
        self.assertEqual(decision(req, 'REFINE 0 0 0 0 0 0')['kind'], 'move')

    def test_execution_without_command_stops_scheduling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root / 'requests.json', [{'id': 'recorded_close', 'label': 'close'}])
            write(root / 'close_execution.json', {'executed': True})
            with self.assertRaisesRegex(RuntimeError, 'scheduling must stop'):
                command_files(root)

    def test_orphan_execution_is_not_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); write(root / 'requests.json', [])
            write(root / 'd_refine_001_execution.json', {'executed': True})
            with self.assertRaises(RuntimeError): command_files(root)

    def test_resource_admission_does_not_place_sim_with_model(self):
        inventory = [{'uuid': 'A', 'free_mb': 10000}, {'uuid': 'B', 'free_mb': 9600}, {'uuid': 'C', 'free_mb': 6500}]
        roles = choose_roles(inventory)
        self.assertNotIn(roles['sim_uuid'], (roles['policy_uuid'], roles['compute_uuid']))
        self.assertIsNone(choose_roles([{'uuid': 'A', 'free_mb': 23000}]))
        self.assertIsNotNone(choose_roles([{'uuid': 'A', 'free_mb': 14000}, {'uuid': 'B', 'free_mb': 9000}]))

    def test_interrupted_slot_is_separate_from_unexecuted_and_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'groups/g/trials/started').mkdir(parents=True)
            reg = {'seed_trials': [{'trial_id': 'started'}, {'trial_id': 'not_started'}]}
            result = summarize(root, reg)
            self.assertEqual(result['interrupted_unknown'], ['started'])
            self.assertEqual(result['unexecuted'], ['not_started'])
            self.assertEqual(result['ordinary_failed_completed'], 0)

    def consume(self, token, moves, mutate=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root / 'model_request.json'
            observation = root / 'observation.json'; write(observation, {'frame_time': 4.})
            write(root / 'model_reply.json', {})
            req = {'request_id': 'model', 'raw_index': 0, 'observation': str(observation),
                   'inputs': {'eef_world': np.eye(4).tolist(), 'stopped': True}}
            action = interpret(np.eye(4), token)
            if mutate:
                action['goal_world'][2][3] += .001
            reply = {'prediction': token, 'action': action}
            env = Environment.__new__(Environment)
            env.pending_policy = {'path': path, 'request': req, 'snapshot': {}, 'started': 0}
            env.group = SimpleNamespace(campaign=root)
            env.episode = root; env.proof = lambda _: {'pass': True}
            env.calls = 2 if moves == 2 else 0; env.neural_calls = env.calls
            env.traces = []; env.local_index = moves; env.session = SimpleNamespace(refines=moves)
            events = []; dispatched = []; closed = []; finished = []
            env.log_event = events.append
            env.submit = lambda label, goal: dispatched.append((label, goal))
            env.request_closure = lambda ident: closed.append(ident)
            env.finish = finished.append
            with patch('scripts.curobo_arx.nearfield.runtime_advisor40.read_bound_reply', return_value=(req, reply)), \
                 patch('scripts.curobo_arx.nearfield.runtime_advisor40.stationary_dispatch', return_value={'pass': True}):
                env.consume_prediction({})
            return dispatched, closed, finished

    def test_budget_exhaustion_does_not_trigger_closure(self):
        dispatched, closed, finished = self.consume('REFINE 0 0 1 0 0 0', 2)
        self.assertFalse(dispatched); self.assertFalse(closed)
        self.assertEqual(finished, ['pure_model_motion_budget_exhausted_without_GRASP'])

    def test_explicit_third_GRASP_after_two_moves_is_allowed(self):
        dispatched, closed, finished = self.consume('GRASP', 2)
        self.assertFalse(dispatched); self.assertEqual(len(closed), 1); self.assertFalse(finished)

    def test_changed_axial_component_cannot_be_dispatched(self):
        with self.assertRaisesRegex(RuntimeError, 'raw REFINE coordinates were altered'):
            self.consume('REFINE 3 4 80 1 2 3', 0, mutate=True)


if __name__ == '__main__':
    unittest.main()
