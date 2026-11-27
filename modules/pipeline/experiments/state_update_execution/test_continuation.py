"""Prove no movement before transferring an interrupted execution handoff."""
import json
from pathlib import Path
import tempfile
import unittest

from arx_project.layout import write_json
from .continuation import no_motion, interrupted_before_approach
from .test_opening import selection


class Continuation(unittest.TestCase):
    def fixture(self, root):
        for name in ['execution.jsonl', 'physics_trace.jsonl']: (root / name).write_text('')
        write_json(root / 'result.json', {'reason': 'V3 opening must match the registered planned finger locks',
            'physical_attempt': True, 'stages': {}})
        write_json(root / 'plan_result.json', {'success': True, 'selected': {**selection(.027480319176659697), 'arm': 'right'}})
        write_json(root / 'arms/right/plan_result.json', {'success': True})
        for name in ['initial_metrics.json', 'execution_planner_initial.json']: write_json(root / name, {})
        (root / 'request_ledger.jsonl').write_text('\n'.join(json.dumps({'stage': 'initial', 'event': e}) for e in ['started','complete']))

    def test_raw_attempt_flag_does_not_replace_actual_motion_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fixture(root); self.assertTrue(no_motion(root))

    def test_any_trace_or_executed_stage_prevents_transfer(self):
        for kind in ['trace', 'physics', 'stage', 'state']:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); self.fixture(root)
                if kind == 'trace': (root / 'execution.jsonl').write_text('{}')
                if kind == 'physics': (root / 'physics_trace.jsonl').write_text('{}')
                if kind == 'stage': write_json(root / 'result.json', {'stages': {'open': {}}})
                if kind == 'state': write_json(root / 'opened_state.json', {})
                with self.assertRaises(ValueError): no_motion(root)

    def test_missing_prior_command_prevents_transfer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fixture(root); (root / 'plan_result.json').unlink()
            with self.assertRaises(ValueError): no_motion(root)

    def test_missing_arm_subplan_is_rejected_before_any_motion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fixture(root); (root / 'arms/right/plan_result.json').unlink()
            with self.assertRaises(ValueError): no_motion(root)

    def interruption(self, root):
        write_json(root / 'result.json', {'reason': "Measured-state planning RPC failed: FileNotFoundError /arms/right/plan_result.json"})
        (root / 'execution.jsonl').write_text('\n'.join(json.dumps({'stage': s}) for s in ['open','pregrasp']))
        for name in ['pregrasp_state.json','infrastructure_error.json']: write_json(root / name, {})
        write_json(root / 'independent_evaluation.json', {'passed': True})

    def test_interruption_is_not_reclassified_as_grasp_failure_or_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.interruption(root)
            d = interrupted_before_approach(root)
            self.assertFalse(d['approach_command_generated']); self.assertEqual(d['physical_grasps_repeated'], 0)

    def test_unknown_fault_or_later_motion_is_not_accepted_as_known_interruption(self):
        for mutation in ['reason', 'stage', 'command']:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); self.interruption(root)
                if mutation == 'reason': write_json(root / 'result.json', {'reason': 'unknown fault'})
                if mutation == 'stage':
                    with (root / 'execution.jsonl').open('a') as stream: stream.write('\n'+json.dumps({'stage':'close'}))
                if mutation == 'command': write_json(root / 'approach_plan_result.json', {})
                with self.assertRaises(ValueError): interrupted_before_approach(root)


if __name__ == '__main__': unittest.main()
