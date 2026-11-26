"""Offline faults must stop a claimed physical comparison, without robot access."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from modules.curobo.indexed_planning.execution_service import configure
from .contract import CASES, schedule, changed_cases, check_executed_stages


class PhysicalContractTests(unittest.TestCase):
    def test_exactly_twenty_unique_alternating_slots(self):
        rows = schedule(CASES)
        self.assertEqual(len(rows), 20);self.assertEqual(len({r['id'] for r in rows}), 20)
        for i in range(10):
            pair = rows[2*i:2*i+2]
            self.assertEqual({r['case_id'] for r in pair}, {CASES[i]})
            self.assertEqual([r['variant'] for r in pair],
                ['baseline', 'optimized'] if i%2 == 0 else ['optimized', 'baseline'])

    def test_missing_or_duplicate_scene_is_rejected(self):
        for cases in [CASES[:-1], CASES+[CASES[0]], [CASES[0]]*10]:
            with self.assertRaises(ValueError): schedule(cases)

    def test_variant_toggles_only_explicit_optimizations(self):
        pool = SimpleNamespace(endpoint_sequence=None, stage_timing=SimpleNamespace(active=False))
        pool.mesh_vote_mode = lambda enabled: setattr(pool, 'mesh_enabled', enabled)
        configure(pool, 'baseline')
        self.assertEqual((pool.ik_batch_size, pool.fast_binding_enabled, pool.mesh_enabled), (0, False, False))
        configure(pool, 'optimized')
        self.assertEqual((pool.ik_batch_size, pool.fast_binding_enabled, pool.mesh_enabled), (16, True, True))
        self.assertFalse(pool.vectorized_fallback)

    def test_variant_switch_in_active_search_is_rejected(self):
        pool = SimpleNamespace(endpoint_sequence=object(), stage_timing=SimpleNamespace(active=False))
        with self.assertRaises(RuntimeError): configure(pool, 'optimized')
        pool.endpoint_sequence = None;pool.stage_timing.active = True
        with self.assertRaises(RuntimeError): configure(pool, 'optimized')

    def test_unknown_variant_is_rejected(self):
        with self.assertRaises(ValueError): configure(None, 'other')

    def test_executed_lift_without_command_or_state_is_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);(root/'execution.jsonl').write_text(json.dumps({'stage': 'lift'})+'\n')
            with self.assertRaises(ValueError): check_executed_stages(root)

    def test_execution_receipt_without_command_is_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ['closed_state.json', 'execution_planner_lift.json']:(root/name).write_text('{}')
            with self.assertRaises(ValueError): check_executed_stages(root)

    def test_failed_close_does_not_require_unrequested_lift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);(root/'execution.jsonl').write_text(json.dumps({'stage': 'close'})+'\n')
            self.assertEqual(check_executed_stages(root), {'close'})

    def test_complete_lift_evidence_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);(root/'execution.jsonl').write_text(json.dumps({'stage': 'hold'})+'\n')
            for name in ['closed_state.json', 'execution_planner_lift.json', 'lift_plan_result.json']:
                (root/name).write_text('{}')
            self.assertEqual(check_executed_stages(root), {'hold'})


if __name__ == '__main__': unittest.main()
