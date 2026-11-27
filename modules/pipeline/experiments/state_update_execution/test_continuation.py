"""Prove no movement before transferring an interrupted execution handoff."""
import json
from pathlib import Path
import tempfile
import unittest

from arx_project.layout import write_json
from .continuation import no_motion
from .test_opening import selection


class Continuation(unittest.TestCase):
    def fixture(self, root):
        for name in ['execution.jsonl', 'physics_trace.jsonl']: (root / name).write_text('')
        write_json(root / 'result.json', {'reason': 'V3 opening must match the registered planned finger locks',
            'physical_attempt': True, 'stages': {}})
        write_json(root / 'plan_result.json', {'success': True, 'selected': selection(.027480319176659697)})
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


if __name__ == '__main__': unittest.main()
