"""Offline faults for stage evidence and one-shot complete scheduling."""
import json
from pathlib import Path
import tempfile
import unittest

from .contract import check_requests, schedule, stage_name
from .review import validate_path
import numpy as np


class Contracts(unittest.TestCase):
    def test_full_schedule_once(self):
        cases = [{'id': str(i)} for i in range(74)]; rows = schedule(cases)
        self.assertEqual(len(rows), 148); self.assertEqual(len({r['id'] for r in rows}), 148)
        self.assertEqual([r['variant'] for r in rows[:4]], ['baseline', 'optimized', 'optimized', 'baseline'])

    def test_incomplete_and_duplicate_cohorts_rejected(self):
        for ids in [list(range(73)), [0] * 74]:
            with self.assertRaises(ValueError): schedule([{'id': str(i)} for i in ids])

    def test_stages(self):
        for flags, expected in [({}, 'initial'), ({'approach': True}, 'approach'), ({'lift': True}, 'lift')]:
            self.assertEqual(stage_name(flags), expected)
        for flags in [{'observe': True}, {'approach': True, 'lift': True}]:
            with self.assertRaises(ValueError): stage_name(flags)

    def build(self, root, stage):
        names = [stage + '_metrics.json', 'execution_planner_' + stage + '.json']
        names += ['plan_result.json'] if stage == 'initial' else [stage + '_plan_result.json',
            'pregrasp_state.json' if stage == 'approach' else 'closed_state.json']
        for name in names: (root / name).write_text('{}')
        (root / 'request_ledger.jsonl').write_text('\n'.join(json.dumps({'stage': stage, 'event': e}) for e in ['started', 'complete']))
        return names

    def test_each_required_record_is_independently_required(self):
        for stage in ['initial', 'approach', 'lift']:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); names = self.build(root, stage)
                self.assertEqual(check_requests(root), [stage])
                for name in names:
                    (root / name).unlink()
                    with self.subTest(stage=stage, missing=name), self.assertRaises(ValueError): check_requests(root)
                    (root / name).write_text('{}')

    def test_lost_command_with_surviving_receipt_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'execution_planner_lift.json').write_text('{}')
            with self.assertRaises(ValueError): check_requests(root)

    def test_trace_without_receipts_stops(self):
        for label in ['approach', 'lift', 'hold']:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); (root / 'execution.jsonl').write_text(json.dumps({'stage': label}))
                with self.assertRaises(ValueError): check_requests(root)

    def test_repeated_or_incomplete_ledger_stops(self):
        for events in [['started'], ['started', 'complete', 'started', 'complete']]:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); self.build(root, 'initial')
                (root / 'request_ledger.jsonl').write_text('\n'.join(json.dumps({'stage': 'initial', 'event': e}) for e in events))
                with self.assertRaises(ValueError): check_requests(root)

    def test_corrupt_motion_commands(self):
        limits = np.array([[-1., 1.]] * 6)
        validate_path({'q': [[0.] * 6, [.001] * 6], 'times': [0., 1.]}, limits)
        for path in [{'q': [[0.] * 6, [2.] * 6], 'times': [0., 1.]},
                     {'q': [[0.] * 6, [.1] * 6], 'times': [0., 0.]},
                     {'q': [[0.] * 6, [float('nan')] * 6], 'times': [0., 1.]},
                     {'q': [[0.] * 6, [.1] * 6], 'times': [0., .00001]}]:
            with self.assertRaises(ValueError): validate_path(path, limits)


if __name__ == '__main__': unittest.main()
