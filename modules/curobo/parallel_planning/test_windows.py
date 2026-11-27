"""Offline checks that selection never serially seeks a full batch."""
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np
import torch

from .endpoint_windows import EndpointWindow, same_width_indices, padded_starts


def candidates(widths):
    result = []
    for i, width in enumerate(widths):
        pose = np.eye(4); pose[0, 3] = i + 1
        result.append({'width': width, 'pose': pose.tolist()})
    return result


class WindowTests(unittest.TestCase):
    def test_raw_window_does_not_expand_to_find_valid_candidates(self):
        rows = candidates([.088] * 40)
        self.assertEqual(same_width_indices(rows, 0), list(range(16)))
        self.assertEqual(same_width_indices(rows, 15), [15])
        self.assertEqual(same_width_indices(rows, 16), list(range(16, 32)))

    def test_widths_are_not_rounded(self):
        rows = candidates([.06, .06000000001, .06])
        self.assertEqual(same_width_indices(rows, 0), [0, 2])

    def test_each_target_keeps_its_own_grasp_seed(self):
        seeds = torch.stack([torch.zeros(6), torch.ones(6)])
        padded = padded_starts(seeds)
        self.assertTrue(torch.equal(padded[:2], seeds))
        self.assertTrue(torch.equal(padded[2:], seeds[:1].expand(14, -1)))
        padded[0, 0] = 9
        self.assertEqual(float(seeds[0, 0]), 0.)

    def test_invalid_seed_count_is_rejected(self):
        for values in [torch.zeros(0, 6), torch.zeros(17, 6), torch.zeros(2, 7)]:
            with self.assertRaises(ValueError): padded_starts(values)

    def test_pre_and_lift_keep_the_grasp_seed_of_each_row(self):
        rows = candidates([.088, .088])
        recorded = []
        class Engine:
            capacity = 16
            calls = []
            def bind(self, context): pass
            def native(self, worlds, start): return [torch.zeros(6), torch.ones(6)]
            def _solve(self, solver, worlds, starts, returns, stage):
                recorded.append((stage, starts.clone()))
                return torch.stack([starts[0], starts[1]])[:, None].expand(-1, 4, -1).clone(), torch.ones((2, 4), dtype=torch.bool)
            full_solver = None
        pool = SimpleNamespace(prepare_endpoint_engine=lambda context: Engine(),
            stage_timing=SimpleNamespace(stage=lambda name: __import__('contextlib').nullcontext()))
        context = SimpleNamespace(checker=SimpleNamespace(validate=lambda q: torch.ones(q.shape[:-1], dtype=torch.bool)))
        seq = EndpointWindow(pool, rows, 4)
        seq.compute(context, {}, [0, 1], torch.zeros(6), None)
        self.assertEqual(len(recorded), 2)
        for stage, seeds in recorded:
            self.assertTrue(torch.equal(seeds[0], torch.zeros(6)))
            self.assertTrue(torch.equal(seeds[1], torch.ones(6)))
        self.assertTrue(torch.equal(seq.rows[1]['q_pre'], torch.ones(6)))

    def test_no_grasp_solution_does_not_open_a_new_window(self):
        class Engine:
            capacity = 16
            calls = []
            def bind(self, context): pass
            def native(self, worlds, start): return [None] * len(worlds)
            def pure(self, worlds, start): return [torch.empty((0, 6)) for _ in worlds]
        pool = SimpleNamespace(prepare_endpoint_engine=lambda context: Engine(),
            stage_timing=SimpleNamespace(stage=lambda name: __import__('contextlib').nullcontext()))
        seq = EndpointWindow(pool, candidates([.088] * 40), 4)
        seq.compute(None, {}, list(range(16)), torch.zeros(6), None)
        self.assertEqual(set(seq.rows), set(range(16)))
        self.assertFalse(any(r['q_grasp'] is not None for r in seq.rows.values()))
        self.assertEqual(seq.stats['endpoint_groups'][0]['pre_and_lift_rows'], 0)

    def test_single_candidate_is_served_immediately(self):
        rows = candidates([.06, .07, .08])
        seq = EndpointWindow(None, rows, 4)
        seq.identify = lambda *a: 0
        calls = []
        def original(*a): calls.append(0); return torch.zeros(6), {'kind': 'original'}
        seq.grasp(None, None, None, None, None, original)
        self.assertEqual(calls, [0]); self.assertEqual(seq.rows, {})
        self.assertEqual(seq.summary()['serial_extra_endpoint_evaluations'], 0)

    def test_one_ready_result_does_not_wait_for_another(self):
        rows = candidates([.088] * 30)
        pool = SimpleNamespace(prepare_parallel=lambda *a: self.fail('Must not wait for or plan extra goals'))
        seq = EndpointWindow(pool, rows, 4); seq.current = 0
        seq.rows[0] = {'start': torch.zeros(6), 'q_pre': torch.ones(6), 'q_lift': torch.ones(6)}
        self.assertIsNone(seq.path(None, torch.zeros(6), torch.ones(6)))
        self.assertEqual(set(seq.rows), {0})

    def test_parallel_path_uses_only_already_computed_rows(self):
        rows = candidates([.088] * 30)
        calls = []
        def plan(context, jobs):
            calls.append([j['index'] for j in jobs]); return {0: {'q': [], 'times': []}}, {'real_rows': 2}
        pool = SimpleNamespace(prepare_parallel=lambda *a: SimpleNamespace(plan=plan),
            stage_timing=SimpleNamespace(stage=lambda name: __import__('contextlib').nullcontext()))
        seq = EndpointWindow(pool, rows, 4); seq.current = 0
        for i in [0, 3]:
            seq.rows[i] = {'start': torch.zeros(6), 'q_pre': torch.ones(6) * (i + 1), 'q_lift': torch.ones(6)}
        self.assertIsNotNone(seq.path(None, torch.zeros(6), torch.ones(6)))
        self.assertEqual(calls, [[0, 3]])
        self.assertEqual(set(seq.rows), {0, 3})

    def test_original_ik_fallback_is_cached_for_only_current_target(self):
        seq = EndpointWindow(SimpleNamespace(_pending_ik=[]), candidates([.088, .088]), 4); seq.current = 0
        seq.rows[0] = {'q_pre': None, 'q_lift': None}
        calls = []
        def original(*a): calls.append(1); return torch.ones(6)
        seq.ik(seq.goals[0]['pregrasp'], torch.zeros(6), original)
        seq.ik(seq.goals[0]['pregrasp'], torch.zeros(6), original)
        self.assertEqual(calls, [1])


if __name__ == '__main__': unittest.main()
