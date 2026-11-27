"""CPU fault tests for speculative batches, score order and original fallback."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np
import torch

from .planning import parallel_pool


class Context:
    def __init__(self):
        self.width = .088
        self.failed = set()
        self.checker = SimpleNamespace(validate=lambda q: torch.ones(q.shape[:-1], dtype=torch.bool))

    def measured(self, state): return torch.zeros(6)
    def ik(self, world, start): return torch.ones(6) * float(world[0, 3])
    def path(self, start, goal):
        if round(float(goal[0]), 4) in self.failed: return None
        return torch.stack([start, goal])
    def optimize(self, label, raw, duration):
        return {'q': raw.tolist(), 'times': [0., duration], 'source': 'original'}


class Base:
    def __init__(self):
        self.stage_timing = SimpleNamespace(stage=lambda name: __import__('contextlib').nullcontext())
        self.contact_profile = SimpleNamespace(profile_id='unchanged')
        self.context = Context()
        self.batch_calls = []
        self.batch_success = set()

    def grasp_endpoint(self, context, state, grasp, start, validator):
        return torch.ones(6) * float(grasp[0, 3]), {'kind': 'original'}

    def validate_narrowing(self, work, state, width): return {'passed': True, 'samples': []}


class SelectionTests(unittest.TestCase):
    def run_case(self, widths, *, batch_success=(), failed=()):
        pool = parallel_pool(Base)()
        pool.parallel_capacity = 4
        pool.context.failed = set(failed)
        validator = SimpleNamespace(straight_approach=lambda *a: {'q': [], 'times': []})
        def bind(work, state, candidate):
            pool.context.width = candidate['width']
            return pool.context, validator, state, {'candidate_width_m': candidate['width']}
        pool._candidate_context = bind
        def plan(context, jobs):
            if batch_success == 'resource_error':
                raise RuntimeError('CUDA resource failure')
            pool.batch_calls.append([j['index'] for j in jobs])
            return {j['index']: {'source': 'parallel'} for j in jobs if j['index'] in batch_success}, {}
        pool.prepare_parallel = lambda *a: SimpleNamespace(plan=plan)
        candidates = []
        for i, width in enumerate(widths):
            pose = np.eye(4); pose[0, 3] = float(i + 1)
            candidates.append({'pose': pose.tolist(), 'width': width, 'score': 1. - .1 * i})
        with patch('modules.curobo.parallel_planning.planning.candidate_pool',
                   return_value={'candidates': candidates}), patch('modules.curobo.parallel_planning.planning.write_json'):
            result = pool.select_contact(pool.context, {}, Path('/unused'), validator)
        return pool, result

    def test_parallel_later_success_cannot_skip_earlier_original_success(self):
        pool, result = self.run_case([.088] * 4, batch_success=[3])
        self.assertEqual(pool.batch_calls, [[0, 1, 2, 3]])
        self.assertEqual(result['selected']['index'], 0)
        self.assertEqual(result['paths']['pregrasp']['source'], 'original')
        self.assertEqual(len(result['candidate_events']), 1)
        self.assertEqual(result['parallel_paths']['extra_endpoint_evaluations'], 3)

    def test_earlier_path_failure_advances_without_repeating_endpoints(self):
        pool, result = self.run_case([.088] * 4, batch_success=[1], failed=[1.])
        self.assertEqual(result['selected']['index'], 1)
        self.assertEqual(result['rejected']['path'], 1)
        self.assertEqual(len(result['candidate_events']), 2)
        self.assertEqual(result['paths']['pregrasp']['source'], 'parallel')

    def test_different_widths_never_share_geometry(self):
        pool, result = self.run_case([.06, .0600001, .07, .08])
        self.assertEqual(pool.batch_calls, [])
        self.assertAlmostEqual(pool.context.width, .06)
        self.assertEqual(result['selected']['open_finger_m'], .03)

    def test_partial_real_batch_and_tail_identity(self):
        pool, result = self.run_case([.088] * 2, batch_success=[0, 1])
        self.assertEqual(pool.batch_calls, [[0, 1]])
        self.assertEqual(result['selected']['index'], 0)
        self.assertEqual(result['parallel_paths']['endpoint_evaluations'], 2)

    def test_all_failed_rows_are_normal_planning_failure(self):
        pool, result = self.run_case([.088] * 2, failed=[1., 2.])
        self.assertFalse(result['success'])
        self.assertEqual(result['rejected']['path'], 2)
        self.assertEqual(len(result['candidate_events']), 2)

    def test_resource_failure_propagates_without_fallback(self):
        with self.assertRaisesRegex(RuntimeError, 'CUDA resource failure'):
            self.run_case([.088] * 2, batch_success='resource_error')


if __name__ == '__main__': unittest.main()
