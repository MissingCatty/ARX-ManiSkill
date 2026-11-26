"""Check row ownership, ordering, fallback mapping and unchanged contact gates."""
from pathlib import Path
from dataclasses import dataclass, field
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from arx_project.layout import ROOT
from modules.curobo.fingertip_contact.planning import contact_pool
from modules.curobo.fingertip_contact.profile import ContactProfile
from .batches import EndpointSequence, copy_optimizer_settings, padded_worlds, row_safe_solve
from .pool import batched_pool


def candidates(count):
    rows = []
    for i in range(count):
        pose = np.eye(4); pose[0, 3] = i*.01
        rows.append({'pose': pose.tolist(), 'score': float(count-i)})
    return rows


class Engine:
    def __init__(self, capacity=3, valid=()):
        self.capacity, self.valid, self.calls = capacity, set(valid), []

    def bind(self, context):
        self.calls = []

    def native(self, goals, start):
        indices = [round(g[0, 3]/.01) for g in goals]
        self.calls.append(('native', indices))
        return [torch.full((6,), float(i)) if i in self.valid else None for i in indices]

    def pure(self, goals, start):
        indices = [round(g[0, 3]/.01) for g in goals]
        self.calls.append(('pure', indices))
        return [torch.full((1, 6), float(i)) for i in indices]


class SequenceTest(unittest.TestCase):
    def test_fixed_capacity_padding_keeps_target_order_and_only_repeats_first(self):
        worlds = np.stack([np.eye(4),np.eye(4)])
        worlds[1,0,3] = .1
        values = padded_worlds(worlds,16)
        self.assertEqual(values.shape,(16,4,4))
        np.testing.assert_array_equal(values[:2],worlds)
        np.testing.assert_array_equal(values[2:],np.repeat(worlds[:1],14,axis=0))
        self.assertFalse(np.shares_memory(values,worlds))

    def test_empty_or_oversize_target_batches_are_record_faults(self):
        for values in [[],np.stack([np.eye(4)]*17)]:
            with self.assertRaisesRegex(ValueError,'fixed batch capacity'):padded_worlds(values,16)

    def test_solver_buffer_shape_is_separate_from_numerical_optimizer_options(self):
        @dataclass
        class Settings:
            num_iters: int = 100
            history: int = 7
            num_problems: int = 1
            debug_info: object = None
            line_search_scale: list = field(default_factory=lambda:[.1,.3,.7,1.])
        original = Settings(num_iters=75,history=6,num_problems=32)
        current = Settings()
        copy_optimizer_settings(current,original)
        self.assertEqual((current.num_iters,current.history,current.num_problems),(75,6,1))
        self.assertEqual(current.line_search_scale,original.line_search_scale)
        self.assertIsNot(current.line_search_scale,original.line_search_scale)

    def test_compacted_fallback_rows_return_to_original_candidates(self):
        engine = Engine(valid=(0, 2, 4))
        sequence = EndpointSequence(candidates(5), torch.zeros(6), engine)
        rows = [sequence.next(g) for g in sequence.goals]
        self.assertEqual(engine.calls, [('native', [0, 1, 2]), ('pure', [1]),
            ('native', [3, 4]), ('pure', [3])])
        self.assertEqual(float(rows[3]['pure'][0, 0]), 3.)
        self.assertEqual(float(rows[4]['native'][0]), 4.)

    def test_early_success_does_not_compute_the_whole_candidate_pool(self):
        engine = Engine(capacity=16, valid=(0,))
        sequence = EndpointSequence(candidates(2000), torch.zeros(6), engine)
        sequence.next(sequence.goals[0])
        self.assertEqual(sequence.summary()['computed_candidates'], 16)
        self.assertEqual(sequence.summary()['prefetched_unconsumed'], 15)

    def test_identical_poses_keep_separate_source_rows(self):
        rows = candidates(2); rows[1]['pose'] = rows[0]['pose']
        sequence = EndpointSequence(rows, torch.zeros(6), Engine())
        for goal in sequence.goals:sequence.next(goal)
        self.assertEqual(sequence.cursor, 2)
        self.assertEqual(len(sequence.rows), 2)

    def test_input_pose_and_score_are_unchanged(self):
        rows = candidates(5)
        before = __import__('json').dumps(rows)
        sequence = EndpointSequence(rows, torch.zeros(6), Engine())
        for goal in sequence.goals:sequence.next(goal)
        self.assertEqual(__import__('json').dumps(rows), before)
        self.assertAlmostEqual(sequence.goals[0][2, 3], -.043)

    def test_wrong_goal_cannot_consume_another_candidate(self):
        sequence = EndpointSequence(candidates(2), torch.zeros(6), Engine())
        with self.assertRaisesRegex(ValueError, 'original grasp'):
            sequence.next(sequence.goals[1])

    def test_missing_solver_rows_stop_instead_of_becoming_grasp_failure(self):
        engine = Engine(); engine.pure = lambda goals, start: []
        sequence = EndpointSequence(candidates(3), torch.zeros(6), engine)
        with self.assertRaisesRegex(RuntimeError, 'fallback IK lost'):
            sequence.next(sequence.goals[0])

    def test_native_resource_error_propagates(self):
        engine = Engine()
        def failed(*args):raise RuntimeError('CUDA resource failure')
        engine.native = failed
        sequence = EndpointSequence(candidates(3), torch.zeros(6), engine)
        with self.assertRaisesRegex(RuntimeError, 'CUDA resource failure'):
            sequence.next(sequence.goals[0])


class ContactGateTest(unittest.TestCase):
    def test_default_native_hook_keeps_the_original_context_call(self):
        cls = contact_pool(object, ContactProfile())
        pool = object.__new__(cls)
        calls = []
        context = SimpleNamespace(ik=lambda goal, start: calls.append((goal,start)) or 'original_result')
        self.assertEqual(pool.native_grasp_ik(context,'goal','start'),'original_result')
        self.assertEqual(calls,[('goal','start')])

    def make_pool(self, *, native=None, pure=None):
        # Run the production ContactPool.grasp_endpoint implementation with CPU
        # endpoint results. Only solver execution and geometry are fixtures.
        cls = batched_pool(contact_pool(object, ContactProfile()))
        pool = object.__new__(cls)
        row = {'native': native, 'pure': pure}
        pool.endpoint_sequence = SimpleNamespace(next=lambda goal: row)
        pool.endpoint_row = None
        goal = np.eye(4)
        checker = SimpleNamespace(validate=lambda q: torch.ones(q.shape[0], dtype=torch.bool))
        context = SimpleNamespace(checker=checker)
        validator = SimpleNamespace(geometry=SimpleNamespace(frames=lambda q: {'right_eef_link':goal[None]}),
            receipts=[], validate=lambda q, stage: np.array([True]))
        return pool, context, validator, goal

    def test_full_world_endpoint_still_precedes_fallback(self):
        q = torch.zeros(6)
        pool, context, validator, goal = self.make_pool(native=q)
        result, evidence = pool.grasp_endpoint(context, {}, goal, q, validator)
        self.assertIs(result, q)
        self.assertEqual(evidence['kind'], 'native_full_world_ik')

    def test_local_contact_rule_still_accepts_a_bounded_fallback(self):
        pool, context, validator, goal = self.make_pool(pure=torch.zeros((1, 6)))
        result, evidence = pool.grasp_endpoint(context, {}, goal, torch.zeros(6), validator)
        self.assertIsNotNone(result)
        self.assertEqual(evidence['kind'], 'bounded_ik_with_local_table_check')

    def test_independent_fk_still_rejects_an_inaccurate_solution(self):
        pool, context, validator, goal = self.make_pool(pure=torch.zeros((1, 6)))
        wrong = goal.copy(); wrong[0, 3] = .003
        validator.geometry.frames = lambda q: {'right_eef_link':wrong[None]}
        result, evidence = pool.grasp_endpoint(context, {}, goal, torch.zeros(6), validator)
        self.assertIsNone(result)
        self.assertIn('independent_fk_error', evidence['reasons'])

    def test_neighbor_collision_is_not_accepted_as_table_contact(self):
        pool, context, validator, goal = self.make_pool(pure=torch.zeros((1, 6)))
        validator.receipts.append({'joint_bounds_clear':True, 'other_world_clear':False})
        validator.validate = lambda q, stage: np.array([False])
        result, evidence = pool.grasp_endpoint(context, {}, goal, torch.zeros(6), validator)
        self.assertIsNone(result)
        self.assertEqual(evidence['reasons'], ['ground_or_other_object_collision'])

    def test_endpoint_row_is_cleared_after_contact_validator_error(self):
        pool, context, validator, goal = self.make_pool(pure=torch.zeros((1, 6)))
        def failed(*args, **kwargs):raise RuntimeError('record chain failed')
        validator.validate = failed
        with self.assertRaisesRegex(RuntimeError, 'record chain failed'):
            pool.grasp_endpoint(context, {}, goal, torch.zeros(6), validator)
        self.assertIsNone(pool.endpoint_row)


class RowSafeTest(unittest.TestCase):
    def fixture(self, flags=(True,False,True), fail=False):
        count = len(flags)
        class State:
            position = torch.zeros((count,6))
            def clone(self):return self
        seeds = torch.arange(count,dtype=torch.float32)[:,None,None].expand(count,2,6).clone()
        seed_solver = SimpleNamespace(solve_batch=lambda **kw:SimpleNamespace(
            js_solution=SimpleNamespace(position=seeds)))
        solver = SimpleNamespace(config=SimpleNamespace(use_lm_seed=True,exit_early=True,num_seeds=2),
            seed_ik_solver=seed_solver,calls=[])
        def solve(goal,**kw):
            self.assertFalse(solver.config.use_lm_seed)
            solver.calls.append(kw)
            if kw['run_optimizer'] and fail:raise RuntimeError('optimizer resource failure')
            values = torch.full((count,1,6),99.) if kw['run_optimizer'] else seeds[:,:1].clone()
            valid = torch.ones((count,1),dtype=torch.bool) if kw['run_optimizer'] else torch.tensor(flags)[:,None]
            return SimpleNamespace(solution=values,success=valid)
        solver.solve_pose = solve
        return solver,State()

    def test_already_solved_rows_never_enter_a_different_rows_optimizer(self):
        solver,state = self.fixture();targets=[]
        q,valid,info = row_safe_solve(solver,None,state,1,3,
            lambda indices:targets.append(indices.tolist()))
        self.assertEqual(q[:,0,0].tolist(),[0.,99.,2.])
        self.assertTrue(valid.all())
        self.assertEqual(targets,[[1,1,1]])
        self.assertEqual(info['refined_source_rows'],[1])
        self.assertEqual(info['seeds_per_original_target'],2)
        self.assertEqual(len(solver.calls),2)

    def test_padding_rows_cannot_force_refinement_of_real_solved_rows(self):
        solver,state = self.fixture(flags=(True,True,False))
        q,valid,info = row_safe_solve(solver,None,state,1,2,lambda indices:None)
        self.assertEqual(info['refined_source_rows'],[])
        self.assertEqual(len(solver.calls),1)

    def test_resource_error_propagates_and_restores_solver_flags(self):
        solver,state = self.fixture(fail=True)
        with self.assertRaisesRegex(RuntimeError,'optimizer resource failure'):
            row_safe_solve(solver,None,state,1,3,lambda indices:None)
        self.assertTrue(solver.config.use_lm_seed)
        self.assertTrue(solver.config.exit_early)

    def test_original_always_optimize_mode_is_preserved(self):
        solver,state = self.fixture(flags=(True,True,True))
        solver.config.exit_early=False
        q,valid,info = row_safe_solve(solver,None,state,1,3,lambda indices:None)
        self.assertEqual(info['refined_source_rows'],[0,1,2])
        self.assertFalse(solver.config.exit_early)


if __name__=='__main__':
    unittest.main()
