"""Exact majority for all votes, and rejection of changed native semantics."""
from itertools import product
import unittest
from types import SimpleNamespace

from .native_contract import early_vote, verify_native_contract
from .backend import mesh_vote_pool


class ContractTests(unittest.TestCase):
    def test_all_eight_vote_patterns(self):
        for values in product([False, True], repeat=3):
            result, count = early_vote(values)
            self.assertEqual(result, sum(values) >= 2)
            self.assertEqual(count, 2 if values[0] == values[1] else 3)

    def test_undecided_votes_cannot_be_accepted(self):
        with self.assertRaises(ValueError): early_vote([False, True])

    def test_different_nearest_point_code_is_rejected(self):
        header = '''mesh_query_point(uint64_t id, const vec3& point, float max_dist, float& inside) { x(); }
mesh_query_point_no_sign(uint64_t id, const vec3& point, float max_dist) { y(); }'''
        with self.assertRaisesRegex(ValueError, 'traversal differs'): verify_native_contract(header)

    def test_changed_ray_voting_is_rejected(self):
        header = '''mesh_query_point(uint64_t id, const vec3& point, float max_dist, float& inside) { x(); }
mesh_query_point_no_sign(uint64_t id, const vec3& point, float max_dist) { x(); }
CUDA_CALLABLE inline float mesh_query_inside_ray_tracing(uint64_t id, const vec3& p)
{ return 1; }'''
        with self.assertRaisesRegex(ValueError, 'majority semantics changed'): verify_native_contract(header)

    def test_control_and_fast_graphs_are_never_shared(self):
        klass = mesh_vote_pool(object);pool = klass.__new__(klass)
        control, fast = object(), object()
        pool.endpoint_sequence = None
        pool.mesh_vote = SimpleNamespace(enabled=False, assert_original_untouched=lambda: None)
        pool.mesh_solver_pairs = {16: (control, fast)}
        pool.batch_engines = {16: SimpleNamespace(full_solver=control)}
        pool.mesh_vote_mode(True);self.assertIs(pool.batch_engines[16].full_solver, fast)
        pool.mesh_vote_mode(False);self.assertIs(pool.batch_engines[16].full_solver, control)

    def test_variant_cannot_change_during_search(self):
        klass = mesh_vote_pool(object);pool = klass.__new__(klass)
        pool.endpoint_sequence = object()
        with self.assertRaises(RuntimeError): pool.mesh_vote_mode(True)


if __name__ == '__main__': unittest.main()
