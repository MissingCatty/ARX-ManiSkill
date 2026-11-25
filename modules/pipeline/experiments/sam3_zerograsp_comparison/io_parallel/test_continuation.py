"""Fault checks for pending-only continuation and collision-cache ownership."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

from modules.curobo.scene_cache import mesh_stores, prune_inactive_meshes
from .continuation import partition, execution_pairs, inherited_terminal_method


class Store:
    def __init__(self, names, cache):
        self.names, self.wp_cache = names, cache
        self.num_envs = len(names)

    def get_names(self, env):
        return self.names[env]


class Continuation(unittest.TestCase):
    def test_finished_method_does_not_load_models_again(self):
        states={'complete':['done'],'interrupted':['interrupted'],'not_started':[]}
        self.assertTrue(inherited_terminal_method(states))
        self.assertFalse(inherited_terminal_method({**states,'not_started':['pending']}))
        self.assertEqual(execution_pairs({'cohort':{'pairs':[{'id':'done'},{'id':'interrupted'}]},
            'continuation':{'pending_case_ids':[]}}),[])

    def test_only_unclaimed_entries_keep_original_order_and_indices(self):
        reg = {'cohort': {'pairs': [{'id': str(i), 'entry': {'seed': 10+i}} for i in range(4)]},
               'continuation': {'pending_case_ids': ['3', '2']}}
        self.assertEqual([(i, p['id'], p['entry']['seed']) for i, p in execution_pairs(reg)],
                         [(2, '2', 12), (3, '3', 13)])

    def test_partition_keeps_interruption_out_of_pending(self):
        reg = {'cohort': {'pairs': [{'id': str(i)} for i in range(3)]}}
        rows = [{'id': str(i), 'method': 'zerograsp', 'state': state} for i, state in
                enumerate(['complete', 'interrupted', 'not_started'])]
        result = partition(reg, {'results': rows}, 'zerograsp')
        self.assertEqual(result, {'complete': ['0'], 'interrupted': ['1'], 'not_started': ['2']})
        for broken in [rows[:-1], rows+[rows[0]], [*rows[:-1], {**rows[-1], 'state': 'unknown'}]]:
            with self.assertRaises(RuntimeError):
                partition(reg, {'results': broken}, 'zerograsp')

    def test_active_meshes_all_environments_and_handles_stay_alive(self):
        active_a, active_b, dead = object(), object(), object()
        store = Store([['a'], ['b']], {'a': active_a, 'b': active_b, 'dead': dead})
        original_names = deepcopy(store.names)
        calls = []
        result = prune_inactive_meshes([store, store], lambda: calls.append(set(store.wp_cache)))
        self.assertEqual(calls, [{'a', 'b', 'dead'}, {'a', 'b'}])
        self.assertIs(store.wp_cache['a'], active_a)
        self.assertIs(store.wp_cache['b'], active_b)
        self.assertEqual(store.names, original_names)
        self.assertEqual(result['removed_inactive_meshes'], 1)
        self.assertEqual(result['stores'], 1)

    def test_missing_active_handle_stops_without_mutating_any_store(self):
        first = Store([['keep']], {'keep': object(), 'old': object()})
        broken = Store([['missing']], {})
        with self.assertRaises(RuntimeError):
            prune_inactive_meshes([first, broken], lambda: None)
        self.assertEqual(set(first.wp_cache), {'keep', 'old'})

    def test_failed_synchronization_never_frees_live_or_inactive_geometry(self):
        store = Store([['keep']], {'keep': object(), 'old': object()})
        def fail():
            raise RuntimeError('CUDA synchronization failed')
        with self.assertRaises(RuntimeError):
            prune_inactive_meshes([store], fail)
        self.assertEqual(set(store.wp_cache), {'keep', 'old'})

    def test_old_geometry_does_not_accumulate_across_a_hundred_worlds(self):
        store = Store([[]], {})
        for i in range(100):
            store.wp_cache[str(i)] = object()
            store.names[0] = [str(i)]
            prune_inactive_meshes([store], lambda: None)
            self.assertEqual(set(store.wp_cache), {str(i)})

    def test_store_discovery_covers_motion_open_closed_and_local_checkers(self):
        store = Store([['a']], {'a': object()})
        scene = SimpleNamespace(data=SimpleNamespace(meshes=store))
        checker = SimpleNamespace(scene_model=scene)
        context = SimpleNamespace(planner=SimpleNamespace(scene_collision_checker=scene),
                                  checker=checker, closed_checker=checker)
        planner = SimpleNamespace(pool=SimpleNamespace(contexts={'open': context, 'closed': context},
            contact_views={'open': {'checker': checker}}))
        self.assertEqual(mesh_stores(planner), [store])


if __name__ == '__main__':
    unittest.main()
