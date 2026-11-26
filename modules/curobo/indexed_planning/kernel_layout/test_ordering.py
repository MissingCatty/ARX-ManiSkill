"""The new indexing must be a permutation of every original collision task."""
import ast
from itertools import product
import unittest

from arx_project.layout import ROOT
from .ordering import reorder_source, task_pair


class OrderingTests(unittest.TestCase):
    def test_every_pair_is_preserved_exactly_once(self):
        for n, m in product([1, 2, 17, 871, 2048], [1, 2, 8]):
            before = [task_pair(i, n, m, obstacle_major=False) for i in range(n*m)]
            after = [task_pair(i, n, m, obstacle_major=True) for i in range(n*m)]
            self.assertEqual(set(before), set(after));self.assertEqual(len(set(after)), n*m)

    def test_invalid_grid_or_index_cannot_be_used(self):
        for i, n, m in [(-1, 1, 8), (8, 1, 8), (0, 0, 8), (0, 1, 0)]:
            with self.assertRaises(ValueError): task_pair(i, n, m, obstacle_major=True)

    def test_actual_sdk_kernel_only_changes_indices(self):
        text = (ROOT/'modules/curobo/source/curobo/_src/geom/collision/wp_collision_kernel.py').read_text()
        source = text[text.index('@wp.kernel\ndef sphere_obstacle_collision_kernel'):]
        changed = ast.parse(reorder_source(source)).body[0]
        self.assertEqual(changed.name, 'obstacle_major_collision')
        self.assertFalse(changed.decorator_list)

    def test_changed_original_indexing_is_rejected(self):
        source = 'def f():\n    sph_flat_idx = tid / max_n_obs\n    obs_local_idx = tid - sph_flat_idx * max_n_obs + 1\n'
        with self.assertRaises(ValueError): reorder_source(source)

    def test_missing_original_indexing_is_rejected(self):
        with self.assertRaises(ValueError): reorder_source('def f():\n    return 0\n')


if __name__ == '__main__': unittest.main()
