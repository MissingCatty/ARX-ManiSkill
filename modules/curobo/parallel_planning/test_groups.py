import unittest
from .groups import width_groups, padded_indices


class GroupTests(unittest.TestCase):
    def test_exact_widths_preserve_identity_and_order(self):
        jobs = [{'index': i, 'width': w} for i, w in enumerate([.088, .06, .088, .06000001, .06])]
        self.assertEqual([[x['index'] for x in g] for g in width_groups(jobs, 4)], [[0, 2], [1, 4], [3]])

    def test_full_and_partial_groups(self):
        jobs = [{'index': i, 'width': .088} for i in range(11)]
        self.assertEqual([len(g) for g in width_groups(jobs, 4)], [4, 4, 3])
        self.assertEqual([len(g) for g in width_groups(jobs, 8)], [8, 3])

    def test_padding_repeats_only_a_valid_row(self):
        self.assertEqual(padded_indices(2, 4), [0, 1, 0, 0])

    def test_empty_never_plans(self):
        self.assertEqual(width_groups([], 4), [])
        with self.assertRaises(ValueError): padded_indices(0, 4)

    def test_invalid_or_duplicate_inputs_stop(self):
        for w in [0., -.01, .089, float('nan'), float('inf')]:
            with self.assertRaises(ValueError): width_groups([{'index': 0, 'width': w}], 4)
        with self.assertRaises(ValueError): width_groups([{'index': 0, 'width': .06}] * 2, 4)


if __name__ == '__main__':
    unittest.main()
