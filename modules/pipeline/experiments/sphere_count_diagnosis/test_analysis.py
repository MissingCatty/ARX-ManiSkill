"""Independent review must catch geometric loss and altered mask dimensions."""
import unittest

from .analysis import cover_margin, mask_difference


class AnalysisTests(unittest.TestCase):
    def test_missing_original_ball_is_rejected(self):
        original = [{'center': [0, 0, 0], 'radius': .01}] * 2
        with self.assertRaises(ValueError):
            cover_margin(original, original[:1], [[0]])

    def test_uncovered_original_ball_is_rejected(self):
        with self.assertRaises(ValueError):
            cover_margin([{'center': [0, 0, 0], 'radius': .02}],
                         [{'center': [0, 0, 0], 'radius': .01}], [[0]])

    def test_added_blocks_and_allows_are_separate(self):
        self.assertEqual(mask_difference([True, False, True], [False, True, True]),
            {'states': 3, 'reference_clear': 2, 'other_clear': 2, 'new_blocks': 1, 'new_allows': 1})

    def test_mismatched_mask_lengths_are_rejected(self):
        with self.assertRaises(ValueError):
            mask_difference([True], [True, False])


if __name__ == '__main__':
    unittest.main()
