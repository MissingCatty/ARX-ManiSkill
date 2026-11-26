"""Cost/gradient guard must reject changed or nonfinite collision responses."""
import unittest

import torch

from .preflight import compare_outputs


class NumericGuardTests(unittest.TestCase):
    def test_roundoff_is_accepted(self):
        first = (torch.tensor([1., 0.]), torch.tensor([.5, -.5]))
        second = (first[0]+1e-8, first[1]+1e-8)
        self.assertEqual(len(compare_outputs(first, second)), 2)

    def test_changed_gradient_cannot_pass(self):
        first = (torch.tensor([1.]), torch.tensor([.5]))
        with self.assertRaises(ValueError):
            compare_outputs(first, (first[0], torch.tensor([.51])))

    def test_nonfinite_values_cannot_pass(self):
        first = (torch.tensor([1.]), torch.tensor([.5]))
        for value in [float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                compare_outputs(first, (first[0], torch.tensor([value])))


if __name__ == '__main__': unittest.main()
