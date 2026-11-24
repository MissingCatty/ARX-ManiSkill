import math
import unittest
import torch
from .multi_view import select_views


class ApproachSelectionTests(unittest.TestCase):
    def fixture(self):
        # The second-highest score is almost the same direction as the first.
        return torch.tensor([[[10.,9.,8.,7.,6.]]]),torch.tensor([[1.,0,0],[.999,.0447,0],[0,1.,0],[0,0,1.],[-1.,0,0]])

    def test_alternatives_change_approach_not_just_neighboring_template_index(self):
        scores,templates=self.fixture();picks=select_views(scores,templates,4,30)
        self.assertEqual([p.item() for p in picks],[0,2,3,4])
        for i in range(len(picks)):
            for j in range(i):self.assertLessEqual(torch.dot(templates[picks[i].item()],templates[picks[j].item()]),math.cos(math.pi/6))

    def test_impossible_separation_is_rejected_instead_of_reusing_the_same_view(self):
        scores,templates=self.fixture()
        with self.assertRaises(ValueError):select_views(scores,templates,4,150)
        with self.assertRaises(ValueError):select_views(scores,templates,6,30)


if __name__=='__main__':unittest.main()
