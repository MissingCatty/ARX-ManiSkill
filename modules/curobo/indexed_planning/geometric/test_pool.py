"""The geometric route cannot bypass physical FK, limits or collision checks."""
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from .pool import geometric_pool


class Original:
    def grasp_endpoint(self, *args):
        self.fallbacks += 1
        return torch.ones(6), {'kind': 'original_endpoint'}


class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.pool = object.__new__(geometric_pool(Original))
        self.pool.fallbacks = 0
        self.pool.contact_profile = SimpleNamespace(position_tolerance_m=.002, rotation_tolerance_rad=.03)
        self.answer = {'q': [[0.]*6], 'admitted_branches': 1}
        self.pool.geometric_sequence = SimpleNamespace(next=lambda *args: self.answer, cursor=1, events=[])
        self.context = SimpleNamespace(checker=SimpleNamespace(validate=lambda q: torch.ones(len(q), dtype=torch.bool)))
        self.state = {'joint_names': [f'right_joint{i}' for i in range(21, 27)],
            'joint_limits': [[-4., 4.]]*6}
        def frames(q):
            pose = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
            pose[:, :3, 3] = q[:, :3]
            return {'right_eef_link': pose}
        self.validator = SimpleNamespace(geometry=SimpleNamespace(frames=frames),
            validate=lambda q, stage: np.ones(len(q), dtype=bool))

    def request(self):
        return self.pool.grasp_endpoint(self.context, self.state, np.eye(4), torch.zeros(6), self.validator)

    def test_admitted_original_target_uses_original_contact_gate(self):
        q, proof = self.request()
        self.assertEqual(self.pool.fallbacks, 0)
        self.assertTrue(torch.equal(q, torch.zeros(6)))
        self.assertIn('geometric_independent_FK', proof['kind'])

    def test_colliding_geometric_branches_fall_back_to_original(self):
        self.validator.validate = lambda q, stage: np.zeros(len(q), dtype=bool)
        _, proof = self.request()
        self.assertEqual(self.pool.fallbacks, 1)
        self.assertEqual(proof['kind'], 'original_endpoint')

    def test_missing_branches_fall_back_to_original(self):
        self.answer.update(q=[], admitted_branches=0)
        self.request(); self.assertEqual(self.pool.fallbacks, 1)

    def test_wrong_physical_FK_cannot_be_accepted(self):
        self.answer['q'][0][0] = .1
        self.request(); self.assertEqual(self.pool.fallbacks, 1)
        self.assertEqual(self.pool.geometric_sequence.events[0]['roundoff_or_independent_FK_rejected'], 1)

    def test_authored_limit_checked_after_float32_conversion(self):
        self.state['joint_limits'][0] = [-.1, .1]
        self.answer['q'][0][0] = .1
        self.request(); self.assertEqual(self.pool.fallbacks, 1)

    def test_placement_witness_keeps_original_priority(self):
        self.state['kinematic_endpoint_witness'] = {'q': [0.]*6}
        self.request(); self.assertEqual(self.pool.fallbacks, 1)
        self.assertFalse(self.pool.geometric_sequence.events)


if __name__ == '__main__': unittest.main()
