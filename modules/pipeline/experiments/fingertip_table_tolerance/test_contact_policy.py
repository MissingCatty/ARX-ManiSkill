"""Fault cases that distinguish distal contact from general collision release."""
import unittest
import warnings

import numpy as np

from contact_policy import TableContactPolicy


class ContactPolicyTest(unittest.TestCase):
    def setUp(self):
        self.policy = TableContactPolicy(.0011, .045)
        self.eef = np.eye(4)[None]
        self.eef[0, :3, :3] = np.diag([1., -1., -1.])
        self.eef[0, 2, 3] = .044
        self.spheres = np.array([[[0., 0., .003, .004], [0., 0., .1, .01]]])
        self.names = ['right_link27', 'right_link26']
        self.table = {'center': [0., 0., -.25], 'dims': [.2, .2, .5]}

    def check(self, stage='final_approach', other=True):
        return self.policy.evaluate(self.spheres, self.eef, self.names, self.table,
                                   stage=stage, other_world_clear=[other])

    def test_local_one_mm_cap_accepted(self):
        self.assertTrue(self.check()['valid'][0])

    def test_zero_tolerance_rejects_same_contact(self):
        self.policy = TableContactPolicy(0., .045)
        self.assertFalse(self.check()['valid'][0])

    def test_over_tolerance_rejected(self):
        self.spheres[0, 0, 2] = .002
        self.assertFalse(self.check()['valid'][0])

    def test_arm_collision_is_never_waived(self):
        self.spheres[0, 1, 2] = .009
        self.assertFalse(self.check()['valid'][0])

    def test_finger_body_is_not_tip(self):
        self.eef[0, 2, 3] = .030
        self.assertFalse(self.check()['valid'][0])

    def test_top_edge_side_contact_is_not_waived(self):
        self.spheres[0, 0, 0] = .101
        self.assertFalse(self.check()['valid'][0])

    def test_table_side_is_not_waived(self):
        self.spheres[0, 0, 2] = -.0001
        self.assertFalse(self.check()['valid'][0])

    def test_nonlocal_stages_stay_strict(self):
        for stage in ['reach', 'pregrasp', 'lift', 'transport', 'unknown']:
            with self.subTest(stage=stage): self.assertFalse(self.check(stage)['valid'][0])

    def test_closure_can_use_local_rule(self):
        self.assertTrue(self.check('closure')['valid'][0])

    def test_neighbor_or_ground_check_cannot_be_bypassed(self):
        self.assertFalse(self.check(other=False)['valid'][0])

    def test_invalid_numeric_state_fails_closed(self):
        self.spheres[0, 0, 0] = np.nan
        self.assertFalse(self.check()['valid'][0])

    def test_invalid_frame_fails_closed(self):
        self.eef[0, 0, 0] = 2.
        self.assertFalse(self.check()['valid'][0])

    def test_incomplete_other_world_receipt_raises(self):
        with self.assertRaises(ValueError):
            self.policy.evaluate(self.spheres, self.eef, self.names, self.table,
                                 stage='closure', other_world_clear=[])

    def test_unregistered_tolerance_rejected(self):
        with self.assertRaises(ValueError): TableContactPolicy(.003, .045)


if __name__ == '__main__': unittest.main()
