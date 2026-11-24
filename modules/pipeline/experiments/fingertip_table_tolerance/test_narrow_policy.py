"""End-to-end gate faults for sphere refinement and real finger contacts."""
from types import SimpleNamespace
import unittest

import numpy as np

from narrow_phase import ConvexFingerPolicy
from test_narrow_phase import cube


class NarrowPolicyTest(unittest.TestCase):
    def setUp(self):
        self.eef = np.eye(4)[None]
        self.eef[0, :3, :3] = np.diag([1., -1., -1.])
        self.eef[0, 2, 3] = .044
        self.spheres = np.array([[[0., 0., .003, .004], [0., 0., .1, .01]]])
        p = object.__new__(ConvexFingerPolicy)
        p.geometry = SimpleNamespace(lower=np.full(6, -2.), upper=np.full(6, 2.),
            names=['right_link27', 'right_link26'],
            frames=lambda q: {'right_eef_link': self.eef, 'right_link27': np.eye(4)[None], 'right_link28': np.eye(4)[None]},
            spheres=lambda f: self.spheres)
        p.table = {'center': [0., 0., -.25], 'dims': [.2, .2, .5]}
        p.center, p.half = np.array(p.table['center']), np.array(p.table['dims'])/2
        p.parts = {'right_link27': [cube([-.002, -.002, -.001], [.002, .002, .004])], 'right_link28': []}
        p.tip, p.distal_length_m, p.tolerance_m = .045, .010, .0011
        self.p = p

    def check(self, **kwargs):
        return self.p.evaluate_one(np.zeros(6), stage=kwargs.get('stage', 'final_approach'),
                                   other_world_clear=kwargs.get('other', True))

    def test_real_distal_one_mm_contact_is_allowed(self):
        result = self.check()
        self.assertTrue(result['valid']); self.assertTrue(result['actual_tip_contact'])

    def test_real_contact_fails_zero_tolerance(self):
        self.p.tolerance_m = 0.
        self.assertFalse(self.check()['valid'])

    def test_sphere_false_positive_is_refined_without_physical_tolerance(self):
        self.p.parts['right_link27'] = [cube([-.002, -.002, .002], [.002, .002, .004])]
        self.p.tolerance_m = 0.
        result = self.check()
        self.assertTrue(result['valid']); self.assertEqual(result['reason'], 'sphere_only_false_positive')

    def test_wrist_sphere_contact_cannot_be_refined(self):
        self.spheres[0, 1, 2] = .009
        self.assertEqual(self.check()['reason'], 'nonfinger_table_collision')

    def test_side_contact_cannot_use_top_face_allowance(self):
        self.p.parts['right_link27'] = [cube([.099, -.002, -.001], [.101, .002, .004])]
        self.assertEqual(self.check()['reason'], 'table_side_contact')

    def test_finger_body_contact_cannot_use_tip_allowance(self):
        self.eef[0, 2, 3] = .030
        self.assertEqual(self.check()['reason'], 'finger_body_contact')

    def test_initial_reach_stays_strict(self):
        self.assertEqual(self.check(stage='reach')['reason'], 'nonlocal_stage_table_collision')

    def test_other_world_collision_stays_strict(self):
        self.assertEqual(self.check(other=False)['reason'], 'other_world_collision')


if __name__ == '__main__': unittest.main()
