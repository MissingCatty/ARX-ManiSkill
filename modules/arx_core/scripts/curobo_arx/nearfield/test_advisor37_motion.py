"""Offline faults and kinematic checks from actual recorded ARX preflights."""
import copy
import unittest
import numpy as np
from scripts.curobo_arx.data_factory.common import ROOT, read
from .advisor37_motion import prepose, seed_arrival, FrozenURDF, cartesian_knots, command_times
from .advisor37_teacher import Teacher
from .advisor36_audit import physical

TOP = ROOT / 'runs/nearfield_advisor36_20261112_01'


class FrozenARXMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.episodes = [next((TOP / 'seed_preflight/groups/seed_00_raytracedlighting').glob(pattern))
                        for pattern in ('trials/*milk*center/episodes/*',
                                        'trials/*raisin*center/episodes/*',
                                        'trials/*tennis*center/episodes/*')]
        cls.state = read(cls.episodes[0] / 'closure_request.json')['measured_hold_snapshot']
        cls.G = np.array(cls.state['link_world']['right_eef_link'])
        cls.urdf = FrozenURDF(ROOT / 'scripts/curobo_arx/static_pick/reference/model/arx.urdf')

    def test_read_only_URDF_matches_real_ARX_links(self):
        for ep in self.episodes:
            self.assertTrue(self.urdf.verify(read(ep / 'closure_request.json')['measured_hold_snapshot'])['pass'])

    def test_corrupted_link_pose_fails_before_command(self):
        snapshot = copy.deepcopy(self.state)
        snapshot['link_world']['right_link27'][0][3] += .005
        self.assertFalse(self.urdf.verify(snapshot)['pass'])

    def test_true_60mm_axis_backoff_and_constant_insertion_rotation(self):
        start = prepose(self.G)
        np.testing.assert_allclose(self.G[:3, 3] - start[:3, 3], self.G[:3, 2] * .06)
        knots, alignment_end = cartesian_knots(start, self.G, insertion=True)
        for k in knots[alignment_end:]:
            np.testing.assert_allclose(k[:3, :3], self.G[:3, :3], atol=1e-12)
            local = self.G[:3, :3].T @ (k[:3, 3] - self.G[:3, 3])
            np.testing.assert_allclose(local[:2], 0, atol=1e-12)
        np.testing.assert_allclose(knots[-1], self.G, atol=1e-12)

    def test_far_start_is_not_swept_directly_into_target(self):
        with self.assertRaises(ValueError):
            cartesian_knots(self.G, self.G, insertion=True)

    def test_lift_fallback_only_world_Z_without_rotation(self):
        target = self.G.copy()
        target[2, 3] += .12
        poses, _ = cartesian_knots(self.G, target, insertion=False)
        self.assertTrue(all(np.allclose(p[:2, 3], self.G[:2, 3]) for p in poses))
        target[0, 3] += .001
        with self.assertRaises(ValueError):
            cartesian_knots(self.G, target, insertion=False)

    def test_real_Cheez_micrometre_rejection_now_passes_arrival_only(self):
        path = next((TOP / 'seed_preflight_02').glob('groups/*/trials/*cheez*X_minus/episodes/*/seed_pose_verification.json'))
        old = read(path)
        self.assertFalse(old['pass'])
        self.assertTrue(seed_arrival(old['target_error'], old['object_drift_before_close']))
        damaged = copy.deepcopy(old['target_error'])
        damaged['translation_m'] = .00301
        self.assertFalse(seed_arrival(damaged, old['object_drift_before_close']))

    def test_Spam_object_drift_still_blocks_closure(self):
        path = next((TOP / 'seed_preflight').glob('groups/*/trials/*spam*center/episodes/*/seed_pose_verification.json'))
        old = read(path)
        self.assertFalse(seed_arrival(old['target_error'], old['object_drift_before_close']))

    def test_timing_keeps_actual_ARX_command_derivative_bounds(self):
        command = read(self.episodes[0] / 'lift_command.json')
        times, maxima = command_times(command['position'], 12.)
        self.assertGreaterEqual(times[-1], 12.)
        for key, bound in [('1', 1.5), ('2', 5.), ('3', 100.)]:
            self.assertLessEqual(maxima[key], bound + 1e-8)

    def test_unregistered_worker_cannot_disable_closed_prediction(self):
        worker = Teacher.__new__(Teacher)
        worker.registration = read(TOP / 'seed_preflight/registration.json')
        with self.assertRaises(RuntimeError):
            worker.authorize()
        worker.registration.update(stage='advisor37_canonical_seed_preflight',
            advisor37_seed_protocol={'remove_nominal_closed_preflight_gate': True, 'training_authorized': False})
        worker.authorize()

    def test_prior_success_is_still_raw_physical_success(self):
        # Re-read independent contact, complete hold and action-union evidence.
        for ep in self.episodes[:2]:
            receipt = physical(ep, persist=False)
            self.assertTrue(receipt['pass'])
            self.assertGreater(receipt['hold_min_rise_m'], .06)
            self.assertGreaterEqual(receipt['recorded_hold_duration_s'], 3.)


if __name__ == '__main__':
    unittest.main()
