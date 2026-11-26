"""Physical-chain and analytic branch checks; no CUDA or simulator imports."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from .kinematics import ArmChain, GeometricSolver, periodic_branches


class GeometricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solvers = {side: GeometricSolver(side) for side in ['right', 'left']}

    def test_family_and_axis_mapping(self):
        for solver in self.solvers.values():
            self.assertEqual(solver.family, '6R-THREE_INNER_PARALLEL')
            np.testing.assert_array_equal(solver.axis_signs, [1, 1, -1, -1, 1, 1])

    def test_FK_and_recovered_targets_across_both_arms(self):
        rng = np.random.default_rng(20261126)
        for side, solver in self.solvers.items():
            values = rng.uniform(solver.chain.limits[:, 0], solver.chain.limits[:, 1], (256, 6))
            for i, q in enumerate(values):
                with self.subTest(side=side, case=i):
                    target = solver.chain.fk(q)[0]
                    np.testing.assert_allclose(solver.robot.fwdkin(q*solver.axis_signs), target,
                        atol=2e-12, rtol=0)
                    answer = solver.solve(target, q)
                    self.assertGreater(answer['admitted_branches'], 0)
                    returned = np.asarray(answer['q'])
                    self.assertTrue((returned >= solver.chain.limits[:, 0]).all())
                    self.assertTrue((returned <= solver.chain.limits[:, 1]).all())
                    actual = solver.chain.fk(returned)
                    self.assertLess(float(np.linalg.norm(actual[:, :3, 3]-target[:3, 3], axis=1).max()), .002)
                    self.assertLess(float(Rotation.from_matrix(actual[:, :3, :3]@target[:3, :3].T).magnitude().max()), .03)

    def test_limit_branches_above_pi_and_wrist_singularities(self):
        for side, solver in self.solvers.items():
            center = solver.chain.limits.mean(axis=1)
            probes = [np.zeros(6), center, np.array([.2, 3.5, 3.2, .4, .5, -.3])]
            for wrist in [0., -1e-8, 1e-8]:
                q = center.copy(); q[4] = wrist; probes.append(q)
            for i, q in enumerate(probes):
                with self.subTest(side=side, case=i):
                    answer = solver.solve(solver.chain.fk(q)[0], q)
                    self.assertGreater(answer['admitted_branches'], 0)

    def test_arms_differ_by_the_actual_mount_translation(self):
        right, left = (self.solvers[side].chain for side in ['right', 'left'])
        q = np.random.default_rng(871).uniform(right.limits[:, 0], right.limits[:, 1], (32, 6))
        rf, lf = right.fk(q), left.fk(q)
        np.testing.assert_allclose(lf[:, :3, 3]-rf[:, :3, 3],
            np.broadcast_to([0., .5, 0.], (len(q), 3)), atol=2e-12, rtol=0)
        np.testing.assert_allclose(lf[:, :3, :3], rf[:, :3, :3], atol=2e-12, rtol=0)

    def test_unreachable_target_produces_unknown_not_success(self):
        target = np.eye(4); target[:3, 3] = [10, 10, 10]
        answer = self.solvers['right'].solve(target, np.zeros(6))
        self.assertEqual(answer['admitted_branches'], 0)
        self.assertEqual(answer['no_admitted_branch_means'], 'fallback_to_original_IK')

    def test_invalid_poses_and_relaxed_tolerances_rejected(self):
        solver = self.solvers['right']
        invalid = [np.eye(3), np.full((4, 4), np.nan), np.diag([-1, 1, 1, 1])]
        for pose in invalid:
            with self.assertRaises(ValueError): solver.solve(pose, np.zeros(6))
        for kwargs in [{'position_tolerance': .003}, {'rotation_tolerance': .04}]:
            with self.assertRaises(ValueError): solver.solve(np.eye(4), np.zeros(6), **kwargs)

    def test_2pi_conversion_does_not_clip_an_unrelated_angle(self):
        limits = np.array([[-1., 1.]]*6)
        values = np.full(6, .2)
        values[0] = 2*np.pi+.2
        np.testing.assert_allclose(periodic_branches(values, limits), [[.2]*6], atol=1e-12)
        values[0] = 2.
        self.assertEqual(len(periodic_branches(values, limits)), 0)

    def test_all_legal_periodic_branches_are_enumerated(self):
        limits = np.array([[-1., 1.]]*6); limits[0] = [-2*np.pi, 2*np.pi]
        values = periodic_branches(np.zeros(6), limits)
        np.testing.assert_allclose(values[:, 0], [-2*np.pi, 0., 2*np.pi], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
