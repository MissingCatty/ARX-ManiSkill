"""Analytic shapes check gap direction, sphere exposure and convex distance."""
import unittest

import numpy as np
import trimesh

from .fit_metrics import ConvexUnion, sphere_gap, sphere_boundary, weighted_stats, support_differences


class FitMetricTests(unittest.TestCase):
    def test_signed_sphere_gap(self):
        np.testing.assert_allclose(sphere_gap(np.array([[0, 0, 0], [2, 0, 0]]),
            np.array([[0, 0, 0, 1.]])), [-1, 1])

    def test_hidden_ball_does_not_contribute_surface_area(self):
        points, weights = sphere_boundary(np.array([[0, 0, 0, 1.], [0, 0, 0, .5]]), 128)
        self.assertEqual(len(points), 134)
        self.assertAlmostEqual(weights.sum(), 4*np.pi)

    def test_duplicate_balls_do_not_double_union_area(self):
        _, weights = sphere_boundary(np.array([[0, 0, 0, 1.], [0, 0, 0, 1.]]), 128)
        self.assertAlmostEqual(weights.sum(), 4*np.pi)

    def test_box_outside_distance_and_inside(self):
        model = ConvexUnion([trimesh.creation.box().vertices])
        values = model.outside_distance(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]]))
        np.testing.assert_allclose(values, [0, .5, np.sqrt(.5)])

    def test_separate_parts_do_not_fill_gap(self):
        cube = trimesh.creation.box().vertices
        model = ConvexUnion([cube + [-1, 0, 0], cube + [1, 0, 0]])
        self.assertFalse(model.inside(np.array([[0, 0, 0]]))[0])
        self.assertAlmostEqual(model.outside_distance(np.array([[0, 0, 0]]))[0], .5)

    def test_planar_collider_stays_zero_thickness(self):
        model = ConvexUnion([np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])])
        self.assertFalse(model.inside(np.array([[.2, .2, .01]]))[0])
        self.assertAlmostEqual(model.outside_distance(np.array([[.2, .2, .01]]))[0], .01)

    def test_weighted_statistics(self):
        result = weighted_stats([0, .001, .002], [98, 1, 1])
        self.assertEqual(result['p95_mm'], 0)
        self.assertAlmostEqual(result['mean_mm'], .03)

    def test_exact_support_direction(self):
        a = np.array([[0, 0, 0, 1.]])
        b = np.array([[0, 0, 0, .5]])
        np.testing.assert_allclose(support_differences(a, b, np.eye(3)), [.5, .5, .5])


if __name__ == '__main__':
    unittest.main()
