"""Geometry checks on top-only contact, table edges and internal finger gaps."""
import unittest

import numpy as np

from narrow_phase import linear_extreme, extreme_points


def cube(lo, hi):
    import itertools
    return np.array(list(itertools.product(*zip(lo, hi))), dtype=float)


class ConvexClipTest(unittest.TestCase):
    def test_mesh_stl_duplicates_do_not_change_convex_shape(self):
        points = cube([-.01]*3, [.01]*3)
        p = extreme_points(np.repeat(points, 3, axis=0))
        self.assertEqual(len(p), 8)

    def test_authored_planar_part_is_retained(self):
        points = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [0, 0, 0]], dtype=float)
        self.assertEqual(len(extreme_points(points)), 4)

    def test_table_edge_triangle_crossing_is_detected(self):
        vertices = np.array([[-2., -2., -.001], [2., -2., -.001], [0., 2., -.001], [0., 0., .1]])
        point = linear_extreme(vertices, [0, 0, 1], np.array([0, 0, -.25]), np.array([.1, .1, .25]))
        self.assertIsNotNone(point)
        self.assertAlmostEqual(point[2], -.001)

    def test_disjoint_finger_gap_is_not_filled(self):
        center, half = np.array([0, 0, -.25]), np.array([.01, .01, .25])
        for lo, hi in [([-.06, -.01, -.01], [-.04, .01, .01]), ([.04, -.01, -.01], [.06, .01, .01])]:
            self.assertIsNone(linear_extreme(cube(lo, hi), [0, 0, 1], center, half))

    def test_side_face_intersection_requires_separate_check(self):
        vertices = cube([.09, -.01, -.001], [.11, .01, .01])
        point = linear_extreme(vertices, [0, 0, 1], np.array([0, 0, -.25]), np.array([.1, .1, .25]),
                               include_z=True, equality=(0, .1))
        self.assertIsNotNone(point)
        self.assertLess(point[2], 0)


if __name__ == '__main__': unittest.main()
