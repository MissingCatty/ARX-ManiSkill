"""Regression checks for the misplaced book and robot-side edge admission."""
from arx_project.assets import load_object_catalog
from arx_project.scenes import current_dataset
import json
from pathlib import Path
import unittest

import numpy as np

import workspace_geometry as geometry


class WorkspaceAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        here = Path(__file__).resolve().parent
        cls.root = current_dataset()
        cls.db = load_object_catalog(cls.root / 'object_catalog.json')['models']
        cls.state = json.loads((cls.root / 'preflight/head_camera.json').read_text())
        cls.model = 'ai2thor__book__Book_11'
        cls.urdf = here.parents[2] / 'arx_core/scripts/curobo_arx/static_pick/reference/model/arx.urdf'
        cls.pivot, cls.radius = geometry.reach_bound(cls.state, cls.urdf)

    def spec(self, x, y):
        b = self.db[self.model]['bbox']
        lo, hi = np.array(b['min']), np.array(b['max'])
        center = (lo + hi) / 2
        return {'name': 'target', 'model_id': self.model,
                'pose': [x - center[0], y - center[1], -lo[2] + .002, 1., 0., 0., 0.]}

    def test_distant_book_rejected_by_geometric_range(self):
        spec = self.spec(.45, 1.)
        check = geometry.placement_check(spec, self.db, self.state, self.pivot, self.radius)
        self.assertFalse(check['passed'])
        self.assertGreater(check['center_distance_m'], self.radius)

    def test_ordinary_nearby_book_accepted(self):
        check = geometry.placement_check(self.spec(-.42, .02), self.db, self.state, self.pivot, self.radius)
        self.assertTrue(check['passed'])
        self.assertGreater(check['center_margin_to_upper_bound_m'], .25)

    def test_reach_bound_uses_shoulder_not_mobile_base(self):
        self.assertAlmostEqual(self.radius, .86783159, places=6)
        base = np.array(self.state['links']['base_link'])[:3, 3]
        self.assertGreater(np.linalg.norm(self.pivot - base), .6)

    def test_outside_camera_rejected_even_within_radius(self):
        check = geometry.placement_check(self.spec(-.66, .0), self.db, self.state, self.pivot, self.radius)
        self.assertLess(check['center_distance_m'], self.radius)
        self.assertFalse(check['fully_in_head_frame'])
        self.assertFalse(check['passed'])

    def test_edge_is_near_side_with_explicit_gap(self):
        spec = self.spec(-.7245 + (self.db[self.model]['bbox']['max'][0] - self.db[self.model]['bbox']['min'][0]) / 2 + .10, .05)
        self.assertTrue(geometry.edge_check(spec, self.db))
        self.assertFalse(geometry.edge_check(self.spec(.30, .05), self.db))


if __name__ == '__main__':
    unittest.main()
