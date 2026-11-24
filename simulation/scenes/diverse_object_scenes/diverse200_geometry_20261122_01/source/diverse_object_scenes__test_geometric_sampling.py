"""Regression checks for independent sampling instead of IK-conditioned poses."""
import json
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

import geometric_sampling as sampling


class Baseline:
    def __init__(self, db):
        self.db = db

    def actor_spec(self, name, model, center, yaw):
        b = self.db[model]['bbox']
        points = np.array([[x, y, z] for x in [b['min'][0], b['max'][0]]
                           for y in [b['min'][1], b['max'][1]] for z in [b['min'][2], b['max'][2]]])
        points = points @ Rotation.from_euler('z', yaw).as_matrix().T
        position = np.r_[np.array(center) - (points.min(0)[:2] + points.max(0)[:2])/2,
                         -points[:, 2].min() + .002]
        return {'name': name, 'model_id': model,
                'pose': [*position.tolist(), np.cos(yaw/2), 0., 0., np.sin(yaw/2)]}

    def bounds(self, spec):
        points = sampling.world_corners(spec, self.db)
        return points.min(0), points.max(0)


class IndependentSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        here = Path(__file__).resolve().parent
        root = here / 'datasets/diverse200_20261122_01'
        cls.state = json.loads((root / 'preflight/head_camera.json').read_text())
        cls.db = json.loads((root / 'object_catalog.json').read_text())['models']
        urdf = here.parents[2] / 'COOLAS-ARX/scripts/curobo_arx/static_pick/reference/model/arx.urdf'
        cls.region = sampling.define_region(cls.state, urdf)
        cls.baseline = Baseline(cls.db)
        cls.model = 'ai2thor__book__Book_11'

    def test_region_comes_from_table_geometry(self):
        np.testing.assert_allclose(self.region['center_roi_min_xy_m'], [-.7245, -.4])
        np.testing.assert_allclose(self.region['center_roi_max_xy_m'], [-.12, .4])

    def test_old_seventy_centimeter_gate_is_removed(self):
        spec = self.baseline.actor_spec('target', self.model, [-.2, .15], 0.)
        result = sampling.check_placement(spec, self.db, self.state, self.region)
        self.assertGreater(result['center_distance_m'], .7)
        self.assertTrue(result['passed'])

    def test_ik_results_and_old_pose_do_not_change_random_draw(self):
        entry = {'model_id': self.model, 'layout': 'S1_open'}
        a = sampling.sample_target(entry, self.baseline, self.region, np.random.default_rng(1234))
        b = sampling.sample_target({**entry, 'ik_success': False, 'old_pose': [999]*7},
                                   self.baseline, self.region, np.random.default_rng(1234))
        self.assertEqual(a, b)

    def test_common_layouts_use_identical_sampling_rule(self):
        poses = [sampling.sample_target({'model_id': self.model, 'layout': layout}, self.baseline,
                                        self.region, np.random.default_rng(9876))
                 for layout in ['S1_open', 'S3_sparse', 'S4_clutter', 'S5_occluded']]
        self.assertTrue(all(pose == poses[0] for pose in poses))

    def test_far_book_is_rejected_by_geometry(self):
        spec = self.baseline.actor_spec('target', self.model, [.318, -.185], 0.)
        result = sampling.check_placement(spec, self.db, self.state, self.region)
        self.assertFalse(result['passed'])
        self.assertFalse(result['not_clearly_beyond_link_length'])


if __name__ == '__main__':
    unittest.main()
