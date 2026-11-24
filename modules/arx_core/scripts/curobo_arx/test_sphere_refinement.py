"""Offline geometry checks from actual ARX RGB-D captures, not new executions."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from autoresearch import ROOT
from sphere_refinement import fit_rgbd_sphere, refine, PAD_CENTER_EEF


class SphereRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=ROOT/'runs/autoresearch_arx_20261019_round2/trials/00_control'
        cls.fit=fit_rgbd_sphere(cls.source)
        cls.seed=json.loads((cls.source/'selected_grasp.json').read_text())

    def test_observed_arx_surface_supports_the_sphere_fit(self):
        self.assertTrue(self.fit['accepted'])
        self.assertGreater(self.fit['point_count'],100)
        self.assertLess(abs(self.fit['radius_m']-.035),.002)

    def test_fit_does_not_read_simulator_pose_or_dimensions(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)
            shutil.copytree(self.source/'camera',path/'camera')
            shutil.copy2(self.source/'camera_world_optical.json',path/'camera_world_optical.json')
            (path/'capture_snapshot.json').write_text('{"objects": "invalid simulator metadata"}')
            (path/'active_asset.json').write_text('{"size_m": [999,999,999]}')
            observed=fit_rgbd_sphere(path)
            np.testing.assert_array_equal(observed['center_world'],self.fit['center_world'])
            self.assertEqual(observed['radius_m'],self.fit['radius_m'])

    def test_pad_target_and_approach_direction_are_preserved(self):
        changed=refine(self.seed,self.fit,height_m=.010)
        grasp=np.asarray(changed['grasp_world']);old=np.asarray(self.seed['grasp_world'])
        np.testing.assert_allclose(grasp[:3,2],old[:3,2],atol=1e-9)
        self.assertLess(abs(grasp[2,1]),1e-9)
        np.testing.assert_allclose(grasp[:3,:3].T@grasp[:3,:3],np.eye(3),atol=1e-6)
        expected=np.asarray(self.fit['center_world'])+[0,0,.010]
        np.testing.assert_allclose(grasp[:3,3]+grasp[:3,:3]@PAD_CENTER_EEF,expected,atol=1e-9)
        np.testing.assert_allclose(np.asarray(changed['lift_world'])[:3,3]-grasp[:3,3],[0,0,.12],atol=1e-9)
        np.testing.assert_allclose(np.asarray(changed['pregrasp_world'])[:3,3]-grasp[:3,3],-.08*grasp[:3,2],atol=1e-9)
        self.assertEqual(self.seed['network_row'],changed['network_row'])

    def test_rejected_fit_cannot_generate_a_grasp(self):
        invalid=copy.deepcopy(self.fit);invalid['accepted']=False
        with self.assertRaises(ValueError):refine(self.seed,invalid)


if __name__=='__main__':unittest.main()
