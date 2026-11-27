"""Exact output checks, including ZeroGrasp's first-outside-point tie case."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from .kernels import native_detect
from .spatial_crop import detect,relevant_points


class SpatialTests(unittest.TestCase):
    def equal(self,poses,points):
        expected=native_detect(poses,points);actual=detect(poses,points)
        for key in expected:np.testing.assert_array_equal(actual[key],expected[key],err_msg=key)

    def test_random_frames_and_background_match_exactly(self):
        rng=np.random.default_rng(17);poses=np.tile(np.eye(4),(50,1,1))
        poses[:,:3,:3]=Rotation.random(50,random_state=rng).as_matrix()
        poses[:,:3,3]=rng.uniform(-.05,.05,(50,3))
        self.equal(poses,rng.uniform(-.4,.4,(20000,3)))

    def test_first_outside_point_is_preserved_for_contact_tie(self):
        points=np.r_[np.array([[1,0,-1]]),np.tile([0,.01,-.02],(20,1)),np.array([[2,2,2]])]
        cropped,ids=relevant_points(np.eye(4)[None],points)
        self.assertIn(0,ids);self.assertNotIn(len(points)-1,ids)
        self.equal(np.eye(4)[None],points)

    def test_nonunit_source_axes_stay_exact(self):
        rng=np.random.default_rng(9);poses=np.tile(np.eye(4),(8,1,1));poses[:,:3,:3]*=1.000001
        self.equal(poses,rng.uniform(-.4,.4,(10000,3)))

    def test_empty_inputs_and_degenerate_frames(self):
        self.equal(np.eye(4)[None],np.empty((0,3)))
        bad=np.eye(4)[None];bad[0,2,2]=0
        with self.assertRaises(ValueError):relevant_points(bad,np.zeros((1,3)))


if __name__=='__main__':unittest.main()
