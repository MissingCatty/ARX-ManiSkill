"""Finer thresholds preserve distinct grasps without changing their poses."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from .processor import Processor,hard_nms


class FineNMSTests(unittest.TestCase):
    def test_position_and_orientation_variants_survive_finer_nms(self):
        poses=np.tile(np.eye(4),(4,1,1));poses[1,0,3]=.02
        poses[2,:3,:3]=Rotation.from_euler('z',20,degrees=True).as_matrix()
        poses[3,0,3]=.001;original=poses.copy()
        args=(poses,np.array([4.,3.,2.,1.]),np.full(4,.06),np.ones(4,bool))
        coarse,_=hard_nms(*args,.03,30);fine,_=hard_nms(*args,.015,15)
        self.assertEqual(coarse,[0]);self.assertEqual(fine,[0,1,2])
        np.testing.assert_array_equal(poses,original)

    def test_default_matches_finer_configuration(self):
        processor=Processor({});self.assertEqual(processor.nms_translation_m,.015)
        self.assertEqual(processor.nms_rotation_deg,15.)
        with self.assertRaises(ValueError):Processor({},nms_translation_m=0)


if __name__=='__main__':unittest.main()
