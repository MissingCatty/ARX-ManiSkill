"""Check that a locally accepted connector also satisfies authored timing limits."""
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from runtime_adapter import LocalContactValidator


class TimingTest(unittest.TestCase):
    def make(self, velocity):
        obj=object.__new__(LocalContactValidator)
        names=[f'right_joint{i}' for i in range(21,27)]
        joints={}
        for name in names:
            node=ET.Element('joint');ET.SubElement(node,'limit',velocity=str(velocity));joints[name]=node
        obj.geometry=SimpleNamespace(config={'cspace':{'joint_names':names}},joints=joints)
        obj.validate=lambda q,**unused:np.ones(len(q),dtype=bool)
        return obj

    def test_long_connection_is_slowed_to_authored_velocity(self):
        result=self.make(.1).straight_approach(np.zeros(6),np.ones(6),duration_s=.1)
        self.assertLessEqual(result['max_authored_velocity_fraction'],.800001)
        self.assertLessEqual(result['max_analytic_acceleration_rad_s2'],5.)
        self.assertLessEqual(result['max_analytic_jerk_rad_s3'],100.)
        np.testing.assert_allclose(result['q'][0],np.zeros(6))
        np.testing.assert_allclose(result['q'][-1],np.ones(6))

    def test_collision_in_dense_connection_rejects_path(self):
        obj=self.make(1.5)
        obj.validate=lambda q,**unused:~((q[:,0]>.49)&(q[:,0]<.51))
        self.assertIsNone(obj.straight_approach(np.zeros(6),np.ones(6)))

    def test_zero_distance_remains_finite(self):
        result=self.make(1.5).straight_approach(np.zeros(6),np.zeros(6))
        self.assertTrue(np.isfinite(result['times']).all())
        self.assertEqual(result['max_analytic_jerk_rad_s3'],0.)


if __name__=='__main__':unittest.main()
