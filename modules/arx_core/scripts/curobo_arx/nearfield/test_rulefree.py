import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from .rulefree_diagnose import raw_action,error


class RawModelTests(unittest.TestCase):
    def test_full_local_translation_and_rotation_are_preserved(self):
        e=np.eye(4);e[:3,:3]=Rotation.from_rotvec([.2,.4,-.1]).as_matrix()
        a=raw_action(e,'REFINE -20.000 10.000 75.000 5.0000 0.0000 -3.0000')
        target=np.asarray(a['goal_world']);delta=np.linalg.inv(e)@target
        np.testing.assert_allclose(delta[:3,3],[-.020,.010,.075],atol=1e-12)
        np.testing.assert_allclose(Rotation.from_matrix(delta[:3,:3]).as_rotvec(degrees=True),[5,0,-3],atol=1e-10)

    def test_zero_refine_is_not_automatic_closure(self):
        self.assertEqual(raw_action(np.eye(4),'REFINE 0 0 0 0 0 0')['kind'],'model_REFINE')
        self.assertEqual(raw_action(np.eye(4),'GRASP')['kind'],'model_GRASP')

    def test_invalid_numbers_stop_interpretation(self):
        with self.assertRaises(ValueError):raw_action(np.eye(4),'REFINE nan 0 1 0 0 0')

    def test_exact_target_has_zero_counterfactual_error(self):
        e=np.eye(4);g=np.asarray(raw_action(e,'REFINE 10 20 60 5 0 0')['goal_world'])
        r=error(e,g,g)
        self.assertLess(r['position_error_mm'],1e-10)
        self.assertLess(r['orientation_error_deg'],1e-10)


if __name__=='__main__':unittest.main()
