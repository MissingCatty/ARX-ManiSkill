"""CPU-only checks for attribution and counterfactual reporting boundaries."""
import unittest
import numpy as np
from .advisor35_diagnose import stats, subtract_gate, target_contacts, vector
from .advisor35_design import symmetric_pair
from scipy.spatial.transform import Rotation


class DiagnosisTests(unittest.TestCase):
    def test_closure_is_not_a_zero_motion_label(self):
        self.assertIsNone(vector('GRASP'))
        with self.assertRaises(ValueError):vector('REFINE 1 2')
        with self.assertRaises(ValueError):vector('REFINE nan 0 0 0 0 0')

    def test_sign_denominators_and_weighted_mean(self):
        a=np.array([[-2.,0,1,0,0,0],[2.,0,3,0,0,0]])
        s=stats(a,[1,3])
        self.assertEqual(s['mean'][:3],[0.,0.,2.])
        self.assertEqual(s['weighted_mean'][:3],[1.,0.,2.5])
        self.assertEqual(s['negative'][:3],[1,0,0])
        self.assertEqual(s['zero'][:3],[0,2,0])
        self.assertEqual(s['positive'][:3],[1,0,2])

    def test_subtraction_never_certifies_unexecuted_grasp(self):
        for passed in (True,False):
            result=subtract_gate({'pass':passed})
            self.assertEqual(result['extra_axial_moves'],0)
            self.assertIsNone(result['closed_collision_check_pass'])
            self.assertIsNone(result['physical_grasp_success'])
        self.assertEqual(subtract_gate({'pass':True})['next'],'independent_closed_model_check')
        self.assertEqual(subtract_gate({'pass':False})['next'],'visual_gate_reject')

    def test_contact_counts_cannot_be_reported_as_force(self):
        contact={'a':'/Robot/right_link28','b':'/Task/assets/asset_x','sim_time':1.,'type':2,'num_contacts':4}
        r=target_contacts({'contacts':[contact,contact]},'x')
        self.assertEqual(r['right_link28']['callbacks'],2)
        self.assertEqual(r['right_link28']['unique_sim_times'],1)
        self.assertFalse(r['right_link28']['force_measured'])
        self.assertIsNone(r['right_link28']['force_N'])
        self.assertEqual(target_contacts({'contacts':[contact]},'other')['right_link28']['callbacks'],0)

    def test_signed_labels_are_symmetric_even_with_rotation(self):
        # Rigid-transform algebra only; this fixture is not robot validation.
        g=np.eye(4);g[:3,3]=[4.05,2.13,.81]
        g[:3,:3]=Rotation.from_rotvec([.3,-.7,.1]).as_matrix()
        pair=symmetric_pair(g,[.020,-.010],np.deg2rad([5,0,15]),.050,.005)
        labels=[np.linalg.inv(p['entry'])@g for p in pair]
        np.testing.assert_allclose(labels[0][:2,3]+labels[1][:2,3],0,atol=1e-12)
        np.testing.assert_allclose(sum(Rotation.from_matrix(a[:3,:3]).as_rotvec() for a in labels),0,atol=1e-12)
        self.assertAlmostEqual(sum(a[2,3] for a in labels)/2,.050)
        with self.assertRaises(ValueError):symmetric_pair(g,[0,0],[0,0,0],.005,.010)


if __name__=='__main__':unittest.main()
