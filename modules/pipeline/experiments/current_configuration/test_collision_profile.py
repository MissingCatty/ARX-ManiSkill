"""Reject old or silently modified collision models at the active loader."""
import copy
import unittest

from configuration import load_profile, require_current_profile


class CollisionProfileTest(unittest.TestCase):
    def setUp(self):
        self.spheres=load_profile()['collision_spheres']
        self.config={'robot_cfg':{'kinematics':{'collision_spheres':copy.deepcopy(self.spheres),
            'collision_link_names':list(self.spheres),'collision_sphere_buffer':0.}}}

    def test_active_profile_has_exact_871_parameters(self):
        self.assertEqual(require_current_profile(self.config)['total_spheres'],871)

    def test_old_model_cannot_pass_by_name(self):
        self.config['robot_cfg']['kinematics']['collision_spheres']['right_link27'].append(copy.deepcopy(self.spheres['right_link27'][0]))
        with self.assertRaises(ValueError):require_current_profile(self.config)

    def test_same_count_but_changed_radius_is_rejected(self):
        self.config['robot_cfg']['kinematics']['collision_spheres']['right_link27'][0]['radius']+=1e-6
        with self.assertRaises(ValueError):require_current_profile(self.config)

    def test_changed_sphere_center_is_rejected(self):
        self.config['robot_cfg']['kinematics']['collision_spheres']['right_link28'][0]['center'][2]+=.00001
        with self.assertRaises(ValueError):require_current_profile(self.config)

    def test_unlisted_link_or_extra_margin_is_rejected(self):
        self.config['robot_cfg']['kinematics']['collision_sphere_buffer']=.001
        with self.assertRaises(ValueError):require_current_profile(self.config)


if __name__=='__main__':unittest.main()
