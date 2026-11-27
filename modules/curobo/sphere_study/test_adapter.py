from copy import deepcopy
import unittest

from .adapter import DiagnosticProfile,opening_with_count
from .profiles import digest


class AdapterTests(unittest.TestCase):
    def test_registered_profile_is_checked_instead_of_accepting_arbitrary_models(self):
        spheres={'finger':[{'center':[0,0,0],'radius':.005}]}
        profile={'profile_id':'test','total_spheres':1,'sphere_parameter_sha256':digest(spheres),
                 'source_parameter_sha256':'approved_source'}
        guard=DiagnosticProfile(profile)
        config={'kinematics':{'collision_spheres':spheres,'collision_link_names':['finger'],'collision_sphere_buffer':0.}}
        self.assertEqual(guard.require(config)['total_spheres'],1)
        bad=deepcopy(config);bad['kinematics']['collision_spheres']['finger'][0]['radius']=.006
        with self.assertRaises(ValueError):guard.require(bad)
        bad=deepcopy(config);bad['kinematics']['collision_sphere_buffer']=.001
        with self.assertRaises(ValueError):guard.require(bad)

    def test_count_override_does_not_modify_the_default_function(self):
        from modules.curobo.gripper_sweep.cached_sweep import opening_sweep
        original=opening_sweep.__code__
        diagnostic=opening_with_count(opening_sweep,256)
        self.assertIs(opening_sweep.__code__,original)
        self.assertIn(871,opening_sweep.__code__.co_consts)
        self.assertIn(256,diagnostic.__code__.co_consts)
        self.assertIsNot(diagnostic.__globals__,opening_sweep.__globals__)


if __name__=='__main__':unittest.main()
