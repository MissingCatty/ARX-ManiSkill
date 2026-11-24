"""Analytical checks of the model-frame shift and the unchanged physical TCP."""
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from repair_contract import descriptor,T_EEF_GENERATOR,generator_to_eef,generator_to_contact,simulation_config


class ContractTests(unittest.TestCase):
    def test_physical_cavity_and_model_cavity_are_equivalent(self):
        d=descriptor()
        physical=np.array(d['physical_eef_parameters']['offset_open'])
        virtual=np.array(d['parameters']['offset_open'])
        for q in Rotation.random(20,random_state=5).as_matrix():
            eef=np.eye(4);eef[:3,:3]=q;eef[:3,3]=[.3,-.2,.1]
            g=eef@T_EEF_GENERATOR
            p1=eef[:3,3]+eef[:3,:3]@T_EEF_GENERATOR[:3,:3]@physical
            p2=g[:3,3]+g[:3,:3]@virtual
            np.testing.assert_allclose(p1,p2,atol=1e-12)
            np.testing.assert_allclose(generator_to_eef(g),eef,atol=1e-12)
            contact=generator_to_contact(g)
            np.testing.assert_allclose(contact[:3,3],eef[:3,3]+.043*eef[:3,2],atol=1e-12)

    def test_contact_override_does_not_mutate_shared_config(self):
        config=SimpleNamespace(scene_config=SimpleNamespace(contact_offset=.02,rest_offset=0.,solver_position_iterations=16))
        new=simulation_config(config)
        self.assertEqual(config.scene_config.contact_offset,.02)
        self.assertEqual(new.scene_config.contact_offset,.001)
        self.assertEqual(new.scene_config.solver_position_iterations,16)
        self.assertEqual(new.scene_config.rest_offset,0.)


if __name__=='__main__':unittest.main()
