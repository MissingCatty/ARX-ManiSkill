import copy
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scripts.curobo_arx.data_factory.common import ROOT,read
from .goal_refinement import offsets,refined_candidate,refining_teacher


class RefinementTests(unittest.TestCase):
    def test_search_has_finite_registered_radius(self):
        values=offsets()
        self.assertEqual(len(values),44)
        self.assertEqual(len({tuple(v) for v in values}),44)
        self.assertTrue(all(0<np.linalg.norm(v)<=.012000001 for v in values))
        self.assertEqual(values,sorted(values,key=lambda p:(round(float(np.linalg.norm(p)),9),tuple(p))))

    def test_actual_arx_record_roundtrip_and_clone_role_translation(self):
        root=ROOT/'runs/graspness_success_20261103_02/M03/wood_block'
        request=read(root/'variant_request.json');observation=read(root/'neural/head_observation.json')
        source=request['pool']['candidates'][0];original=copy.deepcopy(source)
        catalog=read(ROOT/'runs/graspness_success_20261103_02/offline_config.json')['catalog']
        asset=next(a for a in catalog['objects'] if a['object_id']=='wood_block')
        seed=np.array(source['grasp_world']);seed[:3,:3]=seed[:3,:3]@np.diag([-1,-1,1])
        output=refined_candidate(source,seed,[0,0,0],observation,asset)
        self.assertIsNotNone(output);np.testing.assert_allclose(output['grasp_world'],seed,atol=1e-6)
        shifted=copy.deepcopy(observation);translation=np.array([5.,-2.,0.])
        for transform in [shifted['camera_world_optical'],shifted['snapshot']['objects']['test_object']['world']]:
            for axis in range(3):transform[axis][3]+=float(translation[axis])
        second=refined_candidate(source,seed,[0,0,0],shifted,asset)
        np.testing.assert_allclose(second['grasp_world'],output['grasp_world'],atol=1e-6)
        np.testing.assert_allclose(second['network_row'],output['network_row'],atol=1e-6)
        self.assertEqual(source,original)
        with self.assertRaises(ValueError):refined_candidate(source,seed,[.013,0,0],observation,asset)

    def test_existing_qualified_pose_never_starts_fallback(self):
        class Base:
            registration={}
            def qualify_pool(self,request):return {'candidates':[{'goal_id':'unchanged'}]}
        teacher=refining_teacher(Base)()
        with patch('scripts.curobo_arx.graspness_teacher.goal_refinement.generate') as generate:
            for arm in ['baseline','variant']:
                result=teacher.qualify_pool({'pool':{'paired_policy':arm}})
                self.assertEqual(result['candidates'],[{'goal_id':'unchanged'}])
                self.assertEqual(result['model_pool_qualification_calls'],1)
            generate.assert_not_called()


if __name__=='__main__':unittest.main()
