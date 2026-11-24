import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.curobo_arx.data_factory.common import sha
from scripts.curobo_arx.data_factory.teacher import Rejection
from .finger_overlap import FingerOverlap, guarded_teacher, penetration


def cube():
    return {'link':'right_link27','bounds':[[-.01,-.01,-.01],[.01,.01,.01]],
            'equations':[[1,0,0,-.01],[-1,0,0,-.01],[0,1,0,-.01],
                         [0,-1,0,-.01],[0,0,1,-.01],[0,0,-1,-.01]]}


class OverlapTests(unittest.TestCase):
    def test_interior_boundary_and_empty_space(self):
        p=cube()
        self.assertAlmostEqual(penetration([[0,0,0]],p['bounds'],p['equations']),.01)
        self.assertEqual(penetration([[.01,0,0],[1,1,1]],p['bounds'],p['equations']),0)
        with self.assertRaises(ValueError):penetration([[float('nan'),0,0]],p['bounds'],p['equations'])

    def test_live_finger_transform_and_target_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.json';path.write_text(json.dumps({'version':'authored_arx_finger_containment_v1',
                'parts':[cube()],'unresolved_flat_parts':['flat'],'source_files':{}}))
            checker=FingerOverlap({'path':str(path),'sha256':sha(path),'minimum_detected_penetration_m':.0001})
            eef=np.eye(4);eef[:3,:3]=Rotation.from_euler('xyz',[.3,.5,.8]).as_matrix();eef[:3,3]=[2,3,4]
            finger=np.eye(4);finger[1,3]=.04
            goal=np.eye(4);goal[:3,3]=[3,1,2]
            obj=goal@finger
            snapshot={'link_world':{'right_eef_link':eef.tolist(),'right_link27':(eef@finger).tolist()},
                      'objects':{'test_object':{'world':obj.tolist()}}}
            result=checker.inspect(goal,snapshot,[[0,0,0]])
            self.assertTrue(result['reject_proposal']);self.assertFalse(result['collision_free_certificate'])
            snapshot['objects']['test_object']['world'][0][3]+=.1
            result=checker.inspect(goal,snapshot,[[0,0,0]])
            self.assertFalse(result['reject_proposal']);self.assertEqual(result['unresolved_flat_parts'],1)
            with self.assertRaises(RuntimeError):FingerOverlap({'path':str(path),'sha256':'wrong','minimum_detected_penetration_m':.0001})

    def test_variant_rejects_before_original_gates_and_cleans_context(self):
        class Base:
            original_calls=0
            registration={'authored_finger_proposal_filter':{}}
            def qualify_pool(self,request):return self.nominal_closed(np.eye(4))
            def nominal_closed(self,*args,**kwargs):self.original_calls+=1;return 'original gates'
            def visual_target(self,request):
                class Target:
                    class mesh:vertices=np.zeros((1,3))
                return Target()
        teacher=guarded_teacher(Base)()
        with patch('scripts.curobo_arx.graspness_teacher.finger_overlap.FingerOverlap') as model:
            model.return_value.inspect.return_value={'reject_proposal':True}
            with self.assertRaises(Rejection):teacher.qualify_pool({'pool':{'paired_policy':'variant'},'snapshot':{}})
            self.assertIsNone(teacher.proposal_request);self.assertEqual(teacher.original_calls,0)
            self.assertEqual(teacher.qualify_pool({'pool':{'paired_policy':'baseline'}}),'original gates')
            self.assertEqual(teacher.nominal_closed(np.eye(4)),'original gates')
            model.return_value.inspect.return_value={'reject_proposal':False}
            self.assertEqual(teacher.qualify_pool({'pool':{'paired_policy':'variant'},'snapshot':{}}),'original gates')
            self.assertEqual(teacher.original_calls,3)

    def test_production_uses_variant_but_explicit_baseline_stays_original(self):
        class Base:
            registration={'production':True,'authored_finger_proposal_filter':{}}
            def qualify_pool(self,request):return self.nominal_closed(np.eye(4))
            def nominal_closed(self,*args,**kwargs):return 'original'
            def visual_target(self,request):
                class Target:
                    class mesh:vertices=np.zeros((1,3))
                return Target()
        teacher=guarded_teacher(Base)()
        with patch('scripts.curobo_arx.graspness_teacher.finger_overlap.FingerOverlap') as model:
            model.return_value.inspect.return_value={'reject_proposal':True}
            with self.assertRaises(Rejection):teacher.qualify_pool({'pool':{},'snapshot':{}})
            self.assertEqual(teacher.qualify_pool({'pool':{'paired_policy':'baseline'}}),'original')


if __name__=='__main__':unittest.main()
