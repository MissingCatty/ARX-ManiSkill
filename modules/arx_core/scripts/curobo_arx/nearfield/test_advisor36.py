import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from .advisor36_contract import interpret,instruction,boundary_poses,canonical_release,token
from .runtime import audit_actions


class Advisor36Tests(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parents[3]
        items=json.loads((root/'runs/nearfield_advisor35_rulefree_20261112_01/canonical_references.json').read_text())
        self.G=np.asarray(next(v['Gcanonical_object'] for v in items if 'milk' in v['object_id']))

    def test_full_80mm_and_30degree_output_preserved(self):
        result=interpret(self.G,'REFINE 10 -5 80 0 30 0')
        delta=np.linalg.inv(self.G)@np.asarray(result['goal_world'])
        np.testing.assert_allclose(delta[:3,3],[.01,-.005,.08],atol=1e-12)
        np.testing.assert_allclose(Rotation.from_matrix(delta[:3,:3]).as_rotvec(degrees=True),[0,30,0],atol=1e-10)

    def test_anomalies_rejected_not_clipped(self):
        for s in ('REFINE 0 0 121 0 0 0','REFINE 0 0 0 0 46 0'):
            result=interpret(self.G,s);self.assertEqual(result['kind'],'reject');self.assertIsNone(result['goal_world'])
        with self.assertRaises(ValueError):interpret(self.G,'REFINE nan 0 0 0 0 0')

    def test_only_explicit_grasp_closes(self):
        self.assertEqual(interpret(self.G,'REFINE 0 0 0 0 0 0')['kind'],'move')
        self.assertEqual(interpret(self.G,'GRASP')['kind'],'close')

    def test_no_budget_in_either_prompt_component(self):
        s=instruction({'near_m':.035,'far_m':.327})
        self.assertNotIn('budget',s);self.assertNotIn('remaining_corrections',s)
        self.assertIn('Local state: {"stopped":true}',s)

    def test_six_coupled_boundary_cases_and_center(self):
        cases=boundary_poses(self.G);self.assertEqual(len(cases),7)
        for c in cases[1:]:
            delta=np.linalg.inv(self.G)@np.asarray(c['goal_world'])
            self.assertAlmostEqual(np.linalg.norm(delta[:3,3]),.004)
            self.assertAlmostEqual(Rotation.from_matrix(delta[:3,:3]).magnitude(),np.deg2rad(3))
        for a,b in zip(cases[1::2],cases[2::2]):
            np.testing.assert_allclose(np.array(a['translation_m'])+b['translation_m'],0)
            np.testing.assert_allclose(np.array(a['rotation_degrees'])+b['rotation_degrees'],0)

    def test_release_requires_all_seven_actual_successes(self):
        rows=[{'boundary_case':c['case'],'completed':True,'success':True,'record_integrity_pass':True,'seed_pose_verified':True}
              for c in boundary_poses(self.G)]
        self.assertTrue(canonical_release(rows));self.assertFalse(canonical_release(rows[:6]))
        rows[2]['success']=False;self.assertFalse(canonical_release(rows))

    def test_execution_without_command_is_record_fault(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);(p/'close_execution.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'Orphan execution'):audit_actions(p,[])
            with self.assertRaisesRegex(RuntimeError,'Missing occurred-action record'):audit_actions(p,[{'label':'close'}])

    def test_grasp_core_labels_do_not_expand(self):
        e=self.G.copy();e[:3,3]+=self.G[:3,0]*.0041
        with self.assertRaises(ValueError):token(e,self.G,grasp=True)


if __name__=='__main__':unittest.main()
