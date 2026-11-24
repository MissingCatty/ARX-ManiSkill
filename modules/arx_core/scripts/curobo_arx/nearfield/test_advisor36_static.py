import copy
import unittest
from scripts.curobo_arx.data_factory.common import ROOT,read
from .advisor36_samples import generate,grouped_split,require_release
from .advisor36_contract import boundary_poses
from .advisor36_static_launch import pilot_ids
from .advisor36_export import pair_quality


class StaticAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cases=read(ROOT/'runs/nearfield_advisor35_rulefree_20261112_01/canonical_references.json')
        cls.G=next(c['Gcanonical_object'] for c in cases if c['object_id']=='household_hope_milk_carton')
        cls.rows=generate('household_hope_milk_carton',cls.G);grouped_split(cls.rows)

    def test_pilot_belongs_to_original_budget_with_complete_pairs(self):
        ids=pilot_ids(self.rows);self.assertEqual(len(set(ids)),16)
        rows=[r for r in self.rows if r['sample_id'] in ids]
        pairs={r['pair_id'] for r in rows};self.assertEqual(len(pairs),8)
        self.assertEqual({r['category'] for r in rows},{'funnel','boundary_position','boundary_rotation','grasp'})
        self.assertIn('65_90',{r['axial_layer_mm'] for r in rows})
        self.assertEqual(len([r for r in self.rows if r['sample_id'] not in ids]),1008)

    def pair(self):
        return [{'candidate':r,'capture_admitted':True,'split':r['split'],
                 'actual_translation_m':r['desired_translation_m'],
                 'actual_rotation_degrees':r['desired_rotation_degrees']} for r in copy.deepcopy(self.rows[:2])]

    def test_actual_pair_is_not_numerically_forced_to_zero(self):
        rows=self.pair();rows[0]['actual_translation_m'][0]+=.0002
        result=pair_quality(rows);self.assertTrue(result['pass'])
        self.assertAlmostEqual(result['actual_pair_translation_residual_m'][0],.0002)
        rows[0]['actual_translation_m'][0]+=.001
        self.assertFalse(pair_quality(rows)['pass'])

    def test_one_bad_or_missing_member_rejects_whole_pair(self):
        rows=self.pair();rows[1]['capture_admitted']=False
        self.assertFalse(pair_quality(rows)['pass']);self.assertFalse(pair_quality(rows[:1])['pass'])

    def test_split_leak_is_record_fault(self):
        rows=self.pair();rows[1]['split']='different'
        with self.assertRaises(ValueError):pair_quality(rows)

    def test_physical_release_requires_actual_execution_and_exact_seven_cases(self):
        obj='household_hope_milk_carton'
        rows=[{'object_id':obj,'boundary_case':c['case'],'completed':True,'executed':True,
               'success':True,'record_integrity_pass':True,'seed_pose_verified':True} for c in boundary_poses(self.G)]
        self.assertEqual(len(require_release(rows,obj)),7)
        rows[0]['executed']=False
        with self.assertRaises(RuntimeError):require_release(rows,obj)
        rows[0]['executed']=True;rows[0]['boundary_case']='wrong'
        with self.assertRaises(RuntimeError):require_release(rows,obj)


if __name__=='__main__':unittest.main()
