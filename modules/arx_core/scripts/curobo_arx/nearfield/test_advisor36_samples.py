import copy
import unittest
import numpy as np
from scripts.curobo_arx.data_factory.common import ROOT, read
from .advisor36_samples import generate, grouped_split, audit, require_release


class SampleContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cases=read(ROOT/'runs/nearfield_advisor35_rulefree_20261112_01/canonical_references.json')
        cls.G=next(c['Gcanonical_object'] for c in cases if c['object_id']=='household_hope_milk_carton')
        cls.rows=generate('household_hope_milk_carton',cls.G)
        cls.split=grouped_split(cls.rows)

    def test_finite_counts_and_independent_split(self):
        a=audit(self.rows)
        self.assertTrue(a['pass']); self.assertEqual(a['pair_count'],512)
        self.assertEqual(a['categories'],{'funnel':512,'boundary_position':128,'boundary_rotation':128,'grasp':256})
        self.assertEqual(self.split['split_counts'],{'train':768,'validation':128,'test':128})
        groups={}
        for row in self.rows: groups.setdefault(row['neighborhood_group'],set()).add(row['split'])
        self.assertTrue(all(len(v)==1 for v in groups.values()))

    def test_correct_labels_without_expanding_core(self):
        for r in self.rows:
            core=np.linalg.norm(r['desired_translation_m'])<=.004 and np.linalg.norm(r['desired_rotation_degrees'])<=3.
            self.assertEqual(r['candidate_label']=='GRASP',core)

    def test_complete_axial_coverage_and_exact_pair_symmetry(self):
        a=audit(self.rows)
        self.assertEqual(a['funnel_axial_layers_mm'],{'00_05':64,'05_15':64,'15_30':64,'30_40':64,'40_50':64,'50_65':96,'65_90':96})
        self.assertEqual(a['maximum_ideal_pair_sum_xy_m_rot_deg'],[0.]*5)
        z=[r['desired_translation_m'][2] for r in self.rows if r['category']=='funnel']
        self.assertLess(min(z),.001);self.assertGreater(max(z),.089)

    def test_BBQ_Spam_never_enter_training(self):
        for obj in ('household_hot3d_bbq_sauce_bottle','household_ycb_spam_can'):
            # Same actual ARX fixture suffices to check the split rule; this
            # does not certify either object's geometry or physical grasp.
            r=generate(obj,self.G);grouped_split(r,evaluation_only=True)
            self.assertEqual({x['split'] for x in r},{'unseen_evaluation'})

    def test_broken_pair_and_missing_physical_release_stop_capture(self):
        r=copy.deepcopy(self.rows);r[0]['split']='bad'
        with self.assertRaises(ValueError):audit(r)
        with self.assertRaises(RuntimeError):require_release([], 'household_hope_milk_carton')
        r=[{'object_id':'household_hope_milk_carton','boundary_case':str(i),
            'completed':True,'success':True,'record_integrity_pass':True,'seed_pose_verified':True} for i in range(7)]
        r[-1]['success']=False
        with self.assertRaises(RuntimeError):require_release(r,'household_hope_milk_carton')


if __name__=='__main__': unittest.main()
