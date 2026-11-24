"""Offline ARX-record faults and matched-generator accounting; no GPU or simulation."""
import copy
import unittest
from pathlib import Path
import numpy as np
from scripts.curobo_arx.data_factory.common import ROOT,read
from training.nearfield_arx import closure_recovery
from .attribution_spec import make_trials,validate_trials,fixed_entry_candidates,identity,bind_arm,geometry_prediction,verify_generation
from .attribution_report import paired


class AttributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=ROOT/'runs/nearfield_closedloop30_20261110_01'
        cls.previous=read(root/'registration.json');cls.result=read(root/'closedloop30_verification.json')
        cls.trials=make_trials(cls.previous,cls.result);cls.reg={**cls.previous,'seed_trials':cls.trials}
        ep=next((root/'groups').glob('*/trials/r30_00_01_normal_D/episodes/*'))
        cls.request=read(next((ep/'policy').glob('*_request.json')))

    def test_exact_budget_repeats_and_balanced_arm_order(self):
        validate_trials(self.trials)
        self.assertEqual(len(self.trials),72)
        for arm in ('g0','trained','base'):
            self.assertEqual(sum(t['attribution_arm']==arm for t in self.trials),24)
            self.assertEqual([sum(self.trials[i]['attribution_arm']==arm for i in range(pos,72,3)) for pos in (0,1,2)],[8,8,8])

    def test_mismatched_entry_prior_renderer_or_placement_stops(self):
        for key in ('entry_world','G0_world','renderer','placement'):
            trials=copy.deepcopy(self.trials);trials[1][key]='altered'
            with self.assertRaises(ValueError):validate_trials(trials)

    def test_all_arms_share_fixed_short_spam_entries(self):
        for t in self.trials:
            expected=.05 if t['case_id']=='03_entry_offset' else .04 if t['case_id']=='03_joint_prior_entry_offset' else .08
            self.assertEqual(fixed_entry_candidates(t),[{'backoff_m':expected,'goal_world':t['entry_world']}])

    def request_for(self,arm):
        t=next(t for t in self.trials if t['case_id']=='01_normal' and t['attribution_arm']==arm)
        request=copy.deepcopy(self.request);request['trial_id']=t['trial_id'];request['inputs']['G0_world']=copy.deepcopy(t['G0_world'])
        return request

    def test_geometry_decodes_to_G0_from_actual_ARX_pose(self):
        request=self.request_for('g0');i=request['inputs'];token=geometry_prediction(i)
        goal=closure_recovery.decode('D',i['eef_world'],token,i['G0_world'])
        np.testing.assert_allclose(goal,i['G0_world'],atol=2e-6,rtol=0)

    def test_geometry_does_not_consume_Gstar_or_labels(self):
        i=copy.deepcopy(self.request_for('g0')['inputs']);token=geometry_prediction(i)
        i['Gstar_world']='forbidden';i['labels']={'D':'GRASP'}
        self.assertEqual(geometry_prediction(i),token)

    def test_wrong_model_identity_is_rejected(self):
        request=self.request_for('base');reply={**identity(self.reg,'trained'),'prediction':'GRASP'}
        with self.assertRaises(ValueError):verify_generation(self.reg,request,reply)

    def test_geometry_is_not_credited_as_model_inference(self):
        request=self.request_for('g0');reply={**identity(self.reg,'g0'),'prediction':geometry_prediction(request['inputs'])}
        self.assertEqual(verify_generation(self.reg,request,reply),'g0')
        reply['model_inference']=True
        with self.assertRaises(ValueError):verify_generation(self.reg,request,reply)

    def test_foreign_trial_or_modified_G0_stops(self):
        request=self.request_for('trained');request['trial_id']='foreign'
        with self.assertRaises(ValueError):bind_arm(self.reg,request)
        request=self.request_for('trained');request['inputs']['G0_world'][0][3]+=.01
        with self.assertRaises(ValueError):bind_arm(self.reg,request)

    def test_changed_geometry_prediction_rejected(self):
        request=self.request_for('g0');reply={**identity(self.reg,'g0'),'prediction':'GRASP'}
        with self.assertRaises(ValueError):verify_generation(self.reg,request,reply)

    def test_unknown_arm_rejected(self):
        with self.assertRaises(ValueError):identity(self.reg,'unknown')

    def test_missing_pair_is_not_counted_as_failure(self):
        rows=[{'trial_id':'one','attribution_arm':'trained','case_id':'case','repeat':1,'task_success':True}]
        result=paired(rows,'g0');self.assertEqual(result['complete_pairs'],0);self.assertIsNone(result['difference_fraction'])

    def test_rescues_and_regressions_both_retained(self):
        rows=[]
        for case,a,b in [('rescue',False,True),('regression',True,False),('both',True,True)]:
            for arm,ok in [('g0',a),('trained',b)]:rows.append({'trial_id':case+arm,'case_id':case,'repeat':1,'attribution_arm':arm,'task_success':ok})
        result=paired(rows,'g0');self.assertEqual((result['rescued'],result['regressed'],result['net_difference']),(1,1,0))


if __name__=='__main__':unittest.main()
