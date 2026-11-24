"""Offline registration and control-corruption tests using actual ARX evidence."""
import copy
from pathlib import Path
import tempfile
import unittest
from scripts.curobo_arx.data_factory.common import ROOT,read,write
from .visual_grounding_spec import make_trials,validate_trials,expected_reset,alignment,first_decision,verify_reply
from .attribution_spec import geometry_prediction
from .visual_grounding_campaign import prerequisites


class VisualGroundingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=ROOT/'runs/nearfield_closedloop30_20261110_01'
        cls.reg=read(cls.root/'registration.json');cls.trials,cls.seeds=make_trials(cls.reg)

    def test_fixed_prior_entrance_and_exact24_no_retries(self):
        validate_trials(self.trials)
        for t in self.trials:
            old=next(s for s in self.reg['seed_trials'] if s['trial_id']==t['historical_trial_id'])
            self.assertEqual(t['G0_world'],old['G0_world']);self.assertEqual(t['entry_world'],old['entry_world'])
            self.assertAlmostEqual(t['placement']['placement']['position'][0]-old['placement']['placement']['position'][0],t['target_delta_world_m'][0])
            self.assertEqual(t['placement']['placement']['position'][1:],old['placement']['placement']['position'][1:])

    def test_different_branch_placement_or_prior_rejected(self):
        for key in ('G0_world','entry_world','placement'):
            rows=copy.deepcopy(self.trials);rows[1][key]=None
            with self.assertRaises(ValueError):validate_trials(rows)

    def test_reset_changes_only_declared_target_and_placement_metadata(self):
        for t in self.trials:
            seed=self.seeds[t['seed_id']];changed=expected_reset(seed,t)
            from .verify_validation import reset_signature
            old=reset_signature(read(Path(seed['episode'])/'initial_conditions.json'))
            delta=copy.deepcopy(changed);delta['collection_placement']=old['collection_placement']
            delta['objects']['asset:'+t['object_id']]=old['objects']['asset:'+t['object_id']]
            self.assertEqual(delta,old)

    def test_stale_pose_gate_rejects_unmatched_view(self):
        t=next(t for t in self.reg['seed_trials'] if t['method']=='D' and t['condition']=='normal')
        ledger=next(self.root.glob('groups/*/trials/'+t['trial_id']+'/episodes/*/policy_requests.json'))
        p=Path(read(ledger)[0]['request']);req=read(p);donor={'request':str(p)}
        self.assertTrue(first_decision(req));self.assertTrue(alignment(req,donor)['pass'])
        changed=copy.deepcopy(req);changed['inputs']['eef_world'][0][3]+=.01
        self.assertFalse(alignment(changed,donor)['pass'])
        with tempfile.TemporaryDirectory() as tmp:
            changed=copy.deepcopy(req);observation=read(req['observation']);observation['intrinsics'][0][0]+=10
            file=Path(tmp)/'observation.json';write(file,observation);changed['observation']=str(file)
            self.assertFalse(alignment(changed,donor)['pass'])
        changed=copy.deepcopy(req);changed['inputs']['decision_history_by_method']['D']['remaining_refines']=1
        self.assertFalse(alignment(changed,donor)['pass'])

    def test_live_parent_never_opens_next_physical_campaign(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'parent',Path(tmp)/'offline';a.mkdir();b.mkdir()
            write(a/'state.json',{'status':'running'});write(b/'state.json',{'status':'complete'})
            self.assertFalse(prerequisites(a,b))
            write(a/'STOP',{'reason':'test_retained_stop'})
            with self.assertRaises(RuntimeError):prerequisites(a,b)

    def reply_fixture(self,arm):
        trial=next(t for t in self.trials if t['vision_arm']==arm)
        original=next(t for t in self.reg['seed_trials'] if t['trial_id']==trial['historical_trial_id'])
        ledger=next(self.root.glob('groups/*/trials/'+original['trial_id']+'/episodes/*/policy_requests.json'))
        path=Path(read(ledger)[0]['request']);req=read(path);req['trial_id']=trial['trial_id']
        donor={'request':str(path),'images':read(path.with_name(req['request_id']+'_input.json'))['images']}
        reg={**self.reg,'seed_trials':self.trials,'visual_donors':{trial['object_id']:donor}}
        supplied=[str(path.with_name(req['request_id']+'_'+name+'.png')) for name in ('rgb','depth')]
        reply={'vision_arm':arm,'model_inference':arm!='g0','checkpoint':reg['learned_policy']['checkpoints']['D'] if arm!='g0' else None,
            'prediction':geometry_prediction(req['inputs']),
            'visual_control':{'applicable':True,'replaced':arm=='stale_first','alignment':alignment(req,donor),
                              'model_images':donor['images'] if arm=='stale_first' else supplied}}
        return reg,req,reply

    def test_generator_and_fresh_image_substitution_are_rejected(self):
        reg,req,reply=self.reply_fixture('fresh');verify_reply(reg,req,reply)
        reply['visual_control']['model_images']=['unregistered_rgb','unregistered_depth']
        with self.assertRaises(ValueError):verify_reply(reg,req,reply)
        reg,req,reply=self.reply_fixture('g0');verify_reply(reg,req,reply)
        reply['model_inference']=True
        with self.assertRaises(ValueError):verify_reply(reg,req,reply)

    def test_stale_image_cannot_continue_into_second_decision(self):
        reg,req,reply=self.reply_fixture('stale_first');verify_reply(reg,req,reply)
        req['inputs']['decision_history_by_method']['D']['calls_completed']=1
        with self.assertRaises(ValueError):verify_reply(reg,req,reply)


if __name__=='__main__':unittest.main()
