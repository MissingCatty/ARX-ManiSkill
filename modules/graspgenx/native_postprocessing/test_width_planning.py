"""Protect width propagation and real collision-lock consistency."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import contextmanager
import numpy as np
from .width_planning import FINGERS,target_state,binding_proof,width_pool


class WidthTests(unittest.TestCase):
    def state(self):return {'joint_names':['arm_joint',*FINGERS],'qpos':[.4,.044,.044]}

    def test_target_state_only_changes_two_fingers(self):
        old=self.state();copy=deepcopy(old);new=target_state(old,.05)
        self.assertEqual(new['qpos'],[.4,.025,.025]);self.assertEqual(old,copy)

    def test_out_of_stroke_and_nonfinite_widths_fail(self):
        for width in [0,-.01,.089,np.nan]:
            with self.assertRaises(ValueError):target_state(self.state(),width)

    def test_numpy_sweep_width_is_saved_as_plain_float(self):
        import yaml
        value=target_state(self.state(),np.float64(.06))
        self.assertIs(type(value['qpos'][1]),float)
        self.assertEqual(yaml.safe_load(yaml.safe_dump(value)),value)

    def test_wide_collision_model_cannot_claim_narrow_candidate(self):
        validator=SimpleNamespace(geometry=SimpleNamespace(locked={n:.044 for n in FINGERS}))
        with self.assertRaises(RuntimeError):binding_proof(validator,.05,'model.yml')
        self.assertEqual(binding_proof(validator,.088,'model.yml')['open_finger_m'],.044)

    def test_mixed_widths_rebind_after_failure_and_preserve_original_indices(self):
        candidates=[{'width':.05,'metadata':{'original_native_rank':4}},
                    {'width':.088,'metadata':{'original_native_rank':8}}]
        view={'candidates':candidates};bound=[]
        class Base:
            def bind(self,directory,state,**flags):
                value=state['qpos'][1];bound.append(value)
                return SimpleNamespace(model=str(directory),opening=value)
            def contact_view(self,context,state):
                return SimpleNamespace(geometry=SimpleNamespace(locked={n:state['qpos'][1] for n in FINGERS}))
            def select_contact(self,context,state,directory,validator):
                if state['qpos'][1]<.044:return {'success':False,'candidate_events':[{'candidate_index':0}], 'rejected':{'grasp_ik':1}}
                return {'success':True,'selected':{'index':0},'candidate_events':[{'candidate_index':0}], 'rejected':{}}
        @contextmanager
        def window(directory,start,stop):
            old=view['candidates'];view['candidates']=old[start:stop]
            try:yield
            finally:view['candidates']=old
        pool=width_pool(Base)();state=self.state();context=SimpleNamespace(model='open',opening=.044)
        validator=pool.contact_view(context,state)
        with patch('modules.graspgenx.native_postprocessing.width_planning.candidate_pool',lambda _:view), \
             patch('modules.graspgenx.native_postprocessing.width_planning.candidate_window',window), \
             patch('modules.graspgenx.native_postprocessing.width_planning.write_json'):
            result=pool.select_contact(context,state,'/tmp/unused',validator)
        self.assertEqual(bound,[.025,.044]);self.assertEqual(result['selected']['index'],1)
        self.assertEqual(result['selected']['open_finger_m'],.044)
        self.assertEqual([e['candidate_index'] for e in result['candidate_events']],[0,1])


if __name__=='__main__':unittest.main()
