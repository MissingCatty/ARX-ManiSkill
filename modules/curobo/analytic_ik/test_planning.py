"""Whole-candidate fallback, source ordering, and full-open context recovery."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from arx_project.layout import write_json
from modules.curobo.candidate_io.store import candidate_scope,candidate_pool
from modules.curobo.indexed_planning.stage_timing import StageTiming
from .planning import analytic_pool
from .test_bank import FakeSession


class OriginalSelector:
    def select_contact(self,context,state,directory,validator):
        bank=self.analytic_bank;self.calls.append((bank.cursor,bank.enabled))
        if self.scenario=='fallback':
            bank.used=bank.enabled;success=not bank.enabled
        else:
            success=bank.cursor==1
            if not success:validator.geometry.locked={n:.02 for n in validator.geometry.locked}
            if success and any(v!=.044 for v in validator.geometry.locked.values()):
                raise AssertionError('Full-open candidate inherited narrow finger locks')
        result={'success':success,'candidate_events':[{'candidate_index':0}],
            'rejected':{} if success else {'path':1}}
        if success:result['selected']={'index':0}
        return result

    def width_context(self,work,state):
        self.validator.geometry.locked={n:.044 for n in self.validator.geometry.locked}
        self.restores+=1
        return self.context,self.validator


class PlanningTests(unittest.TestCase):
    def run_fixture(self,scenario):
        cls=analytic_pool(OriginalSelector);pool=cls.__new__(cls)
        pool.analytic_enabled=True;pool.analytic_bank=None;pool.analytic_statistics=[]
        pool.stage_timing=StageTiming();pool.prepare_analytic=lambda *args:FakeSession()
        pool.scenario=scenario;pool.calls=[];pool.restores=0
        pool.context=SimpleNamespace(invbase=np.eye(4))
        pool.validator=SimpleNamespace(geometry=SimpleNamespace(locked={'right_joint27':.044,'right_joint28':.044}))
        candidates=[{'pose':np.eye(4).tolist(),'width':width,'score':2-i} for i,width in enumerate([.04,.088])]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_json(root/'candidates/candidate_pool.json',{'candidates':candidates})
            with candidate_scope(root):
                result=pool.select_contact(pool.context,{},root,pool.validator)
                self.assertEqual(candidate_pool(root)['candidates'],candidates)
        self.assertIsNone(pool.analytic_bank)
        return pool,result

    def test_original_full_candidate_fallback_precedes_lower_rank(self):
        pool,result=self.run_fixture('fallback')
        self.assertTrue(result['success']);self.assertEqual(result['selected']['index'],0)
        self.assertEqual(pool.calls,[(0,True),(0,False)])

    def test_rejected_narrow_candidate_restores_full_open_before_next_rank(self):
        pool,result=self.run_fixture('restore')
        self.assertTrue(result['success']);self.assertEqual(result['selected']['index'],1)
        self.assertEqual(pool.calls,[(0,True),(1,True)]);self.assertEqual(pool.restores,1)


if __name__=='__main__':unittest.main()
