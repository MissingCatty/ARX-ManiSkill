"""A failed geometric path must try the same original grasp before advancing."""
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import pool as module


class Sequence:
    def __init__(self,*args):self.events=[];self.cursor=0
    def summary(self):return {'events':self.events,'consumed_candidates':self.cursor}


class OriginalSelector:
    def select_contact(self,context,state,directory,validator):
        candidate=module.candidate_pool(directory)['candidates'][0]
        route='geometric' if self.geometric_sequence is not None else 'original'
        key=(candidate['index'],route);self.calls.append(key)
        if self.fail:raise RuntimeError('original planner fault')
        if route=='geometric':
            self.geometric_sequence.events.append({'candidate_index':candidate['index'],
                'accepted_geometric_branch':self.admitted.get(candidate['index'],True)})
            self.geometric_sequence.cursor+=1
        success=self.outcomes.get(key,False)
        return {'success':success,'selected':{'index':0},
            'candidate_events':[{'candidate_index':0,'kind':route}],
            'rejected':{'path':0 if success else 1}}


class FullCandidateRecoveryTests(unittest.TestCase):
    def setUp(self):
        temp=TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.rows={'candidates':[{'index':0},{'index':1}]}
        self.original=self.rows['candidates']
        @contextmanager
        def window(directory,start,stop):
            old=self.rows['candidates'];self.rows['candidates']=old[start:stop]
            try:yield
            finally:self.rows['candidates']=old
        for patcher in [patch.object(module,'candidate_pool',side_effect=lambda p:self.rows),
                        patch.object(module,'candidate_window',side_effect=window),
                        patch.object(module,'GeometricSequence',Sequence)]:
            patcher.start();self.addCleanup(patcher.stop)
        self.pool=object.__new__(module.geometric_pool(OriginalSelector))
        self.pool.geometric_enabled=True;self.pool.geometric_sequence=None;self.pool.geometric_client=None
        self.pool.calls=[];self.pool.outcomes={};self.pool.admitted={};self.pool.fail=False

    def request(self):
        return self.pool.select_contact(SimpleNamespace(out=self.root),{},self.root,None)

    def test_same_grasp_original_path_is_tried_before_next_rank(self):
        self.pool.outcomes[(0,'original')]=True
        result=self.request()
        self.assertTrue(result['success']);self.assertEqual(result['selected']['index'],0)
        self.assertEqual(self.pool.calls,[(0,'geometric'),(0,'original')])
        self.assertTrue(result['candidate_events'][0]['geometric_attempt']['full_candidate_fallback'])

    def test_both_attempts_fail_before_advancing_rank(self):
        self.pool.outcomes[(1,'geometric')]=True
        result=self.request()
        self.assertEqual(self.pool.calls,[(0,'geometric'),(0,'original'),(1,'geometric')])
        self.assertEqual(result['selected']['index'],1)
        self.assertEqual([e['candidate_index'] for e in result['candidate_events']],[0,1])

    def test_original_endpoint_already_used_is_not_repeated(self):
        self.pool.admitted[0]=False;self.pool.outcomes[(1,'geometric')]=True
        self.request();self.assertEqual(self.pool.calls,[(0,'geometric'),(1,'geometric')])

    def test_complete_geometric_success_does_not_add_native_attempt(self):
        self.pool.outcomes[(0,'geometric')]=True
        result=self.request();self.assertEqual(self.pool.calls,[(0,'geometric')])
        self.assertTrue(result['success'])

    def test_fault_cleans_sequence_and_restores_window(self):
        self.pool.fail=True
        with self.assertRaisesRegex(RuntimeError,'original planner fault'):self.request()
        self.assertIsNone(self.pool.geometric_sequence)
        self.assertIs(self.rows['candidates'],self.original)


if __name__=='__main__':unittest.main()
