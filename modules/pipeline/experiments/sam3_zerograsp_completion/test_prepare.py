"""Fault tests for the explicitly bounded retry authorization."""
from copy import deepcopy
import unittest

from .prepare import completion_slots


class MakeupAuthorizationTest(unittest.TestCase):
    def setUp(self):
        self.ids = [f'case_{i:02d}' for i in range(74)]
        self.previous = {'cohort':{'pairs':[{'id':i} for i in self.ids]}}
        self.authorized = self.ids[-2:]
        self.acceptance = {'passed':True,'record_integrity':True,'results':[
            {'method':m,'id':i,'state':'interrupted' if m=='zerograsp' and i in self.authorized else 'complete'}
            for m in ['graspgenx','zerograsp'] for i in self.ids]}

    def test_only_interrupted_cases_receive_new_attempts(self):
        original,slots=completion_slots(self.previous,self.acceptance,self.authorized)
        self.assertEqual(original['zerograsp']['interrupted'],self.authorized)
        self.assertEqual(slots['zerograsp']['not_started'],self.authorized)
        self.assertEqual(len(slots['zerograsp']['complete']),72)
        self.assertEqual(slots['graspgenx']['not_started'],[])
        self.assertEqual(len(slots['graspgenx']['complete']),74)

    def test_cannot_retry_an_already_completed_case(self):
        with self.assertRaises(RuntimeError):
            completion_slots(self.previous,self.acceptance,[self.ids[0],self.authorized[1]])

    def test_graspgenx_must_finish_first(self):
        accepted=deepcopy(self.acceptance)
        accepted['results'][0]['state']='not_started'
        with self.assertRaises(RuntimeError):
            completion_slots(self.previous,accepted,self.authorized)

    def test_record_fault_or_duplicate_authorization_blocks(self):
        accepted=deepcopy(self.acceptance);accepted['record_integrity']=False
        with self.assertRaises(RuntimeError):
            completion_slots(self.previous,accepted,self.authorized)
        with self.assertRaises(RuntimeError):
            completion_slots(self.previous,self.acceptance,[self.ids[-1]]*2)


if __name__=='__main__':
    unittest.main()
