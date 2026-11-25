"""Check timing populations and the meaning of physical execution."""
from copy import deepcopy
import unittest

from .latency_report import paired_groups


class PairedTimingTest(unittest.TestCase):
    def setUp(self):
        self.ids = ['success','no_candidate','planning_failure','shared_stationing']
        self.rows = {}
        for method in ['graspgenx','zerograsp']:
            self.rows[method] = {sid:{'result':{'success':sid=='success','physical_attempt':sid=='success'},
                'latency':{'common_precondition_failure':sid=='shared_stationing','timings':{
                    'generation_delivery_seconds':1.,'initial_planning_rpc_seconds':2.,
                    'execution_with_replanning_seconds':3.}}} for sid in self.ids}
        del self.rows['zerograsp']['no_candidate']['latency']['timings']['initial_planning_rpc_seconds']

    def test_pairing_intersects_cases_instead_of_unmatched_populations(self):
        groups = paired_groups(self.ids,self.rows)
        row = groups['all_completed']['initial_planning_rpc_seconds']
        self.assertEqual(row['ids'],['success','planning_failure'])
        self.assertEqual(row['graspgenx']['samples'],row['zerograsp']['samples'])
        self.assertEqual(groups['all_completed']['generation_delivery_seconds']['ids'],self.ids[:3])

    def test_execution_requires_real_motion_and_success_group_requires_both(self):
        groups = paired_groups(self.ids,self.rows)
        self.assertEqual(groups['all_completed']['execution_with_replanning_seconds']['ids'],['success'])
        changed = deepcopy(self.rows)
        changed['zerograsp']['success']['result']['success'] = False
        row = paired_groups(self.ids,changed)['both_successful']['execution_with_replanning_seconds']
        self.assertEqual(row['ids'],[])
        self.assertEqual(row['graspgenx']['samples'],0)

    def test_invalid_timing_is_excluded_from_both_columns(self):
        self.rows['graspgenx']['planning_failure']['latency']['timings']['initial_planning_rpc_seconds'] = float('nan')
        row = paired_groups(self.ids,self.rows)['all_completed']['initial_planning_rpc_seconds']
        self.assertEqual(row['ids'],['success'])
        self.assertEqual(row['zerograsp']['samples'],1)


if __name__=='__main__':
    unittest.main()
