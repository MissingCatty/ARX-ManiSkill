"""Offline faults for identity order and the planning-to-execution gate."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from .proxy import original_queue,check_original_order
from .registration import physical_gate
from .planning import comparison


class OriginalOrderTests(unittest.TestCase):
    def setUp(self):
        self.payload={'candidates':[{'pose':[[i]],'score':1-i*.1,'width':.088,'metadata':{'id':i}}
                                    for i in range(4)]}

    def test_neither_group_needs_nms_or_changes_candidate_values(self):
        before=deepcopy(self.payload)
        with patch('modules.graspgenx.postprocessing.ordering.native_nms',side_effect=AssertionError('NMS forbidden')):
            for mode in ['baseline','optimized']:
                queue,manifest=original_queue(self.payload)
                self.assertEqual(queue['candidates'],before['candidates'])
                self.assertEqual(manifest['priority_to_original_source'],list(range(4)))
                self.assertEqual(manifest['deferred'],[])
        self.assertEqual(self.payload,before)

    def test_order_corruption_is_rejected(self):
        _,manifest=original_queue(self.payload)
        manifest['priority_to_original_source']=[1,0,2,3]
        with self.assertRaises(ValueError):check_original_order(manifest)

    def test_hidden_deferred_candidates_are_rejected(self):
        _,manifest=original_queue(self.payload)
        manifest['deferred']=[3]
        with self.assertRaises(ValueError):check_original_order(manifest)

    def test_grasp_or_arm_change_fails_identity(self):
        baseline={'success':True,'selected_arm':'right','selected_original_source_index':3,
            'contact_pose_world':[[1]],'joint_endpoints':{'q_pre':[0]},'event_decisions':{}}
        optimized=deepcopy(baseline);optimized['selected_arm']='left'
        result=comparison({'control_1':baseline,'optimized_1':optimized})
        self.assertFalse(result['same_grasp_and_arm'])

    def test_original_success_loss_fails_gate(self):
        baseline={'success':True,'selected_arm':'right','selected_original_source_index':3,
            'contact_pose_world':[[1]],'joint_endpoints':{},'event_decisions':{}}
        optimized={**baseline,'success':False}
        self.assertFalse(comparison({'control_1':baseline,'optimized_1':optimized})['control_success_retained'])


class FollowupTests(unittest.TestCase):
    def setUp(self):
        self.proof={k:True for k in ['passed','record_integrity','control_successes_retained',
            'original_order_verified','original_grasps_and_arms_retained','joint_endpoints_verified','timing_gate_passed']}
        self.proof['checked_requests']=138

    def test_complete_accepted_replay_allows_only_registered_followup(self):
        self.assertTrue(physical_gate(self.proof))

    def test_each_missing_or_failed_proof_stops_followup(self):
        for key in self.proof:
            bad=dict(self.proof);bad[key]=False
            self.assertFalse(physical_gate(bad),key)
            bad=dict(self.proof);bad.pop(key)
            self.assertFalse(physical_gate(bad),key)

    def test_partial_replay_never_starts_physical_trials(self):
        self.assertFalse(physical_gate({**self.proof,'checked_requests':136}))


if __name__=='__main__':unittest.main()
