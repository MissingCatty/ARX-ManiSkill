"""User D deferral preserves original48 and never turns unrun cases into failures."""
import copy
import os
from pathlib import Path
import unittest
from scripts.curobo_arx.data_factory.common import ROOT,read
from .main48_spec import validate_schedule
from .main48_report import unexecuted_partition,task_counts

FIXTURES=Path(os.environ.get('ARX_TEST_FIXTURE_ROOT',ROOT))


class DeferralTests(unittest.TestCase):
    def setUp(self):
        self.reg=read(FIXTURES/'runs/nearfield_main48_20261110_04/registration.json')
        details=read(FIXTURES/'runs/nearfield_main48_20261110_04/main48_verification.json')['trials']
        detail=read(FIXTURES/'runs/nearfield_main48_boundary_record_20261110_01/boundary_record.json')['detail']
        full=self.reg['registered_trials'];self.reg['carried_details']=details+[detail]
        self.reg['seed_trials']=[t for t in full[22:] if t['method']!='D']
        self.reg['deferred_trials']=[t for t in full[22:] if t['method']=='D']
        self.reg['execution_amendment']={'kind':'user_defer_remaining_D','deferred_methods':['D'],'user_message':'暂停后续D的测试夹具'}

    def test_22_consumed_19_active_7_deferred(self):
        validate_schedule(self.reg)
        self.assertEqual(len(self.reg['carried_details']),22)
        self.assertEqual(len(self.reg['seed_trials']),19)
        self.assertEqual(len(self.reg['deferred_trials']),7)

    def test_paused_D_not_counted_as_failed_or_consumed(self):
        counts=task_counts(self.reg['carried_details'])
        self.assertEqual((counts['physical_successes'],counts['physical_failures'],counts['infrastructure_interruptions']),(8,11,3))
        p=unexecuted_partition(self.reg,{'unexecuted':[t['trial_id'] for t in self.reg['seed_trials']]})
        self.assertEqual(len(p['unexecuted']),26);self.assertEqual(len(p['deferred_unexecuted']),7)
        self.assertEqual(len(unexecuted_partition(self.reg,{'unexecuted':[]})['unexecuted']),7)

    def test_missing_scope_record_or_wrong_method_is_rejected(self):
        for key in ('kind','deferred_methods','user_message'):
            r=copy.deepcopy(self.reg);r['execution_amendment'].pop(key)
            with self.assertRaises(ValueError):validate_schedule(r)
        r=copy.deepcopy(self.reg);r['seed_trials'][0],r['deferred_trials'][0]=r['deferred_trials'][0],r['seed_trials'][0]
        with self.assertRaises(ValueError):validate_schedule(r)

    def test_no_replay_omission_or_duplicate(self):
        for mutation in (lambda r:r['seed_trials'].append(r['registered_trials'][0]),
                         lambda r:r['deferred_trials'].pop(),
                         lambda r:r['deferred_trials'].append(r['deferred_trials'][0])):
            r=copy.deepcopy(self.reg);mutation(r)
            with self.assertRaises(ValueError):validate_schedule(r)

    def test_original_pose_and_relative_execution_order_are_frozen(self):
        # Break shared fixture references before checking content against original48.
        r=copy.deepcopy(self.reg);r['deferred_trials']=copy.deepcopy(r['deferred_trials']);r['deferred_trials'][0]['G0_world'][0][3]+=.001
        with self.assertRaises(ValueError):validate_schedule(r)
        r=copy.deepcopy(self.reg);r['seed_trials'][0],r['seed_trials'][1]=r['seed_trials'][1],r['seed_trials'][0]
        with self.assertRaises(ValueError):validate_schedule(r)

    def test_old_registrations_keep_all_original_methods(self):
        r=read(FIXTURES/'runs/nearfield_main48_20261110_04/registration.json');validate_schedule(r)
        ids=[t['trial_id'] for t in r['seed_trials']]
        self.assertEqual(unexecuted_partition(r,{'unexecuted':ids})['unexecuted'],ids)
        with self.assertRaises(ValueError):unexecuted_partition(self.reg,{'unexecuted':[self.reg['deferred_trials'][0]['trial_id']]})


if __name__=='__main__':unittest.main()
