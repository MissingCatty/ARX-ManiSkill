"""Unchanged ARX hand bounds, isolated rejection, and nonterminal accounting."""
import copy
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
from scripts.curobo_arx.data_factory.common import read,write
from .advisor33_boundary import RANGE_ERROR,REJECTION,range_rejection
from .advisor33_teacher import Teacher
from .advisor33_launch import terminal_isolation


class Boundary(unittest.TestCase):
    def setUp(self):
        self.updater=types.SimpleNamespace(names=['right_joint27','right_joint28'],
            limits={'right_joint27':(0.,.044),'right_joint28':(0.,.044)})
        self.context={m:(None,None,self.updater,None) for m in ('open','closed')}
        self.request={'action':'plan','phase':'local','label':'d_refine_002',
            'snapshot':{'joint_names':self.updater.names,'joint_positions':[.043996945,.044010553509],
                        'joint_limits':[[0.,.044],[0.,.044]]}}

    def test_actual_boundary_rejects_without_clamping(self):
        before=copy.deepcopy(self.request)
        result=range_rejection(ValueError(RANGE_ERROR),self.request,self.context)
        self.assertEqual(result['rejected'],REJECTION);self.assertTrue(result['requires_world_isolation'])
        self.assertFalse(result['new_command_generated']);self.assertEqual(before,self.request)
        self.assertEqual([r['passes_existing_range'] for r in result['measured_gripper']],[True,False])

    def test_within_original_tolerance_does_not_excuse_other_failure(self):
        self.request['snapshot']['joint_positions'][1]=.0440099
        with self.assertRaisesRegex(ValueError,RANGE_ERROR):range_rejection(ValueError(RANGE_ERROR),self.request,self.context)

    def test_unrelated_error_remains_fatal(self):
        with self.assertRaisesRegex(ValueError,'unrelated'):range_rejection(ValueError('unrelated'),self.request,self.context)

    def test_nonfinite_state_remains_fatal(self):
        self.request['snapshot']['joint_positions'][1]=float('nan')
        with self.assertRaisesRegex(ValueError,'Nonfinite'):range_rejection(ValueError(RANGE_ERROR),self.request,self.context)

    def test_changed_limits_remain_fatal(self):
        self.request['snapshot']['joint_limits'][1][1]=.045
        with self.assertRaisesRegex(RuntimeError,'disagree'):range_rejection(ValueError(RANGE_ERROR),self.request,self.context)

    def test_runtime_teacher_returns_rejection_and_bound_record(self):
        with tempfile.TemporaryDirectory() as name:
            self.request['output']=name;teacher=Teacher.__new__(Teacher);teacher.context=self.context
            with patch('scripts.curobo_arx.nearfield.advisor33_teacher.Original.handle',side_effect=ValueError(RANGE_ERROR)):
                result=teacher.handle(self.request)
            self.assertEqual(result,read(Path(name)/'d_refine_002_measured_gripper_rejection.json'))
            self.assertEqual(result['rejected'],REJECTION)

    def test_boundary_failure_requires_unique_world_isolation_marker(self):
        rows=[{'trial_id':'slot','outcome':'planning_rejected: '+REJECTION}]
        with self.assertRaises(RuntimeError):terminal_isolation(rows,['slot'],None)
        marker={'trial_id':'slot','terminal_records_complete':True,'configuration_intact':True,
                'retry':False,'next_slot_requires_new_World':True}
        self.assertTrue(terminal_isolation(rows,['slot'],marker))
        marker['retry']=True
        with self.assertRaises(RuntimeError):terminal_isolation(rows,['slot'],marker)

    def test_started_episode_cannot_be_reported_as_unexecuted(self):
        from .advisor33_report import report
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);ep=root/'groups/g/trials/started/episodes/episode';ep.mkdir(parents=True)
            write(ep/'pregrasp_command.json',{'id':'issued'})
            write(root/'registration.json',{'seed_trials':[{'trial_id':'started'},{'trial_id':'unused'}],
                'execution_source_files':{},'dependencies':{}})
            with patch('scripts.curobo_arx.nearfield.advisor33_report.validate_trials'), \
                 patch('scripts.curobo_arx.nearfield.advisor33_report.exited',return_value={}), \
                 patch('scripts.curobo_arx.nearfield.advisor33_report.native_mask.projector_for'):
                result=report(root)
            self.assertEqual(result['unknown'],1);self.assertEqual(result['unexecuted'],1)
            self.assertFalse(result['record_integrity_pass'])
            self.assertTrue(result['trials'][0]['physical_attempt'])


if __name__=='__main__':unittest.main()
