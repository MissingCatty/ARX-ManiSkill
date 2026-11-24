"""Offline fault injection only. No simulator, policy client or GPU initialization."""
import copy
import json
import shutil
from pathlib import Path
import tempfile
import unittest
from interaction_contract import write, ROOT
from s02_gates import EVENTS, VERSION, trial_gate, pilot_coverage, event_inventory
from audit_s02 import audit
from run_s02 import next_trial

class Gates(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        write(self.root/'grasp_report.json',{'error':None})
        self.good=dict(record_integrity='complete',event_applicability={k:'applicable' for k in EVENTS},key_event_coverage={k:True for k in EVENTS})
        self.schedule=[dict(trial_id=f'p{i}',phase='pilot',arm=['R10','R40'][i%2],case_id=['A1','bottle'][i//2]) for i in range(4)]+[dict(trial_id='main',phase='main')]
    def tearDown(self):self.tmp.cleanup()
    def gate(self,error=None,audits=None,physical=None):
        return trial_gate(self.root,error,audits or [self.good,self.good],physical or {'pass':False},True,True,True)
    def seal(self,i,receipt):
        (self.root/'trials'/f'p{i}').mkdir(parents=True,exist_ok=True)
        write(self.root/'trials'/f'p{i}'/'result.json',dict(cleanup_ok=True,**receipt))
    def rejection(self):
        write(self.root/'grasp_report.json',dict(error='No grasp-relative handoff passed frozen continuation and wrist-frustum checks',local_policy_calls=0))
    def test_legitimate_prepolicy_rejection_is_na_not_coverage(self):
        self.rejection();receipt=audit(self.root)
        self.assertEqual(receipt['record_integrity'],'complete')
        self.assertEqual(receipt['event_applicability']['policy_input'],'not_applicable')
        self.assertFalse(any(receipt['key_event_coverage'].values()))
        result=self.gate({'type':'RuntimeError','message':'Policy pipeline exited 1'},[receipt,receipt])
        self.assertFalse(result['halt']);self.assertEqual(result['task_result'],'failed')
        self.seal(0,result);self.assertEqual(next_trial({'schedule':self.schedule},self.root,'p1')['trial_id'],'p1')
    def test_handoff_rejection_with_missing_observation_is_corruption(self):
        self.rejection();write(self.root/'handoff_actual_entry.json',{'pass':False})
        self.assertEqual(audit(self.root)['record_integrity'],'failed')

    def test_missing_prediction_without_rejection_is_record_failure(self):
        receipt=audit(self.root);self.assertEqual(receipt['record_integrity'],'failed')
        self.assertTrue(self.gate(audits=[receipt,receipt])['halt'])
    def test_request_without_prediction_is_not_na(self):
        self.rejection();write(self.root/'policy_0000_request.json',{'request_id':'unanswered'})
        self.assertFalse(audit(self.root)['pass_check'])
    def test_record_corruption_stops_next_pilot_even_with_success(self):
        bad={**self.good,'record_integrity':'failed'}
        result=self.gate(audits=[bad,self.good],physical={'pass':True})
        self.assertEqual(result['task_result'],'success');self.assertTrue(result['halt'])
        self.seal(0,result)
        with self.assertRaises(ValueError):next_trial({'schedule':self.schedule},self.root,'p1')
    def test_timeout_or_unknown_pipeline_failure_stops_after_clean_cleanup(self):
        for kind in ['TimeoutError','ConnectionError','MemoryError','RuntimeError']:
            with self.subTest(kind=kind):
                result=self.gate({'type':kind,'message':'fault'})
                self.assertTrue(result['infrastructure_failure']);self.assertTrue(result['halt'])
    def test_corrupt_grasp_report_is_a_record_fault_not_an_unhandled_exception(self):
        (self.root/'grasp_report.json').write_text('{broken')
        result=self.gate()
        self.assertEqual(result['record_integrity'],'failed');self.assertTrue(result['halt'])

    def test_oom_log_overrides_old_task_failure(self):
        self.rejection();(self.root/'pipeline.log').write_text('CUDA out of memory')
        self.assertTrue(self.gate({'type':'RuntimeError','message':'exit 1'})['halt'])
    def test_task_failure_with_intact_events_can_continue(self):
        write(self.root/'grasp_report.json',dict(error='No bilateral target contact after original 3s closure'))
        result=self.gate({'type':'RuntimeError','message':'exit 1'})
        self.assertFalse(result['halt']);self.assertEqual(result['record_integrity'],'complete')
        self.assertEqual(result['task_result'],'failed')
    def test_all_na_cannot_unlock_main(self):
        self.rejection();na=audit(self.root);result=self.gate(audits=[na,na])
        for i in range(4):self.seal(i,result)
        with self.assertRaisesRegex(ValueError,'coverage'):next_trial({'schedule':self.schedule},self.root,'main')
    def test_missing_one_arm_or_case_cannot_unlock_main(self):
        for covered in [{0,2},{0,1}]:
            results={f'p{i}':dict(key_event_coverage={k:i in covered for k in EVENTS}) for i in range(4)}
            self.assertFalse(pilot_coverage(self.schedule,results)['pass_check'])
    def test_mixed_na_and_failed_grasps_with_coverage_unlock_main(self):
        for i in range(4):
            result=self.gate()
            if i in [1,2]:
                result['event_applicability']={k:'not_applicable' for k in EVENTS};result['key_event_coverage']={k:False for k in EVENTS}
            self.seal(i,result)
        self.assertEqual(next_trial({'schedule':self.schedule},self.root,'main')['trial_id'],'main')
    def assert_inventory_stops(self,directory,missing):
        for pixels in (False,True):
            receipt=audit(directory,pixels=pixels)
            self.assertEqual(receipt['record_integrity'],'failed')
            self.assertTrue(any(missing in e for e in receipt['errors']),receipt)
        # Defense in depth: even stale/mistaken passing audit receipts cannot
        # hide contradictory filesystem or journal evidence from trial_gate.
        result=trial_gate(directory,None,[self.good,self.good],{'pass':False},True,True,True)
        self.assertEqual(result['record_integrity'],'failed');self.assertTrue(result['halt'])
        self.seal(0,result)
        with self.assertRaises(ValueError):next_trial({'schedule':self.schedule},self.root,'p1')

    def test_executed_close_hold_local_with_lost_command_halts(self):
        source=ROOT/'runs/lilo_P3_guarded_20261028_02/ten_mm_r1/A1'
        for label in ('close','hold','local_0000'):
            with self.subTest(label=label),tempfile.TemporaryDirectory() as folder:
                d=Path(folder)
                write(d/'grasp_report.json',{'error':None})
                for kind in ('command','execution','settling'):
                    shutil.copyfile(source/f'{label}_{kind}.json',d/f'{label}_{kind}.json')
                self.assertFalse(event_inventory(d)['errors'])
                (d/f'{label}_command.json').unlink()
                self.assert_inventory_stops(d,label+'_command.json')

    def test_stage_report_detects_loss_of_all_action_files(self):
        write(self.root/'grasp_report.json',{'error':None,'stage_timing':{'close':{'sim_duration_s':3}}})
        self.assert_inventory_stops(self.root,'close_command.json')

    def test_unlabeled_request_ledger_detects_loss_of_all_action_files(self):
        entries=[dict(event=e,request_id='saved-arx-trajectory',action='trajectory',**({'ok':True} if e=='response' else {})) for e in ('intent','sent','response')]
        (self.root/'requests.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in entries))
        self.assert_inventory_stops(self.root,'unmapped command loss')

    def test_corrupt_request_ledger_fails_closed(self):
        for content in ('{truncated','[]'):
            with self.subTest(content=content):
                (self.root/'requests.jsonl').write_text(content)
                self.assert_inventory_stops(self.root,'Invalid trajectory journal')

    def test_missing_execution_receipt_halts(self):
        source=ROOT/'runs/lilo_P3_guarded_20261028_02/ten_mm_r1/A1'
        for kind in ('command','settling'):shutil.copyfile(source/f'close_{kind}.json',self.root/f'close_{kind}.json')
        self.assert_inventory_stops(self.root,'close_execution.json')

    def test_old_receipt_never_silently_passes_v2_gate(self):
        self.seal(0,dict(instrumentation_pass=True,halt=False))
        with self.assertRaises(ValueError):next_trial({'schedule':self.schedule},self.root,'p1')

if __name__=='__main__':unittest.main()
