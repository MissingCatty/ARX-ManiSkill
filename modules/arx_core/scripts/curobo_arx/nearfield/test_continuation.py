import sqlite3
import tempfile
import unittest
from pathlib import Path
from scripts.curobo_arx.data_factory.common import ROOT,write,sha
from .continuation import admitted_carry,remaining_slots,relocated_dependencies,unused_startup


class ContinuationTests(unittest.TestCase):
    def test_real_milk_is_carried_and_only_unused_mustard_is_scheduled(self):
        root=ROOT/'runs/nearfield_stage1_20261105_02'
        result,files=admitted_carry(root/'replay_02',root/'offline_admission_recheck.json')
        self.assertEqual(result['new_physical_dispatches'],0)
        left=remaining_slots(root/'replay_02',[result])
        self.assertEqual([x['object_id'] for x in left],['mustard_bottle'])
        self.assertIn(str(root/'replay_02/STOP'),files)

    def test_started_or_unknown_slot_cannot_be_replayed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'trial_ledgers').mkdir()
            write(root/'registration.json',{'seed_trials':[{'trial_id':'a'}]})
            db=sqlite3.connect(root/'trial_ledgers/a.sqlite3');db.execute('CREATE TABLE attempts(id TEXT)')
            db.execute("INSERT INTO attempts VALUES ('started')");db.commit();db.close()
            with self.assertRaises(ValueError):remaining_slots(root,[])

    def test_hash_preserving_document_relocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'chat';(p/'gpt回复').mkdir(parents=True)
            new=p/'gpt回复/gpt回应7.md';new.write_text('fixed spec')
            files,receipts=relocated_dependencies({str(p/'gpt回应7.md'):sha(new)})
            self.assertIn(str(new),files);self.assertEqual(len(receipts),1)
            with self.assertRaises(ValueError):relocated_dependencies({str(p/'gpt回应7.md'):'changed'})

    def test_startup_failure_preserves_budget_and_refuses_claimed_slots(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'trial_ledgers').mkdir()
            write(p/'registration.json',{'deadline_unix':123,'seed_trials':[{'trial_id':'a'}]})
            write(p/'state.json',{'status':'fault','unconfirmed_world':False});write(p/'STOP',{'reason':'CUDA'})
            db=sqlite3.connect(p/'trial_ledgers/a.sqlite3');db.execute('CREATE TABLE attempts(id TEXT)');db.commit()
            self.assertEqual(unused_startup(p)['deadline_unix'],123)
            db.execute("INSERT INTO attempts VALUES ('claimed')");db.commit();db.close()
            with self.assertRaises(ValueError):unused_startup(p)

    def test_known_unissued_capture_claim_needs_explicit_verified_repair(self):
        root=ROOT/'runs/nearfield_stage1_20261105_03/mustard_05'
        with self.assertRaises(ValueError):unused_startup(root)
        proof=unused_startup(root,recover_unissued_claim=True)
        self.assertEqual(proof['physical_commands'],0)
        self.assertEqual(proof['claimed_attempts'],1)
        self.assertEqual(len(proof['unissued_claims']),1)

    def test_real_started_mustard_cannot_use_unissued_claim_repair(self):
        root=ROOT/'runs/nearfield_stage1_20261105_03/mustard_07'
        with self.assertRaises(ValueError):unused_startup(root,recover_unissued_claim=True)

    def test_new_authorized_slot_retains_the_original_physical_fault(self):
        from .continuation import authorized_supplement
        from scripts.curobo_arx.data_factory.common import read
        root=ROOT/'runs/nearfield_stage1_20261105_03/mustard_07'
        auth=ROOT/'runs/nearfield_stage1_20261105_04/authorization.json'
        trial,prior,files=authorized_supplement(root,auth)
        self.assertEqual(trial['object_id'],'mustard_bottle')
        self.assertEqual(prior['physical_dispatches'],1)
        self.assertTrue(prior['retained_in_prior_denominator'])
        self.assertIn(str(root/'STOP'),files)
        with tempfile.TemporaryDirectory() as tmp:
            changed=Path(tmp)/'auth.json';write(changed,{**read(auth),'additional_validation_slots':2})
            with self.assertRaises(ValueError):authorized_supplement(root,changed)

    def test_unissued_claim_repair_rejects_command_history_or_missing_requests(self):
        from unittest.mock import patch
        from . import continuation
        original=continuation.read
        root=ROOT/'runs/nearfield_stage1_20261105_03/mustard_05'
        for failure in ('occurred_command','missing_requests','motion_planner_request'):
            def corrupted(path):
                if Path(path).name=='requests.json':
                    if failure=='missing_requests':raise FileNotFoundError(path)
                    if failure=='occurred_command':return [{'label':'close'}]
                value=original(path)
                if failure=='motion_planner_request' and Path(path).suffix=='.done':
                    return {**value,'action':'seed_connect'}
                return value
            with self.subTest(failure=failure),patch.object(continuation,'read',side_effect=corrupted):
                with self.assertRaises((ValueError,FileNotFoundError)):
                    unused_startup(root,recover_unissued_claim=True)

    def test_cuda_failure_receipt_cannot_be_mistaken_for_success(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from .resources import probe_cuda
        with patch('subprocess.run',return_value=SimpleNamespace(stdout='{"pass":true}\n',stderr='CUDA failure',returncode=1)):
            self.assertFalse(probe_cuda({'uuid':'test'}, {})['pass'])

    def test_cuda_timeout_produces_a_terminal_failure_receipt(self):
        from unittest.mock import patch
        import subprocess
        from .resources import probe_cuda
        with patch('subprocess.run',side_effect=subprocess.TimeoutExpired('probe',60)):
            result=probe_cuda({'uuid':'test'}, {})
        self.assertFalse(result['pass']);self.assertEqual(result['timeout_seconds'],60)

    def test_resume_window_cannot_extend_recovery_or_implicitly_restart(self):
        from .prepare_v2 import prepare
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            for kind,startup,seconds,reason in (
                ('recovery',p/'old',7200,'explicit resume'),
                ('replay',None,7200,'explicit resume'),
                ('replay',p/'old',7201,'explicit resume'),
                ('replay',p/'old',7200,None),
            ):
                with self.subTest(kind=kind,startup=startup,seconds=seconds,reason=reason):
                    with self.assertRaises(ValueError):
                        prepare(p/'parent',p/'out',kind,p/'mask',startup_of=startup,
                            resume_window_seconds=seconds,resume_reason=reason)
                    self.assertFalse((p/'out').exists())


if __name__=='__main__':unittest.main()
