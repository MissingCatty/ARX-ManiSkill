"""Offline queue recovery tests; synthetic queue states are not demonstrations."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from subprocess import CompletedProcess
import numpy as np
from .common import sha,write
from .ledger import Ledger
from .resume import copy_and_recover,require_exit_proof,exit_proof


Q={'max_total_dispatches':100,'max_physical_attempts':20,'candidate_reserve':64,
   'goals_per_object':2,'distances_mm':[10,40,60,80],'successes_per_bin':2,
   'physical_per_bin':4,'planning_rejections_per_bin':3}
PROOF={'old_process_exited':True,'environments_idle':True,'containers_absent':True,
       'evidence':{'kind':'offline synthetic exit proof; no simulator involved'}}


class RecoveryTests(unittest.TestCase):
    def late_parent(self,folder):
        parent=self.pre_simulator_parent(folder)
        write(parent/'state.json',{'status':'fault','finished_unix':180,'unresolved_groups':['group_late']})
        write(parent/'group_registrations/group_late.process.json',{'pid':-102,'container':'arx-data-factory-group_late'})
        return parent

    def test_late_ARX_shutdown_requires_receipts_and_preserves_old_unknown_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent=self.late_parent(tmp);before=sha(parent/'state.json')
            with patch('scripts.curobo_arx.data_factory.resume.subprocess.run',return_value=CompletedProcess([],1)):
                with self.assertRaisesRegex(ValueError,'unresolved'):exit_proof(parent)
                with self.assertRaisesRegex(ValueError,'missing durable receipts'):exit_proof(parent,reconcile_late_shutdown=True)
                directory=parent/'groups/group_late'
                write(directory/'block_result.json',{'idle':True})
                write(directory/'shutdown_receipt.json',{'wall':187,'all_trajectories_idle':False,'raw_writer_drained_and_closed':True,'ledger_closed':True})
                with self.assertRaisesRegex(ValueError,'idle and drained'):exit_proof(parent,reconcile_late_shutdown=True)
                write(directory/'shutdown_receipt.json',{'wall':187,'all_trajectories_idle':True,'raw_writer_drained_and_closed':True,'ledger_closed':True})
                proof=exit_proof(parent,reconcile_late_shutdown=True)
            require_exit_proof(proof)
            self.assertEqual(sha(parent/'state.json'),before)
            self.assertTrue(proof['evidence']['late_shutdown_receipts'][0]['old_unresolved_flag_preserved'])

    def test_late_shutdown_never_overrides_existing_ARX_container(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent=self.late_parent(tmp)
            with patch('scripts.curobo_arx.data_factory.resume.subprocess.run',return_value=CompletedProcess([],0)):
                with self.assertRaisesRegex(ValueError,'container is present'):exit_proof(parent,reconcile_late_shutdown=True)

    def test_active_SDG_coordinator_cannot_be_mistaken_for_exited_factory(self):
        import os
        with tempfile.TemporaryDirectory() as tmp:
            parent=self.pre_simulator_parent(tmp)
            write(parent/'coordinator_process.json',{'pid':os.getpid()})
            args=('python -m scripts.curobo_arx.sdg_stage1.run '+str(parent)).encode()
            with patch('scripts.curobo_arx.data_factory.resume.Path.read_bytes',return_value=args):
                with self.assertRaisesRegex(ValueError,'process still exists'):exit_proof(parent)

    def pre_simulator_parent(self,folder):
        parent=Path(folder)
        write(parent/'state.json',{'status':'fault','unresolved_groups':[]})
        write(parent/'STOP',{'reason':'offline fixture'})
        write(parent/'launch.json',{'started_unix':1})
        write(parent/'coordinator_process.json',{'pid':-101})
        return parent

    def test_pre_simulator_failure_requires_positive_coordinator_exit_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent=self.pre_simulator_parent(tmp)
            with patch('scripts.curobo_arx.data_factory.resume.subprocess.run',return_value=CompletedProcess([],0,stdout='')):
                proof=exit_proof(parent)
            require_exit_proof(proof)
            self.assertTrue(proof['evidence']['pre_simulator_exit']['no_simulator_group_registration'])
            (parent/'launch.json').unlink()
            with self.assertRaisesRegex(ValueError,'coordinator/launch evidence'):exit_proof(parent)

    def test_unowned_group_registration_refuses_no_world_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent=self.pre_simulator_parent(tmp)
            write(parent/'group_registrations/group_missing_process.json',{'group_id':'missing'})
            with self.assertRaisesRegex(ValueError,'without virtual environment ownership'):exit_proof(parent)

    def test_orphan_container_or_failed_inventory_refuses_pre_simulator_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent=self.pre_simulator_parent(tmp)
            for result in [CompletedProcess([],0,stdout='arx-data-factory-group_orphan\n'),CompletedProcess([],125,stdout='')]:
                with patch('scripts.curobo_arx.data_factory.resume.subprocess.run',return_value=result):
                    with self.assertRaisesRegex(ValueError,'orphaned factory'):exit_proof(parent)

    def make_parent(self,folder,physical_limit=20):
        source=Path(folder)/'parent.sqlite3';ledger=Ledger(source)
        ledger.initialize({'objects':[{'object_id':'egad_A1'}]},{**Q,'max_physical_attempts':physical_limit})
        ledger.add_goals('egad_A1',{'candidates':[{'goal_id':'g','grasp_world':np.eye(4).tolist(),'group':'top'}]})
        first=ledger.claim('egad_A1','old_owner','env0')
        second=ledger.claim('egad_A1','old_owner','env1')
        ledger.transition(first['id'],'old_owner','executing',physical=True)
        ledger.close();return source,first,second

    def test_missing_exit_or_environment_evidence_refuses_recovery(self):
        for key in ('old_process_exited','environments_idle','containers_absent'):
            for value in (False,'true',None):
                with self.assertRaisesRegex(ValueError,'verified process exit'):
                    require_exit_proof({**PROOF,key:value})

    def test_clone_retains_parent_and_marks_only_child_interrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,first,second=self.make_parent(tmp);before=sha(source)
            out=Path(tmp)/'child.sqlite3';receipt=copy_and_recover(source,out,PROOF)
            self.assertEqual(sha(source),before)
            parent=Ledger(source);child=Ledger(out)
            try:
                self.assertEqual({r['state'] for r in parent.rows()},{'claimed','executing'})
                self.assertEqual({r['state'] for r in child.rows()},{'recovered_interrupted'})
                self.assertEqual(sum(r['physical'] for r in child.rows()),1)
                self.assertEqual(child.db.execute('SELECT COUNT(*) FROM object_leases').fetchone()[0],0)
                self.assertEqual(receipt['inherited_success_ids'],[])
                self.assertEqual({r['meaning'] for r in receipt['recovered']},
                    {'execution_outcome_unknown','interrupted_before_physical_dispatch'})
                new=child.claim('egad_A1','new_owner','new_env')
                self.assertNotIn(new['id'],(first['id'],second['id']))
                self.assertEqual(len(child.rows()),3)
            finally:parent.close();child.close()

    def test_old_physical_budget_is_not_reset_by_continuation(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,_,_=self.make_parent(tmp,physical_limit=1);out=Path(tmp)/'child.sqlite3'
            copy_and_recover(source,out,PROOF);child=Ledger(out)
            try:self.assertIsNone(child.claim('egad_A1','new_owner','new_env'))
            finally:child.close()

    def test_nonempty_destination_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,_,_=self.make_parent(tmp);before=sha(source)
            with self.assertRaisesRegex(ValueError,'new ledger'):
                copy_and_recover(source,source,PROOF)
            other=Path(tmp)/'other';other.mkdir();destination,_,_=self.make_parent(other)
            original=sha(destination)
            with self.assertRaisesRegex(ValueError,'nonempty destination'):
                copy_and_recover(source,destination,PROOF)
            self.assertEqual(sha(source),before);self.assertEqual(sha(destination),original)


if __name__=='__main__':unittest.main()
