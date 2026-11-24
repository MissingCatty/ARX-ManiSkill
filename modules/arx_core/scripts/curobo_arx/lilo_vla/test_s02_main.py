"""Offline adversarial checks for the separately authorized main carryforward."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from interaction_contract import ROOT, read, write, file_sha
from run_s02_main import verify_carryforward, RESOLUTIONS


class MainCarryforwardTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.out=Path(self.temp.name)
        self.schedule=[{**r,'phase':'main'} for r in read(ROOT/'research_plan/configs/s02_design.json')['planned_main_schedule']]
        evidence=self.out/'proof.json';write(evidence,dict(exit_code=0,passed=True,status='passed',simulation_commands=0))
        pilots=[]
        for i,(case,arm) in enumerate((c,a) for c in ('core_egad_A1','core_bbq_bottle') for a in ('R10','R40')):
            result=dict(trial_id=f'pilot__{case}__{arm}',case_id=case,arm=arm,halt=i<3,
                key_event_coverage={k:i!=0 for k in ('observation','policy_input','command_execution')})
            path=self.out/(str(i)+'.json');write(path,result)
            pilots.append(dict(trial_id=result['trial_id'],case_id=case,arm=arm,result_path=str(path),sha256=file_sha(path)))
        carry=dict(main_authorized=True,automatic_retries=0,original_terminal_states_rewritten=False,
            pilots=pilots,evidence_sha256={str(evidence):file_sha(evidence)},offline_checks=str(evidence),warmup_receipt=str(evidence),
            resolutions={k:dict(evidence=[str(evidence)],explanation='Explicit supplemental resolution; old halt retained') for k in RESOLUTIONS})
        self.value=dict(schedule=self.schedule,main_carryforward=carry)

    def tearDown(self):self.temp.cleanup()

    def test_preserves_old_stops_and_accepts_verified_coverage(self):
        before={p['result_path']:file_sha(p['result_path']) for p in self.value['main_carryforward']['pilots']}
        self.assertTrue(verify_carryforward(self.value)['pass_check'])
        self.assertEqual(before,{p:file_sha(p) for p in before})
        self.assertTrue(read(next(iter(before)))['halt'])

    def test_missing_original_receipt_or_missing_coverage_rejected(self):
        bad=copy.deepcopy(self.value);bad['main_carryforward']['pilots'].pop()
        with self.assertRaises(ValueError):verify_carryforward(bad)
        for row in self.value['main_carryforward']['pilots']:
            r=read(row['result_path']);r['key_event_coverage']={};write(row['result_path'],r);row['sha256']=file_sha(row['result_path'])
        with self.assertRaisesRegex(ValueError,'coverage'):verify_carryforward(self.value)

    def test_modified_receipt_or_resolution_evidence_rejected(self):
        p=self.value['main_carryforward']['pilots'][0]['result_path'];write(p,dict(halt=False))
        with self.assertRaisesRegex(ValueError,'receipt'):verify_carryforward(self.value)

    def test_no_authorization_or_fault_resolution_rejected(self):
        for key in ('main_authorized','original_terminal_states_rewritten'):
            bad=copy.deepcopy(self.value);bad['main_carryforward'][key]=not bad['main_carryforward'][key]
            with self.assertRaises(ValueError):verify_carryforward(bad)
        bad=copy.deepcopy(self.value);bad['main_carryforward']['resolutions'].pop('recorder_status_race')
        with self.assertRaises(ValueError):verify_carryforward(bad)

    def test_changed_budget_order_or_pilot_repeat_rejected(self):
        for schedule in (self.schedule[:-1],list(reversed(self.schedule)),[self.schedule[0]]+self.schedule):
            with self.assertRaisesRegex(ValueError,'schedule'):verify_carryforward({**self.value,'schedule':schedule})

    def test_evidence_tampering_rejected(self):
        proof=next(iter(self.value['main_carryforward']['evidence_sha256']))
        write(proof,dict(passed=False))
        with self.assertRaisesRegex(ValueError,'evidence'):verify_carryforward(self.value)


if __name__=='__main__':unittest.main()
