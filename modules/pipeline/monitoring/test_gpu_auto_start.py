"""Offline dispatch tests; no GPU allocation or real process startup."""
import tempfile
from pathlib import Path
import unittest

from arx_project.layout import ROOT, write_json
from modules.pipeline.monitoring.gpu_auto_start import choose, tick


class GPUAutoStartTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root/'claims').mkdir()
        write_json(self.root/'progress.json', {'phase':'registered'})
        write_json(self.root/'registration.json', {'schedule':[{'id':'old'},{'id':'new'}],
            'continuation':{'pending_case_ids':['new']}})
        write_json(self.root/'claims/old.json', {'state':'complete'})
        self.gpus = [{'index':i,'uuid':str(i),'free_mib':free} for i,free in enumerate([24000,16000,15000])]
        self.config = {'run':str(self.root),'controller_unit':'offline-fixture.service',
            'admission_plans_mib':[{'ray':23000,'sam':12000,'plan':10000}],
            'start_commands':[['offline-placeholder','{ray_uuid}']]}
        self.calls = []

    def tearDown(self):
        self.temp.cleanup()

    def check_tick(self, state, active=False, gpus=None):
        return tick(self.config,state,query=lambda:self.gpus if gpus is None else gpus,
            active=lambda _:active,execute=lambda command,**_:self.calls.append(command))

    def test_admission_requires_all_roles(self):
        self.assertIsNone(choose(self.gpus[:2],self.config['admission_plans_mib']))
        self.assertIsNotNone(choose(self.gpus,self.config['admission_plans_mib']))

    def test_inherited_claim_does_not_block_new_task(self):
        result=self.check_tick({})
        self.assertEqual(result['state'],'task_dispatched')
        self.assertEqual(self.calls,[['offline-placeholder','0']])

    def test_dispatch_happens_only_once(self):
        state={};self.check_tick(state);result=self.check_tick(state)
        self.assertEqual(len(self.calls),1)
        self.assertEqual(result['state'],'already_started_not_retried')

    def test_active_task_not_restarted_when_free_memory_low(self):
        result=self.check_tick({},active=True,gpus=[])
        self.assertEqual(result['state'],'preflight_active');self.assertFalse(self.calls)

    def test_stop_marker_prevents_dispatch(self):
        (self.root/'STOP').write_text('fixture')
        self.assertEqual(self.check_tick({})['state'],'stopped_by_experiment_gate')
        self.assertFalse(self.calls)

    def test_new_claim_prevents_dispatch(self):
        write_json(self.root/'claims/new.json',{'state':'claimed'})
        self.assertEqual(self.check_tick({})['state'],'already_started_not_retried')
        self.assertFalse(self.calls)


if __name__=='__main__':
    unittest.main()
