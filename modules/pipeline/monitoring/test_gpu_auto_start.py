"""Offline dispatch tests; no GPU allocation or real process startup."""
import tempfile
from pathlib import Path
import unittest
import subprocess
from unittest.mock import patch
from types import SimpleNamespace

from arx_project.layout import ROOT, write_json
from modules.pipeline.monitoring.gpu_auto_start import choose, tick, inventory


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

    def single_gpu_config(self):
        return {**self.config,'admission_plans_mib':[{'planner':10000}],
            'start_commands':[['offline-placeholder','{planner_uuid}']],
            'cuda_probe_before_dispatch':{'failure_cooldown_seconds':600}}

    def test_unknown_utilization_does_not_crash_memory_monitor(self):
        with patch('modules.pipeline.monitoring.gpu_auto_start.subprocess.run',
                   return_value=SimpleNamespace(stdout='0, gpu0, 24000, [N/A]\n')):
            values=inventory()
        self.assertEqual(values[0]['free_mib'],24000)
        self.assertIsNone(values[0]['utilization_percent'])

    def test_seed_probe_registration_needs_no_grasp_schedule(self):
        write_json(self.root/'registration.json',{'experiment':'index_seed_probe'})
        result=self.check_tick({})
        self.assertEqual(result['state'],'task_dispatched')

    def test_failed_cuda_card_does_not_block_healthy_card(self):
        checks=[]
        def probe(gpu,_):
            checks.append(gpu['uuid']);return {'passed':gpu['uuid']!='0'}
        result=tick(self.single_gpu_config(),{},query=lambda:self.gpus,active=lambda _:False,
            execute=lambda command,**_:self.calls.append(command),probe=probe,clock=lambda:100)
        self.assertEqual(result['selected']['planner']['uuid'],'1')
        self.assertEqual(checks,['0','1'])
        self.assertEqual(self.calls,[['offline-placeholder','1']])
        self.assertTrue((self.root/'monitor/cuda_admission.json').is_file())

    def test_failed_cuda_card_is_not_probed_each_poll(self):
        state={};checks=[]
        def probe(gpu,_):
            checks.append(gpu['uuid']);return {'passed':False}
        config=self.single_gpu_config()
        for at in [100,101,150]:
            result=tick(config,state,query=lambda:self.gpus[:1],active=lambda _:False,
                execute=lambda command,**_:self.calls.append(command),probe=probe,clock=lambda:at)
            self.assertEqual(result['state'],'waiting_for_gpu')
        self.assertEqual(checks,['0']);self.assertFalse(self.calls)

    def test_low_memory_waits_without_cuda_probe(self):
        def forbidden(*_):raise AssertionError('Do not initialize CUDA below the memory gate')
        result=tick(self.single_gpu_config(),{},query=lambda:[{'index':0,'uuid':'0','free_mib':9999}],
            active=lambda _:False,probe=forbidden)
        self.assertEqual(result['state'],'waiting_for_gpu')

    def test_durable_dispatch_marker_blocks_restart_duplicate(self):
        config={**self.single_gpu_config(),'dispatch_marker_files':['dispatch_claim.json']}
        write_json(self.root/'dispatch_claim.json',{'gpu':'0'})
        result=tick(config,{},query=lambda:self.gpus,active=lambda _:False,
            execute=lambda command,**_:self.calls.append(command))
        self.assertEqual(result['state'],'already_started_not_retried')
        self.assertFalse(self.calls)

    def test_temporary_inventory_timeout_waits_and_can_recover(self):
        def unavailable():raise subprocess.TimeoutExpired('nvidia-smi',5)
        state={}
        result=tick(self.config,state,query=unavailable,active=lambda _:False,
            execute=lambda command,**_:self.calls.append(command))
        self.assertEqual(result['state'],'waiting_for_gpu_query')
        self.assertFalse(result['terminal']);self.assertFalse(self.calls)
        recovered=self.check_tick(state)
        self.assertEqual(recovered['state'],'task_dispatched')
        self.assertEqual(len(self.calls),1)

    def test_failed_inventory_query_never_dispatches(self):
        def unavailable():raise subprocess.CalledProcessError(1,'nvidia-smi')
        result=tick(self.config,{},query=unavailable,active=lambda _:False,
            execute=lambda command,**_:self.calls.append(command))
        self.assertEqual(result['state'],'waiting_for_gpu_query')
        self.assertFalse(result['terminal']);self.assertFalse(self.calls)


if __name__=='__main__':
    unittest.main()
