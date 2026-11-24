"""S02 sequencing and failure persistence; response doubles never send ARX commands."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from interaction_contract import ROOT,read,write,resolve,load
from run_s02 import next_trial,execute
from s02_gates import VERSION, EVENTS

SOURCE=ROOT/'runs/lilo_P2_comparison_20261028_02/goal_physical/A1'


class S02Runner(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.out=Path(self.temp.name)
        self.schedule=[dict(trial_id='p'+str(i),phase='pilot',arm=['R10','R40'][i%2],case_id=['A1','bottle'][i//2]) for i in range(4)]+[dict(trial_id='m0',phase='main')]
    def tearDown(self):self.temp.cleanup()
    def seal(self,index,cleanup=True,instrumentation=True):
        trial=self.out/'trials'/('p'+str(index));trial.mkdir(parents=True)
        write(trial/'result.json',dict(cleanup_ok=cleanup,instrumentation_pass=instrumentation,gate_version=VERSION,halt=not instrumentation,record_integrity='complete' if instrumentation else 'failed',key_event_coverage={k:instrumentation for k in EVENTS}))
    def test_cannot_skip_retry_or_start_after_unknown_motion(self):
        value=dict(schedule=self.schedule)
        with self.assertRaises(ValueError):next_trial(value,self.out,'p1')
        self.seal(0,cleanup=False)
        with self.assertRaises(ValueError):next_trial(value,self.out,'p0')
        with self.assertRaises(ValueError):next_trial(value,self.out,'p1')
    def test_task_failure_does_not_replace_instrumentation_gate(self):
        for i in range(4):self.seal(i,instrumentation=i!=2)
        value=dict(schedule=self.schedule)
        with self.assertRaisesRegex(ValueError,'instrumentation'):next_trial(value,self.out,'m0')
        write(self.out/'trials/p2/result.json',dict(cleanup_ok=True,instrumentation_pass=True,physical_pass=False,gate_version=VERSION,halt=False,record_integrity='complete',key_event_coverage={k:True for k in EVENTS}))
        self.assertEqual(next_trial(value,self.out,'m0')['trial_id'],'m0')
    def test_pipeline_and_cleanup_failures_remain_counted(self):
        import batch_runtime,client,egad_evaluate
        setup=read(SOURCE/'setup.json');view=read(SOURCE/'viewport_initial.json')
        case=dict(case_id='core_egad_A1',asset='egad_A1',scenario_source=str(SOURCE/'scenario.json'),
            setup_source=str(SOURCE/'setup.json'),source_trial=str(SOURCE),fixed_goal_source=str(SOURCE/'selected_grasp.json'))
        case['resolved']={'R10':resolve(load(ROOT/'configs/lilo_vla/handoff_10mm.json'),'egad_A1','policy',
            goal_source=SOURCE/'selected_grasp.json',adapter=ROOT/'runs/lilo_P2_comparison_20261028_02/goal_adapter')}
        registration=dict(cases={'core_egad_A1':case},dependencies={})
        fixture=dict(fixture=dict(mailbox=str(self.out/'mailbox'),ops_url='http://unused',model_reference='unused'),
            resources=dict(neural_python='unused',planner_python='unused',gpu_uuid='unused'))
        calls=[];pipeline_calls=[]
        class ResponseOnlyClient:
            def __init__(self,*args,**kwargs):pass
            def request(self,action,**kwargs):
                calls.append(action)
                if action=='dynamic_setup':return copy.deepcopy(setup)
                if action=='grasp_viewport_state':return view
                raise AssertionError('Unexpected RPC in response double: '+action)
        def fake_process(command,env,log,*args,**kwargs):
            if Path(log).name=='pipeline.log':
                pipeline_calls.append(Path(log))
                verification=read(Path(log).parent/'setup_verification.json')
                self.assertIs(verification['historical_physical_properties_equal'],True)
                self.assertEqual(verification['historical_physical_properties_equal'],verification['pass_check'])
                self.assertEqual(env['ARX_LILO_TRACE'],'1')
                self.assertNotIn('ARX_LILO_TEACHER_DIR',env)
                self.assertEqual(command[-2:],['--fixed-selection',str(SOURCE/'selected_grasp.json')])
                raise TimeoutError('injected owned pipeline timeout')
            return 1
        row=dict(trial_id='fault_trial',case_id='core_egad_A1',arm='R10',phase='pilot')
        with patch.object(client,'Client',ResponseOnlyClient),patch.object(batch_runtime,'run_owned',fake_process),\
             patch.object(batch_runtime.Runtime,'cleanup',side_effect=RuntimeError('injected pending motion')),\
             patch.object(egad_evaluate,'evaluate',return_value={'pass':False}):
            result=execute(registration,row,self.out,fixture,'http://unused',dict(session='response-double',policy={}))
        self.assertTrue(result['attempted']);self.assertFalse(result['cleanup_ok'])
        self.assertFalse(result['instrumentation_pass']);self.assertFalse(result['physical_pass'])
        self.assertIn('pipeline timeout',result['error']['message'])
        self.assertTrue((self.out/'trials/fault_trial/result.json').exists())
        self.assertNotIn('trajectory',calls)
        count=len(pipeline_calls)
        with patch.object(client,'Client',ResponseOnlyClient),patch.object(batch_runtime,'run_owned',fake_process),\
             patch.object(batch_runtime.Runtime,'cleanup',side_effect=RuntimeError('injected pending motion')),\
             patch.object(egad_evaluate,'evaluate',return_value={'pass':False}),\
             patch('run_s02.check_setup',return_value=dict(pass_check=False,mismatches=['frozen mismatch'])):
            denied=execute(registration,{**row,'trial_id':'denied_setup'},self.out,fixture,'http://unused',dict(session='response-double',policy={}))
        verification=read(self.out/'trials/denied_setup/setup_verification.json')
        self.assertIs(verification['historical_physical_properties_equal'],False)
        self.assertFalse(verification['pass'])
        self.assertEqual(len(pipeline_calls),count)
        self.assertTrue(denied['halt'])

    def test_clean_cleanup_does_not_hide_audit_or_resource_failure(self):
        import batch_runtime,client,egad_evaluate
        setup=read(SOURCE/'setup.json');view=read(SOURCE/'viewport_initial.json')
        case=dict(case_id='core_egad_A1',asset='egad_A1',scenario_source=str(SOURCE/'scenario.json'),
            setup_source=str(SOURCE/'setup.json'),source_trial=str(SOURCE),fixed_goal_source=str(SOURCE/'selected_grasp.json'))
        case['resolved']={'R10':resolve(load(ROOT/'configs/lilo_vla/handoff_10mm.json'),'egad_A1','policy',
            goal_source=SOURCE/'selected_grasp.json',adapter=ROOT/'runs/lilo_P2_comparison_20261028_02/goal_adapter')}
        fixture=dict(fixture=dict(mailbox=str(self.out/'mailbox'),ops_url='http://unused',model_reference='unused'),
            resources=dict(neural_python='unused',planner_python='unused',gpu_uuid='unused'))
        class ResponseOnlyClient:
            def __init__(self,*args,**kwargs):pass
            def request(self,action,**kwargs):
                if action=='dynamic_setup':return copy.deepcopy(setup)
                if action=='grasp_viewport_state':return view
                if action=='grasp_research_fingerprint':return copy.deepcopy(setup['fingerprint'])
                raise AssertionError('No live RPC allowed: '+action)
        for fault in ['record_corruption','resource_timeout']:
            with self.subTest(fault=fault):
                trial_id=fault
                def process(command,env,log,*args,**kwargs):
                    if Path(log).name=='pipeline.log':
                        if fault=='resource_timeout':raise TimeoutError('owned pipeline timed out')
                        return 0
                    path=Path(command[command.index('--output')+1])
                    ok=fault!='record_corruption'
                    write(path,dict(pass_check=ok,record_integrity='complete' if ok else 'failed',
                        event_applicability={k:'applicable' for k in EVENTS},key_event_coverage={k:ok for k in EVENTS}))
                    return 0 if ok else 1
                with patch.object(client,'Client',ResponseOnlyClient),patch.object(batch_runtime,'run_owned',process), \
                     patch.object(batch_runtime.Runtime,'cleanup',return_value=None), \
                     patch.object(egad_evaluate,'evaluate',return_value={'pass':False}):
                    result=execute(dict(cases={'core_egad_A1':case},dependencies={}),
                        dict(trial_id=trial_id,case_id='core_egad_A1',arm='R10',phase='pilot'),
                        self.out,fixture,'http://unused',dict(session='offline-double',policy={}))
                self.assertTrue(result['cleanup_ok']);self.assertTrue(result['halt'])
                if fault=='record_corruption':self.assertEqual(result['record_integrity'],'failed')
                else:self.assertTrue(result['infrastructure_failure'])


if __name__=='__main__':unittest.main()
