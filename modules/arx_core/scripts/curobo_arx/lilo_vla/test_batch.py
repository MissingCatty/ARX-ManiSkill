"""Offline ARX protocol/fault checks; mocked responses are not physical trials."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from batch_manifest import ROOT, GPU3, resolve_manifest, rehearsal, materialize_fixture
from batch_ledger import Ledger, write, previous, process_identity, receipt, append
from batch_runtime import Runtime, status, confirm_idle, execute_attempt, UnsafeInstance, FrozenMismatch, PipelineFailed, InfrastructureFailure, run_owned
from collect_batch import collect, encode_results
from interaction_contract import digest, read
from episode_quality import inspect_episode, recording_identity

FIXTURE=ROOT/'servers/lilo_vla/tests/fixtures'
REAL=read(FIXTURE/'arx_fixture.json')['observation']


def case(name='rubik_j0',mode='fresh',group='rubik'):
    return {'case_id':name,'object_name':'rubiks_cube','split':'train','goal_mode':mode,'goal_group_id':group,
            'placement':{'position':[4.048,2.08,.72],'orientation':[1,0,0,0]},'contract':'near','entry_delta_world':[0]*6}


def registration(cases=None,budget=20):
    return {'manifest_sha256':'offline-protocol-fixture','cases':cases or [case()],
            'budget':{'max_attempts':budget},'contracts':{},'roles':{'rubiks_cube':'train'},
            'algorithm_sha256':{},'dependency_sha256':{},'resources':{'neural_python':sys.executable}}


class FakeRuntime:
    """Only a response protocol, built around the saved actual ARX observation."""
    fail=None;calls=None;session='protocol-session';physical=True;planned='frozen_before_motion';qpass=True
    def __init__(self,*args):self.calls=[]
    def step(self,name):
        self.calls.append(name)
        if self.fail==name:raise RuntimeError('injected '+name)
        if self.fail=='timeout' and name=='pipeline':raise TimeoutError('injected deadline')
        if self.fail=='frozen' and name=='reset':raise FrozenMismatch('injected frozen mismatch')
        if self.fail=='infrastructure' and name=='pipeline':raise InfrastructureFailure('injected infrastructure')
    def begin(self,trial,case,goal,deadline,heartbeat):
        self.trial=trial;self.goal=goal;self.step('prepare');write(trial/'saved_arx_observation.json',REAL)
    def check_idle(self):self.step('idle')
    def reset(self):self.step('reset')
    def viewport(self):self.step('viewport')
    def pipeline(self):self.step('pipeline')
    def cleanup(self,deadline):self.step('cleanup');write(self.trial/'teacher_recording.json',{'samples':50})
    def evaluate(self):self.step('evaluate');return {'pass':self.physical,'scope':'mock protocol only'}
    def planned_goal(self):return self.planned if 'pipeline' in self.calls else None
    def quality(self):self.step('quality');return {'pass':self.qpass,'content_sha256':self.trial.name}


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.silence=contextlib.redirect_stdout(io.StringIO());self.silence.__enter__()
    def tearDown(self):self.silence.__exit__(None,None,None);self.tmp.cleanup()
    def ledger(self,cases=None,budget=20,name='campaign'):return Ledger(self.root/name,registration(cases,budget))

    def test_all_stage_failures_are_sealed_and_counted(self):
        for stage in ('prepare','idle','reset','viewport','pipeline','cleanup','evaluate','quality'):
            with self.subTest(stage=stage):
                ledger=self.ledger(name=stage);runtime=FakeRuntime();runtime.fail=stage
                result=execute_attempt(ledger,case(),runtime)
                self.assertFalse(result['demonstration_ready']);self.assertEqual(len(ledger.results),1)
                self.assertEqual(receipt(Path(result['source'])/'result.json'),result)
                if stage!='cleanup':self.assertIn('injected '+stage,result['error']['message'])
                else:self.assertTrue(result['stop_campaign'])

    def test_cleanup_does_not_replace_primary_error(self):
        runtime=FakeRuntime();runtime.fail='pipeline'
        runtime.cleanup=lambda deadline:(_ for _ in ()).throw(UnsafeInstance('still active'))
        result=execute_attempt(self.ledger(),case(),runtime)
        self.assertEqual(result['error']['message'],'injected pipeline');self.assertTrue(result['cleanup_errors']);self.assertTrue(result['stop_campaign'])

    def test_timeout_cannot_become_demo_even_with_positive_physical_response(self):
        runtime=FakeRuntime();runtime.fail='timeout'
        result=execute_attempt(self.ledger(),case(),runtime)
        self.assertEqual(result['status'],'timeout');self.assertTrue(result['pass']);self.assertFalse(result['demonstration_ready'])

    def test_frozen_mismatch_stops_campaign(self):
        runtime=FakeRuntime();runtime.fail='frozen'
        self.assertTrue(execute_attempt(self.ledger(),case(),runtime)['stop_campaign'])

    def test_duplicate_content_not_admitted(self):
        runtime=FakeRuntime();runtime.quality=lambda:{'pass':True,'content_sha256':'same'}
        result=execute_attempt(self.ledger(),case(),runtime,seen_recordings={'same'})
        self.assertFalse(result['demonstration_ready']);self.assertEqual(result['status'],'quality_rejected')

    def test_same_goal_survives_first_physical_failure(self):
        ledger=self.ledger([case(),case('rubik_j1','group')]);runtimes=[]
        def factory(*args):
            runtime=FakeRuntime();runtime.physical=False;runtimes.append(runtime);return runtime
        collect(ledger,factory,'protocol-session')
        self.assertEqual(len(ledger.results),2);self.assertEqual(runtimes[1].goal,'frozen_before_motion')
        self.assertTrue(all('pipeline' in r.calls for r in runtimes))

    def test_no_goal_skips_dependent_cases_without_counting_as_attempts(self):
        ledger=self.ledger([case(),case('rubik_j1','group'),case('other',group='other')])
        def factory(*args):r=FakeRuntime();r.planned=None;r.fail='reset';return r
        collect(ledger,factory,'protocol-session')
        self.assertEqual(len(ledger.results),2);self.assertEqual(ledger.unexecuted,[{'case_id':'rubik_j1','reason':'group_goal_unavailable'}])

    def test_budget_is_cumulative_and_does_not_expand(self):
        ledger=self.ledger([case(),case('rubik_j1','group')],budget=1)
        collect(ledger,FakeRuntime,'protocol-session');self.assertEqual(len(ledger.results),1)
        self.assertEqual(ledger.unexecuted[0]['reason'],'cumulative_attempt_budget')

    def test_three_identical_infrastructure_failures_stop(self):
        ledger=self.ledger([case(str(i),group=str(i)) for i in range(4)])
        def factory(*args):r=FakeRuntime();r.fail='infrastructure';return r
        collect(ledger,factory,'protocol-session');self.assertEqual(len(ledger.results),3)
        self.assertEqual(ledger.unexecuted[0]['reason'],'three_consecutive_identical_infrastructure_errors')

    def test_terminal_before_aggregate_crash_does_not_repeat(self):
        ledger=self.ledger();result=execute_attempt(ledger,case(),FakeRuntime())
        (ledger.output/'all_attempts.json').unlink();(ledger.output/'dataset_sources.json').unlink()
        write(ledger.output/'owner.json',{'pid':999999999})
        rows=previous(ledger.output,ledger.registration)
        self.assertEqual(rows,[result])

    def test_started_without_terminal_becomes_interrupted_without_old_writes(self):
        ledger=self.ledger();trial,started=ledger.begin(case())
        write(ledger.output/'owner.json',{'pid':999999999})
        before={str(p):p.read_bytes() for p in ledger.output.rglob('*') if p.is_file()}
        rows=previous(ledger.output,ledger.registration)
        self.assertEqual(rows[0]['status'],'interrupted')
        self.assertEqual(before,{str(p):p.read_bytes() for p in ledger.output.rglob('*') if p.is_file()})
        resumed=Ledger(self.root/'resumed',ledger.registration,rows)
        collect(resumed,FakeRuntime,'protocol-session')
        self.assertEqual(len(resumed.results),1);self.assertEqual(list((resumed.output/'attempts').iterdir()),[])

    def test_live_owner_and_changed_registration_reject_resume(self):
        ledger=self.ledger()
        with self.assertRaisesRegex(RuntimeError,'still owns'):previous(ledger.output,ledger.registration)
        with self.assertRaisesRegex(ValueError,'identity changed'):previous(ledger.output,{**ledger.registration,'manifest_sha256':'changed'})

    def test_terminal_tampering_is_rejected(self):
        result=execute_attempt(self.ledger(),case(),FakeRuntime());path=Path(result['source'])/'result.json'
        value=read(path);value['result']['pass']=False;write(path,value)
        with self.assertRaisesRegex(ValueError,'hash mismatch'):receipt(path)

    def test_alive_orphan_child_prevents_resume(self):
        ledger=self.ledger();trial,_=ledger.begin(case())
        write(ledger.output/'owner.json',{'pid':999999999})
        write(trial/'pipeline.process.json',{'child':process_identity(),'owner':process_identity(),'process_group':os.getpgrp()})
        with self.assertRaisesRegex(RuntimeError,'child still running'):previous(ledger.output,ledger.registration)

    def test_partial_recording_receipt_still_seals_primary_error(self):
        runtime=FakeRuntime();runtime.fail='pipeline'
        def cleanup(deadline):(runtime.trial/'teacher_recording.json').write_text('{partial')
        runtime.cleanup=cleanup
        result=execute_attempt(self.ledger(),case(),runtime)
        self.assertEqual(result['error']['message'],'injected pipeline');self.assertTrue(result['stop_campaign'])
        self.assertFalse(result['demonstration_ready'])

    def test_budget_includes_quality_and_result_seal(self):
        ledger=self.ledger();runtime=FakeRuntime()
        with patch('batch_ledger.time.monotonic',return_value=1000):
            result=execute_attempt(ledger,case(),runtime)
        # Independently test a result crossing the deadline just before seal.
        trial,start=ledger.begin(case('later'))
        final={**start,'status':'completed','demonstration_ready':True,'budget_expires_monotonic':time.monotonic()-1,'source':str(trial),'pass':True}
        ledger.finish(trial,final)
        self.assertEqual(receipt(trial/'result.json')['status'],'timeout')
        self.assertFalse(final['demonstration_ready'])

    def test_encoding_failure_keeps_qualified_result_and_frames(self):
        ledger=self.ledger();result=execute_attempt(ledger,case(),FakeRuntime());trial=Path(result['source'])
        (trial/'frames').mkdir();write(trial/'frames/frames.json',{'frames':[]})
        before=(trial/'result.json').read_bytes()
        with patch('collect_batch.run_owned',side_effect=TimeoutError('encoding')):encode_results(ledger)
        self.assertEqual(read(trial/'video_status.json')['status'],'failed');self.assertEqual(before,(trial/'result.json').read_bytes())
        self.assertTrue((trial/'frames/frames.json').exists())
        # A second encoding attempt preserves a failed partial and never runs RPC.
        partial=trial/(trial.name+'.mp4');partial.write_bytes(b'interrupted encoder output')
        with patch('collect_batch.run_owned',return_value=0),patch('client.Client.request',side_effect=AssertionError('No RPC')):
            encode_results(ledger)
        self.assertEqual(before,(trial/'result.json').read_bytes())
        self.assertEqual(len(list(trial.glob('*.partial_*.mp4'))),1)
        self.assertEqual(len(list(trial.glob('encoding_*.json'))),2)

    def test_owned_child_timeout_is_bounded_and_recorded(self):
        with self.assertRaises(TimeoutError):run_owned([sys.executable,'-c','import time;time.sleep(10)'],os.environ,self.root/'owned.log',time.monotonic()+.1)
        record=read(self.root/'owned.process.json');self.assertEqual(record['owner'],process_identity())


class StatusTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.box=self.root/'mailbox';self.box.mkdir()
        self.trial=self.root/'attempt';self.trial.mkdir()
        self.fixture={'input_sha256':{'initial_setup':'a','catalog':'b','setup':'c'},'renderer':'RaytracedLighting'}
        self.value={'version':'arx_collection_status_v1','session_id':'s','wall_time':time.time(),'gpu_uuid':GPU3,
                    'input_sha256':self.fixture['input_sha256'],'renderer':'RaytracedLighting',
                    'active_trajectory_request_id':None,'controller_busy':False,'setup_pending':None,'streaming':False,
                    'recorders':{'teacher':None,'video':None}}
    def tearDown(self):self.tmp.cleanup()
    def save(self):write(self.box/'stream_state.json',{'collection_status':self.value})
    def progress(self,delay):self.value['wall_time']=time.time();self.save()
    def confirm(self):
        # Exercise state transitions without charging host fsync latency to a
        # 30ms test deadline. Production deadlines remain unchanged.
        clock=[0.]
        def progress(delay):
            self.progress(delay);clock[0]+=delay
        with patch('batch_runtime.time.sleep',side_effect=progress),patch('batch_runtime.time.monotonic',side_effect=lambda:clock[0]):
            return confirm_idle(self.box,self.fixture,'s',1.,self.trial)

    def test_stale_ready_never_proves_idle(self):
        write(self.box/'ready.json',REAL)
        with self.assertRaises(FileNotFoundError):self.confirm()
        self.value['wall_time']-=10;self.save()
        with self.assertRaises(UnsafeInstance):self.confirm()

    def test_ordinary_trajectory_blocks_even_without_stream_command_id(self):
        self.value['active_trajectory_request_id']='trajectory';self.save()
        with self.assertRaises(UnsafeInstance):self.confirm()

    def test_busy_controller_reset_and_stream_block(self):
        for key,value in (('controller_busy',True),('setup_pending','reset'),('streaming',True)):
            original=self.value[key];self.value[key]=value;self.save()
            with self.subTest(key=key),self.assertRaises(UnsafeInstance):self.confirm()
            self.value[key]=original

    def test_two_distinct_fresh_statuses_required(self):
        self.save()
        with patch('batch_runtime.time.sleep',return_value=None),self.assertRaises(UnsafeInstance):confirm_idle(self.box,self.fixture,'s',time.monotonic()+.01,self.trial)
        self.assertEqual(self.confirm()['session_id'],'s')

    def test_unknown_request_is_preserved(self):
        self.save();write(self.box/'request.json',{'id':'unknown'})
        with self.assertRaisesRegex(UnsafeInstance,'Unknown'):self.confirm()
        self.assertEqual(read(self.box/'request.json')['id'],'unknown')

    def test_late_matching_response_resolves_own_request(self):
        self.save();append(self.trial/'requests.jsonl',{'event':'intent','request_id':'own','action':'trajectory'})
        with self.assertRaises(UnsafeInstance):self.confirm()
        write(self.box/'response.json',{'id':'own','ok':True,'result':REAL})
        self.assertEqual(self.confirm()['session_id'],'s')
        self.assertTrue((self.trial/'late_response_own.json').exists())

    def test_other_response_does_not_resolve_own_pending(self):
        self.save();append(self.trial/'requests.jsonl',{'event':'intent','request_id':'own','action':'trajectory'})
        write(self.box/'response.json',{'id':'other','ok':True,'result':REAL})
        with self.assertRaises(UnsafeInstance):self.confirm()

    def test_unowned_recording_is_not_stopped(self):
        self.value['recorders']['teacher']='/another/user/recording';self.save()
        with self.assertRaisesRegex(UnsafeInstance,'Unowned'):self.confirm()
        self.assertFalse((self.box/'request.json').exists())

    def test_wrong_input_session_gpu_rejected(self):
        for key,value in (('session_id','other'),('gpu_uuid','GPU-forbidden'),('renderer','PathTracing')):
            original=self.value[key];self.value[key]=value;self.save()
            with self.subTest(key=key),self.assertRaises((UnsafeInstance,FrozenMismatch)):self.confirm()
            self.value[key]=original

    def test_busy_native_trajectory_can_finish_before_next_reset(self):
        self.value['active_trajectory_request_id']='native';self.save()
        append(self.trial/'requests.jsonl',{'event':'intent','request_id':'native','action':'trajectory'})
        def finish(delay):
            self.value['active_trajectory_request_id']=None
            write(self.box/'response.json',{'id':'native','ok':True,'result':REAL});self.progress(delay)
        with patch('batch_runtime.time.sleep',side_effect=finish):
            confirm_idle(self.box,self.fixture,'s',time.monotonic()+1,self.trial)
        self.assertFalse((self.box/'request.json').exists())

    def test_status_producer_reports_ordinary_trajectory_without_mutating_controller(self):
        import importlib.util
        from types import SimpleNamespace
        path=ROOT/'arx-sim/arx_sim/diagnostics/collection_status.py'
        spec=importlib.util.spec_from_file_location('collection_status_test',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        bridge=SimpleNamespace(active={'request':{'id':'ordinary'}},streaming=False,
            collection_identity={'session_id':'s'},context=SimpleNamespace(controller=SimpleNamespace(active=None),config={'simulation':{'renderer':'RaytracedLighting'}}))
        original=copy.deepcopy(bridge.__dict__)
        result=module.snapshot(bridge)
        self.assertEqual(result['active_trajectory_request_id'],'ordinary');self.assertEqual(bridge.__dict__,original)

    def test_client_journals_intent_before_send_and_honors_shared_deadline(self):
        from client import Client
        self.save();write(self.box/'ready.json',REAL)
        def respond(delay):
            request=read(self.box/'request.json');events=(self.trial/'requests.jsonl').read_text()
            self.assertIn(request['id'],events);(self.box/'request.json').unlink()
            write(self.box/'response.json',{'id':request['id'],'ok':True,'result':REAL})
        client=Client(self.box,journal=self.trial/'requests.jsonl',session='s',explicit_root=True)
        with patch('client.time.sleep',side_effect=respond):self.assertEqual(client.request('snapshot'),REAL)
        client.deadline=time.monotonic()-1
        with self.assertRaises(TimeoutError):client.request('dynamic_setup')
        self.assertFalse((self.box/'request.json').exists())


class ManifestTests(unittest.TestCase):
    def test_prepare_then_start_preserves_materialized_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            value={'fixture':{'runtime':d,'bundle':str(ROOT/'configs/lilo_vla/fixtures/tabletop_rt_v1')}}
            first=materialize_fixture(value)
            self.assertEqual(materialize_fixture(value),first)
            (first/'initial_setup.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'input differs'):materialize_fixture(value)

    def test_wrong_restored_viewport_is_a_frozen_mismatch(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as d:
            fixture={'bundle':str(ROOT/'configs/lilo_vla/fixtures/tabletop_rt_v1')}
            runtime=Runtime({'fixture':fixture},'protocol-session');runtime.trial=Path(d)
            wrong=read(Path(fixture['bundle'])/'viewport_initial.json');wrong['world_matrix'][3][0]+=.1
            runtime.client=Mock();runtime.client.request.return_value=wrong
            with self.assertRaises(FrozenMismatch):runtime.viewport()

    def test_twenty_cases_no_devices_no_rpc_no_legacy_runs(self):
        value=resolve_manifest(ROOT/'configs/lilo_vla/teacher_pilot20.json')
        self.assertEqual(len(value['cases']),20)
        self.assertEqual(set(value['roles'].values()),{'train','validation'})
        self.assertFalse(any('/runs/' in p for p in value['dependency_sha256']))
        for name in value['roles']:
            self.assertEqual(len([r for r in value['cases'] if r['object_name']==name]),4)
        with tempfile.TemporaryDirectory() as d:
            with patch('subprocess.Popen',side_effect=AssertionError('No process launch')),patch('client.Client.request',side_effect=AssertionError('No RPC')):
                rehearsal(value,Path(d)/'dry')
            self.assertEqual(read(Path(d)/'dry/rehearsal.json')['physical_attempts'],0)

    def test_resolve_in_layout_without_any_historical_runs(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);config=base/'configs/lilo_vla';config.mkdir(parents=True)
            shutil.copytree(ROOT/'configs/lilo_vla/fixtures',config/'fixtures')
            for name in ('handoff_10mm.json','handoff_40_60_80mm.json'):shutil.copy2(ROOT/'configs/lilo_vla'/name,config/name)
            original=read(ROOT/'configs/lilo_vla/teacher_pilot20.json')
            for key in ('model_reference','asset_root'):original['fixture'][key]=str((ROOT/'configs/lilo_vla'/original['fixture'][key]).resolve())
            for key in ('neural_python','planner_python','zerograsp_checkpoint','curobo_source'):original['resources'][key]=str((ROOT/'configs/lilo_vla'/original['resources'][key]).resolve())
            write(config/'manifest.json',original)
            resolved=resolve_manifest(config/'manifest.json');self.assertEqual(len(resolved['cases']),20)
            self.assertFalse((base/'runs').exists())
            (config/'fixtures/tabletop_rt_v1/initial_setup.json').unlink()
            with self.assertRaises(FileNotFoundError):resolve_manifest(config/'manifest.json')

    def test_duplicate_case_unknown_object_wrong_group_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'configs/lilo_vla') as d:
            # Absolute paths isolate structural mutations from path resolution.
            source=ROOT/'configs/lilo_vla/teacher_pilot20.json';original=read(source)
            for key in ('bundle','mailbox','runtime','model_reference','asset_root'):original['fixture'][key]=str((source.parent/original['fixture'][key]).resolve())
            for key in ('neural_python','planner_python','zerograsp_checkpoint','curobo_source'):original['resources'][key]=str((source.parent/original['resources'][key]).resolve())
            original['contracts']={k:str(source.parent/p) for k,p in original['contracts'].items()}
            for mode in ('duplicate','object','group'):
                value=copy.deepcopy(original)
                if mode=='duplicate':value['cases'][1]['case_id']=value['cases'][0]['case_id']
                if mode=='object':value['cases'][0]['object_name']='unregistered'
                if mode=='group':value['cases'][0]['goal_mode']='group'
                path=Path(d)/'m.json';write(path,value)
                with self.subTest(mode=mode),self.assertRaises(ValueError):resolve_manifest(path)


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);(self.root/'teacher_raw').mkdir()
        write(self.root/'verification.json',{'pass':True,'scope':'Mock protocol receipt; no physical success claim'})
        write(self.root/'teacher_recording.json',{'samples':51})
        write(self.root/'selected_grasp.json',{'grasp_world':REAL['snapshot']['link_world']['right_eef_link']})
        self.rows=[]
        for i,stage in enumerate(['pregrasp']+['approach']*10+['close']*16+['lift']*10+['hold']*16):
            row=copy.deepcopy(REAL);row.update(rgb_path=str(FIXTURE/'arx_wrist_pregrasp.png'),frame_time=i*.2,capture_sim_time=i*.2,
                 stage=stage,teacher_command={'id':stage,'gripper_target':-3.2 if stage in ('pregrasp','approach') else .1})
            self.rows.append(row)
        self.save()
    def tearDown(self):self.tmp.cleanup()
    def save(self):
        (self.root/'teacher_raw/samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        write(self.root/'teacher_recording.json',{'samples':len(self.rows)})
    def inspect(self):return inspect_episode(self.root,require_entry=False)

    def test_quality_is_readonly_and_checks_closure_chunks(self):
        before=set(self.root.rglob('*'));result=self.inspect()
        self.assertEqual(before,set(self.root.rglob('*')));self.assertGreater(result['quality']['included_closure_start_intervals'],0)
        self.assertGreater(result['quality']['contiguous_eight_action_chunks'],0)

    def test_missing_onset_more_than_eight_frames_rejected(self):
        for row in self.rows:row['teacher_command']['id']='unchanged'
        self.save()
        with self.assertRaisesRegex(ValueError,'pre-command'):self.inspect()

    def test_no_contiguous_eight_action_block_rejected(self):
        for i,row in enumerate(self.rows):row['frame_time']=row['capture_sim_time']=i*.201
        self.save()
        with self.assertRaisesRegex(ValueError,'eight-action'):self.inspect()

    def test_failed_physical_receipt_rejected(self):
        write(self.root/'verification.json',{'pass':False})
        with self.assertRaisesRegex(ValueError,'Failed execution'):self.inspect()

    def test_raw_recording_count_mismatch_rejected(self):
        write(self.root/'teacher_recording.json',{'samples':len(self.rows)+1})
        with self.assertRaisesRegex(ValueError,'frame count'):self.inspect()

    def test_time_mismatch_missing_final_image_nonfinite_rejected(self):
        original=copy.deepcopy(self.rows)
        for mode in ('time','missing','nonfinite'):
            self.rows=copy.deepcopy(original)
            if mode=='time':self.rows[-1]['capture_sim_time']+=.1
            if mode=='missing':self.rows[-1]['rgb_path']='/missing/actual/wrist.png'
            if mode=='nonfinite':self.rows[-1]['state'][0]=float('nan')
            self.save()
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.inspect()

    def test_content_duplicate_survives_path_and_command_id_changes(self):
        first=recording_identity(self.rows);image=self.root/'copied.png';shutil.copy2(FIXTURE/'arx_wrist_pregrasp.png',image)
        for row in self.rows:row['rgb_path']=str(image);row['teacher_command']['id']='renamed-'+row['teacher_command']['id']
        self.assertEqual(first,recording_identity(self.rows))

    def test_legacy_cannot_become_new_contract_data(self):
        with self.assertRaisesRegex(ValueError,'requires a recorded'):inspect_episode(self.root,require_contract=True)


if __name__=='__main__':unittest.main()
