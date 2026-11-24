"""Behavioral checks for module boundaries, orchestration and local RPC faults."""
import json
import multiprocessing
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace

from arx_project.client import Client
from arx_project.contracts import Stage
from arx_project.layout import ROOT,ensure_run_path,read_json,relocated,write_json
from arx_project.pipeline import Pipeline,stages_from_config
from arx_project.registry import load_adapter
from arx_project.service import serve


class FakeWorker:
    def request(self,directory,**options):
        if options.get('fault'):raise RuntimeError('injected worker failure')
        return {'success':True,'target_found':True}


class ArchitectureTest(unittest.TestCase):
    def setUp(self):
        folder=ROOT/'tmp/tests';folder.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='arch_',dir=folder)
        self.root=Path(self.temp.name);self.process=None

    def tearDown(self):
        if self.process:
            if self.process.is_alive():self.process.terminate()
            self.process.join(5)
        self.temp.cleanup()

    def test_paths_reject_escape_and_module_outputs(self):
        for path in [ROOT/'modules/sam3/result.json',ROOT/'tmp/file.json',ROOT/'tmp/x/../../escape']:
            with self.assertRaises(ValueError):ensure_run_path(path)
        self.assertEqual(ensure_run_path(self.root/'a/b'),self.root/'a/b')

    def test_frozen_metadata_relocated_without_rewriting(self):
        p=self.root/'frozen.json';p.write_text('{"asset":"/old/data/a.obj"}\n');raw=p.read_bytes()
        mapped=relocated(json.loads(raw),[('/old/data','simulation/assets')])
        self.assertEqual(mapped['asset'],str(ROOT/'simulation/assets/a.obj'))
        self.assertEqual(p.read_bytes(),raw)

    def test_chained_paths_resolve_without_compatibility_directories(self):
        old='/removed/project';intermediate=str(ROOT/'modules/legacy/assets')
        result=relocated(old+'/assets/a.obj',[(old,str(ROOT/'modules/legacy')),
            (intermediate,'simulation/assets')])
        self.assertEqual(result,str(ROOT/'simulation/assets/a.obj'))
        with self.assertRaises(ValueError):relocated('/a/file',[('/a','/b'),('/b','/a')])

    def test_registry_lazy_imports_and_pipeline_config(self):
        import sys
        before='torch' in sys.modules
        for name in ['sam3','rayst3r','graspgenx','curobo']:self.assertTrue(callable(load_adapter(name).create))
        self.assertEqual('torch' in sys.modules,before)
        self.assertEqual([s.module for s in stages_from_config('sam3_graspgenx')],['sam3','rayst3r','graspgenx'])

    def test_failure_stops_downstream_and_keeps_failed_stage(self):
        calls=[]
        class Worker:
            def request(inner,directory,**options):
                calls.append(1);return {'success':True,'target_found':False}
        outcomes=Pipeline([Stage('a','one'),Stage('b','two')],{'one':Worker(),'two':Worker()}).run(self.root)
        self.assertEqual(len(calls),1);self.assertEqual(outcomes[0].status,'failed')
        self.assertEqual(read_json(self.root/'pipeline_progress.json')['stages'][0]['status'],'failed')

    def test_missing_artifact_and_rpc_fault_raise(self):
        class Worker:
            def request(inner,*args,**kwargs):return {'success':True}
        with self.assertRaisesRegex(RuntimeError,'missing output'):
            Pipeline([Stage('x','a',produced_files=('missing.npy',))],{'a':Worker()}).run(self.root)
        class Fault:
            def request(inner,*args,**kwargs):return {'status':'error','error':'resource fault'}
        with self.assertRaises(RuntimeError):Pipeline([Stage('x','a')],{'a':Fault()}).run(self.root)

    def test_all_artifacts_prevalidated_before_worker_calls(self):
        class Worker:
            def request(inner,*args,**kwargs):raise AssertionError('Must not be called')
        for path in ['../escape','/etc/passwd']:
            with self.assertRaises(ValueError):Pipeline([Stage('x','a',produced_files=(path,))],{'a':Worker()}).run(self.root)
        (self.root/'outside').symlink_to(ROOT/'modules',target_is_directory=True)
        with self.assertRaises(ValueError):Pipeline([Stage('x','a',produced_files=('outside/result',))],{'a':Worker()}).run(self.root)

    def test_rpc_health_fault_finalized_trial_and_shutdown(self):
        args=SimpleNamespace(root=self.root/'service',kind='fake',warmup_trial=self.root/'warm',socket=self.root/'s.sock')
        self.process=multiprocessing.Process(target=serve,args=(args,FakeWorker()));self.process.start()
        deadline=time.monotonic()+10
        while not (self.root/'service/fake_ready.json').exists():
            if not self.process.is_alive() or time.monotonic()>deadline:self.fail('RPC did not become ready')
            time.sleep(.02)
        client=Client(args.socket,timeout=5)
        self.assertEqual(client.call({'command':'health'})['status'],'ready')
        trial=self.root/'trial';trial.mkdir()
        self.assertEqual(client.request(trial)['status'],'ok')
        self.assertEqual(client.request(trial,fault=True)['status'],'error')
        write_json(trial/'result.json',{'success':False})
        self.assertIn('already finalized',client.request(trial)['error'])
        self.assertEqual(client.request(ROOT/'modules')['status'],'error')
        self.assertEqual(client.call({'command':'shutdown'})['status'],'stopped')
        self.process.join(5);self.assertEqual(self.process.exitcode,0)
        self.assertFalse(args.socket.exists())


if __name__=='__main__':unittest.main()
