"""Real Unix-socket, independent-process tests of asynchronous candidate handoff."""
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from arx_project.layout import ROOT, read_json
from modules.curobo.candidate_io.loader import load_staged

STAGED = ROOT/'tmp/candidate_io/refactor_20261125_01'


class RPCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Client = load_staged(STAGED, planning=False)['client'].Client

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='rpc_', dir=ROOT/'tmp/candidate_io')
        self.root = Path(self.temp.name)
        self.children, self.logs = [], []
        self.sockets = {}
        for kind in ['zerograsp', 'curobo_dual_arm']:
            sock = self.root/(kind+'.sock')
            folder = self.root/'resident'/kind
            folder.mkdir(parents=True)
            stream = (folder/'service.log').open('w')
            self.logs.append(stream)
            command = [sys.executable, '-m', 'modules.curobo.candidate_io.rpc_fixture',
                '--staged', str(STAGED), '--kind', kind, '--root', str(folder),
                '--socket', str(sock), '--warmup-trial', str(self.root/'warmup'),
                '--descriptor', str(self.root/'descriptor.json')]
            child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            self.children.append(child); self.sockets[kind] = sock
            deadline = time.monotonic()+15
            while not sock.exists():
                if child.poll() is not None or time.monotonic() > deadline:
                    self.fail((folder/'service.log').read_text())
                time.sleep(.02)

    def tearDown(self):
        for sock, child in zip(self.sockets.values(), self.children):
            if child.poll() is None:
                try:self.Client(sock).call({'command':'shutdown'})
                except Exception:child.terminate()
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:child.kill(); child.wait()
        for stream in self.logs:stream.close()
        self.temp.cleanup()

    def test_consumer_runs_before_generator_snapshot_finishes(self):
        directory = self.root/'test_trial'
        self.Client(self.sockets['zerograsp']).request(directory)
        response = self.Client(self.sockets['curobo_dual_arm'],candidate_transport=True).request(directory)
        self.assertTrue(response['result']['success'])
        self.assertFalse(response['result']['snapshot_present_at_entry'])
        self.assertEqual(response['result']['candidate_count'], 5)
        self.assertTrue((directory/'candidates/candidate_pool.json').is_file())
        self.assertTrue(read_json(directory/'candidate_snapshot_commit.json')['persisted'])
        self.assertEqual(read_json(directory/'candidate_io.json')['decode_count'], 1)

    def test_registered_consumer_negotiates_transport_without_override(self):
        from arx_project.artifacts import FORMAT
        from arx_project.registry import module_records
        self.assertEqual(module_records()['curobo_dual_arm'].get('candidate_transport_format'), FORMAT)
        directory = self.root/'test_trial_registered_consumer'
        self.Client(self.sockets['zerograsp']).request(directory)
        result = self.Client(self.sockets['curobo_dual_arm']).request(directory)
        self.assertTrue(result['result']['success'])
        self.assertFalse(result['result']['snapshot_present_at_entry'])
        self.assertEqual(read_json(directory/'candidate_io.json')['source'], 'shared_memory')
        self.assertTrue(read_json(directory/'candidate_snapshot_commit.json')['persisted'])

    def test_background_failure_fails_request_after_consumer_received_candidates(self):
        from arx_project.artifacts import ArtifactWriteError
        directory = self.root/'test_trial_write_failure'
        self.Client(self.sockets['zerograsp']).request(directory)
        with self.assertRaises(ArtifactWriteError):
            self.Client(self.sockets['curobo_dual_arm'],candidate_transport=True).request(directory)
        self.assertTrue((directory/'consumer_started.json').exists())
        self.assertFalse((directory/'candidate_snapshot_commit.json').exists())

    def test_empty_pool_commits_without_waiting_for_a_planner_call(self):
        directory = self.root/'test_trial_empty_pool'
        result = self.Client(self.sockets['zerograsp']).request(directory)
        self.assertEqual(result['result']['candidate_count'], 0)
        self.assertTrue(read_json(directory/'candidate_snapshot_commit.json')['persisted'])
        from arx_project.artifacts import candidate_transport
        self.assertIsNone(candidate_transport(directory))

    def test_file_only_consumer_keeps_existing_contract(self):
        directory = self.root/'test_trial_legacy_consumer'
        self.Client(self.sockets['zerograsp']).request(directory)
        result = self.Client(self.sockets['curobo_dual_arm'],candidate_transport=False).request(directory)
        self.assertTrue(result['result']['success'])
        self.assertTrue(result['result']['snapshot_present_at_entry'])
        self.assertEqual(read_json(directory/'candidate_io.json')['source'],'single_snapshot_read')


if __name__ == '__main__':
    unittest.main()
