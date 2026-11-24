"""Offline scheduling-fault checks with the registered ARX trial ledgers."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts.curobo_arx.data_factory.common import ROOT, read, write
from .advisor37_seed import import_probe
from .advisor37_orchestrator import require_unstarted, service_state

FIXTURE = ROOT / 'runs/nearfield_advisor37_20261112_01/seed_preflight_02'


class ContinuationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def arx_prepared(self):
        shutil.copy2(FIXTURE / 'registration.json', self.root / 'registration.json')
        shutil.copytree(FIXTURE / 'trial_ledgers', self.root / 'trial_ledgers')
        write(self.root / 'state.json', {'status': 'prepared'})
        write(self.root / 'full_preflight.json', {'pass': True})
        write(self.root / 'container_preflight.json', {'pass': True})

    def test_never_replay_started_prepared_child(self):
        self.arx_prepared()
        write(self.root / 'STARTED.json', {'wall': 1})
        with self.assertRaisesRegex(RuntimeError, 'execution evidence'):
            require_unstarted(self.root)

    def test_prepared_ARX_ledgers_need_no_execution_history(self):
        import sqlite3
        self.arx_prepared()
        path = next((self.root / 'trial_ledgers').glob('*.sqlite3'))
        with sqlite3.connect(path) as db:
            db.execute('INSERT INTO events (attempt_id,event,value,unix) VALUES (?,?,?,?)',
                       ('unexpected', 'dispatch', '{}', 1.))
        with self.assertRaisesRegex(RuntimeError, 'execution history'):
            require_unstarted(self.root)

    def test_prepared_ARX_waits_without_service_or_dispatch(self):
        self.arx_prepared()
        with patch('scripts.curobo_arx.nearfield.advisor37_orchestrator.subprocess.run') as run:
            self.assertEqual(service_state(self.root)['status'], 'prepared')
            run.assert_not_called()

    def test_launched_child_is_not_dispatched_twice_during_startup(self):
        self.arx_prepared()
        write(self.root / 'STARTED.json', {'wall': 1})
        write(self.root / 'supervisor_service.json', {'unit': 'arx-registered-test.service'})
        with patch('scripts.curobo_arx.nearfield.advisor37_orchestrator.subprocess.run',
                   return_value=subprocess.CompletedProcess([], 0, 'active\n', '')):
            self.assertEqual(service_state(self.root)['status'], 'launching')

    def test_STOP_blocks_even_if_prepared(self):
        self.arx_prepared()
        (self.root / 'STOP').write_text('preserve')
        with self.assertRaisesRegex(RuntimeError, 'fault/STOP'):
            service_state(self.root)

    def test_import_timeout_has_cleanup_and_failure_receipt(self):
        process = Mock(returncode=-9)
        process.communicate.side_effect = [subprocess.TimeoutExpired('podman', 600), ('', '')]
        outcomes = [subprocess.CompletedProcess([], 1, '', ''),
                    subprocess.CompletedProcess([], 0, '', ''),
                    subprocess.CompletedProcess([], 1, '', '')]
        with patch('scripts.curobo_arx.nearfield.advisor37_seed.subprocess.Popen', return_value=process), \
             patch('scripts.curobo_arx.nearfield.advisor37_seed.subprocess.run', side_effect=outcomes) as run:
            with self.assertRaisesRegex(RuntimeError, 'no automatic retry'):
                import_probe(self.root, FIXTURE / 'source_snapshot/COOLAS-ARX')
        receipt = read(self.root / 'container_preflight.json')
        self.assertFalse(receipt['pass'])
        self.assertTrue(receipt['container_absent'])
        self.assertTrue(receipt['timed_out'])
        process.kill.assert_called_once()
        self.assertEqual(run.call_args_list[1].args[0][:3], ['podman', 'rm', '--force'])

    def test_successful_import_cannot_hide_failed_cleanup(self):
        process = Mock(returncode=0)
        process.communicate.return_value = ('seed_runtime_import_ok\n', '')
        outcomes = [subprocess.CompletedProcess([], 1, '', ''),
                    subprocess.CompletedProcess([], 125, '', 'cleanup failed'),
                    subprocess.CompletedProcess([], 0, '', '')]
        with patch('scripts.curobo_arx.nearfield.advisor37_seed.subprocess.Popen', return_value=process), \
             patch('scripts.curobo_arx.nearfield.advisor37_seed.subprocess.run', side_effect=outcomes):
            with self.assertRaisesRegex(RuntimeError, 'no automatic retry'):
                import_probe(self.root, FIXTURE / 'source_snapshot/COOLAS-ARX')
        self.assertFalse(read(self.root / 'container_preflight.json')['pass'])


if __name__ == '__main__':
    unittest.main()
