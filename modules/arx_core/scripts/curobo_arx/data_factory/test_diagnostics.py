"""Offline logger checks; real SDK pressure evidence is stored separately."""
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock
from .diagnostics import FaultDiagnostics


class DiagnosticTests(unittest.TestCase):
    def test_periodic_sampling_is_synchronous_without_watchdog_thread(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'trace.log';backend=Mock();clock=[0.]
            diag=FaultDiagnostics(path,backend=backend,clock=lambda:clock[0])
            try:
                backend.enable.assert_called_once_with(file=diag.stream,all_threads=False)
                clock[0]=119.;self.assertFalse(diag.tick())
                clock[0]=120.;self.assertTrue(diag.tick())
                self.assertIn('SAFE_MAIN_THREAD_STACK',path.read_text())
                backend.dump_traceback_later.assert_not_called()
            finally:diag.close()

    def test_cross_thread_sampling_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            diag=FaultDiagnostics(Path(tmp)/'trace.log',backend=Mock());errors=[]
            def worker():
                try:diag.tick()
                except RuntimeError as exc:errors.append(str(exc))
            thread=threading.Thread(target=worker);thread.start();thread.join()
            self.assertEqual(len(errors),1);self.assertIn('main thread',errors[0]);diag.close()

    def test_fatal_handler_is_disabled_before_its_file_is_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend=Mock();diag=FaultDiagnostics(Path(tmp)/'trace.log',backend=backend)
            backend.disable.side_effect=lambda:self.assertFalse(diag.stream.closed)
            diag.close();self.assertTrue(diag.stream.closed);diag.close()
            backend.disable.assert_called_once()
            with self.assertRaisesRegex(RuntimeError,'closed'):diag.tick()


if __name__=='__main__':unittest.main()
