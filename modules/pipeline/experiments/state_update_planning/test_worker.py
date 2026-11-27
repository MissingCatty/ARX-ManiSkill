from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from modules.curobo.indexed_planning.stage_timing import StageTiming
from .worker import Worker, OriginalWorker


class WorkerFaultTests(unittest.TestCase):
    def worker(self, finish=lambda: {'receipts': {'all_written': True}}):
        pool = SimpleNamespace(memory_updates_enabled=False, stage_timing=StageTiming(),
            binding_codec=SimpleNamespace(reset=lambda: None),
            memory_bindings=SimpleNamespace(begin=lambda: None, finish=finish))
        return Worker(SimpleNamespace(pool=pool))

    def test_background_write_failure_prevents_a_success_response(self):
        def failed():
            raise OSError('injected background disk failure')
        worker = self.worker(failed)
        with tempfile.TemporaryDirectory() as temp, patch.object(OriginalWorker, 'request', return_value={'success': True}):
            with self.assertRaisesRegex(OSError, 'disk failure'):
                worker.request(Path(temp), variant='memory')
            self.assertFalse((Path(temp) / 'planning_metrics.json').exists())
        self.assertFalse(worker.planner.pool.memory_updates_enabled)

    def test_original_planning_error_is_propagated_after_receipt_drain(self):
        drained = []
        worker = self.worker(lambda: drained.append(True))
        with tempfile.TemporaryDirectory() as temp, patch.object(OriginalWorker, 'request', side_effect=RuntimeError('injected planning error')):
            with self.assertRaisesRegex(RuntimeError, 'planning error'):
                worker.request(Path(temp), variant='memory')
        self.assertEqual(drained, [True])
        self.assertFalse(worker.planner.pool.stage_timing.active)

    def test_control_does_not_use_the_memory_writer(self):
        def forbidden():
            raise AssertionError('control entered memory implementation')
        worker = self.worker(forbidden)
        with tempfile.TemporaryDirectory() as temp, patch.object(OriginalWorker, 'request', return_value={'success': True}):
            value = worker.request(Path(temp), variant='control')
        self.assertIsNone(value['memory_updates'])

    def test_unknown_variant_does_not_start_a_request(self):
        worker = self.worker()
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            worker.request('/unused', variant='other')


if __name__ == '__main__':
    unittest.main()
