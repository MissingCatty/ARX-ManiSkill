"""Execution stage correctness and background-write failure propagation."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from arx_project.layout import read_json, write_json
from .worker import Worker


class ExecutionRequests(unittest.TestCase):
    def invoke(self, root, mode, stage, finish_error=None, request_error=None):
        write_json(root / 'registration.json', {'cases': [{'id': 'fixture'}]})
        directory = root / 'trial'; directory.mkdir()
        write_json(directory / 'plan_result.json', {'success': True, 'selected': {'sentinel': 'initial'}})
        pool = MagicMock(); pool.contexts = {}; pool.stage_timing.report.return_value = {'stages': {}}
        pool.memory_bindings.finish.return_value = {'receipts': 'drained'}
        pool.memory_bindings.finish.side_effect = finish_error
        def request(out, **flags):
            if request_error: raise request_error
            result = {'success': True, 'paths': {stage: 'measured command'}}
            write_json(Path(out) / (stage + '_plan_result.json'), result)
            return result
        planner = SimpleNamespace(pool=pool, request=request)
        worker = Worker(planner, root); torch = MagicMock()
        torch.cuda.mem_get_info.return_value = (2 * 1024**3, 24 * 1024**3)
        with patch.dict('sys.modules', {'torch': torch}), \
                patch('modules.curobo.scene_cache.mesh_stores', return_value=[]), \
                patch('modules.curobo.scene_cache.prune_inactive_meshes'):
            if finish_error or request_error:
                with self.assertRaises(type(finish_error or request_error)):
                    worker.request(directory, execution_variant=mode, case_id='fixture', **{stage: True})
            else:
                worker.request(directory, execution_variant=mode, case_id='fixture', **{stage: True})
        return directory, pool

    def test_each_stage_preserves_initial_selection_and_records_actual_variant(self):
        for mode in ['baseline', 'optimized']:
            for stage in ['approach', 'lift']:
                with tempfile.TemporaryDirectory() as tmp:
                    directory, pool = self.invoke(Path(tmp), mode, stage)
                    self.assertEqual(read_json(directory / 'plan_result.json')['selected']['sentinel'], 'initial')
                    receipt = read_json(directory / ('execution_planner_' + stage + '.json'))
                    self.assertTrue(receipt['measured_execution_state_used'])
                    self.assertEqual(receipt['memory_state_updates_enabled'], mode == 'optimized')
                    self.assertFalse(pool.memory_updates_enabled)
                    self.assertEqual(pool.memory_bindings.begin.call_count, int(mode == 'optimized'))
                    self.assertEqual(pool.memory_bindings.finish.call_count, int(mode == 'optimized'))

    def test_disk_failure_cannot_return_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory, _ = self.invoke(Path(tmp), 'optimized', 'lift', finish_error=OSError('disk full'))
            self.assertTrue((directory / 'infrastructure_error.json').exists())
            self.assertFalse((directory / 'execution_planner_lift.json').exists())
            self.assertNotIn('complete', (directory / 'request_ledger.jsonl').read_text())

    def test_program_fault_drains_then_propagates(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory, pool = self.invoke(Path(tmp), 'optimized', 'approach', request_error=RuntimeError('bad state'))
            pool.memory_bindings.finish.assert_called_once()
            self.assertTrue((directory / 'infrastructure_error.json').exists())
            self.assertFalse((directory / 'execution_planner_approach.json').exists())


if __name__ == '__main__': unittest.main()
