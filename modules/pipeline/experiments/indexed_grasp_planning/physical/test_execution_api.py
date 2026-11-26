"""The physical API must preserve initial plans during measured-stage replanning."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from arx_project.layout import write_json
from modules.curobo.indexed_planning.execution_service import ExecutionPlanner
from modules.curobo.indexed_planning.stage_timing import StageTiming


def fake_pool():
    pool = SimpleNamespace(endpoint_sequence=None, stage_timing=StageTiming(),
        contexts={}, batch_engines={}, pure_solver=SimpleNamespace(reset_seed=lambda: None),
        binding_codec=SimpleNamespace(reset=lambda: None, report=lambda: {}),
        mesh_vote=SimpleNamespace(enabled=False, assert_original_untouched=lambda: None))
    pool.mesh_vote_mode = lambda value: setattr(pool.mesh_vote, 'enabled', value)
    return pool


class ExecutionAPITests(unittest.TestCase):
    def test_measured_replanning_keeps_initial_plan_and_full_response(self):
        for stage in ['approach', 'lift']:
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp);original = {'success': True, 'selected': {'arm': 'left', 'index': 9}}
                write_json(out/'plan_result.json', original)
                result = {'success': True, 'paths': {stage: {'q': [[0.]*6], 'times': [0.]}}}
                def request(directory, **flags):
                    write_json(Path(directory)/(stage+'_plan_result.json'), result);return result
                planner = SimpleNamespace(pool=fake_pool(), request=request)
                torch = SimpleNamespace(manual_seed=lambda seed: None, cuda=SimpleNamespace(
                    synchronize=lambda: None, mem_get_info=lambda: (16*1024**3, 24*1024**3)))
                with patch.dict('sys.modules', torch=torch), \
                     patch('modules.curobo.scene_cache.mesh_stores', return_value=[]), \
                     patch('modules.curobo.scene_cache.prune_inactive_meshes', return_value={}):
                    actual = ExecutionPlanner(planner).request(out, execution_variant='optimized', **{stage: True})
                self.assertEqual(actual, result)
                self.assertEqual(json.loads((out/'plan_result.json').read_text()), original)
                self.assertTrue((out/('execution_planner_'+stage+'.json')).exists())

    def test_resource_fault_does_not_call_planner(self):
        planner = SimpleNamespace(pool=fake_pool(), request=lambda *a, **k: self.fail('Planner ran without reserve'))
        torch = SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda: (1024**3, 24*1024**3)))
        with tempfile.TemporaryDirectory() as tmp, patch.dict('sys.modules', torch=torch):
            with self.assertRaises(RuntimeError):
                ExecutionPlanner(planner).request(Path(tmp), execution_variant='baseline')


if __name__ == '__main__': unittest.main()
