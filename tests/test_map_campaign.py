"""A fresh 100-scene authorization keeps history and uses the optional service."""
from copy import deepcopy
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from arx_project.layout import ROOT, read_json, sha256, write_json
from modules.pipeline.experiments.mobile_dual_arm.campaign import (
    load_variant, preserved_predecessor, verify_preserved_and_resources)
from modules.pipeline.experiments.mobile_dual_arm.run import CONFIG, summarize
from modules.pipeline.experiments.dual_arm_grasp.run import Services, allocation_probes, choose_gpus
import subprocess


class CampaignTests(unittest.TestCase):
    def setUp(self):
        (ROOT/'tmp/tests').mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='map_campaign_', dir=ROOT/'tmp/tests')
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_map_variant_retains_scenes_perception_and_explicit_budget(self):
        variant = load_variant('configs/experiments/sam3_graspgenx_v4_map100.json', CONFIG)
        self.assertEqual(variant['dataset'], read_json(CONFIG)['dataset'])
        self.assertEqual(variant['physical_slot_budget'], 100)
        self.assertEqual(variant['component_modules'], {'curobo_dual_arm': 'curobo_dual_arm_map'})
        self.assertEqual(variant['station_search_options']['planner_module'], 'curobo_dual_arm')
        self.assertNotIn('validation', variant['station_search_options'])
        self.assertEqual(variant['station_search_options']['trajopt_policy'], 'interactive')

    def test_scope_drift_is_rejected(self):
        original = read_json(ROOT/'configs/experiments/sam3_graspgenx_v4_map100.json')
        for changes in [{'physical_slot_budget': 101}, {'dataset': 'simulation/scenes/latest200'},
                        {'component_modules': {'sam3': 'rayst3r'}}]:
            path = self.root/'variant.json'; write_json(path, {**original, **changes})
            with self.assertRaises(ValueError):
                load_variant(path, CONFIG)

    def predecessor(self):
        old = self.root/'prior'; (old/'claims').mkdir(parents=True); (old/'trials').mkdir()
        current_source = self.root/'model.py'; current_source.write_text('original')
        archived = old/'source_registered'/current_source.relative_to(ROOT)
        archived.parent.mkdir(parents=True); shutil.copy2(current_source, archived)
        write_json(old/'registration.json', {'dataset_freeze_sha256': 'frozen',
            'source_hashes': {str(current_source): sha256(current_source)}})
        write_json(old/'acceptance.json', {'record_integrity': True, 'completed': 0,
            'interrupted': 0, 'not_started': 100, 'successes': 0})
        write_json(old/'STOP', {'reason': 'prior resource fault'})
        write_json(old/'services_stopped.json', {'all_owned_exited': True})
        current_source.write_text('new optional implementation')
        return old

    def test_old_source_archive_and_stop_are_preserved(self):
        old = self.predecessor()
        result = preserved_predecessor({'predecessor': str(old.relative_to(ROOT))}, 'frozen')
        self.assertTrue(result['historical_source_archive_verified'])
        self.assertTrue(result['original_STOP_retained'])
        verify_preserved_and_resources({'predecessor': result})
        (old/'STOP').unlink()
        with self.assertRaises(OSError):
            verify_preserved_and_resources({'predecessor': result})

    def test_scene_mismatch_and_running_predecessor_stop_new_batch(self):
        old = self.predecessor()
        with self.assertRaisesRegex(ValueError, 'scenes differ'):
            preserved_predecessor({'predecessor': str(old.relative_to(ROOT))}, 'other')
        write_json(old/'services_stopped.json', {'all_owned_exited': False})
        with self.assertRaisesRegex(ValueError, 'have not exited'):
            preserved_predecessor({'predecessor': str(old.relative_to(ROOT))}, 'frozen')

    def test_same_length_resource_mutation_is_detected(self):
        path = self.root/'index.bin'; path.write_bytes(b'abc')
        stat = path.stat()
        reg = {'map_resources': {str(path): {'size': 3, 'mtime_ns': stat.st_mtime_ns, 'sha256': sha256(path)}}}
        verify_preserved_and_resources(reg, full_resources=True)
        path.write_bytes(b'xyz')
        with self.assertRaises(RuntimeError):
            verify_preserved_and_resources(reg, full_resources=True)

    def test_partial_result_is_not_a_completed_grasp_failure(self):
        (self.root/'claims').mkdir(); (self.root/'trials/a').mkdir(parents=True)
        schedule = [{'id': 'a', 'layout': 'S1_open'}, {'id': 'b', 'layout': 'S1_open'}]
        write_json(self.root/'registration.json', {'schedule': schedule})
        write_json(self.root/'claims/a.json', {'state': 'claimed'})
        write_json(self.root/'trials/a/result.json', {'id': 'a', 'layout': 'S1_open', 'physical_attempt': True,
                   'success': False, 'reason': 'CUDA fault'})
        stopped = summarize(self.root, 'stopped')
        self.assertEqual((stopped['completed'], stopped['interrupted'], stopped['not_started']), (0, 1, 1))
        self.assertEqual(stopped['failure_reasons'], {})
        self.assertEqual(stopped['recorded_physical_attempts'], 1)
        running = summarize(self.root, 'running')
        self.assertEqual((running['interrupted'], running['in_progress']), (0, 1))

    def test_new_report_does_not_overwrite_original_report(self):
        from modules.pipeline.experiments.mobile_dual_arm import finish
        docs = self.root/'modules/pipeline/docs/mainline_versions'; docs.mkdir(parents=True)
        old = docs/'V4_100场景测试结果.md'; old.write_text('preserved')
        write_json(self.root/'registration.json', {'variant': 'map', 'report_document': 'V4_地图主动站位100场景测试结果.md'})
        write_json(self.root/'summary.json', {'results': [], 'station_ready': 0, 'planning_passes': 0,
            'physical_attempts': 0, 'successes': 0, 'by_layout': {}})
        report = {'results': [], 'completed': 0, 'interrupted': 0, 'not_started': 100, 'passed': True, 'video_count': 0}
        with patch.object(finish, 'ROOT', self.root), patch.dict(sys.modules,
                {'publish_mobile_v4': SimpleNamespace(REASONS={})}):
            finish.write_report(self.root, report)
        self.assertEqual(old.read_text(), 'preserved')
        self.assertIn('地图主动站位', (docs/'V4_地图主动站位100场景测试结果.md').read_text())


class ServiceBindingTests(unittest.TestCase):
    def test_one_probe_timeout_does_not_discard_healthy_devices(self):
        def probe(*args, **kwargs):
            if kwargs['env']['CUDA_VISIBLE_DEVICES'] == 'bad':
                raise subprocess.TimeoutExpired('minimal_cuda_probe', 60)
            return SimpleNamespace(returncode=0, stdout='working', stderr='')
        with patch('modules.pipeline.experiments.dual_arm_grasp.run.subprocess.run', side_effect=probe):
            results = allocation_probes(['good', 'bad'])
        self.assertTrue(results['good']['passed'])
        self.assertFalse(results['bad']['passed'])
        self.assertEqual(results['bad']['reason'], 'allocation_probe_timeout')

    def test_deferred_device_is_not_used_without_a_fresh_probe(self):
        available = {name: 24000 for name in ['a', 'b', 'c', 'deferred']}
        checked = {name: {'passed': True} for name in ['a', 'b', 'c']}
        with patch('modules.pipeline.experiments.dual_arm_grasp.run.gpu_inventory', return_value=available), \
             patch('modules.pipeline.experiments.dual_arm_grasp.run.allocation_probes', return_value=checked) as probe:
            roles, _, _ = choose_gpus(candidate_gpus=['a', 'b', 'c'])
        self.assertEqual(set(probe.call_args.args[0]), {'a', 'b', 'c'})
        self.assertNotIn('deferred', roles.values())

    def test_service_alias_uses_registered_factory_and_logical_rpc_role(self):
        (ROOT/'tmp/tests').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='service_binding_', dir=ROOT/'tmp/tests') as folder:
            root = Path(folder)
            (root/'resident').mkdir(); (root/'preflight').mkdir()
            reg = {'required_components': ['curobo_dual_arm'],
                   'component_modules': {'curobo_dual_arm': 'curobo_dual_arm_map'},
                   'gpu_roles': {'curobo_dual_arm': 'gpu-test', 'simulation': 'gpu-test'},
                   'nonce': folder}
            write_json(root/'registration.json', reg)
            services = Services(root, reg)
            ready = Mock(); child = Mock(); child.poll.return_value = None; child.returncode = 0
            try:
                with patch.dict(sys.modules, {'arx_service_readiness': SimpleNamespace(wait_ready=ready)}), \
                     patch('modules.pipeline.experiments.dual_arm_grasp.run.subprocess.Popen', return_value=child) as popen, \
                     patch('modules.pipeline.experiments.dual_arm_grasp.run.gpu_inventory', return_value={'gpu-test': 5000}):
                    services.start()
                command = popen.call_args.args[0]
                self.assertEqual(command[command.index('--kind')+1], 'curobo_dual_arm_map')
                self.assertEqual(ready.call_args.args[3], 'curobo_dual_arm_map')
                self.assertEqual(ready.call_args.args[1].name, 'curobo_dual_arm_map_ready.json')
                with patch('modules.pipeline.experiments.dual_arm_grasp.run.Client') as client:
                    client.return_value.call.return_value = {'status': 'ok', 'returncode': 2, 'result': {'passed': True}}
                    answer = services.request('curobo_dual_arm', root/'trial', station_probe=True)
                    self.assertTrue(answer['result']['passed'])
            finally:
                child.poll.return_value = 0
                services.stop()
                services.sockets['curobo_dual_arm'].parent.rmdir()


if __name__ == '__main__':
    unittest.main()
