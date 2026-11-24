"""GPU placement contracts; mocked inventory, no CUDA initialization."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'scripts/curobo_arx'), str(ROOT / 'servers/lilo_vla')]
import gpu_policy
from backend import Policy
from evaluate_p2_policy import selected_mailboxes

DEVICES = [
    {'index': 0, 'uuid': 'GPU-zero', 'name': 'A100', 'compute_cap': '8.0'},
    {'index': 1, 'uuid': 'GPU-one', 'name': 'A100', 'compute_cap': '8.0'},
    {'index': 2, 'uuid': gpu_policy.POLICY_GPU, 'name': 'RTX 4090', 'compute_cap': '8.9'},
    {'index': 3, 'uuid': gpu_policy.COMPUTE_GPU, 'name': 'RTX 4090', 'compute_cap': '8.9'}]


class GpuPlacement(unittest.TestCase):
    def setUp(self):
        self.inventory = patch.object(gpu_policy, 'inventory', return_value=DEVICES)
        self.inventory.start(); self.addCleanup(self.inventory.stop)

    def test_physical_gpu_zero_and_one_are_authorized(self):
        for index in (0,1):
            self.assertEqual(gpu_policy.resolve(str(index))['uuid'], DEVICES[index]['uuid'])
            self.assertEqual(gpu_policy.resolve(DEVICES[index]['uuid'])['index'], index)

    def test_numeric_and_uuid_requests_map_to_the_same_device(self):
        self.assertEqual(gpu_policy.resolve('2'), gpu_policy.resolve(gpu_policy.POLICY_GPU))
        self.assertEqual(gpu_policy.resolve('3'), gpu_policy.resolve(gpu_policy.COMPUTE_GPU))

    def test_missing_or_multiple_visible_devices_are_rejected(self):
        for value in ('', '2,3', 'GPU-missing'):
            with self.subTest(value=value), self.assertRaises(ValueError):gpu_policy.resolve(value)

    def test_uuid_binding_survives_visible_renumbering(self):
        remapped=[{**DEVICES[2],'index':0},{**DEVICES[3],'index':1}]
        for original,visible in ((2,0),(3,1)):
            actual=gpu_policy.resolve(DEVICES[original]['uuid'],remapped)
            self.assertEqual(actual['visible_index'],visible)
            self.assertEqual(actual,gpu_policy.resolve(str(visible),remapped))

    def test_obsolete_uuid_cannot_select_new_card_at_same_index(self):
        with self.assertRaisesRegex(ValueError,'not visible'):
            gpu_policy.resolve('GPU-old',DEVICES)

    def test_architecture_follows_selected_card(self):
        result=gpu_policy.compute_env({},'0')
        self.assertEqual(result['TORCH_CUDA_ARCH_LIST'],'8.0')
        self.assertTrue(result['TORCH_EXTENSIONS_DIR'].endswith('sm80'))

    def test_old_architecture_and_visibility_cannot_leak_into_child(self):
        result = gpu_policy.compute_env({'CUDA_VISIBLE_DEVICES': '0', 'TORCH_CUDA_ARCH_LIST': '8.0'}, '3')
        self.assertEqual(result['CUDA_VISIBLE_DEVICES'], gpu_policy.COMPUTE_GPU)
        self.assertEqual(result['TORCH_CUDA_ARCH_LIST'], '8.9')
        self.assertTrue(result['TORCH_EXTENSIONS_DIR'].endswith('torch_extensions_sm89'))

    def test_cuda_ordinal_zero_can_represent_physical_gpu_two(self):
        with patch.object(gpu_policy.subprocess, 'check_output', return_value=f'{gpu_policy.POLICY_GPU}, 123\n'):
            proof = gpu_policy.verify_process(123, '2')
        self.assertEqual(proof['physical_gpu_index'], 2)
        self.assertTrue(proof['gpu_binding_verified'])

    def test_actual_forbidden_gpu_allocation_is_rejected(self):
        with patch.object(gpu_policy.subprocess, 'check_output', return_value='GPU-zero, 123\n'):
            with self.assertRaisesRegex(ValueError, 'differs'):gpu_policy.verify_process(123, '2')

    def test_health_accepts_4090_only_with_verified_binding(self):
        health = {'device': 'RTX 4090', 'gpu_uuid': gpu_policy.POLICY_GPU,
                  'gpu_process_pid': 123, 'gpu_binding_verified': True}
        with patch.object(gpu_policy.subprocess, 'check_output', return_value=f'{gpu_policy.POLICY_GPU}, 123\n'):
            self.assertEqual(gpu_policy.verify_health(health, '2')['physical_gpu_index'], 2)
        with self.assertRaisesRegex(ValueError, 'differs'):gpu_policy.verify_health(health, '3')
        health['gpu_binding_verified'] = False
        with self.assertRaisesRegex(ValueError, 'did not verify'):gpu_policy.verify_health(health)

    def test_backend_rejects_missing_device_before_model_loading(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-missing'}):
            with self.assertRaisesRegex(ValueError, 'not visible'):Policy('/missing/checkpoint')

    def test_insufficient_memory_stops_before_cuda_or_model_loading(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': gpu_policy.POLICY_GPU}), \
             patch('backend.memory_preflight', return_value={'weights_fit_lower_bound': False}):
            with self.assertRaisesRegex(MemoryError, 'before CUDA'):Policy('/missing/checkpoint')

    def test_memory_preflight_is_only_a_necessary_lower_bound(self):
        with patch.object(gpu_policy, 'checkpoint_minimum_bytes', return_value=20 * 1024 ** 2), \
             patch.object(gpu_policy.subprocess, 'check_output', return_value='10\n'):
            report = gpu_policy.memory_preflight('/unused', '2')
        self.assertFalse(report['weights_fit_lower_bound'])
        self.assertTrue(report['additional_head_and_activation_memory_required'])

    def test_single_scene_does_not_require_paused_fixture(self):
        self.assertEqual(selected_mailboxes('A1', '/arx/main', '/arx/paused'), [Path('/arx/main')])
        self.assertEqual(selected_mailboxes('bottle', '/arx/main', '/arx/paused'), [Path('/arx/paused')])
        self.assertEqual(selected_mailboxes(None, '/arx/main', '/arx/paused'), [Path('/arx/main'), Path('/arx/paused')])


class LauncherContracts(unittest.TestCase):
    def launch(self, gpu, stale_perception=False, remapped=False):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            devices=[{**DEVICES[2],'index':0},{**DEVICES[3],'index':1}] if remapped else DEVICES
            inventory = '\n'.join(f"{d['index']}, {d['uuid']}, {d['name']}, {d['compute_cap']}" for d in devices)
            smi = folder / 'nvidia-smi'
            smi.write_text('#!' + sys.executable + '\nprint(' + repr(inventory) + ')\n'); smi.chmod(0o755)
            marker = folder / 'started.json'
            worker = folder / 'worker'
            worker.write_text('#!' + sys.executable + '\nimport os,json\nfrom pathlib import Path\nPath(' +
                repr(str(marker)) + ').write_text(json.dumps({k:os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES","TORCH_CUDA_ARCH_LIST")}))\n')
            worker.chmod(0o755)
            env = {**os.environ, 'PATH': str(folder) + os.pathsep + os.environ['PATH'],
                   'LILO_GPU': gpu, 'LILO_PYTHON': str(worker)}
            command = ['bash', str(ROOT / 'scripts/run_lilo_vla_server.sh')]
            if stale_perception:
                env['ARX_NEURAL_GPU'] = '0'
                command = ['bash', '-c', 'source "$1" && exec "$2"', 'arx-env',
                           str(ROOT / 'scripts/curobo_arx/environment.sh'), str(worker)]
            result = subprocess.run(command,
                                    env=env, capture_output=True, text=True, timeout=30)
            return result, json.loads(marker.read_text()) if marker.exists() else None

    def test_rejected_gpu_never_starts_worker(self):
        result, marker = self.launch('GPU-missing')
        self.assertNotEqual(result.returncode, 0); self.assertIsNone(marker)
        self.assertIn('not visible', result.stderr)

    def test_allowed_gpu_is_exported_as_uuid_before_worker_start(self):
        result, marker = self.launch('2')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(marker, {'CUDA_VISIBLE_DEVICES': gpu_policy.POLICY_GPU, 'TORCH_CUDA_ARCH_LIST': '8.9'})

    def test_sourced_environment_allows_gpu_zero(self):
        result, marker = self.launch('2', stale_perception=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIsNotNone(marker)

    def test_server_launch_survives_gpu_visibility_remapping(self):
        result,marker=self.launch(gpu_policy.POLICY_GPU,remapped=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(marker['CUDA_VISIBLE_DEVICES'],gpu_policy.POLICY_GPU)

    def test_simulator_dry_run_binds_one_uuid_and_uses_local_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);smi=folder/'nvidia-smi'
            inventory='\n'.join(f"{i}, {d['uuid']}, {d['name']}, {d['compute_cap']}" for i,d in enumerate(DEVICES[2:]))
            smi.write_text('#!'+sys.executable+'\nprint('+repr(inventory)+')\n');smi.chmod(0o755)
            env={**os.environ,'PATH':str(folder)+os.pathsep+os.environ['PATH'],'ARX_SIM_GPU':gpu_policy.COMPUTE_GPU,
                 'ARX_CUROBO_VALIDATION_DIR':'/recorded/mailbox','ARX_TABLETOP_CATALOG':'/recorded/catalog',
                 'ARX_HISTORICAL_SETUP':'/recorded/initial','ARX_REPRO_REFERENCE':'/recorded/reference'}
            env.pop('ARX_SIM_DEVICE',None)
            result=subprocess.run(['bash',str(ROOT/'scripts/curobo_arx/start_sim_container.sh'),'--dry-run'],
                                  env=env,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('nvidia.com/gpu='+gpu_policy.COMPUTE_GPU,result.stdout)
        self.assertIn('ARX_SIM_GPU=0',result.stdout)
        self.assertIn('CUDA_VISIBLE_DEVICES='+gpu_policy.COMPUTE_GPU,result.stdout)
        self.assertNotIn('nvidia.com/gpu=all',result.stdout)

    def test_teacher_only_requires_explicit_goal_before_any_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/'unstarted'
            command=[sys.executable,str(ROOT/'scripts/curobo_arx/lilo_vla/probe_transfer.py'),
                     '--teacher-only','--output',str(output),'--object-name','apple_01',
                     '--adapter','/unused','--policy-gpu','2','--planner-gpu','3','--neural-gpu','3']
            result=subprocess.run(command,env=os.environ,capture_output=True,text=True,timeout=30)
            self.assertFalse(output.exists())
        self.assertNotEqual(result.returncode,0)
        self.assertIn('--teacher-only requires --goal-source',result.stderr)


if __name__ == '__main__':unittest.main()
