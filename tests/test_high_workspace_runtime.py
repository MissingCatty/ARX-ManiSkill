"""Exercise preparation order and arm binding without a simulator or model."""
from copy import deepcopy
from pathlib import Path
import tempfile
import json
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from arx_project.layout import read_json, write_json, sha256
from modules.pipeline.experiments.mobile_dual_arm.workspace_priority import runtime


class PreparationTest(unittest.TestCase):
    def exercise(self, root, *, fail_shared=False, corrupt_motion=False):
        calls = []
        state = {'joint_names': ['joint4'], 'qpos': [.4],
                 'links': {'base_link': np.eye(4).tolist(), 'body4': np.eye(4).tolist()},
                 'intrinsics': np.eye(3).tolist(), 'camera_world_optical': np.eye(4).tolist()}
        env = SimpleNamespace(get_obs=lambda: None, initial_q=np.array([.4]), spawn={}, object_bbox={})
        executor = SimpleNamespace(snapshot=lambda *a: deepcopy(state))

        class Search:
            step = .05
            def proposals(self, s, target, side):
                return [{'cost': 0 if side == 'left' else 1}]
            def select(self, s, pool, target, directory, *, geometry_only, scopes):
                assert geometry_only and pool == {'candidates': []}
                scope = scopes[0]
                calls.append('geometry:'+scope)
                if scope == 'shared' and fail_shared:
                    return None, {'failed_stage': 'high_workspace_or_body_transit'}
                return {'proposal': {'base_world': np.eye(4).tolist(), 'lift_m': .4,
                                    'workspace_scope': scope}, 'path': [], 'transit': {'passed': True}}, {}
            def workspace_gate(self, *args):
                return {'passed': True, 'cell': [0, 0, 1]}

        class Services:
            def request(self, role, directory, **flags):
                calls.append(role)
                result = {'target_found': True, 'point_count': 100}
                if role == 'sam3':
                    Image.fromarray(np.ones((2, 2), dtype=np.uint8)*255).save(directory/'camera/mask.png')
                if role == 'curobo_dual_arm':
                    arm = flags.get('arm', 'right')
                    assert arm == ('left' if fail_shared else 'right')
                    result = {'success': True, 'selected': {'arm': arm}}
                    write_json(directory/'plan_result.json', result)
                return {'result': result}

        def capture(unused, directory, *args):
            (directory/'camera').mkdir(parents=True)
            write_json(directory/'state.json', state)
            Image.fromarray(np.ones((2, 2, 3), dtype=np.uint8)).save(directory/'camera/color.png')
            Image.fromarray(np.ones((2, 2, 3), dtype=np.uint8)).save(directory/'overview.jpg')
            np.save(directory/'camera/depth_m.npy', np.ones((2, 2)))

        def candidates(directory):
            write_json(directory/'candidates/candidate_pool.json', {'candidates': [{'index': 7}]})

        station = runtime.HighWorkspaceStationing(Services(), executor, None, {}, 7, station_search=Search())
        def move(unused, directory, *args):
            calls.append('body_motion')
            for name in ['receipt.json', 'settled_state.json', 'contacts.json']:
                write_json(directory/name, state)
            for name in ['execution.jsonl', 'motion.mp4', 'head.mp4']:
                (directory/name).write_bytes(b'fixture')
            return {'success': True, 'motion_started': True}
        station.execute = move
        def audit(*args, **kwargs):
            if corrupt_motion:
                raise RuntimeError('station not idle')
        with patch.object(runtime, 'capture', side_effect=capture), \
             patch.object(runtime, 'backproject', return_value=np.ones((4, 3))*.1), \
             patch.object(runtime, 'audit_segment', side_effect=audit), \
             patch.dict('sys.modules', {'repair_contract': SimpleNamespace(make_candidates=candidates)}):
            receipt = station.prepare(env, root, {'base_pose': [0, 0, 0]})
        return calls, receipt

    def test_move_then_fresh_reconstruction_plan_and_handoff(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            calls, receipt = self.exercise(root)
            self.assertEqual(calls, ['sam3', 'geometry:shared', 'body_motion',
                                     'sam3', 'rayst3r', 'graspgenx', 'curobo_dual_arm'])
            self.assertTrue(receipt['prepared_plan'])
            self.assertEqual(receipt['workspace_scope'], 'shared')
            self.assertEqual(read_json(root/'plan_result.json')['selected']['arm'], 'right')
            self.assertEqual(len(receipt['stationing_segments']), 1)

    def test_single_fallback_binds_better_arm(self):
        with tempfile.TemporaryDirectory() as folder:
            calls, receipt = self.exercise(Path(folder), fail_shared=True)
            self.assertEqual(calls[:3], ['sam3', 'geometry:shared', 'geometry:left'])
            self.assertEqual(receipt['preferred_arm'], 'left')
            self.assertEqual(receipt['workspace_scope'], 'left')
            self.assertEqual(sum(x == 'curobo_dual_arm' for x in calls), 1)

    def test_motion_verification_fault_blocks_inference_and_planning(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, 'not idle'):
                self.exercise(Path(folder), corrupt_motion=True)


class ResourceTest(unittest.TestCase):
    def test_validated_ray_memory_requirement_allows_shared_gpus(self):
        from modules.pipeline.experiments.mobile_dual_arm.workspace_priority import resources
        memory = {g: 16198 for g in ['a', 'b', 'c']}
        with patch.object(resources, 'gpu_inventory', return_value=memory), \
             patch.object(resources, 'allocation_probes', return_value={g: {'passed': True} for g in memory}):
            roles, floors, _ = resources.select_resources(ray_minimum_mib=15360)
        self.assertEqual(len(set(roles.values())), 3)
        self.assertEqual(sorted(floors.values()), [10000, 12000, 15360])

    def test_two_cards_keep_ray_separate_and_require_empty_shared_card(self):
        from modules.pipeline.experiments.mobile_dual_arm.workspace_priority import resources
        memory = {'a': 24000, 'b': 24000, 'busy': 7000, 'broken': 24000}
        probes = {g: {'passed': g != 'broken'} for g in memory}
        with patch.object(resources, 'gpu_inventory', return_value=memory), \
             patch.object(resources, 'allocation_probes', return_value=probes):
            roles, floors, _ = resources.select_resources()
        self.assertNotEqual(roles['rayst3r'], roles['simulation'])
        self.assertEqual(roles['sam3'], roles['graspgenx'])
        self.assertEqual(roles['sam3'], roles['curobo_dual_arm'])
        self.assertNotIn('broken', roles.values())
        self.assertNotIn('busy', roles.values())
        self.assertEqual(sorted(floors.values()), [18000, 23000])

    def test_resource_shortage_stops_before_any_service_or_slot(self):
        from modules.pipeline.experiments.mobile_dual_arm.workspace_priority import resources
        with patch.object(resources, 'gpu_inventory', return_value={'a': 24000, 'b': 17000}), \
             patch.object(resources, 'allocation_probes', return_value={'a': {'passed': True}, 'b': {'passed': True}}):
            with self.assertRaisesRegex(RuntimeError, 'Need three admitted GPUs'):
                resources.select_resources()


class FailedPreparationAuditTest(unittest.TestCase):
    def fixture(self, root):
        from modules.pipeline.experiments.mobile_dual_arm.workspace_priority.runtime import audit_segment
        trial = root/'trials/case'
        station = trial/'stationing'
        segment = station/'segments/00_shared'
        segment.mkdir(parents=True)
        state = {'joint_names': ['joint4'], 'qpos': [.4],
                 'links': {'base_link': np.eye(4).tolist()},
                 'camera_world_optical': np.eye(4).tolist()}
        write_json(segment/'command.json', {'goal': {'lift_m': .4, 'base_world': np.eye(4).tolist()}})
        write_json(segment/'receipt.json', {'success': True, 'idle_samples': 60})
        write_json(segment/'settled_state.json', state)
        write_json(segment/'observation/state.json', state)
        write_json(segment/'contacts.json', [])
        (segment/'execution.jsonl').write_text((json.dumps({'qvel': [0.]})+'\n')*60)
        for name in ['motion.mp4', 'head.mp4']:
            (segment/name).write_bytes(b'video decoded by patched verifier')
        audit_segment(segment, record=True)
        for source in segment.iterdir():
            if source.is_file():
                shutil.copy2(source, station/source.name)
        write_json(station/'coarse/state.json', state)
        result = {'success': False, 'reason': 'unable_to_plan', 'physical_attempt': False,
                  'stationing': {'motion_started': True, 'stationing_segments': ['stationing/segments/00_shared']}}
        write_json(trial/'result.json', result)
        write_json(root/'claims/case.json', {'state': 'complete', 'result_sha256': sha256(trial/'result.json')})
        write_json(root/'dataset/frozen_dataset.json', {'files': {}})
        write_json(root/'registration.json', {'schedule': [{'id': 'case'}],
            'station_search_options': {'placement_method': 'high_workspace_priority'},
            'dataset': str(root/'dataset'), 'source_hashes': {}})
        return segment

    def check(self, *, missing_command):
        from modules.pipeline.experiments.mobile_dual_arm.finish import audit
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            segment = self.fixture(root)
            if missing_command:
                (segment/'command.json').unlink()
            with patch('modules.pipeline.experiments.mobile_dual_arm.finish.subprocess.run',
                       return_value=SimpleNamespace(returncode=0, stderr='')):
                return audit(root)

    def test_motion_then_no_grasp_uses_last_segment_observation(self):
        report = self.check(missing_command=False)
        self.assertTrue(report['passed'], report['issues'])
        self.assertEqual(report['completed'], 1)
        self.assertEqual(report['successes'], 0)

    def test_receipt_without_command_is_record_damage(self):
        report = self.check(missing_command=True)
        self.assertFalse(report['record_integrity'])
        self.assertTrue(any('damaged station segment' in issue for issue in report['issues']))


if __name__ == '__main__':
    unittest.main()
