"""ARX-only archived-data tests of invariance checks and shape fallback."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from client import Client
from research_guard import check, invariant_digest
from research_layout import compare_layout
from shape_policy import select_refinement

ROOT = Path(__file__).resolve().parents[2]


class Constraints(unittest.TestCase):
    def test_unchanged_archived_arx_layout_passes(self):
        snap = json.loads((ROOT / 'runs/bottle_tests_20261019/01_mustard_bottle_upright/initial_snapshot.json').read_text())
        state = {**snap, 'robot_base_world': snap['link_world']['body4']}
        self.assertTrue(compare_layout(state, copy.deepcopy(state))['pass'])
        changed = copy.deepcopy(state)
        changed['objects']['test_object']['world'][0][3] += .01
        self.assertFalse(compare_layout(state, changed)['pass'])

    def test_robot_base_move_rejected(self):
        snap = json.loads((ROOT / 'runs/bottle_tests_20261019/01_mustard_bottle_upright/initial_snapshot.json').read_text())
        state = {**snap, 'robot_base_world': snap['link_world']['body4']}
        changed = copy.deepcopy(state)
        changed['robot_base_world'][0][3] += .01
        self.assertIn('robot_base', compare_layout(state, changed)['failures'])

    def test_forbidden_config_actions_never_create_request(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {'ARX_RESEARCH_GUARD_FILE': d + '/guard.json'}):
            client = Client(d)
            for action in ('grasp_arm_gains', 'grasp_solver_iterations', 'obstacle'):
                with self.assertRaisesRegex(RuntimeError, 'forbidden'):
                    client.request(action)
            self.assertFalse((Path(d) / 'request.json').exists())

    def test_nonfinite_default_is_explicit_not_ignored(self):
        path = ROOT / 'arx-sim/arx_sim/diagnostics/research_fingerprint.py'
        spec = importlib.util.spec_from_file_location('fingerprint_test', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        self.assertNotEqual(module.digest({'limit': float('inf')}), module.digest({'limit': 10.}))
        self.assertEqual(module.plain(float('inf')), '+Inf')
        self.assertEqual(module.plain(float('-inf')), '-Inf')

    def test_live_fingerprint_configuration_change_rejected(self):
        path = ROOT / 'runs/research_contract_20261019/live_fingerprint.json'
        if not path.exists():
            self.skipTest('Live read-only ARX fingerprint not available yet')
        evidence = json.loads(path.read_text())
        with tempfile.TemporaryDirectory() as d:
            guard = Path(d) / 'guard.json'
            guard.write_text(json.dumps({'invariant_sha256': invariant_digest(evidence), 'file_sha256': {}}))
            class Reader:
                def request(self, *args, **kwargs): return evidence
            self.assertIsNotNone(check(Reader(), guard, 'same', 'test'))
            changed = copy.deepcopy(evidence)
            changed['invariant']['runtime_robot']['get_dof_gains'][0][0][0] += 1
            class ChangedReader:
                def request(self, *args, **kwargs): return changed
            with self.assertRaisesRegex(RuntimeError, 'guard failed'):
                check(ChangedReader(), guard, 'changed', 'test')

    def test_renderer_world_pose_is_state_but_physical_configuration_remains_frozen(self):
        path = ROOT / 'runs/research_contract_20261019/live_fingerprint.json'
        if not path.exists(): self.skipTest('Live ARX capture not available')
        evidence = json.loads(path.read_text()); original = invariant_digest(evidence)
        moved = copy.deepcopy(evidence)
        moved['invariant']['usd']['/World/COOLASSensors/head']['attributes']['xformOp:translate'][2] += .1
        self.assertEqual(invariant_digest(moved), original)
        modified = copy.deepcopy(evidence)
        modified['invariant']['configured']['controllers']['camera']['resolution'][0] += 10
        self.assertNotEqual(invariant_digest(modified), original)

    def test_actual_nonsphere_rgbd_preserves_original_proposal(self):
        # Real archived ARX inputs: strong successful bottle and two known failures.
        for relative in ('bottle_tests_20261019/01_mustard_bottle_upright',
                         'object_failure_search_20261019/01_bagel_00',
                         'object_failure_search_20261019/04_dry_erase_marker'):
            source = ROOT / 'runs' / relative
            with self.subTest(case=relative), tempfile.TemporaryDirectory() as d:
                folder = Path(d)
                shutil.copytree(source / 'camera', folder / 'camera')
                shutil.copy2(source / 'camera_world_optical.json', folder / 'camera_world_optical.json')
                original = json.loads((source / 'selected_grasp.json').read_text())
                route = object()
                def forbidden(*args): raise AssertionError('Non-sphere invoked refinement IK')
                proposal, returned = select_refinement(original, route, folder, torch.zeros(6),
                                                       forbidden, forbidden, None, None)
                self.assertIs(proposal, original)
                self.assertIs(returned, route)
                self.assertFalse(json.loads((folder / 'shape_policy.json').read_text())['applied'])


if __name__ == '__main__':
    unittest.main()
