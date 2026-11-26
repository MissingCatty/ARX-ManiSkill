"""Fault checks for stale worlds, arm isolation and request lifetime."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from arx_project.layout import write_json, sha256
from . import context_reuse as reuse


class FakeBase:
    def __init__(self, root, warmup):
        self.contexts = {role: SimpleNamespace(out=None, planner=SimpleNamespace(
            reset_seed=lambda: self.resets.append(role))) for role in ['open', 'closed']}
        self.loads = {'open': 1, 'closed': 1}
        self.contact_views = {}
        self.pure_solver = SimpleNamespace(reset_seed=lambda: None)
        self.calls, self.resets = [], []
        self.failure = False

    def bind(self, directory, state, closed=False, include_target=False):
        self.calls.append((deepcopy(state), closed, include_target))
        if self.failure:
            raise RuntimeError('Simulated binding fault')
        directory.mkdir(parents=True, exist_ok=True)
        context = self.contexts['closed' if closed else 'open']
        context.out = directory
        for name in ['arx.yml', 'scene.json', 'resident_rebind.json', 'fk_validation.json']:
            write_json(directory/name, {'state': state})
        return context

    def contact_view(self, context, state):
        return SimpleNamespace(update=lambda *args: self.calls.append('validator_update'))


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.pool = reuse.reusable_pool(FakeBase)(self.root, None)
        self.pool.reuse_enabled = True
        self.pool.reuse_scope = 'one_request'
        self.pool.model_epoch = 'frozen_model'
        self.state = {'planning_arm': 'right', 'qpos': [0., .022],
            'links': {'body4': [1, 2, 3]}, 'joint_limits': [[-1, 1]],
            'scene_cuboids': {'table': {'center': [0, 0, .7], 'dims': [1, 1, .1]}}}
        self.gate = patch.object(reuse, 'verify_cached_start', return_value={'passed': True})
        self.check = self.gate.start()
        self.addCleanup(self.gate.stop)
        self.addCleanup(self.temp.cleanup)

    def bind(self, **kwargs):
        return self.pool.bind(self.root/'plan', self.state, **kwargs)

    def test_identical_window_reuses_binding_and_rechecks_start(self):
        self.assertIs(self.bind(), self.bind())
        self.assertEqual(len(self.pool.calls), 1)
        self.check.assert_called_once()
        self.assertEqual(len(self.pool.resets), 1)
        self.assertTrue(self.pool.binding_records[-1]['cache_hit'])

    def test_measured_state_changes_force_full_bind(self):
        changes = [('qpos', [.1, .022]), ('joint_limits', [[-2, 1]]),
                   ('links', {'body4': [2, 2, 3]}), ('planning_arm', 'left'),
                   ('scene_cuboids', {'table': {'center': [0, 0, .71], 'dims': [1, 1, .1]}})]
        for key, value in changes:
            with self.subTest(key=key):
                self.pool.binding_cache.clear()
                self.bind()
                count = len(self.pool.calls)
                original = self.state[key]
                self.state[key] = value
                self.bind()
                self.assertEqual(len(self.pool.calls), count+1)
                self.state[key] = original

    def test_target_inclusion_cannot_reuse_target_excluded_world(self):
        self.bind(include_target=False)
        self.bind(include_target=True)
        self.assertEqual(len(self.pool.calls), 2)

    def test_open_and_closed_contexts_are_separate(self):
        first = self.bind()
        self.assertIsNot(first, self.bind(closed=True))
        self.assertIs(first, self.bind())
        self.assertEqual(len(self.pool.calls), 2)

    def test_model_epoch_change_invalidates_cache(self):
        self.bind()
        self.pool.model_epoch = 'changed_model_or_policy'
        self.bind()
        self.assertEqual(len(self.pool.calls), 2)

    def test_different_output_gets_complete_original_receipts(self):
        self.bind()
        self.pool.bind(self.root/'different', self.state)
        self.assertTrue((self.root/'different/arx.yml').is_file())
        self.assertEqual(len(self.pool.calls), 2)

    def test_missing_receipt_stops_and_invalidates(self):
        self.bind()
        (self.root/'plan/scene.json').unlink()
        with self.assertRaisesRegex(RuntimeError, 'disappeared'):
            self.bind()
        self.assertFalse(self.pool.binding_cache)

    def test_changed_receipt_stops_and_invalidates(self):
        self.bind()
        (self.root/'plan/arx.yml').write_text('unexpected model')
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            self.bind()
        self.assertFalse(self.pool.binding_cache)

    def test_bind_fault_does_not_leave_valid_entry(self):
        self.bind()
        self.state['qpos'][0] = .2
        self.pool.failure = True
        with self.assertRaisesRegex(RuntimeError, 'binding fault'):
            self.bind()
        self.assertFalse(self.pool.binding_cache)

    def test_cached_start_gate_fault_invalidates_entry(self):
        self.bind()
        self.check.side_effect = RuntimeError('start FK or collision changed')
        with self.assertRaisesRegex(RuntimeError, 'start FK'):
            self.bind()
        self.assertFalse(self.pool.binding_cache)

    def test_mesh_hash_checked_before_cache_reuse(self):
        mesh = self.root/'mesh.ply'; mesh.write_text('first mesh')
        self.state['scene_meshes'] = {'idle_arm': {'file_path': str(mesh), 'file_sha256': sha256(mesh)}}
        self.bind()
        mesh.write_text('changed idle arm')
        with self.assertRaisesRegex(RuntimeError, 'mesh changed'):
            self.bind()
        self.assertFalse(self.pool.binding_cache)

    def test_mesh_without_declared_hash_is_still_rechecked(self):
        mesh = self.root/'mesh.ply'; mesh.write_text('first mesh')
        self.state['scene_meshes'] = {'idle_arm': {'file_path': str(mesh)}}
        self.bind(); mesh.write_text('second mesh'); self.bind()
        self.assertEqual(len(self.pool.calls), 2)

    def test_disabled_or_outside_scope_never_reuses(self):
        for attribute, value in [('reuse_enabled', False), ('reuse_scope', None)]:
            with self.subTest(attribute=attribute):
                self.pool.binding_cache.clear()
                old = getattr(self.pool, attribute)
                setattr(self.pool, attribute, value)
                count = len(self.pool.calls)
                self.bind(); self.bind()
                self.assertEqual(len(self.pool.calls), count+2)
                setattr(self.pool, attribute, old)

    def test_validator_receipts_reset_per_window(self):
        context = self.bind()
        self.pool.contact_view(context, self.state)
        self.pool.contact_view(context, self.state)
        self.assertEqual(self.pool.calls.count('validator_update'), 2)

    def test_nonfinite_input_is_rejected(self):
        self.state['qpos'][0] = float('nan')
        with self.assertRaises(ValueError):
            self.bind()


class RoutingTests(unittest.TestCase):
    def setUp(self):
        # Instantiate the CPU routing contract without constructing native solvers.
        self.router = object.__new__(reuse.routed_pool(FakeBase))
        self.router.workers = {side: reuse.reusable_pool(FakeBase)(None, None) for side in ['right', 'left']}
        self.router.active_arm = 'right'
        self.router.in_request = False
        self.router.reuse_enabled = True

    def test_two_arms_retain_distinct_contexts(self):
        right = self.router.worker({'planning_arm': 'right'})
        left = self.router.worker({'planning_arm': 'left'})
        self.assertIsNot(right.contexts['open'], left.contexts['open'])
        self.assertEqual(len(self.router.contexts), 4)

    def test_control_routes_both_arms_through_original_shared_worker(self):
        with self.router.request_scope('control', enabled=False, model_epoch='model'):
            self.assertIs(self.router.worker({'planning_arm': 'right'}),
                          self.router.worker({'planning_arm': 'left'}))

    def test_request_scope_clears_entries_on_exception(self):
        with self.assertRaisesRegex(RuntimeError, 'planned fault'):
            with self.router.request_scope('case1', enabled=True, model_epoch='model'):
                self.router.workers['right'].binding_cache['open'] = 'old'
                raise RuntimeError('planned fault')
        self.assertFalse(self.router.in_request)
        for worker in self.router.workers.values():
            self.assertFalse(worker.binding_cache)
            self.assertIsNone(worker.reuse_scope)

    def test_overlapping_requests_are_rejected(self):
        with self.router.request_scope('case1', enabled=True, model_epoch='model'):
            with self.assertRaisesRegex(RuntimeError, 'cannot overlap'):
                with self.router.request_scope('case2', enabled=True, model_epoch='model'):
                    pass

    def test_unknown_arm_is_a_fault(self):
        with self.assertRaises(ValueError):
            self.router.worker({'planning_arm': 'unknown'})


if __name__ == '__main__':
    unittest.main()
