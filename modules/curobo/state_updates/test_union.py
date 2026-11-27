from copy import deepcopy
from types import SimpleNamespace
import unittest

from .scenes import UnionSceneUpdates


class UnionTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.checker = SimpleNamespace(
            update_obstacle_pose=lambda n, p: self.calls.append(('pose', n, p)),
            enable_obstacle=lambda n, flag: self.calls.append(('enable', n, flag)))
        self.owner = SimpleNamespace(scene_model=self.checker,
            clear_scene_cache=lambda: self.calls.append(('clear',)),
            update_world=lambda s: self.calls.append(('load', deepcopy(s))))
        self.updates = UnionSceneUpdates(deepcopy, list)
        self.scene = {'cuboid': {n: {'dims': [.1] * 3, 'pose': [i, 0, 0, 1, 0, 0, 0]}
                                  for i, n in enumerate(['table', 'target'])}}

    def test_target_exclusion_and_reinclusion_use_exact_enable_flags(self):
        self.updates.update(self.owner, self.scene)
        excluded = deepcopy(self.scene)
        del excluded['cuboid']['target']
        self.assertEqual(self.updates.update(self.owner, excluded)['operation'], 'update_enabled')
        self.assertEqual(self.calls[-1], ('enable', 'target', False))
        self.updates.update(self.owner, self.scene)
        self.assertEqual(self.calls[-1], ('enable', 'target', True))
        self.assertEqual(sum(c[0] == 'load' for c in self.calls), 1)

    def test_moved_reactivated_obstacle_updates_its_pose(self):
        self.updates.update(self.owner, self.scene)
        excluded = {'cuboid': {'table': deepcopy(self.scene['cuboid']['table'])}}
        self.updates.update(self.owner, excluded)
        moved = deepcopy(self.scene)
        moved['cuboid']['target']['pose'][0] = .5
        result = self.updates.update(self.owner, moved)
        self.assertEqual(result['updated_obstacles'], 1)
        self.assertIn(('pose', 'target', moved['cuboid']['target']['pose']), self.calls)
        self.assertEqual(self.calls[-1], ('enable', 'target', True))

    def test_unseen_geometry_expands_bank_and_disables_absent_entries(self):
        self.updates.update(self.owner, self.scene)
        other = deepcopy(self.scene)
        other['cuboid']['left_arm'] = other['cuboid'].pop('target')
        result = self.updates.update(self.owner, other)
        self.assertEqual(result['resident_obstacles'], 3)
        self.assertEqual(self.calls[-1], ('enable', 'target', False))

    def test_bank_never_exceeds_original_capacity(self):
        self.updates.update(self.owner, self.scene)
        many = {'cuboid': {str(i): deepcopy(self.scene['cuboid']['target']) for i in range(8)}}
        result = self.updates.update(self.owner, many)
        self.assertEqual(result['resident_obstacles'], 8)
        self.assertEqual(set(self.calls[-1][1]['cuboid']), set(many['cuboid']))

    def test_same_name_geometry_change_is_not_a_pose_update(self):
        self.updates.update(self.owner, self.scene)
        changed = deepcopy(self.scene)
        changed['cuboid']['table']['dims'][0] = .2
        self.assertEqual(self.updates.update(self.owner, changed)['operation'], 'replace_geometry')

    def test_failed_enable_cannot_leave_a_reusable_entry(self):
        self.updates.update(self.owner, self.scene)
        def fail(*args):
            raise RuntimeError('injected enable error')
        self.checker.enable_obstacle = fail
        excluded = {'cuboid': {'table': deepcopy(self.scene['cuboid']['table'])}}
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            self.updates.update(self.owner, excluded)
        self.assertEqual(self.updates.update(self.owner, self.scene)['operation'], 'replace_geometry')

    def test_request_reset_reloads_geometry_after_control_mutations(self):
        self.updates.update(self.owner, self.scene)
        self.updates.reset()
        self.assertEqual(self.updates.update(self.owner, self.scene)['operation'], 'replace_geometry')


if __name__ == '__main__':
    unittest.main()
