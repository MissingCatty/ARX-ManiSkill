from copy import deepcopy
from types import SimpleNamespace
import unittest

from .scenes import SceneUpdates


class SceneTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        checker = SimpleNamespace(update_obstacle_pose=lambda n, p: self.calls.append(('pose', n, p)))
        self.owner = SimpleNamespace(scene_collision_checker=checker,
            clear_scene_cache=lambda: self.calls.append(('clear',)),
            update_world=lambda scene: self.calls.append(('load', scene)))
        self.updates = SceneUpdates(deepcopy, list)
        self.scene = {'cuboid': {'table': {'dims': [1., 2., .1], 'pose': [0., 0., 0., 1., 0., 0., 0.]}},
                      'mesh': {'object_abcd': {'file_path': '/geometry/a', 'pose': [0., 0., 1., 1., 0., 0., 0.]}}}

    def test_same_geometry_keeps_storage(self):
        self.updates.update(self.owner, self.scene)
        result = self.updates.update(self.owner, deepcopy(self.scene))
        self.assertEqual(result['operation'], 'reuse_scene')
        self.assertEqual(len(self.calls), 2)

    def test_moved_target_uses_its_name(self):
        self.updates.update(self.owner, self.scene)
        changed = deepcopy(self.scene)
        changed['mesh']['object_abcd']['pose'][0] = .03
        self.updates.update(self.owner, changed)
        self.assertEqual(self.calls[-1], ('pose', 'object_abcd', changed['mesh']['object_abcd']['pose']))

    def test_table_size_change_replaces_geometry(self):
        self.updates.update(self.owner, self.scene)
        changed = deepcopy(self.scene)
        changed['cuboid']['table']['dims'][2] = .2
        self.assertEqual(self.updates.update(self.owner, changed)['operation'], 'replace_geometry')

    def test_target_inclusion_change_replaces_geometry(self):
        self.updates.update(self.owner, self.scene)
        changed = deepcopy(self.scene)
        del changed['mesh']
        self.assertEqual(self.updates.update(self.owner, changed)['operation'], 'replace_geometry')

    def test_mesh_digest_name_change_replaces_geometry(self):
        self.updates.update(self.owner, self.scene)
        changed = deepcopy(self.scene)
        changed['mesh']['object_efgh'] = changed['mesh'].pop('object_abcd')
        self.assertEqual(self.updates.update(self.owner, changed)['operation'], 'replace_geometry')

    def test_failed_pose_update_invalidates_the_entry(self):
        self.updates.update(self.owner, self.scene)
        def fail(*args):
            raise RuntimeError('injected pose update error')
        self.owner.scene_collision_checker.update_obstacle_pose = fail
        changed = deepcopy(self.scene)
        changed['cuboid']['table']['pose'][0] = 2.
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            self.updates.update(self.owner, changed)
        self.assertEqual(self.updates.update(self.owner, self.scene)['operation'], 'replace_geometry')

    def test_distinct_owners_do_not_share_scene_state(self):
        self.updates.update(self.owner, self.scene)
        other = SimpleNamespace(scene_model=SimpleNamespace(update_obstacle_pose=lambda *a: None),
            clear_scene_cache=lambda: None, update_world=lambda s: None)
        self.assertEqual(self.updates.update(other, self.scene)['operation'], 'replace_geometry')

    def test_request_reset_does_not_reuse_a_control_groups_scene(self):
        self.updates.update(self.owner, self.scene)
        self.updates.reset()
        self.assertEqual(self.updates.update(self.owner, self.scene)['operation'], 'replace_geometry')

    def test_nonfinite_pose_is_rejected_before_mutation(self):
        self.scene['cuboid']['table']['pose'][0] = float('nan')
        with self.assertRaises(ValueError):
            self.updates.update(self.owner, self.scene)
        self.assertFalse(self.calls)


if __name__ == '__main__':
    unittest.main()
