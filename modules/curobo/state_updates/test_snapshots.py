from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .snapshots import GeometryFiles, InputBuilder, collision_inputs


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.mesh = Path(self.temp.name) / 'mesh.obj'
        self.mesh.write_text('frozen geometry')
        self.names = ['finger1', 'finger2']
        self.models = {r: {'kinematics': {'urdf_path': '/models/' + r + '/robot.urdf',
                            'lock_joints': dict.fromkeys(self.names, .044), 'collision_spheres': [1, 2, 3]}}
                       for r in ['open', 'closed']}
        pose7 = lambda t: [*t[:3, 3].tolist(), 1., 0., 0., 0.]
        self.builder = InputBuilder(self.models, {'open': {}, 'closed': {}}, self.names, pose7)
        self.files = GeometryFiles()
        self.state = {'links': {'body4': np.eye(4).tolist()}, 'joint_names': self.names, 'qpos': [.04, .03],
            'scene_cuboids': {'table': {'center': [0., 0., .7], 'dims': [1., 1., .1]}},
            'scene_meshes': {'target': {'file_path': str(self.mesh), 'target': True,
                'file_sha256': hashlib.sha256(self.mesh.read_bytes()).hexdigest(), 'world': np.eye(4).tolist()}}}

    def test_snapshot_has_measured_lock_values(self):
        value = self.builder.build(self.state, files=self.files)
        self.assertEqual(value['cfg']['robot_cfg']['kinematics']['lock_joints'], {'finger1': .04, 'finger2': .03})
        self.assertEqual(self.models['open']['kinematics']['lock_joints']['finger1'], .044)

    def test_later_width_cannot_mutate_an_earlier_receipt(self):
        first = self.builder.build(self.state, files=self.files)
        self.state['qpos'] = [.01, .02]
        second = self.builder.build(self.state, files=self.files)
        self.assertEqual(first['cfg']['robot_cfg']['kinematics']['lock_joints']['finger1'], .04)
        self.assertEqual(second['cfg']['robot_cfg']['kinematics']['lock_joints']['finger1'], .01)

    def test_target_is_included_only_under_the_original_flag(self):
        excluded = self.builder.build(self.state, files=self.files)
        included = self.builder.build(self.state, include_target=True, files=self.files)
        self.assertNotIn('mesh', excluded['scene'])
        self.assertIn('target', included['scene']['mesh'])

    def test_both_roles_keep_actual_table_and_no_nominal_geometry(self):
        opened = self.builder.build(self.state, files=self.files)
        closed = self.builder.build(self.state, closed=True, files=self.files)
        self.assertEqual(opened['scene']['cuboid']['table']['pose'][2], .7)
        self.assertEqual(closed['scene']['cuboid']['table']['pose'][2], .7)
        self.assertNotIn('nominal_scene', opened)
        self.assertNotIn('nominal_cfg', opened)

    def test_changed_mesh_is_rejected_at_request_boundary(self):
        self.builder.build(self.state, include_target=True, files=self.files)
        self.mesh.write_text('modified geometry')
        with self.assertRaisesRegex(RuntimeError, 'changed during'):
            self.files.verify()

    def test_new_request_rechecks_the_same_path(self):
        self.mesh.write_text('modified geometry')
        with self.assertRaisesRegex(RuntimeError, 'registered content'):
            self.builder.build(self.state, include_target=True, files=GeometryFiles())

    def test_content_names_equal_original_suffixes(self):
        value = self.builder.build(self.state, include_target=True, files=self.files)
        scene = collision_inputs(value['scene'], value['mesh_digests'])
        name = 'target_' + self.state['scene_meshes']['target']['file_sha256'][:20]
        self.assertEqual(scene['mesh'][name], value['scene']['mesh']['target'])

    def test_more_than_registered_obstacles_is_an_error(self):
        self.state['scene_cuboids'] = {str(i): {'center': [0., 0., 0.], 'dims': [1., 1., 1.]} for i in range(9)}
        with self.assertRaisesRegex(RuntimeError, 'capacity'):
            self.builder.build(self.state, files=self.files)

    def test_base_motion_changes_all_obstacle_poses(self):
        original = deepcopy(self.state)
        self.state['links']['body4'][0][3] = .2
        old = self.builder.build(original, include_target=True, files=self.files)
        new = self.builder.build(self.state, include_target=True, files=self.files)
        self.assertAlmostEqual(new['scene']['cuboid']['table']['pose'][0] - old['scene']['cuboid']['table']['pose'][0], -.2)
        self.assertAlmostEqual(new['scene']['mesh']['target']['pose'][0] - old['scene']['mesh']['target']['pose'][0], -.2)


if __name__ == '__main__':
    unittest.main()
