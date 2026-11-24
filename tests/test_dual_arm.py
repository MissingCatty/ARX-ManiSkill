"""V3 arm binding, native-pose preservation and failed-service behavior."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from arx_project.layout import ROOT, read_json, write_json
from arx_project.scenes import current_dataset
from modules.arx_core.arms import get_arm
from modules.curobo.dual_arm.adapter import DualArmPlanner, choose_plan
from modules.curobo.dual_arm.model import ArmModel
from modules.curobo.fingertip_contact.geometry import Geometry, collision_meshes
from modules.curobo.fingertip_contact.physical_contact import classify_contact
from modules.curobo.fingertip_contact import ContactProfile


class DeviceAdmissionTests(unittest.TestCase):
    def test_reported_free_but_unallocatable_device_is_excluded(self):
        from modules.pipeline.experiments.dual_arm_grasp.run import choose_gpus
        free = {'gpu0': 14756, 'gpu1': 18725, 'bad': 24075, 'gpu3': 12861}
        probes = {gpu: {'passed': gpu != 'bad'} for gpu in free}
        with patch('modules.pipeline.experiments.dual_arm_grasp.run.gpu_inventory', return_value=free), \
             patch('modules.pipeline.experiments.dual_arm_grasp.run.allocation_probes', return_value=probes):
            roles, floors, _ = choose_gpus()
        self.assertNotIn('bad', roles.values())
        self.assertEqual(roles['rayst3r'], 'gpu1')
        self.assertTrue(all(free[gpu] >= minimum for gpu, minimum in floors.items()))

    def test_two_healthy_devices_do_not_admit_three_resident_roles(self):
        from modules.pipeline.experiments.dual_arm_grasp.run import choose_gpus
        free = {'gpu0': 24000, 'gpu1': 24000, 'bad': 24000}
        with patch('modules.pipeline.experiments.dual_arm_grasp.run.gpu_inventory', return_value=free), \
             patch('modules.pipeline.experiments.dual_arm_grasp.run.allocation_probes',
                   return_value={gpu: {'passed': gpu != 'bad'} for gpu in free}):
            with self.assertRaisesRegex(RuntimeError, 'three CUDA-allocatable'):
                choose_gpus()


class ArmSelectionTests(unittest.TestCase):
    def setUp(self):
        self.state = {'joint_names': list(get_arm('right').joints+get_arm('right').finger_joints+
                                         get_arm('left').joints+get_arm('left').finger_joints),
                      'qpos': [0.]*16}

    def plan(self, index, travel=.1):
        return {'success': True, 'selected': {'index': index, 'q_pre': [travel]*6,
                                            'score': 1.-index*.1, 'contact_pose_world': np.eye(4).tolist()}}

    def test_model_order_before_motion_cost(self):
        self.assertEqual(choose_plan({'right': self.plan(2, .001), 'left': self.plan(1, 1.)}, self.state), 'left')

    def test_same_grasp_shorter_motion(self):
        self.assertEqual(choose_plan({'right': self.plan(2, .5), 'left': self.plan(2, .1)}, self.state), 'left')

    def test_one_arm_unavailable(self):
        self.assertEqual(choose_plan({'right': {'success': False}, 'left': self.plan(2)}, self.state), 'left')

    def test_neither_arm_available(self):
        self.assertIsNone(choose_plan({'right': {'success': False}, 'left': {'success': False}}, self.state))

    def test_unknown_arm_rejected(self):
        with self.assertRaises(ValueError): get_arm('both')

    def test_local_contact_gate_binds_selected_physical_arm(self):
        eef = np.diag([1., -1., -1., 1.]); eef[2, 3] = .044
        args = {'finger': 'left_link17', 'is_table': True, 'stage': 'close',
                'points': [{'world': [0., 0., -.001], 'normal': [0., 0., 1.],
                            'separation_m': -.001, 'impulse_ns': .01}],
                'eef': eef, 'table': {'center': [0., 0., -.25], 'dims': [.2, .2, .5]}, 'tip_plane': .045}
        self.assertTrue(classify_contact(ContactProfile(), **args, arm='left')['allowed'])
        self.assertFalse(classify_contact(ContactProfile(), **args, arm='right')['allowed'])

    def test_arm_change_after_selection_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp', prefix='dual-arm-test-') as folder:
            path = Path(folder)
            write_json(path/'plan_result.json', {'selected': {'arm': 'left'}})
            write_json(path/'pregrasp_state.json', {'execution_arm': 'left'})
            planner = DualArmPlanner(None, None, path)
            with self.assertRaisesRegex(ValueError, 'switching'):
                planner.request(path, approach=True, arm='right')

    def test_wrong_measured_arm_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp', prefix='dual-arm-test-') as folder:
            path = Path(folder)
            write_json(path/'plan_result.json', {'selected': {'arm': 'left'}})
            write_json(path/'closed_state.json', {'execution_arm': 'right'})
            with self.assertRaisesRegex(ValueError, 'Measured execution arm'):
                DualArmPlanner(None, None, path).request(path, lift=True)

    def test_opening_sweep_checks_intermediate_widths_and_target(self):
        class Pool:
            def __init__(self): self.values = []
            def bind(self, directory, state, **flags):
                self.values.append(state['qpos'][0]); assert flags['include_target']
                assert type(state['qpos'][0]) is float
                yaml.safe_dump(state)
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp', prefix='dual-arm-test-') as folder:
            pool = Pool()
            state = {'joint_names': ['right_joint27', 'right_joint28'], 'qpos': [.044, .044],
                     'planned_opening': {'target_per_finger_m': .044,
                         'observed_solver_finger_positions_m': {'right_joint27': 0., 'right_joint28': .001}}}
            result = DualArmPlanner(pool, None, folder).opening_sweep(Path(folder), state)
            self.assertTrue(result['passed'])
            self.assertEqual(pool.values[0], 0.)
            self.assertEqual(pool.values[-1], .044)
            self.assertLessEqual(np.diff(pool.values).max(), .0005+1e-12)

    def test_opening_collision_is_planning_failure(self):
        class Pool:
            def bind(self, directory, state, **flags):
                raise AssertionError('Measured state collides in unchanged ARX model')
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp', prefix='dual-arm-test-') as folder:
            state = {'joint_names': ['right_joint27', 'right_joint28'], 'qpos': [.044, .044],
                     'planned_opening': {'target_per_finger_m': .044,
                         'observed_solver_finger_positions_m': {'right_joint27': 0., 'right_joint28': 0.}}}
            result = DualArmPlanner(Pool(), None, folder).opening_sweep(Path(folder), state)
            self.assertFalse(result['passed'])
            self.assertEqual(result['reason'], 'gripper_opening_sweep_in_collision')

    def test_infrastructure_failure_is_not_arm_unreachability(self):
        class Model:
            def bind(self, state, arm, directory): return state
        class Pool:
            def request(self, directory, **flags): raise RuntimeError('CUDA allocator failed')
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp', prefix='dual-arm-test-') as folder:
            path = Path(folder)
            write_json(path/'state.json', self.state)
            write_json(path/'candidates/candidate_pool.json', {'candidates': []})
            with self.assertRaisesRegex(RuntimeError, 'CUDA allocator'):
                planner = DualArmPlanner(Pool(), Model(), path)
                planner.opening_sweep = lambda *args: {'passed': True}
                planner.request(path)
            self.assertFalse((path/'plan_result.json').exists())


class ActualArmModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = ArmModel()
        # This regression fixture belongs to the preserved V3 collection;
        # selecting V4's 100 new scenes must not redirect the V3 test fixture.
        cls.state = read_json(ROOT/'simulation/scenes/latest200/cases/D0008_S1_open_teapot/state.json')

    def test_authored_chain_equivalence(self):
        self.assertLess(self.model.proof['maximum_transform_error'], 2e-6)
        self.assertEqual(self.model.proof['paired_states'], 32)

    def test_collision_spheres_match_per_link_and_parameters(self):
        proof = self.model.sphere_proof
        self.assertEqual(proof['right_total'], 871)
        self.assertEqual(proof['left_total'], 871)
        self.assertEqual(proof['right_parameter_sha256'], proof['normalized_left_parameter_sha256'])
        self.assertTrue(proof['local_centers_and_radii_identical'])
        self.assertLess(proof['maximum_world_center_error_m'], 2e-6)

    def test_both_arm_bindings_match_independent_fk(self):
        frames = self.model.frames(self.state)
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp', prefix='dual-arm-test-') as folder:
            for side in ['right', 'left']:
                directory = Path(folder)/side
                state = self.model.bind(self.state, side, directory)
                reference = ROOT/'modules/arx_core/scripts/curobo_arx/static_pick/reference/model'
                cfg = yaml.safe_load((reference/'arx.yml').read_text())
                cfg['robot_cfg']['kinematics']['urdf_path'] = str(reference/'arx.urdf')
                for name in get_arm('right').finger_joints:
                    cfg['robot_cfg']['kinematics']['lock_joints'][name] = state['qpos'][state['joint_names'].index(name)]
                path = directory/'model.yml'; path.write_text(yaml.safe_dump(cfg))
                q = [state['qpos'][state['joint_names'].index(name)] for name in get_arm('right').joints]
                actual = Geometry(path, state).frames([q])['right_eef_link'][0]
                np.testing.assert_allclose(actual, frames[get_arm(side).eef], atol=2e-6)
                self.assertIn('parked_robot_arm', state['scene_meshes'])
                self.assertEqual(read_json(directory/'arm_binding.json')['parked_arm'], 'left' if side == 'right' else 'right')
                self.assertEqual(state['object_world'], self.state['object_world'])

    def test_closure_receipt_cannot_be_used_for_other_arm(self):
        state = deepcopy(self.state)
        state['fingertip_table_contact'] = {'arm': 'right', 'qpos': state['qpos']}
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp', prefix='dual-arm-test-') as folder:
            with self.assertRaisesRegex(ValueError, 'other arm'):
                self.model.bind(state, 'left', folder)

    def test_saved_robot_frame_mismatch_stops_binding(self):
        state = deepcopy(self.state)
        state['links']['body4'][0][3] += .01
        with self.assertRaisesRegex(ValueError, 'FK differs'):
            self.model.frames(state)

    def test_left_contact_mesh_uses_left_decomposition(self):
        right, _ = collision_meshes(self.model.asset, 'right')
        left, _ = collision_meshes(self.model.asset, 'left')
        self.assertEqual(len(right['right_link27']), 19)
        self.assertEqual(len(left['right_link27']), 1)
        np.testing.assert_allclose(np.concatenate(right['right_link27']).min(0), left['right_link27'][0].min(0))
        np.testing.assert_allclose(np.concatenate(right['right_link27']).max(0), left['right_link27'][0].max(0))


if __name__ == '__main__':
    unittest.main()
