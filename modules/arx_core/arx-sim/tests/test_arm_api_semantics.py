from __future__ import annotations

from types import MethodType, SimpleNamespace
import unittest

import numpy as np

from arx_shared.schemas import CaptureRequest, EndPoseRequest
from arx_sim.backend.backend import IsaacBackend
from arx_sim.backend.command_queue import Command
from arx_sim.geometry import rpy_to_matrix
from arx_sim.robot.controller import ActiveCommand, IsaacRobotController


class ArmApiSemanticsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = IsaacRobotController.__new__(IsaacRobotController)
        self.controller.cfg = {
            "right_arm": {
                "position_tolerance_m": 0.01,
                "orientation_tolerance_rad": 0.06,
                "stable_steps": 1,
                "home_min_duration_sec": 2.0,
            },
            "gripper": {
                "api_open": -3.2,
                "api_closed": 0.0,
                "api_close_after_pose": 0.1,
                "open_m": 0.044,
                "closed_m": 0.0,
            },
        }
        self.controller.context = SimpleNamespace(
            config={"motion": {"gripper_close_duration_sec": 3.0}}
        )
        self.controller.gripper_indices = [0, 1]
        self.controller._positions = MethodType(
            lambda _self: np.zeros(2), self.controller
        )
        self.completed = []
        self.gripper_targets = []
        self.controller._complete = MethodType(
            lambda _self, command, result: self.completed.append((command, result)),
            self.controller,
        )
        self.controller._set_gripper_api_target = MethodType(
            lambda _self, value, side="right": self.gripper_targets.append(value), self.controller
        )
        self.controller._ee_home_world = MethodType(
            lambda _self, _side: np.eye(4), self.controller
        )

    @staticmethod
    def command(**payload):
        return SimpleNamespace(payload=SimpleNamespace(**payload))

    def active(self, command, **changes):
        values = {
            "command": command,
            "target": np.zeros(6),
            "indices": list(range(6)),
            "started_sim_time": 0.0,
            "phase": "move",
            "phase_started_sim_time": 0.0,
        }
        values.update(changes)
        active = ActiveCommand(**values)
        self.controller.active = active
        return active

    def set_ik_result(self, reached: bool = True) -> None:
        self.controller._step_pose_ik = MethodType(
            lambda _self, _active, **_kwargs: (reached, 0.001, 0.002),
            self.controller,
        )

    def test_move_exits_immediately_when_exit_on_pose_is_set(self) -> None:
        command = self.command(
            exit_after_sec=None,
            exit_on_pose=True,
            close_delay=0.5,
        )
        active = self.active(command)
        self.set_ik_result()

        self.controller._update_move_end_pose(active, 1.0)

        self.assertEqual(len(self.completed), 1)
        self.assertTrue(self.completed[0][1].ok)

    def test_move_waits_then_runs_the_real_api_auto_close_branch(self) -> None:
        command = self.command(
            exit_after_sec=None,
            exit_on_pose=False,
            close_delay=0.5,
        )
        active = self.active(command, hold_gripper=False)
        self.set_ik_result()

        self.controller._update_move_end_pose(active, 1.0)
        self.controller._update_move_end_pose(active, 1.49)
        self.assertEqual(self.gripper_targets, [])

        self.controller._update_move_end_pose(active, 1.5)
        self.assertEqual(self.gripper_targets, [0.1])
        self.assertEqual(self.completed, [])

    def test_move_hold_gripper_suppresses_auto_close(self) -> None:
        command = self.command(
            exit_after_sec=None,
            exit_on_pose=False,
            close_delay=0.0,
        )
        active = self.active(command, hold_gripper=True)
        self.set_ik_result()

        self.controller._update_move_end_pose(active, 1.0)

        self.assertEqual(self.gripper_targets, [])
        self.assertEqual(self.completed, [])

    def test_move_exit_after_uses_elapsed_simulation_time(self) -> None:
        command = self.command(
            exit_after_sec=2.0,
            exit_on_pose=False,
            close_delay=0.5,
        )
        active = self.active(command, started_sim_time=4.0)
        self.set_ik_result(reached=False)

        self.controller._update_move_end_pose(active, 6.0)

        self.assertEqual(len(self.completed), 1)
        self.assertAlmostEqual(self.completed[0][1].data["elapsed_sec"], 2.0)

    def test_home_precloses_then_waits_for_q_style_minimum_duration(self) -> None:
        command = SimpleNamespace(
            payload={
                "gripper_close_duration_sec": None,
                "linear_threshold": 0.006,
                "angular_threshold": 0.03,
            }
        )
        active = self.active(
            command,
            phase="preclose",
            phase_started_sim_time=0.0,
            hold_gripper=False,
            gripper_target_api=0.0,
        )
        self.set_ik_result()

        self.controller._update_arm_home(active, 2.9)
        self.assertEqual(active.phase, "preclose")
        self.assertEqual(self.gripper_targets, [0.0])

        self.controller._update_arm_home(active, 3.0)
        self.assertEqual(active.phase, "home")
        self.assertEqual(self.completed, [])
        self.assertEqual(self.gripper_targets, [0.0, 0.0])

        self.controller._update_arm_home(active, 4.99)
        self.assertEqual(self.completed, [])
        self.assertEqual(self.gripper_targets, [0.0, 0.0, 0.0])
        self.controller._update_arm_home(active, 5.0)
        self.assertEqual(len(self.completed), 1)
        self.assertEqual(self.gripper_targets, [0.0, 0.0, 0.0, 0.0])

    def test_gripper_feedback_uses_real_api_range(self) -> None:
        self.assertAlmostEqual(
            self.controller._gripper_feedback_api(np.array([0.044, 0.044])), -3.2
        )
        self.assertAlmostEqual(
            self.controller._gripper_feedback_api(np.array([0.0, 0.0])), 0.0
        )

    def test_api_xyz_uses_fixed_arm_base_axes_with_ee_home_as_zero(self) -> None:
        arm_base_prim = object()
        self.controller.context.arm_base_prim = arm_base_prim
        arm_base_world = np.eye(4)
        arm_base_world[:3, :3] = rpy_to_matrix(0.1, -0.2, 0.3)
        arm_base_world[:3, 3] = [1.0, 2.0, 3.0]
        home = np.eye(4)
        home[:3, :3] = rpy_to_matrix(0.0, 0.0, np.pi / 2.0)
        home[:3, 3] = [0.4, -0.2, 0.6]
        self.controller.ee_home_in_arm_base = {"right": home}
        self.controller._world_pose = MethodType(
            lambda _self, prim: arm_base_world
            if prim is arm_base_prim
            else (_ for _ in ()).throw(AssertionError("unexpected prim")),
            self.controller,
        )
        requested = [0.15, 0.02, -0.03, 0.1, -0.2, 0.3]

        target_world = self.controller._ee_pose_from_api("right", requested)
        target_in_arm_base = np.linalg.inv(arm_base_world) @ target_world

        np.testing.assert_allclose(
            target_in_arm_base[:3, 3],
            home[:3, 3] + np.asarray(requested[:3]),
            atol=1e-9,
        )
        np.testing.assert_allclose(
            self.controller._ee_pose_to_api("right", target_world),
            requested,
            atol=1e-9,
        )


class ArmBackendSemanticsTest(unittest.TestCase):
    def test_default_capture_matches_real_right_arm_camera(self) -> None:
        submitted = []
        queue = SimpleNamespace(
            submit=lambda kind, payload, timeout: submitted.append(
                (kind, payload, timeout)
            )
        )
        context = SimpleNamespace(
            command_queue=queue,
            config={
                "simulation": {"command_timeout_sec": 30.0},
                "runtime": {"run_dir": "/tmp/coolas-test"},
                "camera": {"serial": "352122272245"},
                "controllers": {
                    "camera": {
                        "logical_serials": {
                            "352122270890": "head",
                            "352122273322": "left_wrist",
                            "352122272245": "right_wrist",
                        },
                        "definitions": {
                            "head": {"serial": "352122270890"},
                            "right_wrist": {"serial": "352122272245"},
                        },
                    }
                },
            },
            cameras={"head": object(), "right_wrist": object()},
            snapshot=lambda: SimpleNamespace(lifecycle="READY", error=""),
        )
        backend = IsaacBackend(context)

        backend.capture_rgbd(CaptureRequest(output_dir="/tmp/capture"))

        self.assertEqual(submitted[0][0], "capture_rgbd")
        self.assertEqual(submitted[0][1]["role"], "right_wrist")
        self.assertEqual(submitted[0][1]["serial_number"], "352122272245")
        self.assertEqual(submitted[0][1]["camera_id"], "default")

    def test_head_camera_remains_addressable_for_mapping(self) -> None:
        submitted = []
        queue = SimpleNamespace(
            submit=lambda kind, payload, timeout: submitted.append(
                (kind, payload, timeout)
            )
        )
        context = SimpleNamespace(
            command_queue=queue,
            config={
                "simulation": {"command_timeout_sec": 30.0},
                "runtime": {"run_dir": "/tmp/coolas-test"},
                "camera": {"serial": "352122272245"},
                "controllers": {
                    "camera": {
                        "logical_serials": {
                            "352122270890": "head",
                            "352122272245": "right_wrist",
                        },
                        "definitions": {
                            "head": {"serial": "352122270890"},
                            "right_wrist": {"serial": "352122272245"},
                        },
                    }
                },
            },
            cameras={"head": object(), "right_wrist": object()},
            snapshot=lambda: SimpleNamespace(lifecycle="READY", error=""),
        )
        backend = IsaacBackend(context)

        backend.capture_rgbd(CaptureRequest(serial="352122270890", output_dir="/tmp/capture"))

        self.assertEqual(submitted[0][1]["role"], "head")
        self.assertEqual(submitted[0][1]["serial_number"], "352122270890")
        self.assertIsNone(submitted[0][1]["camera_id"])

    def test_move_without_timeout_preserves_real_indefinite_wait(self) -> None:
        submitted = []
        queue = SimpleNamespace(
            submit=lambda kind, payload, timeout: submitted.append(
                (kind, payload, timeout)
            )
        )
        context = SimpleNamespace(
            command_queue=queue,
            config={
                "simulation": {"command_timeout_sec": 30.0},
                "motion": {
                    "x_range": [0.0, 0.7],
                    "y_range": [-0.4, 0.4],
                    "z_range": [-0.3, 0.6],
                },
            },
        )
        backend = IsaacBackend(context)

        backend.move_end_pose(EndPoseRequest(target_pose=[0.0] * 6))

        self.assertEqual(submitted[0][0], "move_end_pose")
        self.assertIsNone(submitted[0][2])

    def test_command_without_timeout_has_no_deadline(self) -> None:
        command = Command(kind="move_end_pose", payload={}, generation=0, timeout_sec=None)
        self.assertIsNone(command.deadline)


if __name__ == "__main__":
    unittest.main()
