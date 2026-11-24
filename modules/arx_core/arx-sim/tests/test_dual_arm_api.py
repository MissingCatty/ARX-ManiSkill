from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from urllib.parse import urlencode

import numpy as np
import yaml

from arx_shared.schemas import CaptureRequest, ChassisMoveRequest, EndPoseRequest, ExecutionResult
from arx_sim.api import create_app
from arx_sim.backend.backend import IsaacBackend
from arx_sim.backend.capture import write_rgbd_bundle
from arx_sim.backend.command_queue import Command
from arx_sim.backend.context import SimulationState
from arx_sim.errors import SimNotReadyError
from arx_sim.robot.controller import IsaacRobotController


ROOT = Path(__file__).resolve().parents[2]


class ApiClient:
    """Exercise the ASGI boundary without an optional HTTP client dependency."""

    def __init__(self, app):
        self.app = app

    def request(self, method, path, *, params=None, json_body=None):
        async def run():
            messages = []
            body = json.dumps(json_body).encode() if json_body is not None else b""

            async def receive():
                return {"type": "http.request", "body": body, "more_body": False}

            async def send(message):
                messages.append(message)

            await self.app({"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method, "scheme": "http", "path": path, "raw_path": path.encode(), "root_path": "", "query_string": urlencode(params or {}).encode(), "headers": [(b"content-type", b"application/json")], "client": ("127.0.0.1", 1), "server": ("test", 80)}, receive, send)
            status = next(m["status"] for m in messages if m["type"] == "http.response.start")
            data = json.loads(b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body"))
            return SimpleNamespace(status_code=status, json=lambda: data)

        return asyncio.run(run())

    def get(self, path, *, params=None):
        return self.request("GET", path, params=params)

    def post(self, path, *, params=None, json=None):
        return self.request("POST", path, params=params, json_body=json)


class DualArmTest(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load((ROOT / "configs/arx.yaml").read_text())
        self.config.update(yaml.safe_load((ROOT / "arx-sim/config/controllers.yaml").read_text()))
        self.config["simulation"] = {"command_timeout_sec": 30.0}
        self.config["runtime"] = {"run_dir": "/tmp/arx-api-test"}
        self.context = SimpleNamespace(
            config=self.config,
            command_queue=SimpleNamespace(submit=Mock(return_value=ExecutionResult(ok=True)), finish=Mock(), generation=0),
            snapshot=lambda: SimulationState(
                lifecycle="READY", end_pos=(1.0,) * 6, arm_joint_pos=(1.0,) * 7,
                left_end_pos=(2.0,) * 6, left_arm_joint_pos=(2.0,) * 7,
                gripper=-1.0, left_gripper=-2.0,
            ),
            cameras={"head": object(), "left_wrist": object(), "right_wrist": object()},
        )
        self.backend = IsaacBackend(self.context)
        self.client = ApiClient(create_app(self.backend))
        self.controller = IsaacRobotController.__new__(IsaacRobotController)
        c = self.controller
        c.context = self.context
        c.cfg = self.config["controllers"]
        c.arm_indices = list(range(6))
        c.left_arm_indices = list(range(6, 12))
        c.gripper_indices = [12, 13]
        c.left_gripper_indices = [14, 15]
        c.robot = Mock()
        c._positions = Mock(return_value=np.zeros(16))
        c._ee_pose_from_api = Mock(return_value=np.eye(4))
        c._ee_home_world = Mock(return_value=np.eye(4))
        c.active = None

    def test_status_selects_side(self):
        for side, expected in (("left", 2.0), ("right", 1.0)):
            with self.subTest(side=side):
                response = self.client.get("/arm/status", params={"arm_id": side})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["arm_id"], side)
                self.assertEqual(response.json()["end_pos"], [expected] * 6)

    def test_gripper_body_and_default(self):
        for action in ("open", "close"):
            response = self.client.post(f"/gripper/{action}", json={"arm_id": "left", "duration_sec": 1.7})
            self.assertEqual(response.status_code, 200)
            kind, payload, timeout = self.context.command_queue.submit.call_args.args
            self.assertEqual(kind, "gripper")
            self.assertEqual(payload, {"open": action == "open", "arm_id": "left", "duration_sec": 1.7})
            self.assertGreater(timeout, 1.7)
        self.client.post("/gripper/open")
        self.assertEqual(self.context.command_queue.submit.call_args.args[1]["arm_id"], "right")

    def test_home_parameters_reach_queue(self):
        self.client.post("/arm/home", params={"arm_id": "left", "close_gripper": False, "speed": 0.3, "gripper_close_duration_sec": 1.2})
        kind, payload, timeout = self.context.command_queue.submit.call_args.args
        self.assertEqual(kind, "arm_home")
        self.assertEqual(payload["arm_id"], "left")
        self.assertFalse(payload["close_gripper"])
        self.assertEqual(payload["speed"], 0.3)
        self.assertEqual(payload["gripper_close_duration_sec"], 1.2)
        self.assertEqual(timeout, 120.0)

    def test_invalid_arm_does_not_submit(self):
        with self.assertRaisesRegex(ValueError, "unknown arm_id"):
            self.backend.move_end_pose(EndPoseRequest(target_pose=[0.0] * 6, arm_id="invalid"))
        self.context.command_queue.submit.assert_not_called()

    def test_motion_timeout_returns_failure_result(self):
        self.context.command_queue.submit.side_effect = SimNotReadyError("command timed out")
        for path, body in (
            ("/arm/move-end-pose", {"arm_id": "left", "target_pose": [0.0] * 6, "timeout_sec": 0.1}),
            ("/gripper/open", {"arm_id": "left"}),
            ("/arm/home", None),
        ):
            if body is None:
                response = self.client.post(path, params={"arm_id": "left"})
            else:
                response = self.client.post(path, json=body)
            self.assertEqual(response.status_code, 200)
            self.assertFalse(response.json()["ok"])
            self.assertEqual(response.json()["data"]["arm_id"], "left")

    def test_left_move_delayed_close_and_home_preclose(self):
        c = self.controller
        command = Command(kind="move_end_pose", payload=EndPoseRequest(target_pose=[0.0] * 6, arm_id="left", target_gripper=-3.2, close_delay=0.5), generation=0, timeout_sec=None)
        c._start_move_end_pose(command)
        c._step_pose_ik = Mock(return_value=(True, 0.001, 0.001))
        c.robot.reset_mock()
        c._update_move_end_pose(c.active, 1.0)
        c.robot.set_dof_position_targets.assert_not_called()
        c._update_move_end_pose(c.active, 1.5)
        self.assertEqual(c.robot.set_dof_position_targets.call_args.kwargs["dof_indices"], c.left_gripper_indices)
        self.assertFalse(command.future.done())
        command = Command(kind="arm_home", payload={"arm_id": "left", "gripper_close_duration_sec": 0.5}, generation=0, timeout_sec=None)
        c._start_arm_home(command)
        c._step_pose_ik.reset_mock()
        c._update_arm_home(c.active, 0.49)
        c._step_pose_ik.assert_not_called()
        c._update_arm_home(c.active, 0.5)
        c._step_pose_ik.assert_called_once()
        self.assertFalse(command.future.done())
        c._update_arm_home(c.active, 2.5)
        result = command.future.result(timeout=1)
        self.assertEqual(result.data["arm_id"], "left")
        self.assertEqual(result.data["close_result"]["data"]["arm_id"], "left")

    def test_chassis_move_applies_sim_velocity_scale(self):
        c = self.controller
        c.chassis_indices = [0, 1, 2]
        c.cfg = {
            "chassis": {
                "linear_x_scale": 1.0,
                "linear_y_scale": 1.0,
                "angular_z_scale": 0.5,
            }
        }
        c._positions = Mock(return_value=np.zeros(3))
        c.context.snapshot = Mock(return_value=SimulationState(sim_time=10.0))

        command = Command(
            kind="chassis_move",
            payload=ChassisMoveRequest(linear_x=0.2, angular_z=0.3, duration=1.2),
            generation=0,
            timeout_sec=None,
        )
        c._start_chassis_move(command)

        velocity = c.robot.set_dof_velocity_targets.call_args.args[0]
        np.testing.assert_allclose(velocity, [0.2, 0.0, 0.15])
        self.assertEqual(c.robot.set_dof_velocity_targets.call_args.kwargs["dof_indices"], [0, 1, 2])
        self.assertAlmostEqual(c.active.stop_sim_time, 11.2)

    def test_chassis_move_optionally_clamps_duration(self):
        c = self.controller
        c.chassis_indices = [0, 1, 2]
        c.cfg = {
            "chassis": {
                "linear_x_scale": 1.0,
                "linear_y_scale": 1.0,
                "angular_z_scale": 1.0,
                "max_command_duration_sec": 0.5,
            }
        }
        c._positions = Mock(return_value=np.zeros(3))
        c.context.snapshot = Mock(return_value=SimulationState(sim_time=3.0))

        command = Command(
            kind="chassis_move",
            payload=ChassisMoveRequest(angular_z=0.3, duration=1.2),
            generation=0,
            timeout_sec=None,
        )
        c._start_chassis_move(command)

        self.assertAlmostEqual(c.active.stop_sim_time, 3.5)

    def test_snapshot_publishes_independent_left_feedback(self):
        c = self.controller
        self.context.ee_prim = object()
        self.context.left_ee_prim = object()
        self.context.cameras = {}
        self.context.publish = Mock()
        c._world_pose = Mock(return_value=np.eye(4))
        c._ee_pose_to_api = Mock(side_effect=lambda side, pose: [1.0 if side == "right" else 2.0] * 6)
        positions = np.zeros(20)
        positions[6:12] = 0.5
        positions[14:16] = 0.044
        c._positions.return_value = positions
        c._velocities = Mock(return_value=np.zeros(20))
        c.chassis_indices = [16, 17, 18]
        c.lift_index = 19
        c._ready_cameras = set()
        c._camera_errors = {}
        c.publish_snapshot(1.0)
        data = self.context.publish.call_args.kwargs
        self.assertEqual(data["end_pos"], (1.0,) * 6)
        self.assertEqual(data["left_end_pos"], (2.0,) * 6)
        self.assertEqual(data["left_arm_joint_pos"][:6], (0.5,) * 6)
        self.assertEqual(data["gripper"], 0.0)
        self.assertEqual(data["left_gripper"], -3.2)

    def test_move_selects_only_requested_arm_and_gripper(self):
        c = self.controller
        for side in ("right", "left"):
            with self.subTest(side=side):
                c.robot.reset_mock()
                request = EndPoseRequest(target_pose=[0.0] * 6, arm_id=side, target_gripper=-3.2)
                c._start_move_end_pose(Command(kind="move_end_pose", payload=request, generation=0, timeout_sec=None))
                self.assertEqual(c.active.arm_id, side)
                self.assertEqual(c.active.indices, c._arm_indices(side))
                c._ee_pose_from_api.assert_called_with(side, request.target_pose)
                self.assertEqual(c.robot.set_dof_position_targets.call_args.kwargs["dof_indices"], c._gripper_indices(side))
                np.testing.assert_allclose(c.robot.set_dof_position_targets.call_args.args[0], [0.044, 0.044])
        c.robot.reset_mock()
        c._start_move_end_pose(Command(kind="move_end_pose", payload=EndPoseRequest(target_pose=[0.0] * 6, arm_id="left", hold_gripper=True), generation=0, timeout_sec=None))
        c.robot.set_dof_position_targets.assert_not_called()

    def test_start_clears_cancelled_active_motion_before_new_command(self):
        c = self.controller
        c.chassis_indices = []
        c.lift_index = 15
        old = Command(
            kind="move_end_pose",
            payload=EndPoseRequest(target_pose=[0.0] * 6, arm_id="right"),
            generation=0,
            timeout_sec=5.0,
        )
        c._start_move_end_pose(old)
        old.cancelled.set()
        new = Command(
            kind="move_end_pose",
            payload=EndPoseRequest(target_pose=[0.1, 0.0, 0.0, 0.0, 0.0, 0.0], arm_id="right"),
            generation=0,
            timeout_sec=5.0,
        )

        c.start(new)

        self.assertIs(c.active.command, new)
        old_result = old.future.result(timeout=1)
        self.assertFalse(old_result.ok)
        self.assertEqual(old_result.message, "command cancelled before next command")
        self.assertEqual(self.context.command_queue.finish.call_args.args[0], old)
        self.assertFalse(new.future.done())

    def test_left_ik_uses_left_jacobian_and_joints(self):
        c = self.controller
        command = Command(kind="move_end_pose", payload=EndPoseRequest(target_pose=[0.1, 0, 0, 0, 0, 0], arm_id="left"), generation=0, timeout_sec=None)
        c._start_move_end_pose(command)
        c.active.target_world = np.eye(4)
        c.active.target_world[0, 3] = 0.1
        self.context.left_ee_prim = object()
        c._world_pose = Mock(return_value=np.eye(4))
        c.ee_jacobian_index = 0
        c.left_ee_jacobian_index = 1
        jacobians = np.zeros((1, 2, 6, 16))
        jacobians[0, 1, :, 6:12] = np.eye(6)
        c.robot.get_jacobian_matrices.return_value.numpy.return_value = jacobians
        c.robot.get_dof_limits.return_value = (
            SimpleNamespace(numpy=lambda: np.full((1, 6), -3.0)),
            SimpleNamespace(numpy=lambda: np.full((1, 6), 3.0)),
        )
        result = c._step_pose_ik(c.active, position_tolerance_m=0.01, orientation_tolerance_rad=0.06, stable_steps=1)
        self.assertIsNotNone(result)
        c._world_pose.assert_called_once_with(self.context.left_ee_prim)
        self.assertEqual(c.robot.set_dof_position_targets.call_args.kwargs["dof_indices"], c.left_arm_indices)
        self.assertGreater(c.robot.set_dof_position_targets.call_args.args[0][0], 0)

    def test_left_home_and_gripper_completion(self):
        c = self.controller
        command = Command(kind="arm_home", payload={"arm_id": "left", "close_gripper": False}, generation=0, timeout_sec=None)
        c._start_arm_home(command)
        c._ee_home_world.assert_called_with("left")
        self.assertEqual(c.active.indices, c.left_arm_indices)
        self.assertTrue(c.active.hold_gripper)
        command = Command(kind="gripper", payload={"arm_id": "left", "open": True, "duration_sec": 0.1}, generation=0, timeout_sec=None)
        c._start_gripper(command)
        self.assertEqual(c.active.indices, c.left_gripper_indices)
        self.context.snapshot = lambda: SimulationState(sim_time=0.2)
        c.update()
        result = command.future.result(timeout=1)
        self.assertTrue(result.ok)
        self.assertEqual(result.data["arm_id"], "left")
        self.assertEqual(result.data["target_gripper"], -3.2)

    def test_camera_names_and_serial_priority(self):
        for name, role in (("left_hand", "left_wrist"), ("right_hand", "right_wrist"), ("head", "head")):
            self.backend.capture_rgbd(CaptureRequest(camera_id=name))
            payload = self.context.command_queue.submit.call_args.args[1]
            self.assertEqual(payload["role"], role)
            self.assertEqual(payload["camera_id"], name)
            self.assertEqual(payload["serial_number"], self.config["camera"]["cameras"][name]["serial"])
        serial = self.config["camera"]["cameras"]["left_hand"]["serial"]
        self.backend.capture_rgbd(CaptureRequest(serial=serial, camera_id="head"))
        payload = self.context.command_queue.submit.call_args.args[1]
        self.assertEqual(payload["role"], "left_wrist")
        self.assertEqual(payload["camera_id"], "head")
        with self.assertRaisesRegex(ValueError, "unknown camera_id"):
            self.backend.capture_rgbd(CaptureRequest(camera_id="invalid"))

    def test_capture_metadata_keeps_name_and_serial_separate(self):
        with TemporaryDirectory() as temp:
            result = write_rgbd_bundle(rgb=np.zeros((2, 2, 3), dtype=np.uint8), depth_m=np.ones((2, 2)), intrinsics_matrix=np.eye(3), output_dir=Path(temp) / "capture", camera_id="left_hand", serial_number="123", sim_time=0, frame_id="left")
            self.assertEqual(result.camera_id, "left_hand")
            self.assertEqual(result.serial_number, "123")
            params = json.loads(Path(result.params_path).read_text())
            self.assertEqual(params["camera_id"], "left_hand")
            self.assertEqual(params["serial_number"], "123")


if __name__ == "__main__":
    unittest.main()
