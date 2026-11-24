from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from arx_shared.schemas import ExecutionResult
from arx_sim.backend.capture import write_rgbd_bundle
from arx_sim.backend.command_queue import Command
from arx_sim.geometry import matrix_to_rpy, pose_matrix, rotation_vector_error, rpy_to_matrix


@dataclass
class ActiveCommand:
    command: Command
    target: np.ndarray
    indices: list[int]
    stable: int = 0
    target_world: np.ndarray | None = None
    stop_sim_time: float | None = None
    stopping: bool = False
    started_sim_time: float = 0.0
    phase: str = ""
    phase_started_sim_time: float = 0.0
    pose_reached_sim_time: float | None = None
    hold_gripper: bool = False
    gripper_target_api: float | None = None
    arm_id: str = "right"


class IsaacRobotController:
    """Main-thread-only feedback controller for the Lift2 articulation."""

    def __init__(self, context: Any) -> None:
        self.context = context
        self.robot = context.robot
        self.cfg = context.config["controllers"]
        self.dof_index = {name: index for index, name in enumerate(self.robot.dof_names)}
        self.link_index = {name: index for index, name in enumerate(self.robot.link_names)}
        required = set(
            self.cfg["right_arm"]["joints"]
            + self.cfg["left_arm"]["joints"]
            + self.cfg["gripper"]["joints"]
            + self.cfg["left_gripper"]["joints"]
        )
        required.update([self.cfg["lift"]["joint"], self.cfg["chassis"]["x_joint"], self.cfg["chassis"]["y_joint"], self.cfg["chassis"]["yaw_joint"]])
        missing = sorted(required - self.dof_index.keys())
        if missing:
            raise RuntimeError(f"configured joints absent from articulation: {missing}; available={self.robot.dof_names}")
        ee_links = {
            "right": self.cfg["right_arm"]["ee_link"],
            "left": self.cfg["left_arm"]["ee_link"],
        }
        missing_ee_links = sorted(set(ee_links.values()) - self.link_index.keys())
        if missing_ee_links:
            raise RuntimeError(
                f"configured EE links absent: {missing_ee_links}; available={self.robot.link_names}"
            )
        self.arm_indices = [self.dof_index[name] for name in self.cfg["right_arm"]["joints"]]
        self.left_arm_indices = [self.dof_index[name] for name in self.cfg["left_arm"]["joints"]]
        self.gripper_indices = [self.dof_index[name] for name in self.cfg["gripper"]["joints"]]
        self.left_gripper_indices = [self.dof_index[name] for name in self.cfg["left_gripper"]["joints"]]
        self.lift_index = self.dof_index[self.cfg["lift"]["joint"]]
        self.chassis_indices = [self.dof_index[self.cfg["chassis"][key]] for key in ("x_joint", "y_joint", "yaw_joint")]
        # Fixed-base Jacobians omit the root link row.
        self.ee_jacobian_index = self.link_index[ee_links["right"]] - 1
        self.left_ee_jacobian_index = self.link_index[ee_links["left"]] - 1
        if self.ee_jacobian_index < 0:
            raise RuntimeError("right EE link cannot be the articulation root")
        if self.left_ee_jacobian_index < 0:
            raise RuntimeError("left EE link cannot be the articulation root")
        configured_home = np.asarray(
            self.cfg["right_arm"]["home_joint_positions_rad"], dtype=float
        )
        if configured_home.shape != (len(self.arm_indices),):
            raise RuntimeError(
                "right_arm.home_joint_positions_rad must contain exactly "
                f"{len(self.arm_indices)} values"
            )
        if not np.all(np.isfinite(configured_home)):
            raise RuntimeError("right_arm.home_joint_positions_rad must contain finite values")
        self.home_arm = configured_home.copy()
        configured_left_home = np.asarray(
            self.cfg["left_arm"]["home_joint_positions_rad"], dtype=float
        )
        if configured_left_home.shape != (len(self.left_arm_indices),):
            raise RuntimeError(
                "left_arm.home_joint_positions_rad must contain exactly "
                f"{len(self.left_arm_indices)} values"
            )
        if not np.all(np.isfinite(configured_left_home)):
            raise RuntimeError("left_arm.home_joint_positions_rad must contain finite values")
        self.home_left_arm = configured_left_home.copy()
        self._configure_drives()
        self._reset_arm_to_home()
        self.capture_ee_home_frames()
        self._hold_current_controlled()
        self.active: ActiveCommand | None = None
        self._ready_cameras: set[str] = set()
        self._camera_errors: dict[str, str] = {}

    def _configure_drives(self) -> None:
        """Apply configured gains after initialization and every World reset."""
        # The exported articulation carries drives, but not every drive has a
        # useful gain after unit conversion. Configure only declared joints.
        for section, indices in (
            ("right_arm", self.arm_indices),
            ("left_arm", self.left_arm_indices),
            ("gripper", self.gripper_indices),
            ("left_gripper", self.left_gripper_indices),
            ("lift", [self.lift_index]),
            ("chassis", self.chassis_indices),
        ):
            settings = self.cfg[section]
            self.robot.set_dof_gains(
                stiffnesses=float(settings["drive_stiffness"]),
                dampings=float(settings["drive_damping"]),
                dof_indices=indices,
            )
            self.robot.set_dof_max_efforts(float(settings["max_effort"]), dof_indices=indices)

    def _positions(self) -> np.ndarray:
        return np.asarray(self.robot.get_dof_positions().numpy()[0], dtype=float)

    def _velocities(self) -> np.ndarray:
        return np.asarray(self.robot.get_dof_velocities().numpy()[0], dtype=float)

    def _hold_current_controlled(self) -> None:
        positions = self._positions()
        controlled = self._controlled_indices()
        self.robot.set_dof_position_targets(positions[controlled], dof_indices=controlled)
        self.robot.set_dof_velocity_targets(np.zeros(len(controlled)), dof_indices=controlled)

    def _reset_arm_to_home(self) -> None:
        """Put both arms at deterministic home during startup or a World reset."""
        self.robot.set_dof_positions(self.home_arm, dof_indices=self.arm_indices)
        self.robot.set_dof_position_targets(self.home_arm, dof_indices=self.arm_indices)
        self.robot.set_dof_velocity_targets(
            np.zeros(len(self.arm_indices)), dof_indices=self.arm_indices
        )
        self.robot.set_dof_positions(self.home_left_arm, dof_indices=self.left_arm_indices)
        self.robot.set_dof_position_targets(self.home_left_arm, dof_indices=self.left_arm_indices)
        self.robot.set_dof_velocity_targets(
            np.zeros(len(self.left_arm_indices)), dof_indices=self.left_arm_indices
        )

    def _controlled_indices(self) -> list[int]:
        return (
            self.arm_indices
            + self.left_arm_indices
            + self.gripper_indices
            + self.left_gripper_indices
            + [self.lift_index]
            + self.chassis_indices
        )

    @staticmethod
    def _world_pose(prim: Any) -> np.ndarray:
        positions, orientations = prim.get_world_poses()
        return pose_matrix(positions.numpy()[0], orientations.numpy()[0])

    def capture_ee_home_frames(self) -> None:
        """Store each arm's joint-home EE pose in the moving arm-base frame."""
        arm_base_world = self._world_pose(self.context.arm_base_prim)
        world_to_arm_base = np.linalg.inv(arm_base_world)
        self.ee_home_in_arm_base = {
            "right": world_to_arm_base @ self._world_pose(self.context.ee_prim),
            "left": world_to_arm_base @ self._world_pose(self.context.left_ee_prim),
        }

    def _ee_home_world(self, side: str) -> np.ndarray:
        return self._world_pose(self.context.arm_base_prim) @ self.ee_home_in_arm_base[side]

    def _ee_pose_from_api(self, side: str, pose: Any) -> np.ndarray:
        """Map EE-home-relative API components through the fixed arm-base axes."""
        values = np.asarray(pose, dtype=float)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise ValueError("target_pose must contain six finite values")
        home = self.ee_home_in_arm_base[side]
        target_in_arm_base = np.eye(4)
        target_in_arm_base[:3, 3] = home[:3, 3] + values[:3]
        target_in_arm_base[:3, :3] = rpy_to_matrix(*values[3:]) @ home[:3, :3]
        return self._world_pose(self.context.arm_base_prim) @ target_in_arm_base

    def _ee_pose_to_api(self, side: str, ee_world: np.ndarray) -> list[float]:
        """Report an EE pose using the real API's fixed-axis, home-zero convention."""
        arm_base_world = self._world_pose(self.context.arm_base_prim)
        current_in_arm_base = np.linalg.inv(arm_base_world) @ ee_world
        home = self.ee_home_in_arm_base[side]
        position = current_in_arm_base[:3, 3] - home[:3, 3]
        rotation = current_in_arm_base[:3, :3] @ home[:3, :3].T
        return [*position.tolist(), *matrix_to_rpy(rotation)]

    def _arm_indices(self, side: str) -> list[int]:
        return self.left_arm_indices if side == "left" else self.arm_indices

    def _gripper_indices(self, side: str) -> list[int]:
        return self.left_gripper_indices if side == "left" else self.gripper_indices

    def _arm_config(self, side: str) -> dict[str, Any]:
        return {**self.cfg["right_arm"], **self.cfg.get(f"{side}_arm", {})}

    def _gripper_config(self, side: str) -> dict[str, Any]:
        return {**self.cfg["gripper"], **(self.cfg.get("left_gripper", {}) if side == "left" else {})}

    def _gripper_feedback_api(self, positions: np.ndarray | None = None, side: str = "right") -> float:
        values = self._positions() if positions is None else positions
        grip_m = float(np.mean(values[self._gripper_indices(side)]))
        cfg = self._gripper_config(side)
        open_m = float(cfg["open_m"])
        closed_m = float(cfg["closed_m"])
        fraction = float(np.clip((grip_m - closed_m) / max(open_m - closed_m, 1e-9), 0.0, 1.0))
        return float(cfg.get("api_closed", 0.0)) + fraction * (
            float(cfg.get("api_open", -3.2)) - float(cfg.get("api_closed", 0.0))
        )

    def _set_gripper_api_target(self, api_value: float, side: str = "right") -> None:
        cfg = self._gripper_config(side)
        indices = self._gripper_indices(side)
        api_open = float(cfg.get("api_open", -3.2))
        api_closed = float(cfg.get("api_closed", 0.0))
        denominator = api_open - api_closed
        fraction = 0.0 if abs(denominator) < 1e-9 else (float(api_value) - api_closed) / denominator
        fraction = float(np.clip(fraction, 0.0, 1.0))
        target_m = float(cfg["closed_m"]) + fraction * (float(cfg["open_m"]) - float(cfg["closed_m"]))
        self.robot.set_dof_position_targets(
            [target_m] * len(indices), dof_indices=indices
        )

    @staticmethod
    def _payload_value(payload: Any, name: str, default: Any) -> Any:
        return payload.get(name, default) if isinstance(payload, dict) else getattr(payload, name, default)

    def _gripper_close_duration(self, payload: Any) -> float:
        configured = self.context.config["motion"].get("gripper_close_duration_sec", 3.0)
        requested = self._payload_value(payload, "gripper_close_duration_sec", configured)
        return max(0.1, float(configured if requested is None else requested))

    def _hold_active_gripper_target(self, active: ActiveCommand) -> None:
        if not active.hold_gripper and active.gripper_target_api is not None:
            self._set_gripper_api_target(active.gripper_target_api, active.arm_id)

    def _complete(self, command: Command, result: ExecutionResult) -> None:
        if command.kind in {"move_end_pose", "arm_home", "gripper"}:
            result.data.setdefault("arm_id", self._payload_value(command.payload, "arm_id", "right"))
        result.data.setdefault("command_id", command.id)
        result.data.setdefault("generation", command.generation)
        if not command.future.done():
            command.future.set_result(result)
        self.context.command_queue.finish(command)
        if self.active is not None and self.active.command is command:
            self.active = None

    def _fail(self, command: Command, message: str) -> None:
        self._complete(command, ExecutionResult(ok=False, message=message))

    def _clear_cancelled_active(self) -> bool:
        if self.active is None or not self.active.command.cancelled.is_set():
            return False
        command = self.active.command
        self.safe_stop()
        self._fail(command, "command cancelled before next command")
        return True

    def start(self, command: Command) -> None:
        self._clear_cancelled_active()
        if self.active is not None:
            replace_chassis_stream = (
                command.kind == "chassis_move"
                and self.active.command.kind == "chassis_move"
                and not bool(command.payload.wait_completion)
            )
            if replace_chassis_stream:
                previous = self.active.command
                self.safe_stop()
                self.context.command_queue.finish(previous)
                self.active = None
            elif command.kind in {"chassis_stop", "lift_stop", "simulation_pause", "simulation_reset"}:
                previous = self.active.command
                self.safe_stop()
                self._fail(previous, f"{previous.kind} interrupted by {command.kind}")
            else:
                self._fail(command, f"command rejected while {self.active.command.kind} is active")
                return
        try:
            handler = getattr(self, f"_start_{command.kind}")
        except AttributeError:
            self._fail(command, f"unsupported Isaac command: {command.kind}")
            return
        try:
            handler(command)
        except Exception as exc:
            self._fail(command, f"{command.kind} failed to start: {exc}")

    def _start_capture_rgbd(self, command: Command) -> None:
        payload = command.payload
        camera = self.context.cameras[payload["role"]]
        age=float(self.context.world.current_time)-float(camera.frame_time or 0.0)
        if camera.frame_time is None or age>0.5:
            self._fail(command, f"camera {payload['role']} frame is stale ({age:.2f}s): {self._camera_errors.get(payload['role'], 'no fresh sensor frame')}")
            return
        rgba = camera.get_rgba(device="cpu")
        depth = camera.get_depth(device="cpu")
        if rgba is None or depth is None:
            self._fail(command, f"camera {payload['role']} sensor data is not ready")
            return
        rgb = np.asarray(rgba)
        if np.issubdtype(rgb.dtype, np.floating):
            rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        else:
            rgb = rgb.astype(np.uint8, copy=False)
        intrinsics = np.asarray(camera.get_intrinsics_matrix(device="cpu"))
        result = write_rgbd_bundle(
            rgb=rgb, depth_m=np.asarray(depth), intrinsics_matrix=intrinsics,
            output_dir=Path(payload["output_dir"]), camera_id=payload["camera_id"],
            serial_number=payload["serial_number"],
            sim_time=float(camera.frame_time), frame_id=f"{payload['role']}_camera_optical_frame",
            depth_scale=float(self.cfg["camera"]["depth_scale"]),
        )
        if not command.future.done():
            command.future.set_result(result)
        self.context.command_queue.finish(command)

    def _start_move_end_pose(self, command: Command) -> None:
        request = command.payload
        side = self._payload_value(request, "arm_id", "right")
        indices = self._arm_indices(side)
        target_world = self._ee_pose_from_api(side, request.target_pose)
        current = self._positions()
        hold_gripper = bool(request.hold_gripper)
        target_gripper = float(
            self._gripper_config(side).get("api_closed", 0.0)
            if request.target_gripper is None
            else request.target_gripper
        )
        if not hold_gripper:
            self._set_gripper_api_target(target_gripper, side)
        sim_time = float(self.context.snapshot().sim_time)
        self.active = ActiveCommand(
            command,
            current[indices].copy(),
            indices,
            arm_id=side,
            target_world=target_world,
            started_sim_time=sim_time,
            phase="move",
            phase_started_sim_time=sim_time,
            hold_gripper=hold_gripper,
            gripper_target_api=target_gripper,
        )

    def _start_arm_home(self, command: Command) -> None:
        payload = command.payload
        side = self._payload_value(payload, "arm_id", "right")
        indices = self._arm_indices(side)
        grip_cfg = self._gripper_config(side)
        close_gripper = bool(self._payload_value(payload, "close_gripper", True))
        sim_time = float(self.context.snapshot().sim_time)
        phase = "preclose" if close_gripper else "home"
        if close_gripper:
            self._set_gripper_api_target(float(grip_cfg.get("api_closed", 0.0)), side)
        current = self._positions()
        self.robot.set_dof_position_targets(current[indices], dof_indices=indices)
        self.active = ActiveCommand(
            command,
            current[indices].copy(),
            indices,
            arm_id=side,
            target_world=self._ee_home_world(side),
            started_sim_time=sim_time,
            phase=phase,
            phase_started_sim_time=sim_time,
            hold_gripper=not close_gripper,
            gripper_target_api=(
                float(grip_cfg.get("api_closed", 0.0))
                if close_gripper else self._gripper_feedback_api(current, side)
            ),
        )

    def _start_gripper(self, command: Command) -> None:
        side = command.payload.get("arm_id", "right")
        indices = self._gripper_indices(side)
        grip_cfg = self._gripper_config(side)
        open_gripper = bool(command.payload["open"])
        target = float(
            grip_cfg["open_m"]
            if open_gripper
            else grip_cfg["closed_m"]
        )
        configured_duration = self.context.config["motion"].get(
            "gripper_open_duration_sec" if open_gripper else "gripper_close_duration_sec",
            2.0 if open_gripper else 3.0,
        )
        duration = command.payload.get("duration_sec")
        duration = configured_duration if duration is None else duration
        duration = max(0.1, float(duration))
        values = np.full(len(indices), target)
        self.robot.set_dof_position_targets(values, dof_indices=indices)
        sim_time = float(self.context.snapshot().sim_time)
        self.active = ActiveCommand(
            command,
            values,
            indices,
            arm_id=side,
            started_sim_time=sim_time,
            stop_sim_time=sim_time + duration,
            gripper_target_api=float(
                grip_cfg["api_open"]
                if open_gripper
                else grip_cfg["api_closed"]
            ),
        )

    def _native_to_lift(self, native: float) -> float:
        cfg = self.cfg["lift"]
        return float(cfg["joint_zero_m"]) + (native - float(cfg["native_zero"])) / float(cfg["native_per_meter"])

    def _start_lift_height(self, command: Command) -> None:
        target = self._native_to_lift(float(command.payload.height))
        if not float(self.cfg["lift"]["min_m"]) <= target <= float(self.cfg["lift"]["max_m"]):
            self._fail(command, f"mapped lift target {target:.4f}m is outside asset limits")
            return
        self.robot.set_dof_position_targets([target], dof_indices=[self.lift_index])
        self.active = ActiveCommand(command, np.array([target]), [self.lift_index])

    def _start_lift_stop(self, command: Command) -> None:
        current = self._positions()[self.lift_index]
        self.robot.set_dof_position_targets([current], dof_indices=[self.lift_index])
        self._complete(command, ExecutionResult(ok=True, message="lift stopped at current joint feedback", data={"joint_position_m": float(current)}))

    def _start_chassis_move(self, command: Command) -> None:
        request = command.payload
        current = self._positions()[self.chassis_indices]
        yaw = float(current[2])
        cosine, sine = math.cos(yaw), math.sin(yaw)
        chassis_cfg = self.cfg["chassis"]
        linear_x = float(request.linear_x) * float(chassis_cfg.get("linear_x_scale", 1.0))
        linear_y = float(request.linear_y) * float(chassis_cfg.get("linear_y_scale", 1.0))
        angular_z = float(request.angular_z) * float(chassis_cfg.get("angular_z_scale", 1.0))
        duration = float(request.duration)
        max_duration = chassis_cfg.get("max_command_duration_sec")
        if max_duration is not None:
            duration = min(duration, float(max_duration))
        # HTTP linear velocities use base frame semantics; the exported planar
        # x/y joints are world-aligned.
        velocity = np.array([
            cosine * linear_x - sine * linear_y,
            sine * linear_x + cosine * linear_y,
            angular_z,
        ], dtype=float)
        self.robot.set_dof_velocity_targets(velocity, dof_indices=self.chassis_indices)
        self.active = ActiveCommand(
            command, np.zeros(3), self.chassis_indices,
            stop_sim_time=float(self.context.snapshot().sim_time) + duration,
        )
        if not bool(request.wait_completion) and not command.future.done():
            command.future.set_result(ExecutionResult(
                ok=True,
                message="Isaac chassis move command accepted; feedback control remains active",
                data={"command_id": command.id, "generation": command.generation},
            ))

    def _start_chassis_stop(self, command: Command) -> None:
        current = self._positions()[self.chassis_indices]
        self.robot.set_dof_position_targets(current, dof_indices=self.chassis_indices)
        self.robot.set_dof_velocity_targets([0.0, 0.0, 0.0], dof_indices=self.chassis_indices)
        self._complete(command, ExecutionResult(ok=True, message="Isaac chassis stopped from articulation feedback", data={"x": float(current[0]), "y": float(current[1]), "yaw": float(current[2])}))

    def _start_simulation_pause(self, command: Command) -> None:
        self.safe_stop()
        self.context.command_queue.set_paused()
        self.context.publish(lifecycle="PAUSED")
        self._complete(command, ExecutionResult(ok=True, message="simulation paused"))

    def _start_simulation_resume(self, command: Command) -> None:
        self.context.command_queue.set_ready()
        self.context.publish(lifecycle="READY")
        self._complete(command, ExecutionResult(ok=True, message="simulation resumed"))

    def _start_simulation_reset(self, command: Command) -> None:
        self.context.publish(lifecycle="RESETTING")
        generation = self.context.command_queue.begin_reset(exclude=command)
        self.safe_stop()
        for camera in self.context.cameras.values():
            camera.destroy()
        self.context.world.reset()
        if self.context.contact_materials_applier is not None:
            self.context.contact_materials_applier()
        self._configure_drives()
        self._reset_arm_to_home()
        self.capture_ee_home_frames()
        self._hold_current_controlled()
        if self.context.camera_factory is None:
            self._fail(command, "simulation reset cannot restore cameras: camera factory unavailable")
            return
        self.context.cameras = self.context.camera_factory()
        self._ready_cameras.clear()
        self._camera_errors.clear()
        self.active = None
        self.context.command_queue.set_ready()
        self.publish_snapshot(0.0)
        self._complete(command, ExecutionResult(
            ok=True,
            message="simulation reset from asset default state",
            data={"generation": generation},
        ))

    def _step_pose_ik(
        self,
        active: ActiveCommand,
        *,
        position_tolerance_m: float,
        orientation_tolerance_rad: float,
        stable_steps: int,
    ) -> tuple[bool, float, float] | None:
        """Advance the existing Jacobian IK once and report pose convergence."""
        cfg = self._arm_config(active.arm_id)
        if active.target_world is None:
            self._fail(active.command, "IK target pose is unavailable")
            return None
        ee_prim = self.context.left_ee_prim if active.arm_id == "left" else self.context.ee_prim
        ee_world = self._world_pose(ee_prim)
        position_error = active.target_world[:3, 3] - ee_world[:3, 3]
        rotation_error = rotation_vector_error(active.target_world[:3, :3], ee_world[:3, :3])
        linear_norm = float(np.linalg.norm(position_error))
        angular_norm = float(np.linalg.norm(rotation_error))
        if linear_norm <= position_tolerance_m and angular_norm <= orientation_tolerance_rad:
            active.stable += 1
            return active.stable >= stable_steps, linear_norm, angular_norm
        active.stable = 0
        jacobians = np.asarray(self.robot.get_jacobian_matrices().numpy()[0])
        jacobian_index = self.left_ee_jacobian_index if active.arm_id == "left" else self.ee_jacobian_index
        if jacobian_index >= jacobians.shape[0]:
            self._fail(active.command, f"EE Jacobian row missing: index={jacobian_index}, shape={jacobians.shape}")
            return None
        jacobian = jacobians[jacobian_index][:, active.indices]
        if not np.isfinite(jacobian).all() or np.linalg.norm(jacobian) < 1e-8:
            self._fail(active.command, "IK unavailable: EE Jacobian is invalid")
            return None
        error = np.concatenate([position_error, rotation_error])
        damping = float(cfg["damping"])
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + np.eye(6) * damping * damping, error)
        speed = float(self._payload_value(active.command.payload, "speed", 1.0))
        max_step = float(cfg["joint_step_rad"]) * max(0.05, min(speed, 2.0))
        delta = np.clip(delta, -max_step, max_step)
        current = self._positions()[active.indices]
        lower, upper = self.robot.get_dof_limits(dof_indices=active.indices)
        lower_np = np.asarray(lower.numpy()[0], dtype=float)
        upper_np = np.asarray(upper.numpy()[0], dtype=float)
        target = np.clip(current + delta, lower_np, upper_np)
        if np.linalg.norm(target - current) < 1e-8:
            self._fail(active.command, "IK cannot progress within joint limits")
            return None
        self.robot.set_dof_position_targets(target, dof_indices=active.indices)
        return False, linear_norm, angular_norm

    def _update_move_end_pose(self, active: ActiveCommand, sim_time: float) -> None:
        request = active.command.payload
        exit_after_sec = self._payload_value(request, "exit_after_sec", None)
        if exit_after_sec is not None and float(exit_after_sec) > 0.0:
            if sim_time - active.started_sim_time >= float(exit_after_sec):
                self._complete(
                    active.command,
                    ExecutionResult(
                        ok=True,
                        message="end-pose command completed after requested duration",
                        data={"elapsed_sec": sim_time - active.started_sim_time},
                    ),
                )
                return

        cfg = self._arm_config(active.arm_id)
        result = self._step_pose_ik(
            active,
            position_tolerance_m=float(cfg["position_tolerance_m"]),
            orientation_tolerance_rad=float(cfg["orientation_tolerance_rad"]),
            stable_steps=int(cfg["stable_steps"]),
        )
        if result is None or self.active is None:
            return
        reached, linear_norm, angular_norm = result
        if not reached:
            active.pose_reached_sim_time = None
            return
        if bool(self._payload_value(request, "exit_on_pose", False)):
            self._complete(
                active.command,
                ExecutionResult(
                    ok=True,
                    message="end pose reached from Isaac articulation feedback",
                    data={
                        "position_error_m": linear_norm,
                        "orientation_error_rad": angular_norm,
                    },
                ),
            )
            return

        if active.pose_reached_sim_time is None:
            active.pose_reached_sim_time = sim_time
        close_delay = max(0.0, float(self._payload_value(request, "close_delay", 0.5)))
        if not active.hold_gripper and sim_time - active.pose_reached_sim_time >= close_delay:
            active.gripper_target_api = float(
                self._gripper_config(active.arm_id).get("api_close_after_pose", 0.1)
            )
            self._set_gripper_api_target(active.gripper_target_api, active.arm_id)

    def _update_arm_home(self, active: ActiveCommand, sim_time: float) -> None:
        payload = active.command.payload
        self._hold_active_gripper_target(active)
        if active.phase == "preclose":
            if sim_time - active.phase_started_sim_time < self._gripper_close_duration(payload):
                return
            active.phase = "home"
            active.phase_started_sim_time = sim_time
            active.stable = 0

        active.target_world = self._ee_home_world(active.arm_id)
        linear_threshold = float(self._payload_value(payload, "linear_threshold", 0.006))
        angular_threshold = float(self._payload_value(payload, "angular_threshold", 0.03))
        result = self._step_pose_ik(
            active,
            position_tolerance_m=linear_threshold,
            orientation_tolerance_rad=angular_threshold,
            stable_steps=int(self._arm_config(active.arm_id)["stable_steps"]),
        )
        if result is None or self.active is None:
            return
        reached, linear_norm, angular_norm = result
        minimum_duration = float(self._arm_config(active.arm_id).get("home_min_duration_sec", 2.0))
        if reached and sim_time - active.phase_started_sim_time >= minimum_duration:
            data = {
                "position_error_m": linear_norm,
                "orientation_error_rad": angular_norm,
                "close_gripper": not active.hold_gripper,
            }
            if not active.hold_gripper:
                data["close_result"] = ExecutionResult(
                    ok=True,
                    message="pre-close gripper duration completed before home",
                    data={
                        "target_gripper": active.gripper_target_api,
                        "final_gripper": self._gripper_feedback_api(side=active.arm_id),
                        "arm_id": active.arm_id,
                        "command_duration_sec": self._gripper_close_duration(payload),
                    },
                ).model_dump()
            self._complete(
                active.command,
                ExecutionResult(
                    ok=True,
                    message="arm home reached from Isaac EE-home-relative feedback",
                    data=data,
                ),
            )

    def update(self) -> None:
        active = self.active
        if active is None:
            return
        if active.command.cancelled.is_set():
            self.safe_stop()
            self._fail(active.command, "command cancelled")
            return
        if active.command.generation != self.context.command_queue.generation:
            self.safe_stop()
            self._fail(active.command, "command belongs to a stale reset generation")
            return
        deadline = active.command.deadline
        if deadline is not None and time.monotonic() >= deadline:
            self.safe_stop()
            self._fail(active.command, f"{active.command.kind} timed out before feedback converged")
            return
        sim_time = float(self.context.snapshot().sim_time)
        if active.command.kind == "move_end_pose":
            self._update_move_end_pose(active, sim_time)
            return
        if active.command.kind == "arm_home":
            self._update_arm_home(active, sim_time)
            return
        if active.command.kind == "chassis_move":
            if active.stop_sim_time is not None and sim_time < active.stop_sim_time:
                return
            if not active.stopping:
                self.robot.set_dof_velocity_targets(np.zeros(3), dof_indices=self.chassis_indices)
                active.stopping = True
            current_velocity = self._velocities()[self.chassis_indices]
            tolerance = float(self.cfg["chassis"]["velocity_tolerance"])
            active.stable = active.stable + 1 if float(np.max(np.abs(current_velocity))) <= tolerance else 0
            if active.stable >= int(self.cfg["chassis"]["stable_steps"]):
                current = self._positions()[self.chassis_indices]
                self._complete(active.command, ExecutionResult(
                    ok=True,
                    message="chassis_move completed from timed Isaac velocity and stop feedback",
                    data={"joint_positions": current.tolist(), "max_velocity": float(np.max(np.abs(current_velocity)))},
                ))
            return
        current = self._positions()[active.indices]
        if active.command.kind == "gripper":
            if active.stop_sim_time is not None and sim_time < active.stop_sim_time:
                return
            self._complete(
                active.command,
                ExecutionResult(
                    ok=True,
                    message="gripper command completed after requested duration",
                    data={
                        "target_gripper": active.gripper_target_api,
                        "final_gripper": self._gripper_feedback_api(side=active.arm_id),
                        "command_duration_sec": sim_time - active.started_sim_time,
                    },
                ),
            )
            return
        elif active.command.kind == "lift_height":
            tolerance = float(self.cfg["lift"]["tolerance_m"]); stable_steps = int(self.cfg["lift"]["stable_steps"])
        else:
            self._fail(active.command, f"no update handler for {active.command.kind}")
            return
        error = float(np.max(np.abs(active.target - current)))
        active.stable = active.stable + 1 if error <= tolerance else 0
        if active.stable >= stable_steps:
            self._complete(active.command, ExecutionResult(ok=True, message=f"{active.command.kind} reached from Isaac joint feedback", data={"max_joint_error": error, "joint_positions": current.tolist()}))

    def publish_snapshot(self, sim_time: float) -> None:
        positions = self._positions(); velocities = self._velocities()
        ee_world = self._world_pose(self.context.ee_prim)
        end_pos = self._ee_pose_to_api("right", ee_world)
        gripper_api = self._gripper_feedback_api(positions)
        arm_joint_pos = tuple(float(positions[i]) for i in self.arm_indices) + (gripper_api,)
        left_end_pos = self._ee_pose_to_api("left", self._world_pose(self.context.left_ee_prim))
        left_gripper = self._gripper_feedback_api(positions, "left")
        left_joint_pos = tuple(float(positions[i]) for i in self.left_arm_indices) + (left_gripper,)
        for role, camera in self.context.cameras.items():
            try:
                if camera.refresh(float(sim_time)):
                    self._ready_cameras.add(role)
                    self._camera_errors.pop(role, None)
            except Exception as exc:
                self._camera_errors[role] = str(exc)
        self.context.publish(
            lifecycle="READY", generation=self.context.command_queue.generation, sim_time=float(sim_time),
            arm_joint_pos=arm_joint_pos, end_pos=tuple(end_pos),
            left_arm_joint_pos=left_joint_pos, left_end_pos=tuple(left_end_pos), left_gripper=left_gripper,
            gripper=gripper_api, chassis=tuple(float(positions[i]) for i in self.chassis_indices),
            chassis_velocity=tuple(float(velocities[i]) for i in self.chassis_indices), lift_m=float(positions[self.lift_index]),
            lift_velocity_mps=float(velocities[self.lift_index]),
            cameras_ready=tuple(sorted(self._ready_cameras)),
            error="; ".join(f"camera {role}: {message}" for role, message in sorted(self._camera_errors.items())),
        )

    def safe_stop(self) -> None:
        positions = self._positions()
        controlled = self._controlled_indices()
        self.robot.set_dof_position_targets(positions[controlled], dof_indices=controlled)
        self.robot.set_dof_velocity_targets(np.zeros(len(controlled)), dof_indices=controlled)

    def fail_active(self, message: str) -> None:
        """Turn a controller exception into an explicit command failure."""
        if self.active is None:
            raise RuntimeError(message)
        command = self.active.command
        try:
            self.safe_stop()
        finally:
            self._fail(command, message)
