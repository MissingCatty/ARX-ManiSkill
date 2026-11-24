from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from uuid import uuid4

from arx_shared.schemas import ArmStatus, CaptureRequest, ChassisMoveRequest, ChassisStopRequest, EndPoseRequest, ExecutionResult, HealthResponse, LiftHeightRequest

from arx_sim.errors import SimBackendError, SimNotReadyError
from .context import IsaacContext


class IsaacBackend:
    """Thread-safe API facade; mutable Isaac objects remain on the main thread."""

    name = "isaac"

    def __init__(self, context: IsaacContext) -> None:
        self.context = context
        self._default_timeout = float(context.config["simulation"].get("command_timeout_sec", 30.0))

    def health(self) -> HealthResponse:
        return HealthResponse(service="arx-ops")

    def _require_snapshot(self):
        snapshot = self.context.snapshot()
        if snapshot.lifecycle != "READY":
            raise SimNotReadyError(snapshot.error or f"simulation is {snapshot.lifecycle}")
        return snapshot

    def _submit(
        self,
        kind: str,
        payload: Any,
        timeout: float | None = None,
        *,
        preserve_none_timeout: bool = False,
    ):
        effective_timeout = (
            timeout
            if timeout is not None or preserve_none_timeout
            else self._default_timeout
        )
        return self.context.command_queue.submit(
            kind,
            payload,
            None if effective_timeout is None else float(effective_timeout),
        )

    def _resolve_arm(self, arm_id: str | None) -> str:
        selected = str(arm_id or self.context.config.get("robot", {}).get("default_arm_id") or "right").strip() or "right"
        if selected not in {"right", "left"}:
            raise ValueError(f"unknown arm_id: {selected}")
        return selected

    def _submit_motion(self, kind: str, payload: Any, timeout: float | None, arm_id: str) -> ExecutionResult:
        try:
            return self._submit(kind, payload, timeout, preserve_none_timeout=True)
        except SimBackendError as exc:
            return ExecutionResult(ok=False, message=str(exc), data={"arm_id": arm_id})

    def arm_status(self, arm_id: str = "right") -> ArmStatus:
        arm_id = self._resolve_arm(arm_id)
        snapshot = self._require_snapshot()
        prefix = "left_" if arm_id == "left" else ""
        end_pos = getattr(snapshot, prefix + "end_pos")
        joint_pos = getattr(snapshot, prefix + "arm_joint_pos")
        gripper = getattr(snapshot, prefix + "gripper")
        if len(end_pos) != 6 or len(joint_pos) != 7:
            raise SimNotReadyError("articulation feedback is not ready")
        return ArmStatus(end_pos=list(end_pos), joint_pos=list(joint_pos), gripper=gripper, source="isaac_articulation", arm_id=arm_id)

    def capture_rgbd(self, request: CaptureRequest):
        self._require_snapshot()
        camera_cfg = self.context.config["controllers"]["camera"]
        serials = camera_cfg.get("logical_serials", {})
        cfg = self.context.config.get("camera", {})
        cameras = cfg.get("cameras", {})
        if request.serial:
            requested = request.serial
            camera_id = request.camera_id or next((name for name, entry in cameras.items() if str(entry.get("serial", "")).strip() == str(requested).strip()), None)
        else:
            camera_id = str(request.camera_id or cfg.get("default") or cfg.get("name") or "").strip()
            if camera_id in cameras:
                requested = cameras[camera_id].get("serial")
                if not requested:
                    raise ValueError(f"camera {camera_id!r} has no serial configured")
            elif request.camera_id:
                raise ValueError(f"unknown camera_id: {camera_id}")
            else:
                requested = cfg.get("serial", "352122272245")
                camera_id = str(cfg.get("name", "default"))
        role = serials.get(str(requested), str(requested))
        if role not in self.context.cameras:
            raise SimNotReadyError(f"unknown or unavailable simulation camera role: {role}")
        if request.output_dir:
            output = Path(request.output_dir)
        else:
            output = Path(self.context.config["runtime"]["run_dir"]) / "artifacts" / f"capture_{uuid4().hex[:12]}"
        definitions = camera_cfg.get("definitions", {})
        serial_number = definitions.get(role, {}).get("serial", str(requested))
        return self._submit("capture_rgbd", {"role": role, "output_dir": str(output), "camera_id": camera_id, "serial_number": serial_number})

    def move_end_pose(self, request: EndPoseRequest) -> ExecutionResult:
        request = request.model_copy(update={"arm_id": self._resolve_arm(request.arm_id)})
        try:
            self._validate_target_pose(request.target_pose)
        except ValueError as exc:
            return ExecutionResult(ok=False, message=str(exc), data={"arm_id": request.arm_id})
        return self._submit_motion(
            "move_end_pose",
            request,
            request.timeout_sec,
            request.arm_id,
        )

    def gripper(self, *, open_gripper: bool, duration_sec: float | None = None, arm_id: str = "right") -> ExecutionResult:
        arm_id = self._resolve_arm(arm_id)
        motion = self.context.config["motion"]
        configured_duration = motion.get(
            "gripper_open_duration_sec" if open_gripper else "gripper_close_duration_sec",
            2.0 if open_gripper else 3.0,
        )
        command_duration = max(
            0.1,
            float(configured_duration if duration_sec is None else duration_sec),
        )
        timeout_margin = max(1.0, float(motion.get("gripper_timeout_margin_sec", 3.0)))
        return self._submit_motion(
            "gripper",
            {"open": open_gripper, "duration_sec": command_duration, "arm_id": arm_id},
            command_duration + timeout_margin,
            arm_id,
        )

    def chassis_move(self, request: ChassisMoveRequest) -> ExecutionResult:
        try:
            cfg = self.context.config.get("chassis", {})
            self._validate_range("linear_x", float(request.linear_x), cfg.get("linear_x_range", (-0.5, 0.5)))
            self._validate_range("linear_y", float(request.linear_y), cfg.get("linear_y_range", (-0.5, 0.5)))
            self._validate_range("angular_z", float(request.angular_z), cfg.get("angular_z_range", (-0.8, 0.8)))
            self._validate_range("duration", float(request.duration), cfg.get("duration_range", (0.0, 10.0)))
            if request.duration <= 0:
                raise ValueError("duration must be > 0 for /chassis/move; use /chassis/stop to stop")
            if request.linear_y and not self.context.config["controllers"]["chassis"].get("supports_linear_y", False):
                raise ValueError("robot asset does not support linear_y")
        except ValueError as exc:
            return ExecutionResult(ok=False, message=str(exc))
        timeout = request.timeout_sec or (float(request.duration) + self._default_timeout)
        return self._submit("chassis_move", request, timeout)

    def chassis_stop(self, request: ChassisStopRequest | None = None) -> ExecutionResult:
        return self._submit("chassis_stop", request or ChassisStopRequest())

    def chassis_status(self) -> ExecutionResult:
        snapshot = self._require_snapshot()
        velocity_tolerance = float(self.context.config["controllers"]["chassis"].get("velocity_tolerance", 0.005))
        cosine, sine = math.cos(snapshot.chassis[2]), math.sin(snapshot.chassis[2])
        world_x, world_y = snapshot.chassis_velocity[:2]
        linear_x = cosine * world_x + sine * world_y
        linear_y = -sine * world_x + cosine * world_y
        return ExecutionResult(ok=True, message="Isaac chassis feedback", data={
            "mode": "isaac_planar_virtual_joints", "moving": any(abs(v) > velocity_tolerance for v in (linear_x, linear_y, snapshot.chassis_velocity[2])),
            "x": snapshot.chassis[0], "y": snapshot.chassis[1], "yaw": snapshot.chassis[2],
            "linear_x": linear_x, "linear_y": linear_y, "angular_z": snapshot.chassis_velocity[2],
        })

    def lift_height(self, request: LiftHeightRequest) -> ExecutionResult:
        mapping = self.context.config["controllers"]["lift"]
        if mapping.get("native_zero") is None or mapping.get("native_per_meter") is None:
            return ExecutionResult(ok=False, message="lift native-height to metre mapping is not configured; refusing to guess")
        return self._submit("lift_height", request)

    def lift_home(self) -> ExecutionResult:
        home = float(self.context.config["lift"].get("home_height", 0.5))
        return self.lift_height(LiftHeightRequest(height=home, hold=False))

    def lift_status(self) -> ExecutionResult:
        snapshot = self._require_snapshot()
        mapping = self.context.config["controllers"]["lift"]
        if snapshot.lift_m is None:
            return ExecutionResult(ok=False, message="Isaac lift feedback is not ready")
        tolerance = float(mapping.get("velocity_tolerance_mps", 0.002))
        data = {
            "joint_position_m": snapshot.lift_m,
            "joint_velocity_mps": snapshot.lift_velocity_mps,
            "running": abs(snapshot.lift_velocity_mps) > tolerance,
            "mode": "isaac_joint_feedback",
        }
        if mapping.get("native_zero") is not None and mapping.get("native_per_meter") is not None:
            native = float(mapping["native_zero"]) + (snapshot.lift_m - float(mapping.get("joint_zero_m", 0.0))) * float(mapping["native_per_meter"])
            data.update({"feedback_height": native, "target_height": native, "height_command": native})
        return ExecutionResult(ok=True, message="Isaac lift joint feedback", data=data)

    def lift_stop(self) -> ExecutionResult:
        return self._submit("lift_stop", {})

    def home_arm(self, **kwargs: Any) -> ExecutionResult:
        kwargs["arm_id"] = self._resolve_arm(kwargs.get("arm_id"))
        return self._submit_motion("arm_home", kwargs, float(kwargs.get("timeout_sec", 120.0)), kwargs["arm_id"])

    def reset_grasp_scene(self) -> ExecutionResult:
        task_layout = self.context.config.get("task_layout", {})
        if task_layout.get("name") != "grasp":
            return ExecutionResult(
                ok=False,
                message="grasp task is not loaded; restart Isaac Sim with --task grasp",
                data={"loaded_task": task_layout.get("name")},
            )
        timeout = float(self.context.config["simulation"].get("reset_timeout_sec", self._default_timeout))
        return self._submit(
            "simulation_reset",
            {"task": "grasp", "reason": "reset_grasp_scene"},
            timeout,
        )

    def _validate_target_pose(self, pose: list[float]) -> None:
        if len(pose) != 6:
            raise ValueError("target_pose must contain 6 values")
        motion = self.context.config["motion"]
        for name, value, key in zip(("x", "y", "z"), pose[:3], ("x_range", "y_range", "z_range")):
            self._validate_range(name, float(value), motion[key])
        for name, value in zip(("roll", "pitch", "yaw"), pose[3:]):
            self._validate_range(name, float(value), (-math.pi, math.pi))

    @staticmethod
    def _validate_range(name: str, value: float, bounds) -> None:
        low, high = map(float, bounds)
        if not low <= value <= high:
            raise ValueError(f"{name}={value:.4f} out of configured range [{low}, {high}]")
