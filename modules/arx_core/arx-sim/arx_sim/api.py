from __future__ import annotations

from fastapi import FastAPI, HTTPException

from arx_shared.schemas import (
    ChassisMoveRequest,
    ChassisStopRequest,
    CaptureRequest,
    EndPoseRequest,
    GripperRequest,
    HealthResponse,
    LiftHeightRequest,
)

from .errors import SimBackendError


def create_app(backend) -> FastAPI:
    """Build the exact public HTTP surface exposed by main's arx-ops."""
    app = FastAPI(title="coolas arx-ops", version="0.2.0-phase2")

    def call(method: str, *args, **kwargs):
        try:
            return getattr(backend, method)(*args, **kwargs)
        except SimBackendError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/health")
    def health() -> HealthResponse:
        return HealthResponse(service="arx-ops")

    @app.get("/arm/status")
    def arm_status(arm_id: str = "right"):
        return call("arm_status", arm_id=arm_id)

    @app.post("/camera/capture-rgbd")
    def capture(request: CaptureRequest):
        return call("capture_rgbd", request)

    @app.post("/arm/move-end-pose")
    def move(request: EndPoseRequest):
        return call("move_end_pose", request)

    @app.post("/gripper/open")
    def open_gripper(request: GripperRequest | None = None):
        request = request or GripperRequest()
        return call("gripper", open_gripper=True, duration_sec=request.duration_sec, arm_id=request.arm_id)

    @app.post("/gripper/close")
    def close_gripper(request: GripperRequest | None = None):
        request = request or GripperRequest()
        return call("gripper", open_gripper=False, duration_sec=request.duration_sec, arm_id=request.arm_id)

    @app.post("/chassis/move")
    def chassis_move(request: ChassisMoveRequest):
        return call("chassis_move", request)

    @app.post("/chassis/stop")
    def chassis_stop(request: ChassisStopRequest | None = None):
        return call("chassis_stop", request)

    @app.get("/chassis/status")
    def chassis_status():
        return call("chassis_status")

    @app.post("/lift/height")
    def lift_height(request: LiftHeightRequest):
        return call("lift_height", request)

    @app.post("/lift/home")
    def lift_home():
        return call("lift_home")

    @app.get("/lift/status")
    def lift_status():
        return call("lift_status")

    @app.post("/lift/stop")
    def lift_stop():
        return call("lift_stop")

    @app.post("/arm/home")
    def home(
        speed: float = 0.5,
        close_delay: float = 0.5,
        timeout_sec: float = 120.0,
        auto_quit_after: float = 5.0,
        close_gripper: bool = True,
        linear_threshold: float = 0.006,
        angular_threshold: float = 0.03,
        gripper_close_duration_sec: float | None = None,
        arm_id: str = "right",
    ):
        return call(
            "home_arm",
            speed=speed,
            close_delay=close_delay,
            timeout_sec=timeout_sec,
            auto_quit_after=auto_quit_after,
            close_gripper=close_gripper,
            linear_threshold=linear_threshold,
            angular_threshold=angular_threshold,
            gripper_close_duration_sec=gripper_close_duration_sec,
            arm_id=arm_id,
        )

    @app.post("/task/reset-grasp-scene")
    def reset_grasp_scene():
        return call("reset_grasp_scene")

    return app
