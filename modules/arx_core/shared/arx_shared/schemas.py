from __future__ import annotations

from typing import Any, Literal

try:
    from pydantic import BaseModel, Field
except Exception:  # pragma: no cover - lets non-API tooling import this module without pydantic.
    class BaseModel:  # type: ignore[no-redef]
        def __init__(self, **data: Any) -> None:
            self.__dict__.update(data)

        def model_dump(self) -> dict[str, Any]:
            return dict(self.__dict__)

    def Field(default: Any = None, **_: Any) -> Any:  # type: ignore[no-redef]
        return default


class HealthResponse(BaseModel):
    service: str
    status: Literal["ok"] = "ok"
    stage: str = "phase2"


class CameraIntrinsics(BaseModel):
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    model: str | None = None
    coeffs: list[float] = Field(default_factory=list)


class CaptureRequest(BaseModel):
    serial: str | None = None
    camera_id: str | None = None
    output_dir: str | None = None


class CaptureResult(BaseModel):
    color_path: str
    depth_path: str
    params_path: str
    intrinsics: CameraIntrinsics | None = None
    depth_scale: float | None = None
    serial_number: str | None = None
    camera_id: str | None = None


class ArmStatus(BaseModel):
    end_pos: list[float] = Field(default_factory=list)
    joint_pos: list[float] = Field(default_factory=list)
    gripper: float | None = None
    source: str = "arm_status"
    arm_id: str = "right"


class EndPoseRequest(BaseModel):
    target_pose: list[float]
    arm_id: str = "right"
    target_gripper: float | None = None
    speed: float = 1.0
    close_delay: float = 0.5
    hold_gripper: bool = False
    exit_on_pose: bool = False
    exit_after_sec: float | None = None
    timeout_sec: float | None = None


class GripperRequest(BaseModel):
    arm_id: str = "right"
    duration_sec: float | None = None


class LiftHeightRequest(BaseModel):
    height: float
    robot_type: int = 1
    warmup: float = 4.0
    feedback_target: bool = True
    ramp_step: float = 0.001
    max_command: float = 0.0
    until_target: bool = True
    hold: bool = True
    stop_existing: bool = True


class ChassisMoveRequest(BaseModel):
    linear_x: float = 0.0
    linear_y: float = 0.0
    angular_z: float = 0.0
    duration: float = 1.0
    height: float | None = None
    read_current_height: bool = True
    lock_after: bool = True
    wait_completion: bool = True
    timeout_sec: float | None = None


class ChassisStopRequest(BaseModel):
    height: float | None = None
    read_current_height: bool = True
    lock: bool = True
    timeout_sec: float | None = None


class ExecutionResult(BaseModel):
    ok: bool
    message: str = ""
    command: list[str] = Field(default_factory=list)
    returncode: int | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class PromptPoint(BaseModel):
    """An image-space point used to select one instance from text candidates."""

    x: float
    y: float
    label: Literal["positive", "negative"] = "positive"


class SegmentRequest(BaseModel):
    image_path: str
    prompt: str = "red cube."
    points: list[PromptPoint] = Field(default_factory=list)
    output_dir: str | None = None
    method: Literal["langsam", "hsv"] = "langsam"
    mask_name: str | None = None
    overlay_name: str | None = None
    info_name: str | None = None
    min_area: int = 50
    keep_largest: bool = True


class SegmentationResult(BaseModel):
    mask_path: str | None = None
    overlay_path: str | None = None
    info_path: str | None = None
    bbox: list[float] | None = None
    centroid: list[float] | None = None
    score: float | None = None
    label: str | None = None
    found: bool = False
    area: int | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class ZeroGraspInputRequest(BaseModel):
    camera_dir: str
    mask_path: str
    label: str = "target"
    output_dir: str | None = None
    depth_image_paths: list[str] = Field(default_factory=list)
    depth_fusion_enabled: bool = False
    depth_fusion_drop_worst: bool = True


class ZeroGraspInputResult(BaseModel):
    input_dir: str
    rgb_path: str
    depth_path: str
    mask_path: str
    camera_info_path: str
    filled_depth_holes: int = 0
    depth_fusion: dict[str, Any] = Field(default_factory=dict)


class GraspPredictRequest(BaseModel):
    input_dir: str | None = None
    img_path: str | None = None
    depth_path: str | None = None
    mask_path: str | None = None
    camera_info_path: str | None = None
    output_dir: str | None = None
    timeout_sec: int = 600
    grasp_criterion: Literal["score", "min_rpy"] = "score"


class GraspPrediction(BaseModel):
    grasp_path: str | None = None
    output_dir: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class GraspPlanRequest(BaseModel):
    arm_id: Literal["right", "left"] = "right"
    prediction_path: str | None = None
    prediction: dict[str, Any] | None = None
    current_end_pos: list[float] | None = None
    allow_debug_zero_pose: bool = False
    execute: bool = False


class GraspPlan(BaseModel):
    approach_pose: list[float] | None = None
    target_pose: list[float] | None = None
    target_gripper: float | None = None
    advance_depth_m: float = 0.0
    safety_passed: bool = False
    safety_failures: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunGraspRequest(BaseModel):
    arm_id: Literal["right", "left"] = "right"
    prompt: str = "red cube."
    points: list[PromptPoint] = Field(default_factory=list)
    execute: bool = False
    mask_method: Literal["langsam", "hsv"] = "langsam"
    mask_color: str = "red"
    mask_target: str | None = None
    run_tag: str = ""
    output_dir: str | None = None
    skip_projection: bool = False
    allow_debug_zero_pose: bool = False
    execute_duration_sec: float | None = 10.0
    timeout_sec: int = 600
    max_target_x_m: float | None = None
    reuse_camera_dir: str | None = None
    reuse_mask_path: str | None = None
    reuse_segmentation: dict[str, Any] | None = None


class RunGraspResult(BaseModel):
    ok: bool
    mode: str
    run_dir: str
    current_pose_source: str
    capture: CaptureResult | None = None
    segmentation: SegmentationResult | None = None
    zg_input: ZeroGraspInputResult | None = None
    prediction: GraspPrediction | None = None
    plan: GraspPlan | None = None
    intermediate_waypoint: dict[str, Any] = Field(default_factory=dict)
    intermediate_execution: ExecutionResult | None = None
    execution: ExecutionResult | None = None
    grasp_projection_path: str | None = None
    depth_fusion: dict[str, Any] = Field(default_factory=dict)
    report_path: str | None = None
    message: str = ""


class DualMapExploreRequest(BaseModel):
    """Structured request for closed-loop frontier exploration."""

    from_scratch: bool = False
    map_id: str | None = None
    execute: bool | None = None
    max_commands: int | None = None
    initial_scan_steps: int | None = None
    rerun_mode: Literal["save", "spawn", "serve", "connect", "none"] | None = None
    timeout_sec: float | None = None


class DualMapSemanticNavigationRequest(BaseModel):
    """Structured request for semantic navigation on a saved DualMap."""

    query: str
    map_id: str | None = None
    execute: bool | None = None
    max_attempts: int | None = None
    max_commands_per_path: int | None = None
    # Internal BrainAgent recovery state: candidates rejected by the main VLM
    # evaluator must not be selected again by a subsequent operator call.
    exclude_candidate_uids: list[str] = Field(default_factory=list)
    rerun_mode: Literal["save", "spawn", "serve", "connect", "none"] | None = None
    timeout_sec: float | None = None


class DualMapTaskStopResult(BaseModel):
    ok: bool
    status: Literal["IDLE", "STOPPING", "STOPPED", "FAILED"]
    active_task: dict[str, Any] | None = None
    chassis_stop: dict[str, Any] = Field(default_factory=dict)
    message: str = ""


class DualMapTaskResult(BaseModel):
    ok: bool
    task: Literal["explore", "semantic_navigation"]
    status: str
    map_id: str
    map_path: str
    execute: bool
    duration_sec: float = 0.0
    report_path: str | None = None
    runner_result_path: str | None = None
    artifacts: dict[str, Any] = Field(default_factory=dict)
    evaluation_image_path: str | None = None
    evaluation_capture: dict[str, Any] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)
    message: str = ""
