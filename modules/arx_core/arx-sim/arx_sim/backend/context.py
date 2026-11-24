from __future__ import annotations

import threading
from dataclasses import dataclass, field, replace
from typing import Any

from .command_queue import CommandQueue, QueueState


@dataclass(frozen=True)
class SimulationState:
    lifecycle: str = QueueState.STARTING.value
    generation: int = 0
    sim_time: float = 0.0
    arm_joint_pos: tuple[float, ...] = ()
    end_pos: tuple[float, ...] = ()
    gripper: float | None = None
    left_arm_joint_pos: tuple[float, ...] = ()
    left_end_pos: tuple[float, ...] = ()
    left_gripper: float | None = None
    chassis: tuple[float, float, float] = (0.0, 0.0, 0.0)
    chassis_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lift_m: float | None = None
    lift_velocity_mps: float = 0.0
    cameras_ready: tuple[str, ...] = ()
    error: str = ""


@dataclass
class IsaacContext:
    simulation_app: Any
    world: Any
    stage: Any
    robot: Any
    base_prim: Any
    lift_prim: Any
    arm_base_prim: Any
    ee_prim: Any
    left_ee_prim: Any
    tcp_prim: Any
    camera_prims: dict[str, Any]
    cameras: dict[str, Any]
    config: dict[str, Any]
    command_queue: CommandQueue
    camera_factory: Any = None
    contact_materials_applier: Any = None
    controller: Any = None
    arm_controller: Any = None
    gripper_controller: Any = None
    chassis_controller: Any = None
    lift_controller: Any = None
    ros2_publishers: Any = None
    _state: SimulationState = field(default_factory=SimulationState, repr=False)
    _state_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> SimulationState:
        with self._state_lock:
            return self._state

    def publish(self, **changes: Any) -> SimulationState:
        with self._state_lock:
            self._state = replace(self._state, **changes)
            return self._state
