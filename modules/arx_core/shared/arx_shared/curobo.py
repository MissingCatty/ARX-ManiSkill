"""HTTP requests for the independent cuRobo planning service."""
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

JointVector = Annotated[list[FiniteFloat], Field(min_length=1)]


class CuroboFKRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    joint_positions: JointVector


class CuroboPlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start_joint_positions: JointVector
    max_attempts: int = Field(default=5, ge=1, le=20)


class CuroboJointPlanRequest(CuroboPlanRequest):
    goal_joint_positions: JointVector


class CuroboPosePlanRequest(CuroboPlanRequest):
    goal_position: tuple[FiniteFloat, FiniteFloat, FiniteFloat]
    # Quaternion convention: w, x, y, z; pose is relative to robot base.
    goal_quaternion: tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]

    @model_validator(mode="after")
    def validate_quaternion(self):
        norm2 = sum(v * v for v in self.goal_quaternion)
        if abs(norm2 - 1.0) > 1e-3:
            raise ValueError("goal_quaternion must be a unit quaternion in wxyz order")
        return self
