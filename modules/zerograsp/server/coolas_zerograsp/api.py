from __future__ import annotations

try:
    from fastapi import FastAPI
except Exception as exc:  # pragma: no cover
    raise RuntimeError("fastapi is required to run zerograsp server") from exc

from arx_shared.schemas import (
    GraspPlanRequest,
    GraspPredictRequest,
    HealthResponse,
    RunGraspRequest,
)

from .grasp import run_predict_only
from .orchestrator import run_grasp_task
from .planning import plan_grasp


app = FastAPI(title="coolas zerograsp server", version="0.2.0-phase2")


@app.get("/health")
def health() -> HealthResponse:
    return HealthResponse(service="zerograsp")


@app.post("/grasp/predict")
def grasp_predict(request: GraspPredictRequest):
    return run_predict_only(request)


@app.post("/grasp/plan")
def grasp_plan(request: GraspPlanRequest):
    return plan_grasp(request)


@app.post("/task/run-grasp")
def task_run_grasp(request: RunGraspRequest):
    return run_grasp_task(request)

