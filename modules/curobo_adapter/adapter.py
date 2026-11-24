"""Isolated cuRobo 0.9 adapter for unchanged, archived ARX planning models."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
from curobo.types import GoalToolPose, JointState
from curobo._src.trajectory.trajectory_interpolation import TrajInterpolationType

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parent
ARX = PROJECTS / "arx_core"
NAMES = [f"right_joint{i}" for i in range(21, 27)]
sys.path.insert(0, str(ARX / "scripts/curobo_arx"))
from optimize_path import optimize_path, flush_planning_artifacts
from collision_profile import require_current_profile


VARIANTS: dict[str, dict[str, Any]] = {
    "official32": {},
    "legacy_lbfgs": {"legacy_lbfgs": True},
    "seeds64": {"num_ik_seeds": 64},
    "particle32": {"particle": True},
    "seed_rotation01": {"seed_orientation_weight": 0.1},
    "without_lm": {"use_lm_seed": False},
}


def save(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ARXPlanner:
    """Reuse frozen models and gates; change only the declared solver variant."""

    def __init__(self, trial: Path, out: Path, variant: str = "official32") -> None:
        out = out.resolve()
        self.out = out
        out.mkdir(parents=True, exist_ok=False)
        self.state = json.loads((trial / "state.json").read_text())
        self.base = np.asarray(self.state["links"]["body4"])
        self.invbase = np.linalg.inv(self.base)
        settings = VARIANTS[variant]
        for name in ("arx.yml", "nominal_closed.yml", "scene.json"):
            shutil.copyfile(trial / "plan" / name, out / name)
            assert digest(out / name) == digest(trial / "plan" / name)
        self.model = out / "arx.yml"
        save(out / 'collision_profile.json', require_current_profile(yaml.safe_load(self.model.read_text())))
        require_current_profile(yaml.safe_load((out / 'nominal_closed.yml').read_text()))
        self.scene = json.loads((out / "scene.json").read_text())
        for mesh in self.state.get("scene_meshes", {}).values():
            assert digest(Path(mesh["file_path"])) == mesh["file_sha256"]
        n_meshes = max(1, len(self.scene.get("mesh", {})))

        task_root = HERE / "source/curobo/content/configs/task"
        trajectory_task = yaml.safe_load((task_root / "trajopt/lbfgs_bspline_trajopt.yml").read_text())
        trajectory_task["rollout"]["cost_cfg"]["cspace_cfg"]["weight"][0] *= 100
        ik_task = yaml.safe_load((task_root / "ik/lbfgs_ik.yml").read_text())
        if settings.get("legacy_lbfgs"):
            old = yaml.safe_load((PROJECTS / "curobo/source/curobo/content/configs/task/ik/lbfgs_ik.yml").read_text())
            # Keep the new pose schema, and match old numerical optimizer settings.
            allowed = set(ik_task["optimizer"])
            ik_task["optimizer"].update({k: v for k, v in old["optimizer"].items() if k in allowed})
            # The 0.9 API rejects the old zero-length line-search candidate.
            ik_task["optimizer"]["line_search_scale"] = [
                v for v in ik_task["optimizer"]["line_search_scale"] if v > 0
            ]
        optimizers = [ik_task]
        if settings.get("particle"):
            optimizers.insert(0, "ik/particle_ik.yml")
        optimizer_receipt = copy.deepcopy(optimizers)
        config = MotionPlannerCfg.create(
            robot=str(self.model), scene_model=self.scene,
            self_collision_check=False,
            collision_cache={"mesh": n_meshes, "primitive": 8},
            interpolation_dt=0.025, optimizer_collision_activation_distance=0.003,
            position_tolerance=0.002, orientation_tolerance=0.03,
            trajopt_optimizer_configs=[trajectory_task], ik_optimizer_configs=optimizers,
            num_ik_seeds=settings.get("num_ik_seeds", 32),
            random_seed=123, use_cuda_graph=True,
            # Preserve the previous explicit solve/validate/time-stretch path.
            enable_timeopt=False,
        )
        config.ik_solver_config.seed_orientation_weight = settings.get("seed_orientation_weight", 1.0)
        config.ik_solver_config.use_lm_seed = settings.get("use_lm_seed", True)
        for rollout in (
            config.prm_graph_planner_config.rollout_config,
            config.ik_solver_config.core_cfg.metrics_rollout_config,
            config.trajopt_solver_config.core_cfg.metrics_rollout_config,
        ):
            for cost in rollout.get_cost_manager_configs():
                cost.disable_self_collision()
        self.planner = MotionPlanner(config)
        self.checker = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
            robot_config=str(self.model), scene_model=self.scene,
            n_meshes=n_meshes, collision_activation_distance=0.0))
        self.checker.self_collision_cost = None
        closed_scene = copy.deepcopy(self.scene)
        closed_scene["cuboid"]["table"]["pose"][:3] = (
            np.asarray(closed_scene["cuboid"]["table"]["pose"][:3])
            + self.invbase[:3, 2] * 0.008).tolist()
        self.closed_checker = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
            robot_config=str(out / "nominal_closed.yml"), scene_model=closed_scene,
            n_meshes=n_meshes, collision_activation_distance=0.0))
        self.closed_checker.self_collision_cost = None
        self.limits = torch.tensor([
            self.state["joint_limits"][self.state["joint_names"].index(n)] for n in NAMES
        ], dtype=torch.float32, device="cuda")
        start = self.measured(self.state)
        actual = torch.tensor([self.state["qpos"][self.state["joint_names"].index(n)] for n in NAMES],
                              dtype=torch.float32, device="cuda")
        kin = self.planner.compute_kinematics(self.js(actual)).tool_poses
        xyz = kin.position.detach().cpu().numpy().reshape(-1, 3)[0]
        q = kin.quaternion.detach().cpu().numpy().reshape(-1, 4)[0]
        expected = self.invbase @ np.asarray(self.state["links"]["right_eef_link"])
        error = {
            "position_error_m": float(np.linalg.norm(xyz - expected[:3, 3])),
            "rotation_error_rad": float(Rotation.from_matrix(
                Rotation.from_quat([*q[1:], q[0]]).as_matrix() @ expected[:3, :3].T).magnitude()),
        }
        save(out / "fk_validation.json", error)
        assert error["position_error_m"] < 2e-5 and error["rotation_error_rad"] < 2e-4, error
        assert bool(self.checker.validate(start[None, None]).all()), "Frozen start is invalid"
        self.config_receipt = {
            "variant": variant, "variant_settings": settings,
            "num_seeds": config.ik_solver_config.num_seeds,
            "use_lm_seed": config.ik_solver_config.use_lm_seed,
            "seed_orientation_weight": config.ik_solver_config.seed_orientation_weight,
            "position_tolerance_m": config.ik_solver_config.position_tolerance,
            "orientation_tolerance_rad": config.ik_solver_config.orientation_tolerance,
            "optimizer_configs": optimizer_receipt, "self_collision_check": False,
            "optimization_dt": config.ik_solver_config.optimization_dt,
            "solver_seed": 123, "use_cuda_graph": True,
            "new_timeopt_enabled": False,
            "model_sha256": digest(self.model), "scene_sha256": digest(out / "scene.json"),
        }
        save(out / "configuration.json", self.config_receipt)

    def measured(self, state: dict) -> torch.Tensor:
        q = torch.tensor([state["qpos"][state["joint_names"].index(n)] for n in NAMES],
                         dtype=torch.float32, device="cuda")
        return torch.maximum(torch.minimum(q, self.limits[:, 1] - 1e-4), self.limits[:, 0] + 1e-4)

    def js(self, q: torch.Tensor) -> JointState:
        return JointState.from_position(q[None].clone(), joint_names=NAMES)

    def goal(self, world: np.ndarray) -> GoalToolPose:
        t = self.invbase @ world
        xyzw = Rotation.from_matrix(t[:3, :3]).as_quat()
        return GoalToolPose(
            tool_frames=["right_eef_link"],
            position=torch.tensor(t[:3, 3], dtype=torch.float32, device="cuda").reshape(1, 1, 1, 1, 3),
            quaternion=torch.tensor([xyzw[3], *xyzw[:3]], dtype=torch.float32,
                                    device="cuda").reshape(1, 1, 1, 1, 4))

    def ik(self, world: np.ndarray, start: torch.Tensor) -> tuple[torch.Tensor | None, Any]:
        result = self.planner.ik_solver.solve_pose(self.goal(world), current_state=self.js(start), return_seeds=4)
        if not result.success.any():
            return None, result
        qs = result.solution[result.success].reshape(-1, 6)
        qs = qs[self.checker.validate(qs[:, None]).flatten()]
        if not len(qs):
            return None, result
        return qs[(qs - start).square().sum(-1).argmin()].clone(), result

    def path(self, start: torch.Tensor, goal: torch.Tensor) -> torch.Tensor | None:
        result = self.planner.graph_planner.find_path(
            start[None].clone(), goal[None].clone(), interpolate_waypoints=True,
            interpolation_steps=201, interpolation_type=TrajInterpolationType.LINEAR,
            validate_interpolated_trajectory=True)
        if not result.success.all():
            return None
        q = result.interpolated_waypoints.reshape(-1, 6).clone()
        assert (q[0] - start).abs().max() < 1e-4
        assert (q[-1] - goal).abs().max() < 1e-4
        a = torch.arange(5, device="cuda") / 5
        dense = torch.cat([(q[:-1, None] * (1 - a[None, :, None])
                           + q[1:, None] * a[None, :, None]).reshape(-1, 6), q[-1:]])
        if not all(self.checker.validate(chunk[:, None]).all() for chunk in dense.split(64)):
            return None
        return q

    def optimize(self, label: str, path: torch.Tensor, seconds: float) -> dict:
        q, times = optimize_path(self.planner, self.checker, path.clone(), NAMES, self.limits,
                                 seconds, self.out, label)
        flush_planning_artifacts()
        return {"q": q.detach().cpu().numpy().tolist(), "times": np.asarray(times).tolist()}
