"""cuRobo adapter using COOLAS-ARX's frozen model and original optimizer gates."""
import argparse
import hashlib
import copy
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
ARX = ROOT / "arx_core"
sys.path.insert(0, str(ARX / "scripts/curobo_arx"))
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import JointState, GoalToolPose
from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
from curobo._src.graph_planner.graph_planner_prm import TrajInterpolationType
from optimize_path import optimize_path, flush_planning_artifacts
from collision_profile import require_current_profile

NAMES = [f"right_joint{i}" for i in range(21, 27)]
FINGERS = ["right_joint27", "right_joint28"]


def pose7(t):
    q = Rotation.from_matrix(t[:3, :3]).as_quat()
    return [*t[:3, 3].tolist(), float(q[3]), *q[:3].tolist()]


class Planner:
    def __init__(self, directory, state, closed=False, include_target=False):
        self.out = Path(directory).resolve()
        self.out.mkdir(exist_ok=True)
        self.base = np.asarray(state["links"]["body4"])
        self.invbase = np.linalg.inv(self.base)
        reference = ARX / "scripts/curobo_arx/static_pick/reference" / ("model_closed" if closed else "model")
        cfg = yaml.safe_load((reference / "arx.yml").read_text())
        profile = require_current_profile(cfg)
        (self.out / 'collision_profile.json').write_text(json.dumps(profile, indent=2))
        self.model = self.out / "arx.yml"
        cfg["robot_cfg"]["kinematics"]["urdf_path"] = str(reference / "arx.urdf")
        measured = dict(zip(state["joint_names"], state["qpos"]))
        for n in FINGERS:
            cfg["robot_cfg"]["kinematics"]["lock_joints"][n] = measured[n]
        self.model.write_text(yaml.safe_dump(cfg, sort_keys=False))
        scene = {"cuboid": {}}
        if "scene_cuboids" not in state:
            raise ValueError("Measured scene geometry missing; refusing a guessed table height")
        for key, geometry in state["scene_cuboids"].items():
            t = np.eye(4); t[:3, 3] = geometry["center"]
            if closed and key == "table":
                t[2, 3] += .008  # Original closed-gripper planning clearance.
            scene["cuboid"][key] = {"pose": pose7(self.invbase @ t), "dims": geometry["dims"]}
        meshes = {n:g for n,g in state.get('scene_meshes',{}).items() if include_target or not g['target']}
        for geometry in meshes.values():
            if 'file_sha256' in geometry:
                assert hashlib.sha256(Path(geometry['file_path']).read_bytes()).hexdigest()==geometry['file_sha256'], 'Scene collision mesh changed'
        if meshes:
            scene['mesh']={n:{'pose':pose7(self.invbase@np.asarray(g['world'])), 'file_path':g['file_path']}
                           for n,g in meshes.items()}
        n_meshes=max(1,len(meshes))
        (self.out / "scene.json").write_text(json.dumps(scene, indent=2))
        task = yaml.safe_load((ROOT / "curobo/source/curobo/content/configs/task/trajopt/lbfgs_bspline_trajopt.yml").read_text())
        task["rollout"]["cost_cfg"]["cspace_cfg"]["weight"][0] *= 100
        config = MotionPlannerCfg.create(robot=str(self.model), scene_model=scene,
            self_collision_check=False, collision_cache={"mesh": n_meshes, "primitive": 8}, interpolation_dt=.025,
            optimizer_collision_activation_distance=.003, position_tolerance=.002, orientation_tolerance=.03,
            trajopt_optimizer_configs=[task])
        for rollout in [config.graph_planner_config.rollout_config,
                        config.ik_solver_config.core_cfg.metrics_rollout_config,
                        config.trajopt_solver_config.core_cfg.metrics_rollout_config]:
            for cost in rollout.get_cost_manager_configs():
                cost.disable_self_collision()
        self.planner = MotionPlanner(config)
        self.checker = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
            robot_config=str(self.model), scene_model=scene, n_meshes=n_meshes, collision_activation_distance=0.0))
        self.checker.self_collision_cost = None
        self.closed_checker = None
        if not closed:
            closed_reference = ARX / "scripts/curobo_arx/static_pick/reference/model_closed"
            nominal = yaml.safe_load((closed_reference / "arx.yml").read_text())
            require_current_profile(nominal)
            nominal["robot_cfg"]["kinematics"]["urdf_path"] = str(closed_reference / "arx.urdf")
            for n in FINGERS:
                nominal["robot_cfg"]["kinematics"]["lock_joints"][n] = .022
            nominal_path = self.out / "nominal_closed.yml"
            nominal_path.write_text(yaml.safe_dump(nominal, sort_keys=False))
            closed_scene = copy.deepcopy(scene)
            closed_scene["cuboid"]["table"]["pose"][:3] = (
                np.array(closed_scene["cuboid"]["table"]["pose"][:3]) + self.invbase[:3, 2]*.008).tolist()
            self.closed_checker = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
                robot_config=str(nominal_path), scene_model=closed_scene, n_meshes=n_meshes, collision_activation_distance=0.0))
            self.closed_checker.self_collision_cost = None
        self.limits = torch.tensor([state["joint_limits"][state["joint_names"].index(n)] for n in NAMES], device="cuda")
        q = torch.tensor([state["qpos"][state["joint_names"].index(n)] for n in NAMES], dtype=torch.float32, device="cuda")
        # Compare actual simulator FK with cuRobo before allowing any motion.
        kin = self.planner.compute_kinematics(self.js(q)).tool_poses
        xyz = kin.position.detach().cpu().numpy().reshape(-1, 3)[0]
        quat = kin.quaternion.detach().cpu().numpy().reshape(-1, 4)[0]
        expected = self.invbase @ np.asarray(state["links"]["right_eef_link"])
        rotation = Rotation.from_quat([*quat[1:], quat[0]]).as_matrix()
        error = {"position_error_m": float(np.linalg.norm(xyz - expected[:3, 3])),
                 "rotation_error_rad": float(Rotation.from_matrix(rotation @ expected[:3, :3].T).magnitude())}
        (self.out / "fk_validation.json").write_text(json.dumps(error, indent=2))
        assert error["position_error_m"] < 2e-5 and error["rotation_error_rad"] < 2e-4, error
        start = self.measured(state)
        (self.out / "start_state.json").write_text(json.dumps({"actual": q.tolist(), "planning_start": start.tolist(),
            "max_clamp_rad": float((start-q).abs().max()), "policy": "Original COOLAS-ARX 1e-4 rad inset from joint limits"}, indent=2))
        assert bool(self.checker.validate(start[None, None]).all()), "Initial ARX state collides in cuRobo model"

    def measured(self, state):
        q = torch.tensor([state["qpos"][state["joint_names"].index(n)] for n in NAMES], dtype=torch.float32, device="cuda")
        return torch.maximum(torch.minimum(q, self.limits[:, 1]-1e-4), self.limits[:, 0]+1e-4)

    def js(self, q):
        # cuRobo retains and updates solve-state buffers across repeated calls.
        # Never expose caller-owned measurements or IK goals to those writes.
        return JointState.from_position(q[None].clone(), joint_names=NAMES)

    def ik(self, world, start):
        t = self.invbase @ world
        q = pose7(t)[3:]
        goal = GoalToolPose(tool_frames=["right_eef_link"],
            position=torch.tensor(t[:3, 3], dtype=torch.float32, device="cuda").reshape(1, 1, 1, 1, 3),
            quaternion=torch.tensor(q, dtype=torch.float32, device="cuda").reshape(1, 1, 1, 1, 4))
        result = self.planner.ik_solver.solve_pose(goal, current_state=self.js(start), return_seeds=4)
        if not result.success.any():
            return None
        qs = result.solution[result.success].reshape(-1, 6)
        qs = qs[self.checker.validate(qs[:, None]).flatten()]
        if not len(qs):
            return None
        return qs[(qs-start).square().sum(-1).argmin()].clone()

    def path(self, start, goal):
        r = self.planner.graph_planner.find_path(start[None].clone(), goal[None].clone(),
            interpolate_waypoints=True, interpolation_steps=201,
            interpolation_type=TrajInterpolationType.LINEAR, validate_interpolated_trajectory=True)
        if not r.success.all():
            return None
        q = r.interpolated_waypoints.reshape(-1, 6).clone()
        assert (q[0]-start).abs().max() < 1e-4, "PRM start differs from requested state"
        assert (q[-1]-goal).abs().max() < 1e-4, "PRM goal differs from requested state"
        a = torch.arange(5, device="cuda") / 5
        dense = torch.cat([(q[:-1, None]*(1-a[None, :, None])+q[1:, None]*a[None, :, None]).reshape(-1, 6), q[-1:]])
        if not all(self.checker.validate(c[:, None]).all() for c in dense.split(64)):
            return None
        return q

    def optimize(self, label, path, seconds):
        expected_start, expected_goal = path[0].clone(), path[-1].clone()
        q, times = optimize_path(self.planner, self.checker, path.clone(), NAMES, self.limits, seconds, self.out, label)
        assert (q[0]-expected_start).abs().max() < .005, "Optimized start differs from requested state"
        assert (q[-1]-expected_goal).abs().max() < .005, "Optimized goal differs from requested state"
        flush_planning_artifacts()
        return {"q": q.detach().cpu().numpy().tolist(), "times": np.asarray(times).tolist()}

    def select(self, state, rows, camera):
        start = self.measured(state)
        obj = np.asarray(state["object_center"])
        axes = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        rejected = {k: 0 for k in ["geometry", "height", "grasp_ik", "closed_collision", "pre_or_lift_ik", "path", "optimizer"]}
        for index, row in enumerate(rows):
            r = camera[:3, :3] @ row[4:13].reshape(3, 3)
            surface = camera[:3, :3] @ row[13:16] + camera[:3, 3]
            contact = surface + row[3] * r[:, 0]
            if r[2, 0] > -.35 or np.linalg.norm(contact-obj) > .065:
                rejected["geometry"] += 1; continue
            raise_z = max(0., float(obj[2]+.015-contact[2]))
            if raise_z > .030:
                rejected["height"] += 1; continue
            contact[2] += raise_z
            for flip in [np.eye(3), np.diag([1, -1, -1])]:
                g = np.eye(4); g[:3, :3] = r @ flip @ axes
                g[:3, 3] = contact - .040 * r[:, 0]
                pre = g.copy(); pre[:3, 3] -= .08 * r[:, 0]
                lift = g.copy(); lift[2, 3] += .12
                qg = self.ik(g, start)
                if qg is None:
                    rejected["grasp_ik"] += 1; continue
                if not self.closed_checker.validate(qg[None, None]).all():
                    rejected["closed_collision"] += 1; continue
                qp, ql = self.ik(pre, qg), self.ik(lift, qg)
                if qp is None or ql is None:
                    rejected["pre_or_lift_ik"] += 1; continue
                reach, approach = self.path(start, qp), self.path(qp, qg)
                if reach is None or approach is None:
                    rejected["path"] += 1; continue
                selected = {"index": index, "score": float(row[0]), "network_row": row.tolist(),
                    "grasp_world": g.tolist(), "pregrasp_world": pre.tolist(), "lift_world": lift.tolist(),
                    "target_contact_world": contact.tolist(), "table_clearance_raise_m": raise_z,
                    "q_grasp": qg.tolist(), "q_pre": qp.tolist(), "q_lift": ql.tolist()}
                try:
                    paths = {"pregrasp": self.optimize("pregrasp", reach, 6), "approach": self.optimize("approach", approach, 5)}
                except RuntimeError:
                    rejected["optimizer"] += 1
                    raise  # retain failed optimizer evidence; do not replace it with another candidate
                return {"success": True, "selected": selected, "paths": paths, "rejected": rejected}
            if index % 100 == 0:
                print("candidates", index, rejected, flush=True)
        return {"success": False, "reason": "no_feasible_network_candidate", "rejected": rejected, "count": len(rows)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    p.add_argument("--lift", action="store_true")
    p.add_argument("--approach", action="store_true")
    p.add_argument("--observe", action="store_true")
    args = p.parse_args()
    directory = args.directory.resolve()
    name = "observation_plan" if args.observe else "lift_plan" if args.lift else "approach_plan" if args.approach else "plan"
    started = time.monotonic()
    result = {"success": False}
    try:
        state = json.loads((directory / ("closed_state.json" if args.lift else "pregrasp_state.json" if args.approach else "state.json")).read_text())
        planner = Planner(directory / name, state, args.lift, include_target=args.observe)
        if args.observe:
            request=json.loads((directory/'observation_request.json').read_text())
            start=planner.measured(state)
            goal=torch.tensor(request['q'],device='cuda',dtype=torch.float32)
            path=planner.path(start,goal)
            result=({'success':True,'paths':{'observe':planner.optimize('observe',path,6.)}}
                    if path is not None else {'success':False,'reason':'observation_path_failed'})
        elif args.lift or args.approach:
            selected = json.loads((directory / "plan_result.json").read_text())["selected"]
            current = planner.measured(state)
            goal = (planner.ik(np.asarray(selected["lift_world"]), current) if args.lift
                    else torch.tensor(selected["q_grasp"], dtype=torch.float32, device="cuda"))
            path = planner.path(current, goal) if goal is not None else None
            if path is None:
                result = {"success": False, "reason": "measured_closure_lift_unreachable" if args.lift else "measured_pregrasp_approach_unreachable"}
            else:
                label = "lift" if args.lift else "approach"
                result = {"success": True, "paths": {label: planner.optimize(label, path, 6 if args.lift else 5)}}
        else:
            rows = np.load(directory / "raw_grasps.npy")
            rows = rows[np.argsort(-rows[:, 0], kind="stable")]
            result = planner.select(state, rows, np.asarray(state["camera_world_optical"]))
    except Exception:
        result = {"success": False, "reason": "planner_error", "error": traceback.format_exc()}
        print(result["error"], flush=True)
    finally:
        flush_planning_artifacts()
        result["wall_seconds"] = time.monotonic()-started
        (directory / (name + "_result.json")).write_text(json.dumps(result, indent=2))
    return 0 if result["success"] else 2


if __name__ == "__main__":
    sys.exit(main())
