"""Observe native IK on registered ARX inputs; never connect to a simulator."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

from adapter import ARXPlanner, HERE, digest, save

sys.path.insert(0, str(HERE.parent / "maniskill/planning_common"))
from grasp_selector import decode_candidate


def array(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().copy()


def capture(opt: torch.Tensor, metrics: Any, result: Any, config: Any) -> dict:
    """Reconstruct and check the native success gate without altering its result."""
    n = opt.numel() // 6
    convergence = {name: array(value).reshape(n, -1)
                   for name, value in zip(metrics.convergence.names, metrics.convergence.values)}
    position = np.concatenate([v for k, v in convergence.items() if "position_tolerance" in k], axis=1).max(axis=1)
    rotation = np.concatenate([v for k, v in convergence.items() if "orientation_tolerance" in k], axis=1).max(axis=1)
    pose = (position < config.position_tolerance) & (rotation < config.orientation_tolerance)
    costs = metrics.costs_and_constraints
    constraints = {name: array(value).reshape(n, -1)
                   for name, value in zip(costs.constraints.names, costs.constraints.values)}
    raw = costs.get_feasible(include_all_hybrid=False, sum_horizon=True)
    feasible = np.full(n, raw, dtype=bool) if isinstance(raw, bool) else array(raw).reshape(n, -1).all(axis=1)
    success = pose & feasible
    assert config.success_requires_convergence
    selected = array(result.debug_info["seed_idx"]).reshape(-1).astype(int)
    assert np.array_equal(success[selected], array(result.success).reshape(-1)), "Native gate mismatch"
    best = int(np.argmin(np.maximum(position / config.position_tolerance,
                                    rotation / config.orientation_tolerance)))
    return {
        "position_error_m": position.tolist(), "orientation_error_rad": rotation.tolist(),
        "pose_converged": pose.tolist(), "feasible": feasible.tolist(),
        "native_full_success": success.tolist(), "selected_seed_indices": selected.tolist(),
        "returned_success": array(result.success).reshape(-1).tolist(),
        "constraint_sum": {k: v.sum(axis=1).tolist() for k, v in constraints.items()},
        "constraint_violating_seeds": {k: int((v.sum(axis=1) > 0).sum()) for k, v in constraints.items()},
        "pose_converged_constraint_violations": {
            k: int((pose & (v.sum(axis=1) > 0)).sum()) for k, v in constraints.items()},
        "best_seed": best, "best_pose_position_m": float(position[best]),
        "best_pose_rotation_rad": float(rotation[best]),
        "optimized_joint_positions": array(opt).reshape(n, 6).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--variant", default="official32")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--indices", default="all")
    parser.add_argument("--known-goals", type=Path)
    parser.add_argument("--paths", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    trial = HERE.parent / "maniskill/outputs/arx_native_postprocessed_retry11_20261117_01/trials" / args.case
    state = json.loads((trial / "state.json").read_text())
    rows = np.load(trial / "postprocessed_grasps.npy", allow_pickle=False)
    rows = rows[np.argsort(-rows[:, 0], kind="stable")]
    if args.known_goals:
        goals = json.loads(args.known_goals.read_text())[args.case]
        queries = [(i, np.asarray(g["world_goal"]), "certified_fk") for i, g in enumerate(goals)]
    else:
        indices = list(range(len(rows))) if args.indices == "all" else [int(x) for x in args.indices.split(",")]
        queries = [(i, decode_candidate(rows[i], state["camera_world_optical"])[0], "postprocessed") for i in indices]
    save(args.out / "inputs.json", {
        "trial": str(trial), "variant": args.variant,
        "state_sha256": digest(trial / "state.json"),
        "postprocessed_sha256": digest(trial / "postprocessed_grasps.npy"),
        "queries": [{"index": i, "world_goal": world.tolist(), "kind": kind} for i, world, kind in queries],
        "physical_trials": 0,
    })
    started = time.monotonic()
    planner = ARXPlanner(trial, args.out / "model", args.variant)
    initialized = time.monotonic() - started
    solver = planner.planner.ik_solver
    native = solver._get_result
    phases: list[dict] = []

    def observe(opt: torch.Tensor, metrics: Any, n_seeds: int, n_return: int, *a: Any, **kw: Any) -> Any:
        result = native(opt, metrics, n_seeds, n_return, *a, **kw)
        phases.append(capture(opt, metrics, result, solver.config))
        return result

    solver._get_result = observe
    start = planner.measured(state)
    records = []
    with (args.out / "requests.jsonl").open("x") as requests, (args.out / "calls.jsonl").open("x") as stream:
        for index, world, kind in queries:
            phases.clear()
            requests.write(json.dumps({"index": index, "kind": kind, "requested_at_monotonic": time.monotonic()}) + "\n")
            requests.flush()
            torch.cuda.synchronize()
            begin = time.monotonic()
            q, result = planner.ik(world, start)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - begin
            assert phases, "Native observation hook not called"
            final = phases[-1]
            returned = bool(result.success.any())
            record = {
                "index": index, "kind": kind,
                "gate": "accepted" if q is not None else "outer_collision" if returned else "native_ik_gate",
                "q": None if q is None else array(q).tolist(),
                "converged_seeds": int(sum(final["pose_converged"])),
                "feasible_seeds": int(sum(final["feasible"])),
                "native_phases": phases.copy(), "wall_seconds_with_diagnostics": elapsed,
                "world_goal": world.tolist(),
            }
            if q is not None:
                # Independent FK and frozen collision checks on returned joints.
                kin = planner.planner.compute_kinematics(planner.js(q)).tool_poses
                goal = planner.goal(world)
                p_error = float(torch.linalg.vector_norm(kin.position.reshape(-1, 3)[0]
                                                        - goal.position.reshape(-1, 3)[0]))
                from scipy.spatial.transform import Rotation
                q_fk = array(kin.quaternion).reshape(-1, 4)[0]
                target = planner.invbase @ world
                r_error = float(Rotation.from_matrix(Rotation.from_quat(
                    [*q_fk[1:], q_fk[0]]).as_matrix() @ target[:3, :3].T).magnitude())
                valid = bool(planner.checker.validate(q[None, None]).all())
                within_limits = bool(((q >= planner.limits[:, 0]) & (q <= planner.limits[:, 1])).all())
                assert p_error < 0.002 and r_error < 0.03 and valid and within_limits
                record["independent"] = {"position_error_m": p_error, "rotation_error_rad": r_error,
                                          "collision_valid": valid, "joint_limits_valid": within_limits}
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            records.append(record)
            print(args.variant, args.case, index, record["gate"], "pose", record["converged_seeds"],
                  "feasible", record["feasible_seeds"], flush=True)
    if args.paths:
        # Integration validation only: archived pregrasp path, no execution.
        original = json.loads((trial / "plan_result.json").read_text())
        target = torch.tensor(original["selected"]["q_pre"], dtype=torch.float32, device="cuda")
        path = planner.path(start, target)
        assert path is not None, "Archived control pregrasp path was not found"
        optimized = planner.optimize("control_pregrasp", path, 6.0)
        save(args.out / "path_integration.json", {"success": True, "physical_execution": False,
             "waypoints": len(path), "optimized_samples": len(optimized["q"]),
             "duration_s": optimized["times"][-1]})
    summary = {
        "case": args.case, "variant": args.variant, "candidate_count": len(records),
        "gates": dict(Counter(r["gate"] for r in records)),
        "pose_converged_candidates": sum(r["converged_seeds"] > 0 for r in records),
        "initialization_seconds": initialized, "total_seconds": time.monotonic() - started,
        "native_gate_reconstructed_exactly": True, "frozen_model_scene_byte_identical": True,
        "physical_trials": 0, "cloud_requests": 0,
    }
    save(args.out / "result.json", summary)
    print("COMPLETE", json.dumps(summary), flush=True)
    planner.planner.destroy()


if __name__ == "__main__":
    main()
