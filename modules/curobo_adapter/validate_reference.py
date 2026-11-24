"""Check independent joint witnesses with unchanged old/new collision models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
NAMES = [f"right_joint{i}" for i in range(21, 27)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["old", "new"], required=True)
    args = parser.parse_args()
    if args.backend == "old":
        sys.path.insert(0, str(HERE.parent / "curobo/source"))
    import curobo
    import torch
    from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
    root = HERE / "research/20261117_01"
    scope = json.loads((root / "scope.json").read_text())
    out = root / ("reference_validation_" + args.backend)
    out.mkdir(exist_ok=False)
    candidates = []
    for line in (root / "cpu_all/calls.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["pose_valid"]:
            candidates.append(r)
    for r in json.loads((root / "cpu_roots/results.json").read_text()):
        candidates += [{"case": r["case"], "index": r["index"], "q": q["q"], "world_goal": r["world_goal"]}
                       for q in r["roots"]]
    first = [json.loads(s) for s in (root / "cpu_reference/calls.jsonl").read_text().splitlines()]
    candidates += [{"case": r["case"], "index":r["index"], "q": r["best"]["q"], "world_goal":r["world_goal"]}
                   for r in first if r["best"]["pose_valid"]]
    records = []
    model_comparisons = []
    for case in scope["cases"]:
        selected = [r for r in candidates if r["case"] == case["id"]]
        if not selected:
            continue
        trial = Path(scope["source_campaign"]) / "trials" / case["id"]
        state = json.loads((trial / "state.json").read_text())
        scene = json.loads((trial / "plan/scene.json").read_text())
        checker = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
            robot_config=str(trial / "plan/arx.yml"), scene_model=scene,
            n_meshes=max(1, len(scene.get("mesh", {}))), collision_activation_distance=0.0))
        checker.self_collision_cost = None
        state_bounds = np.array([state["joint_limits"][state["joint_names"].index(n)] for n in NAMES])
        native_bounds = checker.kinematics.get_joint_limits().position.detach().cpu().numpy()
        assert np.max(np.abs(native_bounds.T-state_bounds)) < 1e-6
        queries = torch.tensor([r["q"] for r in selected], dtype=torch.float32, device="cuda")
        kin = checker.get_kinematics(queries[:, None])
        arrays = {"q": queries.detach().cpu().numpy(), "bounds":native_bounds,
                  "positions":kin.tool_poses.position.detach().cpu().numpy().copy(),
                  "quaternions":kin.tool_poses.quaternion.detach().cpu().numpy().copy(),
                  "spheres":kin.robot_spheres.detach().cpu().numpy().copy()}
        bound_cost = checker.get_bound(queries[:, None]).detach().cpu().numpy().reshape(len(selected), -1)
        world_cost = checker.get_collision_constraint(kin).detach().cpu().numpy().reshape(len(selected), -1)
        valid = checker.validate(queries[:, None]).detach().cpu().numpy().reshape(-1)
        arrays.update(valid=valid, bound_cost=bound_cost, world_cost=world_cost)
        np.savez(out / (case["id"] + ".npz"), **arrays)
        invbase = np.linalg.inv(np.asarray(state["links"]["body4"]))
        for i, row in enumerate(selected):
            target = invbase @ np.asarray(row["world_goal"])
            position = arrays["positions"].reshape(len(selected), -1, 3)[i, 0]
            wxyz = arrays["quaternions"].reshape(len(selected), -1, 4)[i, 0]
            pe = float(np.linalg.norm(position-target[:3, 3]))
            re = float(Rotation.from_matrix(Rotation.from_quat([*wxyz[1:], wxyz[0]]).as_matrix()
                                            @ target[:3, :3].T).magnitude())
            assert pe < .002+1e-6 and re < .03+1e-6
            record = {"case":case["id"],"index":row["index"],"q":row["q"],"world_goal":row["world_goal"],
                      "position_error_m":pe,"rotation_error_rad":re,"pose_valid":pe<.002 and re<.03,
                      "constraints_valid":bool(valid[i]),"bound_cost":float(bound_cost[i].sum()),
                      "scene_collision_spheres":np.flatnonzero(world_cost[i]>0).tolist(),
                      "scene_collision_cost":float(world_cost[i].sum())}
            records.append(record)
        model_comparisons.append({"id":case["id"],"queries":len(selected),"source":curobo.__file__})
        print("CHECK",args.backend,case["id"],"queries",len(selected),"valid",int(valid.sum()),flush=True)
        del checker
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    (out / "records.json").write_text(json.dumps(records,indent=2))
    (out / "models.json").write_text(json.dumps(model_comparisons,indent=2))
    controls={c["id"] for c in scope["cases"] if c["control"]}
    unique={(r["case"],r["index"]) for r in records if r["pose_valid"] and r["constraints_valid"]}
    result={"backend":args.backend,"queries":len(records),"valid_goal_indices":sorted(unique),
            "valid_failed_goal_indices":sorted(k for k in unique if k[0] not in controls),
            "native_ik_calls":0,"physical_trials":0}
    (out / "result.json").write_text(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
