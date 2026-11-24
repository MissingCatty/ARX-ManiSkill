"""Validate independent IK witnesses and save collision-certified ARX FK targets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from adapter import ARXPlanner, HERE, NAMES, save
from urdf_reference import URDFChain


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / "certification"
    out.mkdir(exist_ok=False)
    scope = json.loads((root / "scope.json").read_text())
    reference = [json.loads(s) for s in (root / "cpu_reference/calls.jsonl").read_text().splitlines()]
    known_cases = {"B0001_S1_open_003_cracker_box", "B0008_S2_edge_004_sugar_box",
                   "B0142_S4_clutter_022_windex_bottle"}
    targets = {}
    witnesses = []
    rng = np.random.default_rng(7109)
    model = URDFChain(HERE.parent / "arx_core/scripts/curobo_arx/static_pick/reference/model/arx.urdf")
    for case in scope["cases"]:
        identifier = case["id"]
        candidates = [r for r in reference if r["case"] == identifier and r["best"]["pose_valid"]]
        if not candidates and identifier not in known_cases:
            continue
        trial = Path(scope["source_campaign"]) / "trials" / identifier
        planner = ARXPlanner(trial, out / identifier)
        for record in candidates:
            q = torch.tensor(record["best"]["q"], dtype=torch.float32, device="cuda")
            kin = planner.planner.compute_kinematics(planner.js(q)).tool_poses
            expected = planner.invbase @ np.asarray(record["world_goal"])
            position = kin.position.detach().cpu().numpy().reshape(-1, 3)[0]
            wxyz = kin.quaternion.detach().cpu().numpy().reshape(-1, 4)[0]
            pe = float(np.linalg.norm(position - expected[:3, 3]))
            re = float(Rotation.from_matrix(Rotation.from_quat([*wxyz[1:], wxyz[0]]).as_matrix()
                                            @ expected[:3, :3].T).magnitude())
            valid = bool(planner.checker.validate(q[None, None]).all())
            bounded = bool(((q >= planner.limits[:, 0]) & (q <= planner.limits[:, 1])).all())
            assert pe < 0.002 and re < 0.03 and bounded
            witnesses.append({"case": identifier, "index": record["index"], "q": record["best"]["q"],
                              "position_error_m": pe, "rotation_error_rad": re,
                              "collision_valid": valid, "joint_limits_valid": bounded})
        if identifier in known_cases:
            lower, upper = planner.limits.detach().cpu().numpy().T
            center = planner.measured(planner.state).detach().cpu().numpy()
            possible = [center]
            possible += list(np.clip(center + rng.uniform(-0.45, 0.45, (256, 6)), lower+1e-4, upper-1e-4))
            targets[identifier] = []
            for q_array in possible:
                q = torch.tensor(q_array, dtype=torch.float32, device="cuda")
                if not planner.checker.validate(q[None, None]).all():
                    continue
                own, _ = model.fk(q_array)
                kin = planner.planner.compute_kinematics(planner.js(q)).tool_poses
                position = kin.position.detach().cpu().numpy().reshape(-1, 3)[0]
                wxyz = kin.quaternion.detach().cpu().numpy().reshape(-1, 4)[0]
                rotation = Rotation.from_quat([*wxyz[1:], wxyz[0]]).as_matrix()
                p_error = float(np.linalg.norm(position - own[:3, 3]))
                r_error = float(Rotation.from_matrix(rotation @ own[:3, :3].T).magnitude())
                assert p_error < 2e-5 and r_error < 2e-4
                targets[identifier].append({"q_witness": q_array.tolist(),
                    "world_goal": (planner.base @ own).tolist(), "collision_valid": True,
                    "cross_engine_fk_position_m": p_error, "cross_engine_fk_rotation_rad": r_error})
                if len(targets[identifier]) == 8:
                    break
            assert len(targets[identifier]) == 8
        planner.planner.destroy()
        del planner
        torch.cuda.empty_cache()
        print("CERTIFIED", identifier, "reference_witnesses", len(candidates), flush=True)
    save(root / "known_goals.json", targets)
    save(out / "witnesses.json", witnesses)
    save(out / "result.json", {"reference_witnesses": len(witnesses),
         "reference_collision_valid": sum(w["collision_valid"] for w in witnesses),
         "known_fk_goals": sum(map(len, targets.values())), "native_ik_calls": 0,
         "physical_trials": 0})


if __name__ == "__main__":
    main()
