"""One independent bounded solve per failed candidate, within the CPU budget."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from urdf_reference import URDFChain, NAMES, HERE


def main() -> None:
    root = HERE / "research/20261117_01"
    scope = json.loads((root / "scope.json").read_text())
    spent = json.loads((root / "cpu_roots/result.json").read_text())["cumulative_cpu_solves"]
    count = sum(len(c["indices"]) for c in scope["cases"] if not c["control"])
    assert spent + count <= scope["maximum_cpu_least_squares_solves"]
    out = root / "cpu_all"
    out.mkdir(exist_ok=False)
    (out / "registration.json").write_text(json.dumps({"targets": count, "starts_per_target": 1,
        "initialization": "Best balanced native v0.9 optimized seed, clipped inside original bounds",
        "previous_cpu_solves": spent, "total_budget":scope["maximum_cpu_least_squares_solves"],
        "no_infeasibility_proof_from_failure":True},indent=2))
    model = URDFChain(HERE.parent / "arx_core/scripts/curobo_arx/static_pick/reference/model/arx.urdf")
    records = []
    with (out / "calls.jsonl").open("x") as log:
        for case in scope["cases"]:
            if case["control"]:
                continue
            state = json.loads((Path(scope["source_campaign"])/"trials"/case["id"]/"state.json").read_text())
            bounds = np.array([state["joint_limits"][state["joint_names"].index(n)] for n in NAMES])
            invbase = np.linalg.inv(np.asarray(state["links"]["body4"]))
            native = [json.loads(s) for s in (root/"full_official32"/case["id"]/"calls.jsonl").read_text().splitlines()]
            for row in native:
                phase = row["native_phases"][-1]
                q0 = np.clip(np.asarray(phase["optimized_joint_positions"][phase["best_seed"]]),bounds[:,0]+1e-6,bounds[:,1]-1e-6)
                goal = invbase @ np.asarray(row["world_goal"])
                result = least_squares(lambda q:model.residual(q,goal)[0],q0,
                    jac=lambda q:model.residual(q,goal)[1],bounds=(bounds[:,0],bounds[:,1]),
                    ftol=1e-10,xtol=1e-10,gtol=1e-10,max_nfev=250,x_scale="jac")
                t,_=model.fk(result.x)
                p=float(np.linalg.norm(t[:3,3]-goal[:3,3]));r=float(Rotation.from_matrix(goal[:3,:3].T@t[:3,:3]).magnitude())
                item={"case":case["id"],"index":row["index"],"world_goal":row["world_goal"],"q":result.x.tolist(),
                      "position_error_m":p,"rotation_error_rad":r,"pose_valid":p<.002 and r<.03,"nfev":result.nfev,
                      "scene_collision_checked":False}
                log.write(json.dumps(item)+"\n");log.flush();records.append(item)
            print(case["id"],"pose witnesses",sum(r["pose_valid"] for r in records if r["case"]==case["id"]),flush=True)
    (out/"result.json").write_text(json.dumps({"targets":len(records),"cpu_solves":len(records),
        "cumulative_cpu_solves":spent+len(records),"kinematic_witnesses":sum(r["pose_valid"] for r in records),
        "collision_checked":False,"physical_trials":0},indent=2))


if __name__=="__main__":
    main()
