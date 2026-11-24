"""Enumerate additional kinematic branches within the original CPU budget."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from urdf_reference import URDFChain, NAMES, HERE


def main() -> None:
    root = HERE / "research/20261117_01"
    scope = json.loads((root / "scope.json").read_text())
    prior = [json.loads(s) for s in (root / "cpu_reference/calls.jsonl").read_text().splitlines()]
    controls = {c["id"] for c in scope["cases"] if c["control"]}
    targets = [r for r in prior if r["best"]["pose_valid"] and r["case"] not in controls]
    spent = json.loads((root / "cpu_reference/result.json").read_text())["cpu_solves"]
    assert spent + 32*len(targets) <= scope["maximum_cpu_least_squares_solves"]
    out = root / "cpu_roots"
    out.mkdir(exist_ok=False)
    (out / "registration.json").write_text(json.dumps({"targets": [[r["case"],r["index"]] for r in targets],
        "starts_each":32,"rng_seed":10712,"previous_cpu_solves":spent,"same_frozen_urdf":True},indent=2))
    model = URDFChain(HERE.parent / "arx_core/scripts/curobo_arx/static_pick/reference/model/arx.urdf")
    rng = np.random.default_rng(10712)
    results = []
    for item in targets:
        state = json.loads((Path(scope["source_campaign"]) / "trials" / item["case"] / "state.json").read_text())
        bounds = np.array([state["joint_limits"][state["joint_names"].index(n)] for n in NAMES])
        current = np.array([state["qpos"][state["joint_names"].index(n)] for n in NAMES])
        goal = np.linalg.inv(np.asarray(state["links"]["body4"])) @ np.asarray(item["world_goal"])
        seeds = [np.clip(current,bounds[:,0]+1e-6,bounds[:,1]-1e-6),bounds.mean(axis=1)]
        seeds += list(rng.uniform(bounds[:,0]+1e-6,bounds[:,1]-1e-6,(30,6)))
        trials = []
        roots = []
        began = time.monotonic()
        for seed in seeds:
            solution = least_squares(lambda q:model.residual(q,goal)[0],seed,
                jac=lambda q:model.residual(q,goal)[1],bounds=(bounds[:,0],bounds[:,1]),
                ftol=1e-10,xtol=1e-10,gtol=1e-10,max_nfev=250,x_scale="jac")
            t,_=model.fk(solution.x)
            p=float(np.linalg.norm(t[:3,3]-goal[:3,3]));r=float(Rotation.from_matrix(goal[:3,:3].T@t[:3,:3]).magnitude())
            record={"q":solution.x.tolist(),"position_error_m":p,"rotation_error_rad":r,
                    "pose_valid":p<.002 and r<.03,"nfev":solution.nfev}
            trials.append(record)
            if record["pose_valid"] and all(np.linalg.norm(solution.x-np.asarray(x["q"]))>1e-3 for x in roots):
                roots.append(record)
        result={"case":item["case"],"index":item["index"],"world_goal":item["world_goal"],
                "trials":trials,"roots":roots,"wall_seconds":time.monotonic()-began}
        results.append(result)
        (out / "results.json").write_text(json.dumps(results,indent=2))
        print(item["case"],item["index"],"distinct pose roots",len(roots),flush=True)
    (out / "result.json").write_text(json.dumps({"targets":len(targets),"cpu_solves":32*len(targets),
        "cumulative_cpu_solves":spent+32*len(targets),"roots":sum(len(r["roots"]) for r in results),
        "collision_checked":False,"physical_trials":0},indent=2))


if __name__ == "__main__":
    main()
