"""Recheck saved entry records and frozen sources without starting CUDA."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation

from urdf_reference import URDFChain, NAMES

HERE = Path(__file__).resolve().parent
ROOT = HERE / "research/20261117_02"


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    spec = json.loads((ROOT / "scope.json").read_text())
    hashes = {**spec["protected_hashes"], **spec["source_hashes"]}
    for name in ["collision_audit_registration.json", "runtime_collider_registration_v2.json"]:
        hashes.update(json.loads((ROOT / name).read_text())["hashes"])
    for path, expected in hashes.items():
        assert digest(path) == expected, path
    # Preserve the original failed CPU-audit source snapshot as well as v2.
    v1 = json.loads((ROOT / "runtime_collider_registration.json").read_text())
    assert digest(ROOT / "sources/audit_runtime_colliders.py") == v1["hashes"][str(HERE/"audit_runtime_colliders.py")]
    assert not subprocess.check_output(["git", "-C", str(HERE / "source"), "diff", "--name-only"], text=True).strip()
    assert subprocess.check_output(["git", "-C", str(HERE / "source"), "rev-parse", "HEAD"], text=True).strip() == "42498cf4e7d0d6fbd4ef9fbcec1243c00c888a83"
    chain = URDFChain(HERE.parent / "arx_core/scripts/curobo_arx/static_pick/reference/model/arx.urdf")
    results = json.loads((ROOT / "entry_results.json").read_text())["results"]
    count = phases = paths = dense_samples = 0
    case_gates = []
    max_position = max_rotation = 0.

    def check_fk(state, q, world):
        nonlocal max_position, max_rotation
        values = np.asarray(q)
        limits = np.array([state["joint_limits"][state["joint_names"].index(n)] for n in NAMES])
        assert ((values >= limits[:, 0]-1e-6) & (values <= limits[:, 1]+1e-6)).all()
        own, _ = chain.fk(values)
        goal = np.linalg.inv(np.asarray(state["links"]["body4"])) @ np.asarray(world)
        pe = np.linalg.norm(own[:3, 3]-goal[:3, 3])
        re = Rotation.from_matrix(own[:3, :3] @ goal[:3, :3].T).magnitude()
        assert pe < .002 and re < .03
        max_position, max_rotation = max(max_position, float(pe)), max(max_rotation, float(re))

    def check_path(folder, state, plan):
        nonlocal paths, dense_samples
        assert plan["planning_scope"] == "pregrasp_entry_only"
        assert plan["full_grasp_validated"] is False
        assert set(plan["paths"]) == {"pregrasp"}
        assert plan["selected"]["q_grasp"] is None and plan["selected"]["q_lift"] is None
        q = np.asarray(plan["paths"]["pregrasp"]["q"])
        times = np.asarray(plan["paths"]["pregrasp"]["times"])
        assert np.isfinite(q).all() and (np.diff(times)>0).all()
        limits = np.array([state["joint_limits"][state["joint_names"].index(n)] for n in NAMES])
        assert ((q >= limits[:, 0]-1e-6) & (q <= limits[:, 1]+1e-6)).all()
        measured = np.array([state["qpos"][state["joint_names"].index(n)] for n in NAMES])
        measured = np.clip(measured, limits[:, 0]+1e-4, limits[:, 1]-1e-4)
        assert np.max(np.abs(q[0]-measured)) < .005
        check_fk(state, q[-1], plan["selected"]["pregrasp_world"])
        opt = json.loads((folder/"model/pregrasp_optimization.json").read_text())
        assert opt["success"] and opt["invalid_samples"] == 0
        assert opt["max_limit_exceed_rad"] <= 1e-6 and opt["goal_error_rad"] <= .005
        assert opt["dense_samples"] >= len(q)
        dense_samples += opt["dense_samples"]; paths += 1

    for case, result in zip(spec["cases"], results):
        assert case["id"] == result["id"]
        folder = ROOT / case["id"]
        trial = Path(spec["source_campaign"])/"trials"/case["id"]
        state = json.loads((trial/"state.json").read_text())
        for filename in ["arx.yml", "nominal_closed.yml", "scene.json"]:
            assert (trial/"plan"/filename).read_bytes() == (folder/"model"/filename).read_bytes()
        requests = list(map(json.loads, (folder/"requests.jsonl").read_text().splitlines()))
        calls = list(map(json.loads, (folder/"calls.jsonl").read_text().splitlines()))
        assert len(requests) == len(calls) == result["native_queries"]
        count += len(calls); gates = Counter()
        assert [r["index"] for r in requests if r["kind"]=="H_from_measured_start"] == case["indices"]
        for i, (request, call) in enumerate(zip(requests, calls)):
            assert request["request_id"] == call["request_id"] == i
            assert (request["index"], request["kind"], request["world_goal"]) == (call["index"], call["kind"], call["world_goal"])
            for phase in call["native_phases"]:
                p = np.asarray(phase["position_error_m"])
                r = np.asarray(phase["orientation_error_rad"])
                pose = (p<.002) & (r<.03)
                success = pose & np.asarray(phase["feasible"])
                assert np.array_equal(pose, phase["pose_converged"])
                assert np.array_equal(success, phase["native_full_success"])
                selected = np.asarray(phase["selected_seed_indices"])
                assert np.array_equal(success[selected], phase["returned_success"])
                phases += 1
            phase = call["native_phases"][-1]
            if call["accepted"]:
                check_fk(state, call["q"], call["world_goal"])
            if call["kind"] == "H_from_measured_start":
                label = "accepted" if call["accepted"] else "outer_collision" if any(phase["returned_success"]) else "pose_converged_constraint_rejected" if any(phase["pose_converged"]) else "finite_search_not_converged"
                gates[label] += 1
        case_gates.append({"case":case["id"], "counts":dict(gates)})
        if result["entry_trajectory_valid"]:
            check_path(folder, state, json.loads((folder/"entry_plan.json").read_text()))

    integration = ROOT/"entry_cli_integration"
    plan = json.loads((integration/"plan_result.json").read_text())
    assert plan["success"] and plan["selected"]["index"] == 14
    assert plan["candidate_input"]["stage"] == "zerograsp_collision_refinement_nms"
    assert not plan["candidate_input"]["raw_fallback"] and plan["candidate_input"]["count"] == 65
    state = json.loads((integration/"input/state.json").read_text())
    requests = list(map(json.loads, (integration/"requests.jsonl").read_text().splitlines()))
    receipts = list(map(json.loads, (integration/"calls.jsonl").read_text().splitlines()))
    assert len(requests) == len(receipts) == 15
    assert [bool(c["accepted"]) for c in receipts] == [False]*14+[True]
    assert all(not any(np.asarray(c["native_success"]).reshape(-1)) for c in receipts[:-1])
    assert any(np.asarray(receipts[-1]["native_success"]).reshape(-1))
    count += len(receipts)
    trial = Path(spec["source_campaign"])/"trials/B0209_S5_occluded_048_hammer"
    for filename in ["arx.yml", "nominal_closed.yml", "scene.json"]:
        assert (trial/"plan"/filename).read_bytes() == (integration/"input/plan"/filename).read_bytes() == (integration/"model"/filename).read_bytes()
    check_path(integration, state, plan)
    assert count == sum(len(p.read_text().splitlines()) for p in ROOT.rglob("requests.jsonl"))
    assert count == sum(len(p.read_text().splitlines()) for p in ROOT.rglob("calls.jsonl"))
    assert count <= spec["maximum_native_ik_calls"]
    output = {"passed":True,"protected_hashes_checked":len(hashes), "request_count":count,
              "receipt_count":count, "unmatched_requests":0, "native_phases_reconstructed":phases,
              "qualified_entry_paths":paths, "reported_dense_samples":dense_samples,
              "independent_FK_max_position_m":max_position, "independent_FK_max_rotation_rad":max_rotation,
              "H_gate_counts":case_gates, "physical_trials":0,"simulator_instances":0,"cloud_requests":0,
              "old_default_changed":False,"official_source_modified":False}
    (ROOT/"verification.json").write_text(json.dumps(output,indent=2))
    print(json.dumps({k:v for k,v in output.items() if k!="H_gate_counts"},indent=2))


if __name__ == "__main__":
    main()
