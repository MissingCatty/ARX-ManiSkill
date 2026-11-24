"""Plan native postprocessed ARX grasps with the isolated official cuRobo backend."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from adapter import ARXPlanner, ARX, HERE, digest, save, flush_planning_artifacts

sys.path.insert(0, str(HERE.parent / "maniskill/planning_common"))
from candidates import load_candidates
from grasp_selector import select_postprocessed


def pose7(transform: np.ndarray) -> list[float]:
    q = Rotation.from_matrix(transform[:3, :3]).as_quat()
    return [*transform[:3, 3].tolist(), float(q[3]), *q[:3].tolist()]


def build_frozen_models(state: dict, trial: Path) -> None:
    """Apply the original measured-state serialization, without model tuning."""
    plan = trial / "plan"
    plan.mkdir(parents=True, exist_ok=False)
    save(trial / "state.json", state)
    reference = ARX / "scripts/curobo_arx/static_pick/reference/model"
    cfg = yaml.safe_load((reference / "arx.yml").read_text())
    cfg["robot_cfg"]["kinematics"]["urdf_path"] = str(reference / "arx.urdf")
    measured = dict(zip(state["joint_names"], state["qpos"]))
    for name in ("right_joint27", "right_joint28"):
        cfg["robot_cfg"]["kinematics"]["lock_joints"][name] = measured[name]
    (plan / "arx.yml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    invbase = np.linalg.inv(np.asarray(state["links"]["body4"]))
    scene = {"cuboid": {}}
    assert "scene_cuboids" in state
    for key, geometry in state["scene_cuboids"].items():
        t = np.eye(4)
        t[:3, 3] = geometry["center"]
        scene["cuboid"][key] = {"pose": pose7(invbase @ t), "dims": geometry["dims"]}
    meshes = {name: value for name, value in state.get("scene_meshes", {}).items() if not value["target"]}
    for mesh in meshes.values():
        assert digest(Path(mesh["file_path"])) == mesh["file_sha256"]
    if meshes:
        scene["mesh"] = {name: {"pose": pose7(invbase @ np.asarray(g["world"])), "file_path":g["file_path"]}
                         for name, g in meshes.items()}
    # Exact original formatting supports byte-level equivalence checks.
    (plan / "scene.json").write_text(json.dumps(scene, indent=2))
    closed_ref = reference.parent / "model_closed"
    closed = yaml.safe_load((closed_ref / "arx.yml").read_text())
    closed["robot_cfg"]["kinematics"]["urdf_path"] = str(closed_ref / "arx.urdf")
    for name in ("right_joint27", "right_joint28"):
        closed["robot_cfg"]["kinematics"]["lock_joints"][name] = .022
    (plan / "nominal_closed.yml").write_text(yaml.safe_dump(closed, sort_keys=False))


class SelectorAdapter:
    """Expose the unchanged selector interface and retain native IK receipts."""

    def __init__(self, planner: ARXPlanner, log_path: Path, max_calls: int) -> None:
        self.planner = planner
        self.log_path = log_path
        self.max_calls = max_calls
        self.calls = 0

    def __getattr__(self, name: str):
        return getattr(self.planner, name)

    def ik(self, world: np.ndarray, start):
        if self.calls >= self.max_calls:
            raise RuntimeError("Registered integration IK call cap reached")
        self.calls += 1
        with self.log_path.open("a") as requests:
            requests.write(json.dumps({"world_goal":world.tolist(),"current_q":start.tolist()})+"\n")
        q, result = self.planner.ik(world, start)
        with self.log_path.with_name("calls.jsonl").open("a") as calls:
            calls.write(json.dumps({"accepted":q is not None,"native_success":result.success.tolist(),
                                    "q":None if q is None else q.tolist()})+"\n")
        return q


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", default="official32")
    parser.add_argument("--max-ik-queries", type=int, default=64)
    args = parser.parse_args()
    source = args.input.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    rows, metadata = load_candidates(source)
    state = json.loads((source / "state.json").read_text())
    build_frozen_models(state, out / "input")
    planner = ARXPlanner(out / "input", out / "model", args.variant)
    try:
        selector = SelectorAdapter(planner, out / "requests.jsonl", args.max_ik_queries)
        result = select_postprocessed(selector, state, rows[np.argsort(-rows[:, 0], kind="stable")],
                                     np.asarray(state["camera_world_optical"]))
        result["candidate_input"] = {"stage":metadata["candidate_stage"],"count":len(rows),
                                     "raw_fallback":False,"source":str(source)}
        result.update(wall_seconds=time.monotonic()-began,physical_execution=False,
                      backend="isolated official cuRobo v0.9.0")
        save(out / "plan_result.json", result)
        print(json.dumps({"success":result["success"],"out":str(out)}),flush=True)
    except Exception:
        save(out / "plan_result.json", {"success":False,"reason":"planner_error",
             "error":traceback.format_exc(),"physical_execution":False})
        raise
    finally:
        try:
            flush_planning_artifacts()
        finally:
            planner.planner.destroy()


if __name__ == "__main__":
    main()
