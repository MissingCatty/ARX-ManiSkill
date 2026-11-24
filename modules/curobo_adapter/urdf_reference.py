"""Independent bounded CPU IK using the frozen ARX URDF; no collision bypass."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

NAMES = [f"right_joint{i}" for i in range(21, 27)]
HERE = Path(__file__).resolve().parent


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


class URDFChain:
    """Evaluate URDF origins and joint axes independently of cuRobo."""

    def __init__(self, path: Path) -> None:
        root = ET.parse(path).getroot()
        by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
        link = "right_eef_link"
        chain = []
        while link != "body4":
            joint = by_child[link]
            origin = joint.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            axis = joint.find("axis")
            axis = np.fromstring(axis.get("xyz"), sep=" ") if axis is not None else np.array([1., 0., 0.])
            name = joint.get("name")
            if joint.get("type") != "fixed":
                assert name in NAMES and joint.get("type") == "revolute", name
            chain.append((Rotation.from_euler("xyz", rpy).as_matrix(), xyz,
                          NAMES.index(name) if name in NAMES else None, axis))
            link = joint.find("parent").get("link")
        self.chain = list(reversed(chain))

    def fk(self, q: np.ndarray, jacobian: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
        rot = np.eye(3)
        pos = np.zeros(3)
        axes = []
        for origin_rot, xyz, index, axis in self.chain:
            pos = pos + rot @ xyz
            rot = rot @ origin_rot
            if index is not None:
                axes.append((index, rot @ axis, pos.copy()))
                rot = rot @ Rotation.from_rotvec(q[index] * axis).as_matrix()
        transform = np.eye(4)
        transform[:3, :3] = rot
        transform[:3, 3] = pos
        jac = None
        if jacobian:
            jac = np.zeros((6, 6))
            for index, direction, pivot in axes:
                jac[:3, index] = np.cross(direction, pos - pivot)
                jac[3:, index] = direction
        return transform, jac

    def residual(self, q: np.ndarray, goal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t, jac = self.fk(q, jacobian=True)
        phi = Rotation.from_matrix(goal[:3, :3].T @ t[:3, :3]).as_rotvec()
        theta = np.linalg.norm(phi)
        hat = skew(phi)
        factor = 1. / 12. if theta < 1e-5 else (1. - 0.5 * theta / np.tan(0.5 * theta)) / theta**2
        inv_left = np.eye(3) - 0.5 * hat + factor * (hat @ hat)
        weight = 0.1  # scale in meters/radian for this independent optimizer only
        residual = np.r_[t[:3, 3] - goal[:3, 3], weight * phi]
        derivative = np.vstack((jac[:3], weight * inv_left @ goal[:3, :3].T @ jac[3:]))
        return residual, derivative


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    scope = json.loads(args.scope.read_text())
    original = Path(scope["old_result_root"])
    trialroot = Path(scope["source_campaign"]) / "trials"
    urdf = HERE.parent / "arx_core/scripts/curobo_arx/static_pick/reference/model/arx.urdf"
    model = URDFChain(urdf)
    rng = np.random.default_rng(9817)
    total = 0
    records = []
    with (args.out / "calls.jsonl").open("x") as log:
        for case in scope["cases"]:
            state = json.loads((trialroot / case["id"] / "state.json").read_text())
            measured = np.array([state["qpos"][state["joint_names"].index(n)] for n in NAMES])
            bounds = np.array([state["joint_limits"][state["joint_names"].index(n)] for n in NAMES])
            invbase = np.linalg.inv(np.asarray(state["links"]["body4"]))
            expected = invbase @ np.asarray(state["links"]["right_eef_link"])
            own, _ = model.fk(measured)
            assert np.linalg.norm(own[:3, 3] - expected[:3, 3]) < 2e-5
            assert Rotation.from_matrix(own[:3, :3] @ expected[:3, :3].T).magnitude() < 2e-4
            # Verify the independently derived Jacobian before numerical IK.
            test_q = bounds.mean(axis=1)
            test_goal, _ = model.fk(test_q + 0.01)
            _, analytic = model.residual(test_q, test_goal)
            numerical = np.column_stack([(model.residual(test_q + np.eye(6)[i]*1e-6, test_goal)[0]
                                          - model.residual(test_q - np.eye(6)[i]*1e-6, test_goal)[0]) / 2e-6
                                         for i in range(6)])
            assert np.max(np.abs(analytic - numerical)) < 1e-5
            archived = {r["index"]: r for r in map(json.loads, (original / case["id"] / "calls.jsonl").read_text().splitlines())}
            for index in case["screen_indices"]:
                world = np.asarray(archived[index]["world_goal"])
                goal = invbase @ world
                seeds = [np.clip(measured, bounds[:, 0]+1e-6, bounds[:, 1]-1e-6), bounds.mean(axis=1)]
                seeds += list(rng.uniform(bounds[:, 0]+1e-6, bounds[:, 1]-1e-6, size=(30, 6)))
                trials = []
                best = None
                start_time = time.monotonic()
                for q0 in seeds:
                    cached_q = None
                    cached = None
                    def evaluate(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                        nonlocal cached_q, cached
                        if cached_q is None or not np.array_equal(q, cached_q):
                            cached_q, cached = q.copy(), model.residual(q, goal)
                        return cached
                    solved = least_squares(lambda q: evaluate(q)[0], q0,
                        jac=lambda q: evaluate(q)[1], bounds=(bounds[:, 0], bounds[:, 1]),
                        max_nfev=250, ftol=1e-10, xtol=1e-10, gtol=1e-10, x_scale="jac")
                    total += 1
                    fk, _ = model.fk(solved.x)
                    p = float(np.linalg.norm(fk[:3, 3] - goal[:3, 3]))
                    r = float(Rotation.from_matrix(goal[:3, :3].T @ fk[:3, :3]).magnitude())
                    candidate = {"q": solved.x.tolist(), "position_error_m": p, "rotation_error_rad": r,
                                 "nfev": solved.nfev, "status": solved.status,
                                 "pose_valid": p < .002 and r < .03,
                                 "within_limits": bool(((solved.x >= bounds[:, 0]) & (solved.x <= bounds[:, 1])).all())}
                    trials.append(candidate)
                    if best is None or max(p/.002, r/.03) < max(best["position_error_m"]/.002, best["rotation_error_rad"]/.03):
                        best = candidate
                    if candidate["pose_valid"]:
                        break  # witness found; scene collision still needs independent verification
                record = {"case": case["id"], "index": index, "world_goal": world.tolist(),
                          "best": best, "trials": trials, "wall_seconds": time.monotonic()-start_time,
                          "scene_collision_checked": False, "physical_trials": 0}
                log.write(json.dumps(record, allow_nan=False)+"\n")
                log.flush()
                records.append(record)
                print(case["id"], index, "witness", best["pose_valid"],
                      "mm", round(best["position_error_m"]*1000, 3), "starts", len(trials), flush=True)
    summary = {"targets": len(records), "cpu_solves": total,
               "kinematic_witnesses": sum(r["best"]["pose_valid"] for r in records),
               "scene_collision_verified": False, "physical_trials": 0}
    (args.out / "result.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
