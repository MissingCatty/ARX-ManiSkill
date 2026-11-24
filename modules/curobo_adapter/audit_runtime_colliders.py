"""Static checks using the exact convex pieces loaded by the ManiSkill ARX URDF."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import trimesh
import yaml

from audit_collision_cover import HERE, ROOT, REFERENCE, original_geometry, link_frames, coverage, union_outside


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def runtime_geometry(assets: Path) -> dict[str, list[np.ndarray]]:
    manifest = json.loads((assets / "manifest.json").read_text())
    result = {}
    for link in ET.parse(assets / "robot.urdf").getroot().findall("link"):
        name = link.get("name")
        if name not in manifest["links"] or not name.startswith("right_link"):
            continue
        pieces = []
        for collision in link.findall("collision"):
            mesh = collision.find("geometry/mesh")
            assert mesh is not None
            filename = mesh.get("filename")
            assert digest(assets / filename) == manifest["files"][filename]
            vertices = np.asarray(trimesh.load(assets / filename, force="mesh", process=False).vertices)
            vertices = vertices * np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
            origin = collision.find("origin")
            if origin is not None:
                rot = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
                vertices = vertices @ rot.T + np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            pieces.append(np.unique(vertices, axis=0))
        if pieces:
            result[name] = pieces
    assert len(result) == 8
    assert {n: len(v) for n, v in result.items()} == {f"right_link{i}": 19 if i in (27, 28) else 1 for i in range(21, 29)}
    return result


def pose_rotation(pose):
    return Rotation.from_quat([*pose[4:], pose[3]]).as_matrix()


def obstacles(trial: Path):
    scene = json.loads((trial / "plan/scene.json").read_text())
    state = json.loads((trial / "state.json").read_text())
    result = []
    for name, box in scene["cuboid"].items():
        vertices = np.array([[x, y, z] for x in [-.5, .5] for y in [-.5, .5] for z in [-.5, .5]]) * np.asarray(box["dims"])
        p = box["pose"]
        result.append((name, vertices @ pose_rotation(p).T + np.asarray(p[:3])))
    for name, mesh in scene.get("mesh", {}).items():
        assert digest(Path(mesh["file_path"])) == state["scene_meshes"][name]["file_sha256"]
        parts = list(trimesh.load(mesh["file_path"], force="mesh", process=False).split(only_watertight=False))
        assert len(parts) == state["scene_meshes"][name]["native_convex_parts"]
        p = mesh["pose"]
        for i, part in enumerate(parts):
            result.append((f"{name}:part{i}", np.asarray(part.vertices) @ pose_rotation(p).T + np.asarray(p[:3])))
    return [(n, v, ConvexHull(v).equations) for n, v in result]


def main() -> None:
    spec = json.loads((ROOT / "runtime_collider_registration_v2.json").read_text())
    for path, expected in spec["hashes"].items():
        assert digest(Path(path)) == expected, path
    began = time.monotonic()
    assets = Path(spec["assets"])
    geometry = runtime_geometry(assets)
    authored = original_geometry()
    equivalence = {}
    for name, pieces in geometry.items():
        v = np.vstack(pieces)
        difference = float(np.max(np.abs(np.r_[v.min(0), v.max(0)] - np.r_[authored[name].min(0), authored[name].max(0)])))
        assert difference < 1e-6
        equivalence[name] = {"parts": len(pieces), "source_bounds_max_difference_m": difference}
    cfg = yaml.safe_load((REFERENCE / "arx.yml").read_text())["robot_cfg"]["kinematics"]
    rng = np.random.default_rng(spec["seed"])
    covering = []
    for name, pieces in geometry.items():
        detail = []
        for vertices in pieces:
            rank = int(np.linalg.matrix_rank(vertices-vertices.mean(0), tol=1e-10))
            if rank == 3:
                row = coverage(name, vertices, cfg["collision_spheres"][name], rng)
            else:
                # Keep degenerate source pieces intact. Do not joggle/thicken
                # their geometry to obtain a favorable cooking surrogate.
                weights = rng.dirichlet(np.ones(len(vertices)), size=5000)
                samples = np.vstack([vertices, weights @ vertices])
                spheres = cfg["collision_spheres"][name]
                gaps = union_outside(samples, np.asarray([s["center"] for s in spheres]), np.asarray([s["radius"] for s in spheres]))
                row = {"hull_surface_and_interior_samples":len(samples),
                       "sampled_uncovered_count":int((gaps>1e-7).sum()),
                       "maximum_sampled_outside_gap_mm":float(max(0., gaps.max())*1000),
                       "sampled_2mm_offset_uncovered_count":0,
                       "maximum_2mm_offset_outside_gap_mm":0.,
                       "outward_offset_not_applicable":True}
            row["source_affine_dimension"] = rank
            detail.append(row)
        covering.append({"link": name, "parts": len(pieces),
            "sampled_uncovered_count": sum(d["sampled_uncovered_count"] for d in detail),
            "sample_count": sum(d["hull_surface_and_interior_samples"] for d in detail),
            "maximum_sampled_outside_gap_mm": max(d["maximum_sampled_outside_gap_mm"] for d in detail),
            "sampled_2mm_offset_uncovered_count": sum(d["sampled_2mm_offset_uncovered_count"] for d in detail),
            "maximum_2mm_offset_outside_gap_mm": max(d["maximum_2mm_offset_outside_gap_mm"] for d in detail),
            "degenerate_piece_count":sum(d["source_affine_dimension"]<3 for d in detail),
            "per_piece":detail,
            "interpretation": "Sample diagnostic, not formal coverage or volume estimate; part-level support excess is not the whole-link envelope error"})
    old = HERE / "research/20261117_01"
    scope = json.loads((old / "scope.json").read_text())
    records = json.loads((old / "reference_validation_new/records.json").read_text())
    tasks = {}
    controls = {c["id"] for c in scope["cases"] if c["control"]}
    for record in records:
        key = (record["case"], record["index"], "G_witness")
        tasks.setdefault(key, {"case":record["case"], "index":record["index"], "kind":"G_witness",
                              "q":record["q"], "control":record["case"] in controls})
    entries = json.loads((ROOT / "entry_results.json").read_text())["results"]
    for entry in entries:
        if entry["entry_trajectory_valid"]:
            plan = json.loads((ROOT / entry["id"] / "entry_plan.json").read_text())
            tasks[(entry["id"], entry["selected_index"], "H_endpoint")] = {
                "case":entry["id"], "index":entry["selected_index"], "kind":"H_endpoint",
                "q":plan["paths"]["pregrasp"]["q"][-1], "control":entry["control"]}
    lp_calls = 0
    obstacle_cache = {}
    result_rows = []
    for task in tasks.values():
        assert time.monotonic()-began < spec["maximum_wall_seconds"], "CPU audit wall cap"
        trial = Path(scope["source_campaign"]) / "trials" / task["case"]
        model = yaml.safe_load((trial / "plan/arx.yml").read_text())["robot_cfg"]["kinematics"]
        values = dict(model["lock_joints"])
        values.update(dict(zip([f"right_joint{k}" for k in range(21, 27)], np.asarray(task["q"], dtype=np.float32).astype(float))))
        frames = link_frames(values)
        if task["case"] not in obstacle_cache:
            obstacle_cache[task["case"]] = obstacles(trial)
        hits = []; unresolved = []
        for name, pieces in geometry.items():
            transform = frames[name]
            for part_index, vertices in enumerate(pieces):
                v = vertices @ transform[:3, :3].T + transform[:3, 3]
                planes = None
                for obstacle, other, other_planes in obstacle_cache[task["case"]]:
                    gap = np.maximum(v.min(0)-other.max(0), other.min(0)-v.max(0))
                    if gap.max() > 1e-7:
                        continue
                    if np.linalg.matrix_rank(vertices-vertices.mean(0), tol=1e-10) < 3:
                        unresolved.append({"link":name,"part":part_index,"obstacle":obstacle,
                            "reason":"Degenerate source convex piece; actual PhysX cooking not examined; overlapping AABBs cannot establish intersection"})
                        continue
                    if planes is None:
                        planes = ConvexHull(v).equations
                    all_planes = np.vstack([planes, other_planes])
                    assert lp_calls < spec["maximum_lp_calls"], "Static LP cap"
                    lp_calls += 1
                    lp = linprog([0, 0, 0, -1], A_ub=np.c_[all_planes[:, :3], np.ones(len(all_planes))],
                                 b_ub=-all_planes[:, 3], bounds=[(None, None)]*4,
                                 method="highs", options={"time_limit":2.})
                    assert lp.success, lp.message
                    if lp.x[3] >= -1e-7:
                        hits.append({"link":name, "part":part_index, "obstacle":obstacle,
                                     "intersects":bool(lp.x[3]>1e-7),
                                     "near_contact":bool(abs(lp.x[3])<=1e-7),
                                     "common_interior_ball_radius_mm":float(lp.x[3]*1000)})
        row = {k:v for k,v in task.items() if k != "q"}
        row.update(all_runtime_source_pieces_disjoint=not hits and not unresolved,
                   overlaps_or_near_contacts=hits, unresolved_degenerate_pairs=unresolved,
                   runtime_PhysX_cooking_verified=False)
        result_rows.append(row)
        print(row["case"], row["index"], row["kind"], "pairs", len(hits), "unresolved",len(unresolved), flush=True)
    result = {"original_authored_vs_runtime_bounds":equivalence, "runtime_cover_samples":covering,
              "records":result_rows, "lp_calls":lp_calls, "wall_seconds":time.monotonic()-began,
              "physical_trials":0, "model_changes":0, "runtime_piece_count":sum(map(len,geometry.values())),
              "scope":"Static source convex pieces from existing ManiSkill URDF versus recorded environment geometry. Degenerate pieces with overlapping AABBs remain unresolved. No PhysX cooking, contacts, dynamics, unmodelled links or target collision certification"}
    (ROOT / "runtime_collider_audit.json").write_text(json.dumps(result, indent=2))
    for path, expected in spec["hashes"].items():
        assert digest(Path(path)) == expected, path


if __name__ == "__main__":
    main()
