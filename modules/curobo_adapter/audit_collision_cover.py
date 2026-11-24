"""Read-only coverage diagnostics for the original USD hulls and sphere proxy."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import yaml
from pxr import Usd, UsdGeom, UsdPhysics

HERE = Path(__file__).resolve().parent
ROOT = HERE / "research/20261117_02"
REFERENCE = HERE.parent / "arx_core/scripts/curobo_arx/static_pick/reference/model"


def original_geometry() -> dict[str, np.ndarray]:
    manifest = json.loads((REFERENCE / "model_manifest.json").read_text())
    stage = Usd.Stage.Open(manifest["source"])
    cache = UsdGeom.XformCache()
    geometry = {}
    for link in stage.Traverse():
        if not link.HasAPI(UsdPhysics.RigidBodyAPI) or not link.GetName().startswith("right_link"):
            continue
        inverse = np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(link)))
        pieces = []
        for mesh in Usd.PrimRange(link):
            if not mesh.IsA(UsdGeom.Mesh) or not mesh.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if not UsdPhysics.CollisionAPI(mesh).GetCollisionEnabledAttr().Get():
                continue
            assert mesh.GetAttribute("physics:approximation").Get() == "convexHull"
            points = np.asarray(UsdGeom.Mesh(mesh).GetPointsAttr().Get(), dtype=float)
            transform = np.asarray(cache.GetLocalToWorldTransform(mesh)) @ inverse
            pieces.append((np.c_[points, np.ones(len(points))] @ transform)[:, :3])
        # Source asset has exactly one authored convex collider per link.
        assert len(pieces) == 1, (link.GetName(), len(pieces))
        geometry[link.GetName()] = np.unique(pieces[0], axis=0)
    assert len(geometry) == 8
    return geometry


def link_frames(values: dict[str, float]) -> dict[str, np.ndarray]:
    pending = list(ET.parse(REFERENCE / "arx.urdf").getroot().findall("joint"))
    frames = {"body4": np.eye(4)}
    while pending:
        ready = [j for j in pending if j.find("parent").get("link") in frames]
        assert ready
        for joint in ready:
            origin = joint.find("origin")
            transform = np.eye(4)
            if origin is not None:
                transform[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                transform[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
            motion = np.eye(4)
            axis = joint.find("axis")
            axis = np.fromstring(axis.get("xyz"), sep=" ") if axis is not None else np.array([1., 0., 0.])
            value = values.get(joint.get("name"), 0.)
            if joint.get("type") == "revolute":
                motion[:3, :3] = Rotation.from_rotvec(axis * value).as_matrix()
            elif joint.get("type") == "prismatic":
                motion[:3, 3] = axis * value
            frames[joint.find("child").get("link")] = frames[joint.find("parent").get("link")] @ transform @ motion
        pending = [j for j in pending if j not in ready]
    return frames


def union_outside(points: np.ndarray, centers: np.ndarray, radii: np.ndarray) -> np.ndarray:
    return np.min(np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2) - radii, axis=1)


def coverage(name: str, vertices: np.ndarray, spheres: list[dict], rng: np.random.Generator) -> dict:
    hull = ConvexHull(vertices)
    centers = np.asarray([s["center"] for s in spheres])
    radii = np.asarray([s["radius"] for s in spheres])
    triangles = vertices[hull.simplices]
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    ids = rng.choice(len(triangles), size=5000, p=area / area.sum())
    uv = rng.uniform(size=(5000, 2))
    flip = uv.sum(axis=1) > 1
    uv[flip] = 1 - uv[flip]
    chosen = triangles[ids]
    surface = chosen[:, 0] + uv[:, :1] * (chosen[:, 1] - chosen[:, 0]) + uv[:, 1:] * (chosen[:, 2] - chosen[:, 0])
    # Interpolate between an interior point and sampled hull surfaces. This is
    # deterministic diagnostic sampling, not a formal containment proof.
    interior = hull.points[hull.vertices].mean(axis=0)
    inside = interior + (surface - interior) * rng.uniform(size=(5000, 1)) ** (1 / 3)
    samples = np.vstack([vertices[hull.vertices], surface, inside])
    gap = union_outside(samples, centers, radii)
    offset = surface + .002 * hull.equations[ids, :3]
    offset_gap = union_outside(offset, centers, radii)
    directions = np.vstack([np.eye(3), -np.eye(3), hull.equations[:, :3]])
    k = np.arange(128)
    z = 1 - 2 * (k + .5) / 128
    phi = k * np.pi * (3 - np.sqrt(5))
    directions = np.vstack([directions, np.c_[np.sqrt(1-z*z)*np.cos(phi), np.sqrt(1-z*z)*np.sin(phi), z]])
    support_excess = (centers @ directions.T + radii[:, None]).max(axis=0) - (vertices @ directions.T).max(axis=0)
    return {
        "link": name, "sphere_count": len(spheres), "hull_vertices": len(hull.vertices),
        "hull_surface_and_interior_samples": len(samples),
        "sampled_uncovered_count": int((gap > 1e-7).sum()),
        "maximum_sampled_outside_gap_mm": float(max(0., gap.max()) * 1000),
        "surface_outward_2mm_samples": len(offset),
        "sampled_2mm_offset_uncovered_count": int((offset_gap > 1e-7).sum()),
        "maximum_2mm_offset_outside_gap_mm": float(max(0., offset_gap.max()) * 1000),
        "support_directions": len(directions),
        "support_excess_min_mm": float(support_excess.min() * 1000),
        "support_excess_median_mm": float(np.median(support_excess) * 1000),
        "support_excess_max_mm": float(support_excess.max() * 1000),
        "axis_support_excess_mm": dict(zip(["+X", "+Y", "+Z", "-X", "-Y", "-Z"], (support_excess[:6]*1000).tolist())),
    }


def plot_hammer(geometry: dict[str, np.ndarray], spheres: dict) -> None:
    old = HERE / "research/20261117_01"
    witness = json.loads((old / "hammer14_proxy_counterexample.json").read_text())
    scope = json.loads((old / "scope.json").read_text())
    trial = Path(scope["source_campaign"]) / "trials" / witness["case"]
    cfg = yaml.safe_load((trial / "plan/arx.yml").read_text())["robot_cfg"]["kinematics"]
    q = dict(cfg["lock_joints"])
    q.update(dict(zip([f"right_joint{k}" for k in range(21, 27)], np.asarray(witness["q"], dtype=np.float32).astype(float))))
    frames = link_frames(q)
    table = json.loads((trial / "plan/scene.json").read_text())["cuboid"]["table"]
    pose = table["pose"]
    rotation = Rotation.from_quat([*pose[4:], pose[3]]).as_matrix()
    def tabletop(points):
        result = (points - np.asarray(pose[:3])) @ rotation
        result[:, 2] -= table["dims"][2] / 2
        return result * 100  # cm, table top z=0
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plotted = []
    colors = {"right_link26": "#6a3d9a", "right_link27": "#1f78b4", "right_link28": "#e31a1c"}
    for name in colors:
        transform = frames[name]
        v = tabletop(geometry[name] @ transform[:3, :3].T + transform[:3, 3])
        c = tabletop(np.asarray([s["center"] for s in spheres[name]]) @ transform[:3, :3].T + transform[:3, 3])
        radii = np.asarray([s["radius"] for s in spheres[name]]) * 100
        plotted.append(v)
        for ax, horizontal in zip(axes, [0, 1]):
            hull2 = ConvexHull(v[:, [horizontal, 2]])
            outline = v[hull2.vertices][:, [horizontal, 2]]
            outline = np.vstack([outline, outline[0]])
            ax.fill(outline[:, 0], outline[:, 1], color=colors[name], alpha=.15)
            ax.plot(outline[:, 0], outline[:, 1], color=colors[name], lw=2, label=name + " USD hull")
            for center, radius in zip(c, radii):
                ax.add_patch(Circle(center[[horizontal, 2]], radius, fill=False, color=colors[name], alpha=.12, lw=.7))
    merged = np.vstack(plotted)
    for ax, horizontal in zip(axes, [0, 1]):
        ax.axhspan(-3, 0, color="gray", alpha=.25)
        ax.axhline(0, color="black", lw=1.4, label="Table top")
        ax.set(xlim=(merged[:, horizontal].min()-2, merged[:, horizontal].max()+2),
               ylim=(-2, merged[:, 2].max()+2), xlabel=f"Table-local {'XY'[horizontal]} (cm)", ylabel="Height above table (cm)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Hammer candidate index 14: original collision hulls and unchanged sphere proxy\n"
                 "Static orthographic projections; not camera input or execution evidence\n"
                 "USD minimum table gap 0.403 mm; sphere/table overlap 11.024 mm")
    fig.tight_layout(rect=(0, 0, 1, .87))
    fig.savefig(ROOT / "hammer14_collision_projection.png", dpi=170)
    plt.close(fig)


def main() -> None:
    spec = json.loads((ROOT / "collision_audit_registration.json").read_text())
    for p, h in spec["hashes"].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == h, p
    geometry = original_geometry()
    cfg = yaml.safe_load((REFERENCE / "arx.yml").read_text())["robot_cfg"]["kinematics"]
    spheres = cfg["collision_spheres"]
    assert sum(map(len, spheres.values())) == 956
    rng = np.random.default_rng(spec["seed"])
    rows = [coverage(name, geometry[name], spheres[name], rng) for name in cfg["collision_link_names"]]
    plot_hammer(geometry, spheres)
    result = {"sphere_count": 956, "rows": rows, "model_modified": False,
              "native_ik_queries": 0, "physical_trials": 0,
              "scope": "Read-only sampled coverage and directional support; not a formal containment proof or collision-model replacement",
              "existing_padding": "Builder includes 2mm in sphere radii; support excess includes padding and discretization; do not remove either based on this audit"}
    (ROOT / "collision_cover_audit.json").write_text(json.dumps(result, indent=2))
    for row in rows:
        print(row["link"], row["sphere_count"], "sample gaps", row["sampled_uncovered_count"],
              "support excess max mm", round(row["support_excess_max_mm"], 2), flush=True)


if __name__ == "__main__":
    main()
