#!/usr/bin/env python3
"""Visualize ARX-X5 cuRobo collision spheres on top of the URDF mesh."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from arx_project.assets import resource_path
from arx_project.layout import run_dir

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    imageio = None

try:
    import trimesh
except Exception:  # pragma: no cover
    trimesh = None


ROOT = resource_path("zerograsp_arx_assets")
URDF_PATH = ROOT / "X5A.urdf"
SPHERE_PATH = ROOT / "collision_X5A.yml"
OUT_DIR = run_dir("zerograsp", "legacy_arx_visualizations")


@dataclass
class Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray


@dataclass
class VisualMesh:
    link: str
    path: Path
    origin: np.ndarray
    scale: np.ndarray


def parse_vec(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=float)
    return np.asarray([float(x) for x in text.split()], dtype=float)


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = np.linalg.norm(axis)
    if norm < 1e-9 or abs(angle) < 1e-12:
        return np.eye(3)
    x, y, z = axis / norm
    c, s = math.cos(angle), math.sin(angle)
    c1 = 1.0 - c
    return np.array(
        [
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ],
        dtype=float,
    )


def transform_from_xyz_rpy(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    tf = np.eye(4)
    tf[:3, :3] = rpy_to_matrix(rpy)
    tf[:3, 3] = xyz
    return tf


def transform_points(tf: np.ndarray, points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    hom = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    return (tf @ hom.T).T[:, :3]


def parse_urdf(path: Path) -> tuple[list[str], list[Joint], list[VisualMesh]]:
    tree = ET.parse(path)
    root = tree.getroot()
    links = [link.attrib["name"] for link in root.findall("link")]
    joints: list[Joint] = []
    visuals: list[VisualMesh] = []

    for joint in root.findall("joint"):
        origin_el = joint.find("origin")
        xyz = parse_vec(origin_el.attrib.get("xyz") if origin_el is not None else None, (0, 0, 0))
        rpy = parse_vec(origin_el.attrib.get("rpy") if origin_el is not None else None, (0, 0, 0))
        axis_el = joint.find("axis")
        axis = parse_vec(axis_el.attrib.get("xyz") if axis_el is not None else None, (0, 0, 1))
        joints.append(
            Joint(
                name=joint.attrib["name"],
                joint_type=joint.attrib.get("type", "fixed"),
                parent=joint.find("parent").attrib["link"],
                child=joint.find("child").attrib["link"],
                origin=transform_from_xyz_rpy(xyz, rpy),
                axis=axis,
            )
        )

    for link in root.findall("link"):
        link_name = link.attrib["name"]
        for visual in link.findall("visual"):
            geometry = visual.find("geometry")
            mesh_el = geometry.find("mesh") if geometry is not None else None
            if mesh_el is None:
                continue
            filename = mesh_el.attrib.get("filename", "")
            if not filename:
                continue
            origin_el = visual.find("origin")
            xyz = parse_vec(origin_el.attrib.get("xyz") if origin_el is not None else None, (0, 0, 0))
            rpy = parse_vec(origin_el.attrib.get("rpy") if origin_el is not None else None, (0, 0, 0))
            scale = parse_vec(mesh_el.attrib.get("scale"), (1, 1, 1))
            mesh_path = (path.parent / filename).resolve()
            visuals.append(VisualMesh(link_name, mesh_path, transform_from_xyz_rpy(xyz, rpy), scale))

    return links, joints, visuals


def compute_link_poses(links: list[str], joints: list[Joint], q: dict[str, float]) -> dict[str, np.ndarray]:
    children = {joint.child for joint in joints}
    roots = [link for link in links if link not in children]
    if not roots:
        roots = [links[0]]
    poses = {roots[0]: np.eye(4)}
    unresolved = joints[:]

    while unresolved:
        progressed = False
        next_unresolved: list[Joint] = []
        for joint in unresolved:
            if joint.parent not in poses:
                next_unresolved.append(joint)
                continue
            value = q.get(joint.name, 0.0)
            motion = np.eye(4)
            if joint.joint_type in {"revolute", "continuous"}:
                motion[:3, :3] = axis_angle_to_matrix(joint.axis, value)
            elif joint.joint_type == "prismatic":
                axis = joint.axis / max(np.linalg.norm(joint.axis), 1e-9)
                motion[:3, 3] = axis * value
            poses[joint.child] = poses[joint.parent] @ joint.origin @ motion
            progressed = True
        if not progressed:
            missing = ", ".join(j.child for j in next_unresolved)
            raise RuntimeError(f"Could not resolve URDF tree for links: {missing}")
        unresolved = next_unresolved

    return poses


def load_collision_spheres(path: Path) -> dict[str, list[dict[str, float | list[float]]]]:
    data = yaml.safe_load(path.read_text())
    return data.get("collision_spheres", data)


def sphere_surface(center: np.ndarray, radius: float, resolution: int = 14) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution // 2 + 2)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def draw_meshes(ax, visuals: list[VisualMesh], poses: dict[str, np.ndarray], max_faces_per_mesh: int = 1200) -> list[np.ndarray]:
    points: list[np.ndarray] = []
    if trimesh is None:
        return points
    for visual in visuals:
        if visual.link not in poses or not visual.path.exists():
            continue
        try:
            mesh = trimesh.load_mesh(visual.path, force="mesh", process=False)
        except Exception:
            continue
        if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
            continue
        vertices = np.asarray(mesh.vertices, dtype=float) * visual.scale
        faces = np.asarray(mesh.faces, dtype=int)
        world_vertices = transform_points(poses[visual.link] @ visual.origin, vertices)
        points.append(world_vertices)

        if len(faces) > max_faces_per_mesh:
            step = max(1, len(faces) // max_faces_per_mesh)
            faces = faces[::step]
        collection = Poly3DCollection(
            world_vertices[faces],
            facecolor=(0.7, 0.7, 0.7, 0.12),
            edgecolor=(0.28, 0.28, 0.28, 0.08),
            linewidth=0.15,
        )
        ax.add_collection3d(collection)
    return points


def set_axes_equal(ax, points: np.ndarray, margin: float = 0.07) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0 + margin)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def draw_scene(
    output_path: Path,
    poses: dict[str, np.ndarray],
    visuals: list[VisualMesh],
    collision_spheres: dict[str, list[dict[str, float | list[float]]]],
    title: str,
    elev: float,
    azim: float,
    show_labels: bool = True,
    show_mesh: bool = True,
    sphere_resolution: int = 14,
    figsize: tuple[float, float] = (11, 9),
    dpi: int = 160,
) -> None:
    colors = plt.get_cmap("tab10")
    link_names = list(collision_spheres)
    link_color = {link: colors(i % 10) for i, link in enumerate(link_names)}

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    all_points: list[np.ndarray] = []
    if show_mesh:
        all_points.extend(draw_meshes(ax, visuals, poses))

    for link in link_names:
        if link not in poses:
            continue
        link_centers = []
        for sphere in collision_spheres[link]:
            center_local = np.asarray(sphere["center"], dtype=float).reshape(1, 3)
            radius = float(sphere["radius"])
            center_world = transform_points(poses[link], center_local)[0]
            x, y, z = sphere_surface(center_world, radius, resolution=sphere_resolution)
            ax.plot_surface(x, y, z, color=link_color[link], alpha=0.35, linewidth=0)
            ax.scatter([center_world[0]], [center_world[1]], [center_world[2]], color=link_color[link], s=8)
            link_centers.append(center_world)
            all_points.append(
                np.vstack(
                    [
                        center_world - radius,
                        center_world + radius,
                    ]
                )
            )
        if show_labels and link_centers:
            label_pos = np.mean(np.vstack(link_centers), axis=0)
            ax.text(label_pos[0], label_pos[1], label_pos[2], link, color=link_color[link], fontsize=8)

    if not all_points:
        all_points.append(np.zeros((1, 3)))
    all_points_np = np.vstack(all_points)
    set_axes_equal(ax, all_points_np)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def build_summary(poses: dict[str, np.ndarray], collision_spheres: dict[str, list[dict[str, float | list[float]]]]) -> dict:
    per_link = {}
    all_bounds = []
    for link, spheres in collision_spheres.items():
        centers = []
        radii = []
        for sphere in spheres:
            center = transform_points(poses[link], np.asarray(sphere["center"], dtype=float).reshape(1, 3))[0]
            radius = float(sphere["radius"])
            centers.append(center)
            radii.append(radius)
            all_bounds.append(center - radius)
            all_bounds.append(center + radius)
        centers_np = np.vstack(centers) if centers else np.zeros((0, 3))
        per_link[link] = {
            "count": len(spheres),
            "radius_min_m": min(radii) if radii else None,
            "radius_max_m": max(radii) if radii else None,
            "world_center_mean": centers_np.mean(axis=0).round(6).tolist() if len(centers_np) else None,
        }
    bounds = np.vstack(all_bounds) if all_bounds else np.zeros((1, 3))
    return {
        "urdf": str(URDF_PATH),
        "collision_spheres": str(SPHERE_PATH),
        "pose": "home_q_zero",
        "total_spheres": sum(v["count"] for v in per_link.values()),
        "links_with_spheres": list(per_link),
        "links_without_spheres_note": "URDF links such as camera_base/camera/link7/link8 have no separate entries in collision_X5A.yml.",
        "world_bbox_min": bounds.min(axis=0).round(6).tolist(),
        "world_bbox_max": bounds.max(axis=0).round(6).tolist(),
        "per_link": per_link,
    }


def write_orbit_video(
    output_path: Path,
    poses: dict[str, np.ndarray],
    visuals: list[VisualMesh],
    collision_spheres: dict[str, list[dict[str, float | list[float]]]],
    frames: int = 30,
) -> bool:
    if imageio is None:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        frame_paths = []
        for i in range(frames):
            frame = Path(tmp) / f"frame_{i:03d}.png"
            draw_scene(
                frame,
                poses,
                visuals,
                collision_spheres,
                "ARX-X5 collision spheres, home q=0",
                elev=24,
                azim=45 + 360.0 * i / frames,
                show_labels=False,
                show_mesh=False,
                sphere_resolution=10,
                figsize=(7, 5),
                dpi=110,
            )
            frame_paths.append(frame)
        images = [imageio.imread(path) for path in frame_paths]
        try:
            imageio.mimsave(output_path, images, fps=12)
        except Exception:
            gif_path = output_path.with_suffix(".gif")
            imageio.mimsave(gif_path, images, fps=8)
            return gif_path.exists()
    return output_path.exists()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    links, joints, visuals = parse_urdf(URDF_PATH)
    poses = compute_link_poses(links, joints, q={})
    spheres = load_collision_spheres(SPHERE_PATH)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    view_specs = [
        ("arx_x5_collision_spheres_home_iso.png", "ARX-X5 collision spheres, home q=0, iso", 24, 45),
        ("arx_x5_collision_spheres_home_front.png", "ARX-X5 collision spheres, home q=0, front", 8, 0),
        ("arx_x5_collision_spheres_home_top.png", "ARX-X5 collision spheres, home q=0, top", 90, -90),
    ]
    for filename, title, elev, azim in view_specs:
        draw_scene(out_dir / filename, poses, visuals, spheres, title, elev, azim)

    summary = build_summary(poses, spheres)
    (out_dir / "collision_sphere_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not args.no_video:
        write_orbit_video(out_dir / "arx_x5_collision_spheres_home_orbit.mp4", poses, visuals, spheres)

    print(json.dumps(summary, indent=2))
    print(f"wrote visualizations to {out_dir}")


if __name__ == "__main__":
    main()
