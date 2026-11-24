#!/usr/bin/env python3
"""Project M5 reconstructed geometry back into its RGB-D camera views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from maniskill_curobo_real.multiview_instance_scene import load_multiview_instances
from maniskill_curobo_real.scene_builder import (
    DEFAULT_WORKSPACE_BOUNDS,
    OPENCV_TO_SAPIEN_CAMERA,
    _voxelize_closed_instance,
    depth_to_meters,
    deterministic_downsample,
    transform_points,
)


DEFAULT_ROOT = Path("maniskill_curobo_real/runs/pickclutter_m5_instance_esdf_table_seed1_200")
DEFAULT_STAGE = "m5_multiview_instance_voxel_esdf_table"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--stage", default=DEFAULT_STAGE)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-points-per-instance", type=int, default=2500)
    parser.add_argument("--max-voxels", type=int, default=9000)
    parser.add_argument("--depth-tolerance", type=float, default=0.035)
    parser.add_argument(
        "--high-contrast",
        action="store_true",
        help="Dim RGB backgrounds and draw larger outlined markers for manual inspection.",
    )
    parser.add_argument(
        "--show-target",
        action="store_true",
        help="Also project the excluded target instance as grey points.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.expanduser().resolve()
    seed_name = f"seed{int(args.seed):03d}"
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / "visualization_three_view_esdf" / seed_name / "reprojection"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    instances, input_metadata = load_multiview_instances(
        input_root=root / "multiview_inputs",
        seed=int(args.seed),
        workspace_bounds=DEFAULT_WORKSPACE_BOUNDS,
    )
    views = load_views(root / "multiview_inputs" / seed_name)
    scene_dir = root / str(args.stage) / seed_name / "real_scene"
    metadata_path = scene_dir / "curobo_scene_metadata.json"
    voxel_path = scene_dir / "curobo_scene_voxel.npz"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    colors = instance_colors(instances, high_contrast=bool(args.high_contrast))
    voxel_sets = load_per_instance_voxels(
        voxel_path,
        instances=instances,
        metadata=metadata,
        colors=colors,
        max_voxels=int(args.max_voxels),
    )
    suffix = "_high_visibility" if args.high_contrast else ""
    instance_path = output_dir / f"{seed_name}_{args.stage}_reprojected_instances{suffix}.png"
    voxel_overlay_path = output_dir / f"{seed_name}_{args.stage}_reprojected_esdf_voxels{suffix}.png"
    panel_path = output_dir / f"{seed_name}_{args.stage}_reprojection_panel{suffix}.png"

    render_overlay_grid(
        output_path=instance_path,
        title=f"{seed_name}: fused instance points reprojected to each RGB view",
        views=views,
        instances=instances,
        colors=colors,
        voxel_sets=None,
        max_points_per_instance=int(args.max_points_per_instance),
        depth_tolerance=float(args.depth_tolerance),
        show_target=bool(args.show_target),
        high_contrast=bool(args.high_contrast),
    )
    render_overlay_grid(
        output_path=voxel_overlay_path,
        title=f"{seed_name}: occupied ESDF voxels reprojected to each RGB view",
        views=views,
        instances=[],
        colors={},
        voxel_sets=voxel_sets,
        max_points_per_instance=int(args.max_points_per_instance),
        depth_tolerance=float(args.depth_tolerance),
        show_target=bool(args.show_target),
        high_contrast=bool(args.high_contrast),
    )
    render_panel(
        output_path=panel_path,
        seed_name=seed_name,
        stage=str(args.stage),
        views=views,
        instances=instances,
        colors=colors,
        voxel_sets=voxel_sets,
        max_points_per_instance=int(args.max_points_per_instance),
        depth_tolerance=float(args.depth_tolerance),
        show_target=bool(args.show_target),
        metadata=metadata,
        input_metadata=input_metadata,
        high_contrast=bool(args.high_contrast),
    )

    summary = {
        "seed": int(args.seed),
        "stage": str(args.stage),
        "root": str(root),
        "input_views": [view["name"] for view in views],
        "n_instances": len(instances),
        "n_non_target_instances": int(sum(not item.is_task_target for item in instances)),
        "occupied_voxels_loaded": int(sum(item["points"].shape[0] for item in voxel_sets)),
        "voxel_sets": [
            {
                "segmentation_id": int(item["segmentation_id"]),
                "name": str(item["name"]),
                "shown_voxels": int(item["points"].shape[0]),
            }
            for item in voxel_sets
        ],
        "depth_tolerance_m": float(args.depth_tolerance),
        "outputs": {
            "instances": str(instance_path),
            "esdf_voxels": str(voxel_overlay_path),
            "panel": str(panel_path),
        },
    }
    summary_path = output_dir / f"{seed_name}_{args.stage}_reprojection_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**summary, "summary": str(summary_path)}, ensure_ascii=False, indent=2))
    return 0


def load_views(seed_dir: Path) -> list[dict]:
    views = []
    for metadata_path in sorted(seed_dir.glob("view_*/view_metadata.json")):
        view_dir = metadata_path.parent
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        rgb_path = view_dir / "rgb.png"
        if rgb_path.is_file():
            rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
        else:
            with np.load(view_dir / "rgbd.npz") as payload:
                rgb = np.asarray(payload["rgb"], dtype=np.uint8)
        with np.load(view_dir / "rgbd.npz") as payload:
            depth = depth_to_meters(np.asarray(payload["depth"], dtype=np.float32))
            camera_matrix = np.asarray(payload["cam_K"], dtype=np.float64).reshape(3, 3)
        views.append(
            {
                "name": metadata.get("view", {}).get("name", view_dir.name),
                "role": metadata.get("view", {}).get("role", ""),
                "rgb": rgb,
                "depth_m": depth,
                "camera_matrix": camera_matrix,
                "base_from_camera": np.asarray(metadata["base_from_camera"], dtype=np.float64).reshape(4, 4),
            }
        )
    if not views:
        raise FileNotFoundError(f"No view metadata found in {seed_dir}")
    return views


def load_per_instance_voxels(
    voxel_path: Path,
    *,
    instances,
    metadata: dict,
    colors: dict[int, tuple[float, float, float, float]],
    max_voxels: int,
) -> list[dict]:
    with np.load(voxel_path) as payload:
        sdf = np.asarray(payload["feature_tensor"], dtype=np.float32)
        center = np.asarray(payload["voxel_center"], dtype=np.float64).reshape(3)
        voxel_size = float(payload["voxel_size"])
    axes = [
        center[axis]
        + (np.arange(sdf.shape[axis], dtype=np.float64) - (sdf.shape[axis] - 1) / 2.0)
        * voxel_size
        for axis in range(3)
    ]
    obstacle_ids = [
        int(record["segmentation_id"])
        for record in metadata.get("obstacle_records", [])
        if "segmentation_id" in record
    ]
    instances_by_id = {int(instance.segmentation_id): instance for instance in instances}
    per_instance_limit = max(1, int(max_voxels) // max(1, len(obstacle_ids)))
    voxel_sets = []
    for segmentation_id in obstacle_ids:
        instance = instances_by_id.get(segmentation_id)
        if instance is None:
            continue
        occupied = np.zeros_like(sdf, dtype=bool)
        sampled = deterministic_downsample(
            instance.points_base,
            int(metadata.get("max_points_per_instance", 1500)),
        )
        filled, _ = _voxelize_closed_instance(
            occupied=occupied,
            axes=axes,
            points=sampled,
            voxel_size=voxel_size,
        )
        if filled <= 0:
            continue
        idx = np.argwhere(occupied)
        points = np.column_stack([axes[0][idx[:, 0]], axes[1][idx[:, 1]], axes[2][idx[:, 2]]])
        voxel_sets.append(
            {
                "segmentation_id": segmentation_id,
                "name": short_name(instance.actor_name),
                "points": deterministic_downsample(points, per_instance_limit),
                "color": colors.get(segmentation_id, (1.0, 0.1, 0.05, 0.65)),
            }
        )
    return voxel_sets


def instance_colors(instances, *, high_contrast: bool = False) -> dict[int, tuple[float, float, float, float]]:
    if high_contrast:
        palette = [
            (1.00, 0.05, 0.05, 0.96),
            (0.00, 0.85, 1.00, 0.96),
            (1.00, 0.85, 0.00, 0.96),
            (0.95, 0.00, 1.00, 0.96),
            (0.15, 1.00, 0.15, 0.96),
            (1.00, 0.45, 0.00, 0.96),
            (0.20, 0.35, 1.00, 0.96),
            (1.00, 1.00, 1.00, 0.96),
            (0.55, 1.00, 0.00, 0.96),
            (1.00, 0.35, 0.60, 0.96),
        ]
        colors = {}
        color_index = 0
        for instance in instances:
            if instance.is_task_target:
                colors[int(instance.segmentation_id)] = (0.75, 0.75, 0.75, 0.50)
            else:
                colors[int(instance.segmentation_id)] = palette[color_index % len(palette)]
                color_index += 1
        return colors
    cmap = plt.get_cmap("tab20")
    colors = {}
    color_index = 0
    for instance in instances:
        if instance.is_task_target:
            colors[int(instance.segmentation_id)] = (0.55, 0.55, 0.55, 0.32)
        else:
            colors[int(instance.segmentation_id)] = cmap(color_index % 20)
            color_index += 1
    return colors


def render_panel(
    *,
    output_path: Path,
    seed_name: str,
    stage: str,
    views: list[dict],
    instances,
    colors: dict[int, tuple[float, float, float, float]],
    voxel_sets: list[dict],
    max_points_per_instance: int,
    depth_tolerance: float,
    show_target: bool,
    metadata: dict,
    input_metadata: dict,
    high_contrast: bool,
) -> None:
    n = len(views)
    fig, axes = plt.subplots(2, n, figsize=(6.2 * n, 9.8), dpi=160, squeeze=False)
    for col, view in enumerate(views):
        draw_instance_overlay(
            axes[0, col],
            view=view,
            instances=instances,
            colors=colors,
            max_points_per_instance=max_points_per_instance,
            depth_tolerance=depth_tolerance,
            show_target=show_target,
            high_contrast=high_contrast,
        )
        draw_voxel_overlay(
            axes[1, col],
            view=view,
            voxel_sets=voxel_sets,
            depth_tolerance=depth_tolerance,
            high_contrast=high_contrast,
        )
    handles, labels = collect_legend_entries(instances=instances, colors=colors, voxel_sets=voxel_sets, show_target=show_target)
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(5, len(handles)), fontsize=8 if high_contrast else 7)
    fig.suptitle(
        (
            f"{seed_name} {stage}: M5 reconstruction reprojected into the 3 source RGB-D views\n"
            f"instances={metadata.get('n_instances_seen')} obstacles={metadata.get('n_pointcloud_obstacles')} "
            f"occupied_voxels={metadata.get('occupied_voxels')} "
            f"non_target_points={input_metadata.get('non_target_instance_points_accumulated_over_views')}"
        ),
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def render_overlay_grid(
    *,
    output_path: Path,
    title: str,
    views: list[dict],
    instances,
    colors: dict[int, tuple[float, float, float, float]],
    voxel_sets: list[dict] | None,
    max_points_per_instance: int,
    depth_tolerance: float,
    show_target: bool,
    high_contrast: bool,
) -> None:
    fig, axes = plt.subplots(1, len(views), figsize=(6.2 * len(views), 5.2), dpi=160, squeeze=False)
    for col, view in enumerate(views):
        if voxel_sets is None:
            draw_instance_overlay(
                axes[0, col],
                view=view,
                instances=instances,
                colors=colors,
                max_points_per_instance=max_points_per_instance,
                depth_tolerance=depth_tolerance,
                show_target=show_target,
                high_contrast=high_contrast,
            )
        else:
            draw_voxel_overlay(
                axes[0, col],
                view=view,
                voxel_sets=voxel_sets,
                depth_tolerance=depth_tolerance,
                high_contrast=high_contrast,
            )
    handles, labels = collect_legend_entries(
        instances=instances,
        colors=colors,
        voxel_sets=voxel_sets or [],
        show_target=show_target,
    )
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(5, len(handles)), fontsize=8 if high_contrast else 7)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def draw_instance_overlay(
    ax,
    *,
    view: dict,
    instances,
    colors: dict[int, tuple[float, float, float, float]],
    max_points_per_instance: int,
    depth_tolerance: float,
    show_target: bool,
    high_contrast: bool,
) -> None:
    rgb = np.asarray(view["rgb"], dtype=np.uint8)
    h, w = rgb.shape[:2]
    ax.imshow(dim_rgb(rgb) if high_contrast else rgb)
    for instance in instances:
        if instance.is_task_target and not show_target:
            continue
        points = deterministic_downsample(instance.points_base, max(1, int(max_points_per_instance)))
        if points.size == 0:
            continue
        uv, visible, depth_consistent = project_base_points_to_view(points, view=view, image_shape=(h, w), depth_tolerance=depth_tolerance)
        if not np.any(visible):
            continue
        color = colors[int(instance.segmentation_id)]
        faint = visible & ~depth_consistent
        bright = visible & depth_consistent
        faint_size = 4.0 if high_contrast else 0.25
        bright_size = 12.0 if high_contrast else 0.7
        if np.any(faint):
            if high_contrast:
                ax.scatter(uv[faint, 0], uv[faint, 1], s=faint_size + 7.0, color=(0, 0, 0, 0.42), linewidths=0)
            ax.scatter(uv[faint, 0], uv[faint, 1], s=faint_size, color=color, alpha=0.32 if high_contrast else 0.18, linewidths=0)
        if np.any(bright):
            if high_contrast:
                ax.scatter(uv[bright, 0], uv[bright, 1], s=bright_size + 10.0, color=(0, 0, 0, 0.70), linewidths=0)
            ax.scatter(
                uv[bright, 0],
                uv[bright, 1],
                s=bright_size,
                color=color,
                alpha=0.96 if high_contrast else 0.72,
                linewidths=0,
                label=short_name(instance.actor_name),
            )
    ax.set_title(f"{view['name']}\ninstance cloud: bright=depth-consistent")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")


def draw_voxel_overlay(ax, *, view: dict, voxel_sets: list[dict], depth_tolerance: float, high_contrast: bool) -> None:
    rgb = np.asarray(view["rgb"], dtype=np.uint8)
    h, w = rgb.shape[:2]
    ax.imshow(dim_rgb(rgb) if high_contrast else rgb)
    for voxel_set in voxel_sets:
        voxels = np.asarray(voxel_set["points"], dtype=np.float64).reshape(-1, 3)
        if not voxels.size:
            continue
        uv, visible, depth_consistent = project_base_points_to_view(
            voxels,
            view=view,
            image_shape=(h, w),
            depth_tolerance=max(depth_tolerance, 0.06),
        )
        faint = visible & ~depth_consistent
        bright = visible & depth_consistent
        base_color = tuple(float(v) for v in voxel_set["color"][:3])
        faint_size = 5.0 if high_contrast else 0.35
        bright_size = 16.0 if high_contrast else 1.0
        if np.any(faint):
            if high_contrast:
                ax.scatter(
                    uv[faint, 0],
                    uv[faint, 1],
                    s=faint_size + 7.0,
                    color=(0, 0, 0, 0.40),
                    linewidths=0,
                )
            ax.scatter(
                uv[faint, 0],
                uv[faint, 1],
                s=faint_size,
                color=(*base_color, 0.30 if high_contrast else 0.16),
                linewidths=0,
            )
        if np.any(bright):
            if high_contrast:
                ax.scatter(
                    uv[bright, 0],
                    uv[bright, 1],
                    s=bright_size + 11.0,
                    color=(0, 0, 0, 0.74),
                    linewidths=0,
                )
            ax.scatter(
                uv[bright, 0],
                uv[bright, 1],
                s=bright_size,
                color=(*base_color, 0.96 if high_contrast else 0.68),
                linewidths=0,
                label=str(voxel_set["name"]),
            )
    ax.set_title(f"{view['name']}\nESDF occupied voxels")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")


def dim_rgb(rgb: np.ndarray) -> np.ndarray:
    data = np.asarray(rgb, dtype=np.float32)
    data = data * 0.32 + 18.0
    return np.clip(data, 0, 255).astype(np.uint8)


def collect_legend_entries(
    *,
    instances,
    colors: dict[int, tuple[float, float, float, float]],
    voxel_sets: list[dict],
    show_target: bool,
) -> tuple[list, list[str]]:
    entries: list[tuple[str, tuple[float, float, float, float]]] = []
    if instances:
        for instance in instances:
            if instance.is_task_target and not show_target:
                continue
            entries.append((short_name(instance.actor_name), colors[int(instance.segmentation_id)]))
    else:
        for voxel_set in voxel_sets:
            color = tuple(float(v) for v in voxel_set["color"])
            entries.append((str(voxel_set["name"]), color))
    handles = [
        plt.Line2D([0], [0], marker="o", color="black", markerfacecolor=color, markersize=7, linestyle="")
        for _, color in entries
    ]
    labels = [name for name, _ in entries]
    return handles, labels


def project_base_points_to_view(
    points_base: np.ndarray,
    *,
    view: dict,
    image_shape: tuple[int, int],
    depth_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points_base, dtype=np.float64).reshape(-1, 3)
    camera_from_base = np.linalg.inv(np.asarray(view["base_from_camera"], dtype=np.float64).reshape(4, 4))
    points_camera = transform_points(camera_from_base, points)
    points_cv = points_camera @ OPENCV_TO_SAPIEN_CAMERA.T
    z = points_cv[:, 2]
    k = np.asarray(view["camera_matrix"], dtype=np.float64).reshape(3, 3)
    finite = np.isfinite(points_cv).all(axis=1) & (z > 1e-6)
    uv = np.zeros((points.shape[0], 2), dtype=np.float64)
    uv[:, 0] = k[0, 0] * points_cv[:, 0] / np.where(finite, z, 1.0) + k[0, 2]
    uv[:, 1] = k[1, 1] * points_cv[:, 1] / np.where(finite, z, 1.0) + k[1, 2]
    h, w = image_shape
    in_image = (
        finite
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < float(w))
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < float(h))
    )
    depth_consistent = np.zeros((points.shape[0],), dtype=bool)
    if np.any(in_image):
        xs = np.clip(np.rint(uv[:, 0]).astype(np.int64), 0, w - 1)
        ys = np.clip(np.rint(uv[:, 1]).astype(np.int64), 0, h - 1)
        observed = np.asarray(view["depth_m"], dtype=np.float64)[ys, xs]
        depth_consistent = in_image & np.isfinite(observed) & (observed > 1e-6) & (np.abs(observed - z) <= float(depth_tolerance))
    return uv, in_image, depth_consistent


def short_name(name: str, max_len: int = 22) -> str:
    text = str(name).replace("scene-0_set_0_", "")
    return text if len(text) <= max_len else text[: max_len - 1] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
