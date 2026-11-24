from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from arx_shared.schemas import CameraIntrinsics, CaptureResult


def write_rgbd_bundle(
    *, rgb: np.ndarray, depth_m: np.ndarray, intrinsics_matrix: np.ndarray,
    output_dir: Path, camera_id: str | None, sim_time: float, frame_id: str,
    depth_scale: float = 0.001, serial_number: str | None = None,
) -> CaptureResult:
    rgb = np.asarray(rgb)
    depth = np.asarray(depth_m, dtype=np.float32)
    intrinsics = np.asarray(intrinsics_matrix, dtype=float)
    if rgb.ndim != 3 or rgb.shape[2] < 3 or rgb.dtype != np.uint8:
        raise ValueError(f"RGB frame must be uint8 HxWx3/4, got {rgb.dtype} {rgb.shape}")
    if depth.shape != rgb.shape[:2]:
        raise ValueError(f"RGB/depth dimensions differ: {rgb.shape[:2]} vs {depth.shape}")
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError("camera intrinsics must be a finite 3x3 matrix")
    if depth_scale <= 0:
        raise ValueError("depth_scale must be positive")
    valid = np.isfinite(depth) & (depth > 0)
    depth_raw = np.zeros(depth.shape, dtype=np.uint16)
    depth_raw[valid] = np.clip(np.rint(depth[valid] / depth_scale), 1, 65535).astype(np.uint16)

    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp = output_dir.parent / f".{output_dir.name}.{uuid4().hex}.tmp"
    temp.mkdir()
    try:
        Image.fromarray(rgb[:, :, :3], mode="RGB").save(temp / "color.png")
        Image.fromarray(depth_raw).save(temp / "depth.png")
        height, width = depth.shape
        intrinsics_data = {
            "width": width, "height": height,
            "fx": float(intrinsics[0, 0]), "fy": float(intrinsics[1, 1]),
            "ppx": float(intrinsics[0, 2]), "ppy": float(intrinsics[1, 2]),
            "model": "pinhole", "coeffs": [],
        }
        params = {
            "intrinsics": intrinsics_data, "depth_scale": depth_scale,
            "serial_number": serial_number or camera_id, "camera_id": camera_id,
            "frame_id": frame_id, "sim_time": sim_time,
            "depth_semantics": "linear_distance_to_image_plane_m",
            "color_encoding": "rgb8", "depth_encoding": "16UC1",
        }
        (temp / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
        if output_dir.exists():
            if any(output_dir.iterdir()):
                raise FileExistsError(f"capture output directory is not empty: {output_dir}")
            output_dir.rmdir()
        os.replace(temp, output_dir)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    model = CameraIntrinsics(**intrinsics_data)
    return CaptureResult(
        color_path=str(output_dir / "color.png"), depth_path=str(output_dir / "depth.png"),
        params_path=str(output_dir / "params.json"), intrinsics=model,
        depth_scale=depth_scale, serial_number=serial_number or camera_id, camera_id=camera_id,
    )
