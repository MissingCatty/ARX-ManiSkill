"""D405 标定、镜头映射和深度编码；投影公式与 librealsense rs.cpp 一致。"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json

import cv2
import numpy as np
import yaml

from arx_project.layout import ROOT, module_dir
SOURCE = module_dir("arx_core") / "arx-sim/config/controllers.yaml"
SDK_PROJECTION = "https://github.com/realsenseai/librealsense/blob/master/src/rs.cpp"
SDK_UNITS = "https://github.com/realsenseai/librealsense/blob/master/src/ds/d400/d400-device.cpp"


def camera_profiles():
    original = yaml.safe_load(SOURCE.read_text())["controllers"]["camera"]
    result = {}
    for role, source in original["definitions"].items():
        p = deepcopy(source)
        p.update(model="Intel RealSense D405", model_identity="supported by ARX data-pipeline documentation and project D405 capture implementation; live USB identity not available on GPU host",
                 raw_depth_scale_m=0.0001, raw_depth_scale_source=SDK_UNITS,
                 fps=30, aligned_depth_frame="color_optical",
                 ideal_depth_range_m=[0.07, 0.50],
                 minimum_depth_m=0.07,
                 application_maximum_depth_m=float(source["depth_valid_range_m"][1]),
                 render_clipping_m=[0.05, 10.0],
                 model_input_projection="rectified_pinhole_with_per_device_color_K",
                 model_input_distortion_coeffs=[0.0] * 5)
        result[role] = p
    return result


def intrinsic_matrix(spec):
    k = spec["color_intrinsics"] if "color_intrinsics" in spec else spec
    return np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]], dtype=float)


def project_normalized(xy, intr):
    """Undistorted rays -> raw color pixels (same ordering as rs2_project)."""
    x, y = np.moveaxis(np.asarray(xy, dtype=float), -1, 0)
    k1, k2, p1, p2, k3 = intr["coeffs"]
    r2 = x*x + y*y
    radial = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
    model = intr["distortion_model"]
    if model == "inverse_brown_conrady":
        xq, yq = x*radial, y*radial
        xd = xq + 2*p1*xq*yq + p2*(r2 + 2*xq*xq)
        yd = yq + 2*p2*xq*yq + p1*(r2 + 2*yq*yq)
    elif model == "brown_conrady":
        xd = x*radial + 2*p1*x*y + p2*(r2 + 2*x*x)
        yd = y*radial + 2*p2*x*y + p1*(r2 + 2*y*y)
    else:
        raise ValueError("Unsupported lens model: " + model)
    return np.stack([xd*intr["fx"] + intr["cx"], yd*intr["fy"] + intr["cy"]], axis=-1)


def deproject_pixels(uv, intr):
    """Raw color pixels -> normalized rays; librealsense uses ten iterations."""
    uv = np.asarray(uv, dtype=float)
    xo = (uv[..., 0] - intr["cx"]) / intr["fx"]
    yo = (uv[..., 1] - intr["cy"]) / intr["fy"]
    x, y = xo.copy(), yo.copy()
    k1, k2, p1, p2, k3 = intr["coeffs"]
    for _ in range(10):
        r2 = x*x + y*y
        inverse_radial = 1 / (1 + ((k3*r2 + k2)*r2 + k1)*r2)
        if intr["distortion_model"] == "inverse_brown_conrady":
            xq, yq = x/inverse_radial, y/inverse_radial
        elif intr["distortion_model"] == "brown_conrady":
            xq, yq = x, y
        else:
            raise ValueError("Unsupported lens model")
        dx = 2*p1*xq*yq + p2*(r2 + 2*xq*xq)
        dy = 2*p2*xq*yq + p1*(r2 + 2*yq*yq)
        x, y = (xo-dx)*inverse_radial, (yo-dy)*inverse_radial
    return np.stack([x, y], axis=-1)


def lens_maps(spec):
    intr = spec["color_intrinsics"]
    y, x = np.indices((intr["height"], intr["width"]), dtype=float)
    pixels = np.stack([x, y], -1)
    ideal = np.stack([(x-intr["cx"])/intr["fx"], (y-intr["cy"])/intr["fy"]], -1)
    # Maps are destination-pixel -> source-pixel, as required by cv2.remap.
    raw_from_rectified = deproject_pixels(pixels, intr)
    raw_from_rectified[..., 0] = raw_from_rectified[..., 0]*intr["fx"] + intr["cx"]
    raw_from_rectified[..., 1] = raw_from_rectified[..., 1]*intr["fy"] + intr["cy"]
    rectified_from_raw = project_normalized(ideal, intr)
    return raw_from_rectified.astype(np.float32), rectified_from_raw.astype(np.float32)


def quantize_depth(depth_m, spec):
    depth_m = np.asarray(depth_m, dtype=float)
    valid = np.isfinite(depth_m) & (depth_m >= spec["minimum_depth_m"]) & (depth_m <= spec["application_maximum_depth_m"])
    raw = np.zeros(depth_m.shape, dtype=np.uint16)
    raw[valid] = np.clip(np.rint(depth_m[valid]/spec["raw_depth_scale_m"]), 1, 65535).astype(np.uint16)
    return raw


def sensor_and_model_frames(rgb_ideal, depth_m_ideal, spec, maps=None):
    """Save a distorted raw-sensor view and rectify it for pinhole model inputs."""
    to_raw, to_rectified = lens_maps(spec) if maps is None else maps
    raw_rgb = cv2.remap(rgb_ideal, to_raw[..., 0], to_raw[..., 1], cv2.INTER_LINEAR)
    raw_z = cv2.remap(np.asarray(depth_m_ideal, dtype=np.float32), to_raw[..., 0], to_raw[..., 1], cv2.INTER_NEAREST)
    raw_depth = quantize_depth(raw_z, spec)
    rgb = cv2.remap(raw_rgb, to_rectified[..., 0], to_rectified[..., 1], cv2.INTER_LINEAR)
    depth = cv2.remap(raw_depth, to_rectified[..., 0], to_rectified[..., 1], cv2.INTER_NEAREST).astype(np.float32)*spec["raw_depth_scale_m"]
    return {"raw_rgb": raw_rgb, "raw_depth": raw_depth, "rgb": rgb, "depth_m": depth}


def write_profiles():
    data = {"profiles": camera_profiles(), "calibration_source": str(SOURCE),
            "calibration_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "source_urls": ["https://github.com/ARXroboticsX/ROS2_AC-one_Play", "https://arx-global.us/faq/", "https://www.realsenseai.com/products/stereo-depth-camera-d405/", SDK_PROJECTION, SDK_UNITS],
            "note": "per-device calibration copied from existing project; no live hardware re-calibration; head mounting override stored separately; stereo-matching noise not simulated"}
    p = ROOT / "simulation/cameras/camera_profiles.json"
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    return p


if __name__ == "__main__":
    print(write_profiles())
