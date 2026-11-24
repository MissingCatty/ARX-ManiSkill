"""以官方 SDK 为独立参照核验标定和镜头数学，无实机或抓取动作。"""
import json
from pathlib import Path
import numpy as np
import pyrealsense2 as rs
from camera_imaging import camera_profiles, project_normalized, deproject_pixels, quantize_depth, write_profiles


def main():
    result = {}
    for role, spec in camera_profiles().items():
        k = spec["color_intrinsics"]
        intr = rs.intrinsics()
        intr.width, intr.height = k["width"], k["height"]
        intr.fx, intr.fy, intr.ppx, intr.ppy = k["fx"], k["fy"], k["cx"], k["cy"]
        intr.model = rs.distortion.inverse_brown_conrady
        intr.coeffs = k["coeffs"]
        uv = np.array([[x,y] for y in np.linspace(0,479,9) for x in np.linspace(0,639,11)])
        rays = deproject_pixels(uv, k)
        reference = np.array([rs.rs2_deproject_pixel_to_point(intr, p.tolist(), 1.)[:2] for p in uv])
        deprojection_error = float(np.max(np.abs(rays-reference)))
        xy = np.array([[x,y] for x in np.linspace(-.6,.6,9) for y in np.linspace(-.5,.5,9)])
        projected = project_normalized(xy, k)
        reference_uv = np.array([rs.rs2_project_point_to_pixel(intr, [*p, 1.]) for p in xy])
        projection_error = float(np.max(np.abs(projected-reference_uv)))
        assert deprojection_error < 1e-6, (role, deprojection_error)
        assert projection_error < 2e-4, (role, projection_error)
        values = np.array([np.nan, 0, .069, .07, .10014, .50, .60, 3.])
        codes = quantize_depth(values, spec)
        assert np.array_equal(codes[[0,1,2,3,4,5,6,7]], [0,0,0,700,1001,5000,6000,0])
        result[role] = {"deprojection_max_normalized_error": deprojection_error,
                        "projection_max_pixel_error": projection_error,
                        "checked_pixel_count": len(uv), "checked_ray_count": len(xy),
                        "depth_unit_and_invalid_checks_passed": True}
    config = write_profiles()
    out = config.parent / "camera_math_checks.json"
    out.write_text(json.dumps({"passed": True, "sdk_version": rs.__version__, "roles": result, "physical_trials": 0}, ensure_ascii=False, indent=2)+"\n")
    print(out)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
