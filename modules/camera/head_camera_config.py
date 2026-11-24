"""本实验使用俯视桌面的头部相机；共享历史标定文件不改写。"""
from copy import deepcopy
import numpy as np
import sapien


def head_configuration(controllers, pitch_degrees=45.0):
    spec = deepcopy(controllers["camera"]["definitions"]["head"])
    gl_from_ros = np.array([[0, -1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
    original = np.asarray(spec["xform_rows"]).T @ gl_from_ros
    angle = np.deg2rad(pitch_degrees)
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    updated = original.copy()
    updated[:3, :3] = original[:3, :3] @ rotation
    spec["xform_rows"] = (updated @ np.linalg.inv(gl_from_ros)).T.tolist()
    record = {
        "camera_role": "head", "mount": controllers["camera"]["roles"]["head"],
        "change": "rotate head camera down by 45 degrees about its local SAPIEN +Y axis",
        "downward_pitch_degrees": pitch_degrees,
        "position_and_intrinsics_unchanged": True,
        "original_local_sapien": original.tolist(), "local_sapien": updated.tolist(),
        "camera_definition": spec,
        "scope": "new point-cloud benchmark; historical shared calibration unchanged",
    }
    return {"head": {"pose": sapien.Pose(updated)}}, record
