from __future__ import annotations

import hashlib
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any

import yaml

from arx_sim.config import CONFIG_ROOT


def load_yaml_path(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_registry(config: dict[str, Any], lock: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    asset_root = Path(config["simulation"]["asset_root"])
    registry_path = asset_root / "registry.yaml"
    if not registry_path.is_file():
        return [f"asset registry missing: {registry_path}"]
    try:
        registry = load_yaml_path(registry_path)
    except (OSError, ValueError) as exc:
        return [f"invalid asset registry {registry_path}: {exc}"]
    if registry.get("format") != "coolas-asset-registry-v1":
        errors.append(f"unsupported asset registry format: {registry.get('format')!r}")
    sections = {"environment": "environments", "robot": "robots"}
    if config.get("task"):
        sections["task"] = "tasks"
    manifests: dict[str, dict[str, Any]] = {}
    for kind, section in sections.items():
        entry = lock.get("assets", {}).get(kind, {})
        asset_id, version = entry.get("id"), entry.get("version")
        registry_key = f"{asset_id}@{version}"
        manifest_rel = registry.get(section, {}).get(registry_key)
        if not manifest_rel:
            errors.append(f"locked {kind} absent from registry: {registry_key}")
            continue
        manifest_path = asset_root / str(manifest_rel)
        if not manifest_path.is_file():
            errors.append(f"asset manifest missing ({kind}): {manifest_path}")
            continue
        manifest = load_yaml_path(manifest_path)
        manifests[kind] = manifest
        if manifest.get("id") != asset_id or str(manifest.get("version")) != str(version):
            errors.append(f"asset manifest identity mismatch ({kind}): {manifest_path}")
        manifest_hash = manifest.get("sha256")
        if manifest_hash is not None and manifest_hash != entry.get("sha256"):
            errors.append(f"asset manifest checksum differs from lock ({kind})")

    task = manifests.get("task", {})
    environment = lock.get("assets", {}).get("environment", {})
    robot = lock.get("assets", {}).get("robot", {})
    if task:
        if task.get("environment") != f"{environment.get('id')}@{environment.get('version')}":
            errors.append("task manifest environment does not match assets lock")
        if task.get("robot") != f"{robot.get('id')}@{robot.get('version')}":
            errors.append("task manifest robot does not match assets lock")

    robot_manifest = manifests.get("robot", {})
    if robot_manifest:
        simulation = config["simulation"]
        controllers = config["controllers"]
        if robot_manifest.get("articulation_root") != simulation.get("articulation_root"):
            errors.append("configured articulation_root differs from robot manifest")
        expected_joints = {
            "left_arm": controllers["left_arm"]["joints"],
            "left_gripper": controllers["left_gripper"]["joints"],
            "right_arm": controllers["right_arm"]["joints"],
            "right_gripper": controllers["gripper"]["joints"],
            "lift": controllers["lift"]["joint"],
            "chassis": [controllers["chassis"][key] for key in ("x_joint", "y_joint", "yaw_joint")],
        }
        for name, expected in expected_joints.items():
            if robot_manifest.get("joints", {}).get(name) != expected:
                errors.append(f"configured {name} joints differ from robot manifest")
        configured_frames = controllers.get("frames", {})
        for frame_name in ("base_link", "lift_link", "arm_base_link", "ee_link", "gripper_tcp"):
            if robot_manifest.get("frames", {}).get(frame_name) != configured_frames.get(frame_name):
                errors.append(f"configured {frame_name} differs from robot manifest")
        for role, relative_path in controllers.get("camera", {}).get("roles", {}).items():
            if not isinstance(relative_path, str) or not relative_path.strip():
                errors.append(f"configured camera role {role} must provide a non-empty prim path")
        for side in ("right_arm", "left_arm"):
            ee_link = controllers[side].get("ee_link")
            if not isinstance(ee_link, str) or not ee_link.strip():
                errors.append(f"configured {side} must provide a non-empty ee_link")
            home = controllers[side].get("home_joint_positions_rad")
            if not isinstance(home, list) or len(home) != len(expected_joints[side]):
                errors.append(f"configured {side} home must provide one value per arm joint")
            elif any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in home):
                errors.append(f"configured {side} home values must be numeric")
        if controllers["right_arm"].get("ee_link") != configured_frames.get("ee_link"):
            errors.append("right arm EE controller frame differs from configured ee_link")
        if bool(robot_manifest.get("supports_linear_y")) != bool(controllers["chassis"].get("supports_linear_y")):
            errors.append("configured chassis linear_y capability differs from robot manifest")
    return errors


def validate_install_and_assets(config: dict[str, Any], *, check_port: bool = False) -> list[str]:
    errors: list[str] = []
    simulation = config["simulation"]
    isaac_root = Path(simulation["isaac_root"])
    simulator_lock = load_yaml_path(CONFIG_ROOT / "simulator.lock.yaml")["simulator"]
    version_file = isaac_root / simulator_lock["version_file"]
    if not (isaac_root / "python.sh").is_file():
        errors.append(f"Isaac Sim python.sh not found: {isaac_root / 'python.sh'}")
    if not version_file.is_file():
        errors.append(f"Isaac Sim version file not found: {version_file}")
    else:
        actual = version_file.read_text(encoding="utf-8").strip()
        if not actual.startswith(str(simulator_lock["version_prefix"])):
            errors.append(f"Isaac Sim version mismatch: expected {simulator_lock['version_prefix']}, got {actual}")
    ros = config.get("ros2", {})
    if bool(ros.get("enabled", True)):
        distro = str(ros.get("distro", "jazzy"))
        bundled_ros = isaac_root / "exts" / "isaacsim.ros2.core" / distro / "lib"
        if not bundled_ros.is_dir() and not shutil.which("ros2"):
            errors.append(f"neither system ROS 2 nor Isaac bundled {distro} libraries were found")

    asset_root = Path(simulation["asset_root"])
    lock = load_yaml_path(CONFIG_ROOT / "assets.lock.yaml")
    errors.extend(_validate_registry(config, lock))
    required_assets = {"environment", "robot"}
    if config.get("task"):
        required_assets.add("task")
    for name, entry in lock.get("assets", {}).items():
        if name not in required_assets:
            continue
        path = asset_root / entry["path"]
        if not path.is_file():
            errors.append(f"asset missing ({name}): {path}")
        elif entry.get("sha256") and sha256_file(path) != entry["sha256"]:
            errors.append(f"asset checksum mismatch ({name}): {path}")
    for key in ("environment", "robot"):
        if not Path(simulation[key]).is_file():
            errors.append(f"resolved simulation asset missing ({key}): {simulation[key]}")
    overlay = str(simulation.get("task_overlay", "")).strip()
    if overlay and not Path(overlay).is_file():
        errors.append(f"resolved simulation asset missing (task_overlay): {overlay}")
    if shutil.which("nvidia-smi") is None:
        errors.append("nvidia-smi not found")
    else:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                text=True,
                capture_output=True,
                timeout=5.0,
                check=True,
            )
            available = {}
            for line in result.stdout.splitlines():
                index, free_mb = [part.strip() for part in line.split(",", 1)]
                available[int(index)] = int(free_mb)
            selected = int(simulation.get("active_gpu", 0))
            minimum = int(simulation.get("minimum_free_gpu_memory_mb", 0))
            if selected not in available:
                errors.append(f"configured GPU {selected} not reported by nvidia-smi")
            elif available[selected] < minimum:
                errors.append(
                    f"GPU {selected} has {available[selected]} MiB free; at least {minimum} MiB is required"
                )
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            errors.append(f"could not inspect GPU memory: {exc}")
    if check_port:
        host = str(config["services"]["arx_ops_host"])
        bind_host = "0.0.0.0" if host in {"0.0.0.0", "::"} else host
        port = int(config["services"]["arx_ops_port"])
        with socket.socket() as sock:
            # Match uvicorn and do not reject a port solely because a clean
            # shutdown left client connections in TCP TIME_WAIT.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((bind_host, port))
            except OSError as exc:
                errors.append(f"arx-ops port {host}:{port} unavailable: {exc}")
        signal_port = int(config["webrtc"]["port"])
        stream_port = signal_port + 1
        if not 1 <= signal_port <= 65534:
            errors.append("webrtc port must be in 1..65534")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", signal_port))
            except OSError as exc:
                errors.append(f"WebRTC signaling TCP port {signal_port} unavailable: {exc}")
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            try:
                sock.bind(("0.0.0.0", stream_port))
            except OSError as exc:
                errors.append(f"WebRTC media UDP port {stream_port} unavailable: {exc}")
    return errors
