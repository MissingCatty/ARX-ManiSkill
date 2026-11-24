from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MODULE_ROOT.parent
CONFIG_ROOT = MODULE_ROOT / "config"
TASK_CONFIG_ROOT = CONFIG_ROOT / "tasks"


def _load(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return data


def _merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve(value: str | Path, base: Path = REPOSITORY_ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def available_task_layouts() -> list[str]:
    """Return task layout names that can be selected at simulator startup."""
    if not TASK_CONFIG_ROOT.is_dir():
        return []
    return sorted(path.stem for path in TASK_CONFIG_ROOT.glob("*.yaml") if path.is_file())


def _load_task_layout(name: str) -> dict[str, Any]:
    if not name or Path(name).name != name or not name.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"invalid task layout name: {name!r}")
    path = TASK_CONFIG_ROOT / f"{name}.yaml"
    if not path.is_file():
        available = ", ".join(available_task_layouts()) or "none"
        raise ValueError(f"unknown task layout {name!r}; available: {available}")
    layout = _load(path)
    overlay = str(layout.get("simulation", {}).get("task_overlay", "")).strip()
    if overlay:
        layout["simulation"]["task_overlay"] = str(_resolve(overlay, path.parent))
    layout["task_layout"] = {**layout.get("task_layout", {}), "name": name}
    return layout


def create_session_dir(run_root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = run_root / f"sim_{stamp}"
    suffix = 1
    while candidate.exists():
        candidate = run_root / f"sim_{stamp}_{suffix}"
        suffix += 1
    (candidate / "logs").mkdir(parents=True)
    (candidate / "artifacts").mkdir()
    return candidate


def load_sim_config(
    *,
    headless: bool | None = None,
    active_gpu: int | None = None,
    env: Mapping[str, str] | None = None,
    run_dir: Path | None = None,
    webrtc_port: int | None = None,
    arx_ops_port: int | None = None,
    task: str | None = None,
) -> dict[str, Any]:
    """Resolve standalone simulator config without changing shared loaders."""
    environment = os.environ if env is None else env
    config = _load(REPOSITORY_ROOT / "configs" / "arx.yaml")
    config["calibration"] = _load(REPOSITORY_ROOT / "configs" / "calibration.yaml")
    for filename in ("sim.yaml", "runtime.yaml", "controllers.yaml", "topics.yaml"):
        config = _merge(config, _load(CONFIG_ROOT / filename))
    if task:
        config = _merge(config, _load_task_layout(task))

    simulation = config["simulation"]
    if headless is not None:
        simulation["headless"] = headless
    if active_gpu is not None:
        if active_gpu < 0:
            raise ValueError("active_gpu cannot be negative")
        simulation.update({"active_gpu": active_gpu, "physics_gpu": active_gpu})
    simulation["isaac_root"] = str(
        _resolve(environment.get("ISAAC_SIM_ROOT", simulation["isaac_root"]))
    )
    asset_root = _resolve(environment.get("ARX_SIM_ASSET_ROOT", simulation["asset_root"]))
    simulation["asset_root"] = str(asset_root)
    for key in ("environment", "robot"):
        simulation[key] = str((asset_root / simulation[key]).resolve())

    port = webrtc_port
    if port is None:
        port = int(environment.get("ARX_SIM_WEBRTC_PORT", config["webrtc"]["port"]))
    if not 1 <= int(port) <= 65534:
        raise ValueError("webrtc port must be in 1..65534")
    config["webrtc"]["port"] = int(port)
    config["services"]["arx_ops_port"] = int(
        arx_ops_port
        if arx_ops_port is not None
        else environment.get("ARX_SIM_HTTP_PORT", config["services"]["arx_ops_port"])
    )

    target_run = run_dir or create_session_dir(_resolve(config["runtime"]["run_root"]))
    (target_run / "logs").mkdir(parents=True, exist_ok=True)
    (target_run / "artifacts").mkdir(exist_ok=True)
    config["runtime"]["run_dir"] = str(target_run.resolve())
    return config


def write_resolved_config(config: dict[str, Any]) -> Path:
    path = Path(config["runtime"]["run_dir"]) / "resolved_config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=True, allow_unicode=True), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (path.parent / "manifest.json").write_text(
        json.dumps({"resolved_config_sha256": digest}, indent=2), encoding="utf-8"
    )
    return path
