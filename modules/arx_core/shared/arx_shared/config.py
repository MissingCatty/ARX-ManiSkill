from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


COOLAS_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(name: str) -> dict[str, Any]:
    path = COOLAS_ROOT / "configs" / name
    if yaml is None:
        raise RuntimeError("PyYAML is required to read coolas config files")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config {path} must contain a mapping")
    return data

