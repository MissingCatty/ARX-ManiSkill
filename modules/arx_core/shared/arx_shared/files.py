from __future__ import annotations

from pathlib import Path
from uuid import uuid4


COOLAS_ROOT = Path(__file__).resolve().parents[2]


def project_path(*parts: str) -> Path:
    return COOLAS_ROOT.joinpath(*parts)


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def new_run_dir(prefix: str = "run") -> Path:
    return ensure_dir(project_path("runs", f"{prefix}_{uuid4().hex[:12]}"))

