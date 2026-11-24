from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from arx_shared.config import load_yaml
from arx_shared.schemas import GraspPrediction, GraspPredictRequest

from .storage import new_zerograsp_run_dir


COOLAS_ROOT = Path(__file__).resolve().parents[3]


def _resolve_inputs(request: GraspPredictRequest) -> tuple[Path, Path, Path, Path]:
    if request.input_dir:
        input_dir = Path(request.input_dir)
        return (
            input_dir / "rgb.png",
            input_dir / "depth.png",
            input_dir / "mask.png",
            input_dir / "camera.json",
        )
    required = [request.img_path, request.depth_path, request.mask_path, request.camera_info_path]
    if any(value is None for value in required):
        raise ValueError("provide input_dir or img_path/depth_path/mask_path/camera_info_path")
    return tuple(Path(str(value)) for value in required)  # type: ignore[return-value]


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def predict_grasp(request: GraspPredictRequest) -> GraspPrediction:
    cfg = load_yaml("zerograsp.yaml").get("model", {})
    model_root = COOLAS_ROOT / str(cfg.get("root", "servers/zerograsp/vendor/zerograsp_mainline_minimal"))
    run_mainline = model_root / "run_mainline.py"
    if not run_mainline.exists():
        raise FileNotFoundError(f"ZeroGrasp run_mainline.py not found: {run_mainline}")

    output_dir = Path(request.output_dir) if request.output_dir else new_zerograsp_run_dir() / "zg_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    img_path, depth_path, mask_path, camera_info_path = _resolve_inputs(request)

    cmd = [
        os.environ.get("ZEROGRASP_PYTHON")
        or (
            str(Path(os.environ["CONDA_PREFIX"]) / "bin" / "python")
            if os.environ.get("CONDA_PREFIX")
            else str(cfg.get("python", "python3"))
        ),
        str(run_mainline),
        "--img_path",
        str(img_path),
        "--depth_path",
        str(depth_path),
        "--mask_path",
        str(mask_path),
        "--camera_info_path",
        str(camera_info_path),
        "--output_dir",
        str(output_dir),
    ]
    if cfg.get("checkpoint"):
        checkpoint = Path(str(cfg["checkpoint"]))
        checkpoint = checkpoint if checkpoint.is_absolute() else model_root / checkpoint
        if not checkpoint.is_file():
            raise FileNotFoundError(f"ZeroGrasp checkpoint not found: {checkpoint}")
        cmd.extend(["--checkpoint", str(checkpoint.resolve())])
    if cfg.get("config"):
        cmd.extend(["--config", str(cfg["config"])])

    env = os.environ.copy()
    env["PYTHONPATH"] = str(model_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("OCNN_DISABLE_TRITON", "1")
    completed = subprocess.run(
        cmd,
        cwd=model_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=request.timeout_sec,
    )
    report_path = output_dir / "run_report.json"
    top1_path = output_dir / "recommended_grasp_top1.json"
    raw = {
        "command": cmd,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
        "run_report": _load_json(report_path),
        "recommended_grasp_top1": _load_json(top1_path),
    }
    if completed.returncode != 0:
        raw["error"] = "ZeroGrasp subprocess failed"
    return GraspPrediction(
        grasp_path=str(top1_path) if top1_path.exists() else None,
        output_dir=str(output_dir),
        raw=raw,
    )
