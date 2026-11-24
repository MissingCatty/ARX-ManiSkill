"""Fresh inference through the current COOLAS-ARX ZeroGrasp entry point."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARX = ROOT / "arx_core"
sys.path[:0] = [str(ARX / "shared"), str(ARX / "servers/zerograsp")]
from arx_shared.schemas import ZeroGraspInputRequest
from coolas_zerograsp.grasp import prepare_zerograsp_input

out = Path(sys.argv[1]).resolve()
prepared = prepare_zerograsp_input(ZeroGraspInputRequest(camera_dir=str(out / "camera"),
    mask_path=str(out / "camera/mask.png"), output_dir=str(out / "zg_input"), label="maniskill_target"))
(out / "network_input.json").write_text(json.dumps(prepared.model_dump(), indent=2))
package = ARX / "servers/zerograsp/vendor/zerograsp_mainline_minimal"
subprocess.run([sys.executable, str(package / "run_mainline.py"),
    "--img_path", prepared.rgb_path, "--depth_path", prepared.depth_path,
    "--mask_path", prepared.mask_path, "--camera_info_path", prepared.camera_info_path,
    "--output_dir", str(out / "network"), "--checkpoint",
    str(package / "checkpoints/zerograsp_cvpr2025/zerograsp_synthetic.ckpt"),
    "--config", str(package / "configs/demo_640x480.yaml")], check=True, cwd=package)
files = sorted((out / "network/raw_outputs").glob("*.raw_grasps.npy"))
if not files:
    raise RuntimeError("ZeroGrasp emitted no raw proposals")
rows = np.concatenate([np.load(f) for f in files])
np.save(out / "raw_grasps.npy", rows)
(out / "network_candidates.json").write_text(json.dumps({"fresh_inference": True, "count": len(rows),
    "source_files": list(map(str, files)), "mask": "ManiSkill target instance segmentation"}, indent=2))
