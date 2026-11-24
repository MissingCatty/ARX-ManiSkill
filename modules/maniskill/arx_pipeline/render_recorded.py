"""Render measured execution states from a new camera without running physics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
import sapien
import imageio.v2 as imageio
from environment import make, META, RECORDING_VIEW, SCENE_CUBOIDS


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview-only", action="store_true")
    args = parser.parse_args()
    source, out = args.trial.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    initial = json.loads((source/"state.json").read_text())
    result = json.loads((source/"result.json").read_text())
    recorded = [json.loads(line) for line in (source/"execution.jsonl").read_text().splitlines()]
    rows = [r for r in recorded if r["tick"]%6 == 0]
    assert initial["scene_cuboids"] == SCENE_CUBOIDS, "Scene differs from recorded trial"
    for name, digest in META["files"].items():
        path = Path(__file__).resolve().parents[1]/"assets/arx"/name
        assert sha(path) == digest, "Robot asset changed: " + name
    env = make(result["model_id"], perception_camera=result["camera"], scene_spec=result.get('scene'))
    writer = None
    try:
        env.reset(seed=result["seed"])
        names = [j.name for j in env.agent.robot.active_joints]
        assert names == initial["joint_names"]
        env.agent.robot.set_pose(sapien.Pose(np.asarray(initial["links"]["base_link"])))
        def render(row):
            env.agent.robot.set_qpos(torch.tensor([row["qpos"]], device=env.device))
            p = row["object_pose"]
            env.obj.set_pose(sapien.Pose(p[:3], p[3:]))
            for name,pose in row.get('objects',{}).items():
                if name!='target':env.objects[name].set_pose(sapien.Pose(pose[:3],pose[3:]))
            return env.render().cpu().numpy()[0]
        previews = {"initial": rows[0], "closed": next(r for r in reversed(rows) if r["stage"]=="close"),
                    "held": rows[-1]}
        for name, row in previews.items():
            path = out/(name+".png")
            if not path.exists():
                imageio.imwrite(path, render(row))
        evidence = {"scope": "New viewpoint rendered from original measured joint/object states; no physics steps and no new grasp attempt",
                    "source_trial": str(source), "original_success": result["success"],
                    "execution_trace_sha256": sha(source/"execution.jsonl"),
                    "original_registration_sha256": sha(source.parent/"registration.json"),
                    "recording_camera": RECORDING_VIEW, "gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "preview_ticks": {k:r["tick"] for k,r in previews.items()}, "physics_steps":0,
                    "new_grasp_attempts":0}
        if not args.preview_only:
            destination = out/"left_front.mp4"
            if destination.exists():
                raise FileExistsError(destination)
            writer = imageio.get_writer(destination, fps=20, codec="libx264", quality=8, macro_block_size=2)
            for index, row in enumerate(rows):
                writer.append_data(render(row))
                if index%100 == 0:
                    print(f"Rendered {index}/{len(rows)} frames",flush=True)
            writer.close(); writer = None
            evidence.update(video=str(destination), frames=len(rows), fps=20, duration_seconds=len(rows)/20)
        (out/("preview.json" if args.preview_only else "rendering.json")).write_text(json.dumps(evidence,indent=2))
        print(json.dumps(evidence,indent=2),flush=True)
    finally:
        if writer is not None:
            writer.close()
        env.close()


if __name__=="__main__":
    main()
