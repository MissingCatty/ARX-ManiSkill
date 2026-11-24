"""Fresh perception/planning check for scene setup; no grasp motion execution."""
import argparse
import json
from pathlib import Path
import traceback
import numpy as np
import torch
from PIL import Image
import imageio.v2 as imageio
from environment import make, PERCEPTION_CAMERA, TABLE_HEIGHT, SCENE_CUBOIDS, ALIGNMENT
from run import snapshot, worker, save


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--neural-gpu", required=True)
    p.add_argument("--planner-gpu", required=True)
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {"scope": "Fresh RGB-D/ZeroGrasp and cuRobo planning; hold initial pose only, no grasp execution",
              "table_height_m": TABLE_HEIGHT, "scene_cuboids": SCENE_CUBOIDS, "alignment": ALIGNMENT,
              "grasp_executions": 0, "planning_passed": False}
    env = make()
    try:
        env.reset(seed=20261116)
        initial = env.agent.robot.qpos[0].cpu().numpy().copy()
        command = torch.tensor(env.initial_q[None], device=env.device)
        target = env.obj._objs[0]
        robot_entities = {l._objs[0].entity for l in env.agent.robot.get_links()}
        contacts = []
        def record():
            for c in env.scene.get_contacts():
                a, b = [body.entity for body in c.bodies]
                if (a in robot_entities) == (b in robot_entities):
                    continue
                impulse = sum(float(np.linalg.norm(x.impulse)) for x in c.points)
                if impulse>1e-8:
                    contacts.append({"a":a.name, "b":b.name, "impulse_ns":impulse})
        env.trace_callback = record
        for _ in range(80):
            env._step_action(command)
            if contacts:
                raise RuntimeError("Robot touched environment while holding initial pose")
        env.trace_callback = None
        report["initial_hold_robot_contacts"] = contacts
        report["initial_hold_seconds"] = 2.
        report["initial_hold_max_joint_drift"] = float(np.max(np.abs(env.agent.robot.qpos[0].cpu().numpy()-initial)))
        obs = env.get_obs()
        state = snapshot(env, obs)
        save(out/"state.json", state)
        data = obs["sensor_data"][PERCEPTION_CAMERA]
        rgb = data["rgb"].cpu().numpy()[0]
        depth = data["depth"].cpu().numpy()[0,:,:,0]
        seg = data["segmentation"].cpu().numpy()[0,:,:,0]
        mask = seg==target.per_scene_id
        camera = out/"camera"; camera.mkdir()
        for name, value in [("color.png",rgb), ("depth.png",depth.astype(np.uint16)),
                            ("mask.png",mask.astype(np.uint8))]:
            Image.fromarray(value).save(camera/name)
        imageio.imwrite(out/"overview.png", env.render().cpu().numpy()[0])
        k = np.asarray(state["intrinsics"])
        save(camera/"params.json", {"depth_scale":.001,"intrinsics":{
            "width":rgb.shape[1],"height":rgb.shape[0],"fx":k[0,0],"fy":k[1,1],"ppx":k[0,2],"ppy":k[1,2]}})
        report["target_mask_pixels"] = int(mask.sum())
        report["target_touches_image_border"] = bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any())
        report["target_fully_in_view"] = bool(mask.sum()>=50 and not report["target_touches_image_border"])
        if mask.sum()<50:
            raise RuntimeError("Target has fewer than 50 visible pixels")
        print("Fresh ZeroGrasp",flush=True)
        if worker("neural.py",out,args.neural_gpu,out/"neural.log"):
            raise RuntimeError("ZeroGrasp failed")
        print("cuRobo planning",flush=True)
        worker("planner.py",out,args.planner_gpu,out/"planner.log")
        plan = json.loads((out/"plan_result.json").read_text())
        report["planning_passed"] = plan["success"]
        report["planning_reason"] = plan.get("reason","success")
        if plan["success"]:
            report["selected_candidate_index"] = plan["selected"]["index"]
    except Exception:
        report["error"] = traceback.format_exc()
        print(report["error"],flush=True)
    finally:
        env.trace_callback = None
        env.close()
        save(out/"setup_check.json", report)
    print(json.dumps({k:v for k,v in report.items() if k not in ["alignment","scene_cuboids"]},indent=2),flush=True)
    return 0 if report["planning_passed"] else 2


if __name__=="__main__":
    raise SystemExit(main())
