"""Registered serial ARX/ZeroGrasp/cuRobo physical batch in ManiSkill."""
from arx_project.layout import canonical_path
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image
from scipy.spatial.transform import Rotation
from environment import (make, ROOT, ARM, FINGERS, BASE_POSE, META, ALIGNMENT,
                         PERCEPTION_CAMERA, INITIAL_SOURCE, CONTROLLERS_SOURCE, RUNTIME_SOURCE)
from environment import SCENE_CUBOIDS, TABLE_HEIGHT, alignment_for_camera, RECORDING_VIEW

ARX = ROOT.parent / "arx_core"
CODE = Path(__file__).parent
NEURAL = canonical_path(ROOT.parent / "tmp/arx-fullchain/env/bin/python")
CUROBO = ROOT.parent / "curobo/.venv/bin/python"
CHECKPOINT = ARX / "servers/zerograsp/vendor/zerograsp_mainline_minimal/checkpoints/zerograsp_cvpr2025/zerograsp_synthetic.ckpt"


def save(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2)); tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(env, obs=None):
    robot = env.agent.robot
    obj = env.obj.pose.to_transformation_matrix().cpu().numpy()[0]
    center = (np.array(env.object_bbox["min"]) + np.array(env.object_bbox["max"])) / 2 * env.object_scale
    state = {"joint_names": [j.name for j in robot.active_joints],
             "qpos": robot.qpos.cpu().numpy()[0].tolist(),
             "joint_limits": robot.qlimits.cpu().numpy()[0].tolist(),
             "links": {n: robot.links_map[n].pose.to_transformation_matrix().cpu().numpy()[0].tolist()
                       for n in ["base_link", "body4", "right_eef_link"]},
             "object_world": obj.tolist(), "object_center": (obj[:3, :3] @ center + obj[:3, 3]).tolist(),
             "scene_cuboids": SCENE_CUBOIDS, "table_height_m": TABLE_HEIGHT}
    if env.scene_spec is not None:
        from scene_tools import mesh_state
        state['scene_meshes']=mesh_state(env)
        state['objects']={n:{'model_id':env.object_models[n],
                           'pose':a.pose.raw_pose.cpu().numpy()[0].tolist()}
                          for n,a in env.objects.items()}
    if obs is not None:
        params = obs["sensor_param"][env.perception_camera]
        state["perception_camera"] = env.perception_camera
        state["camera_alignment"] = env.camera_alignment
        state["camera_world_optical"] = (params["cam2world_gl"].cpu().numpy()[0] @ np.diag([1, -1, -1, 1])).tolist()
        state["intrinsics"] = params["intrinsic_cv"].cpu().numpy()[0].tolist()
    return state


def worker(script, out, gpu, log, lift=False, approach=False, observe=False):
    guard=out/'source_guard.json'
    if guard.exists():
        receipt=json.loads(guard.read_text());registration=Path(receipt['registration'])
        if sha(registration)!=receipt['sha256']:raise RuntimeError('Campaign registration changed')
        for name,digest in json.loads(registration.read_text())['source_hashes'].items():
            if sha(Path(name))!=digest:raise RuntimeError('Frozen source changed before '+script+': '+name)
    env = os.environ.copy()
    env.update(ARX_NEURAL_GPU=gpu, ARX_PLANNER_GPU=gpu, ARX_SIM_GPU=gpu, LILO_GPU=gpu,
               CUDA_VISIBLE_DEVICES=gpu, GRASP_TRAJOPT_POLICY="interactive")
    # Clear the ManiSkill venv's Python search path for each dedicated backend.
    env.pop("PYTHONPATH", None)
    python = NEURAL if script == "neural.py" else CUROBO
    cmd = ["bash", "-c", 'source "$1"; shift; exec "$@"', "arx-worker",
           str(ARX / "scripts/curobo_arx/environment.sh"), str(python), str(CODE / script), str(out)]
    if lift:
        cmd.append("--lift")
    if approach:
        cmd.append("--approach")
    if observe:
        cmd.append('--observe')
    with log.open("x") as f:
        process = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            process.wait(timeout=900)
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise
    return process.returncode


class Execution:
    def __init__(self, env, out):
        self.env, self.out = env, out
        self.names = [j.name for j in env.agent.robot.active_joints]
        self.arm_indices = [self.names.index(n) for n in ARM]
        self.finger_indices = [self.names.index(n) for n in FINGERS]
        self.command = env.agent.robot.qpos.cpu().numpy()[0].copy()
        self.tick, self.label = 0, "setup"
        self.initial_z = float(env.obj.pose.p[0, 2])
        save(out/'execution_origin.json',{'object_initial_z_m':self.initial_z,'reference':'After initial settling, before any commanded execution stage'})
        self.events, self.heights, self.trace = [], [], (out / "execution.jsonl").open("x")
        self.writer = imageio.get_writer(out / "execution.mp4", fps=20, codec="libx264", quality=7, macro_block_size=2)
        save(out / "recording_camera.json", RECORDING_VIEW)
        self.sensor_writer = (imageio.get_writer(out / "right_wrist.mp4", fps=20, codec="libx264", quality=7)
                              if env.perception_camera == "right_wrist" else None)
        self.control_ticks = 0
        self.forbidden = []
        self.bilateral = set()
        self.stages = {}
        env.trace_callback = self.record

    def record(self):
        self.tick += 1
        contacts = []
        robot_entities = {link._objs[0].entity for link in self.env.agent.robot.get_links()}
        target = self.env.obj._objs[0]
        for c in self.env.scene.get_contacts():
            a, b = [body.entity for body in c.bodies]
            if a not in robot_entities and b not in robot_entities:
                continue
            if a in robot_entities and b in robot_entities:
                continue
            impulse = sum(float(np.linalg.norm(p.impulse)) for p in c.points)
            if not c.points or impulse <= 1e-8:
                continue
            robot_link = a if a in robot_entities else b
            other = b if a in robot_entities else a
            allowed = self.label!='observe' and other == target and any(n in robot_link.name for n in ["right_link27", "right_link28"])
            event = {"a": a.name, "b": b.name, "impulse_ns": impulse, "allowed": allowed}
            contacts.append(event)
            if not allowed:
                self.forbidden.append({"tick": self.tick, "stage": self.label, **event})
            if allowed and self.label == "close":
                self.bilateral.add("right_link27" if "right_link27" in robot_link.name else "right_link28")
        z = float(self.env.obj.pose.p[0, 2])
        if self.label == "hold":
            self.heights.append(z)
        row = {"tick": self.tick, "sim_seconds": self.tick/120, "stage": self.label,
               "qpos": self.env.agent.robot.qpos[0].tolist(),
               "object_pose": self.env.obj.pose.raw_pose[0].tolist(), "contacts": contacts}
        if self.env.scene_spec is not None:
            row['objects']={n:a.pose.raw_pose[0].tolist() for n,a in self.env.objects.items()}
        self.trace.write(json.dumps(row) + "\n")

    def step(self):
        self.env._step_action(torch.tensor(self.command[None], device=self.env.device))
        self.control_ticks += 1
        if self.control_ticks % 2 == 0:
            self.writer.append_data(self.env.render().cpu().numpy()[0])
            if self.sensor_writer is not None:
                obs = self.env.get_obs()
                self.sensor_writer.append_data(obs["sensor_data"][self.env.perception_camera]["rgb"].cpu().numpy()[0])
        if self.forbidden:
            raise RuntimeError("forbidden_robot_contact")

    def segment(self, label, q, times, finger=.044, goal_world=None):
        self.label = label
        q, times = np.asarray(q), np.asarray(times)
        start = self.env.agent.robot.qpos[0, self.arm_indices].cpu().numpy()
        if np.max(np.abs(start-q[0])) >= .005:
            raise RuntimeError(label + "_trajectory_start_mismatch")
        steps = max(1, int(np.ceil(times[-1] * 40)))
        for k in range(steps + 1):
            t = min(k/40, times[-1])
            self.command[self.arm_indices] = [np.interp(t, times, q[:, j]) for j in range(6)]
            self.command[self.finger_indices] = finger
            self.step()
        # Original strict joint tolerance; allow up to 4 s additional settling.
        for _ in range(160):
            actual = self.env.agent.robot.qpos[0, self.arm_indices].cpu().numpy()
            if np.max(np.abs(actual-q[-1])) < .005:
                break
            self.step()
        actual = self.env.agent.robot.qpos[0, self.arm_indices].cpu().numpy()
        err = float(np.max(np.abs(actual-q[-1])))
        stage = {"joint_error_rad": err, "settled": err < .005, "planned_seconds": float(times[-1])}
        if goal_world is not None:
            actual_pose = self.env.agent.tcp.pose.to_transformation_matrix().cpu().numpy()[0]
            goal = np.asarray(goal_world)
            stage["tool_position_error_m"] = float(np.linalg.norm(actual_pose[:3, 3]-goal[:3, 3]))
            stage["tool_rotation_error_rad"] = float(Rotation.from_matrix(actual_pose[:3, :3] @ goal[:3, :3].T).magnitude())
        self.stages[label] = stage
        save(self.out / "stages.json", self.stages)
        if not stage["settled"]:
            raise RuntimeError(label + "_not_converged")

    def finish(self):
        self.env.trace_callback = None
        self.trace.close(); self.writer.close()
        if self.sensor_writer is not None:
            self.sensor_writer.close()
        save(self.out / "forbidden_contacts.json", self.forbidden)


def trial(entry, out, neural_gpu, planner_gpu, perception_camera=PERCEPTION_CAMERA):
    env = None; execution = None
    result = {**entry, "camera": perception_camera, "success": False, "planning_success": False, "physical_attempt": False}
    started = time.monotonic()
    out.mkdir(exist_ok=False)
    try:
        if entry.get('campaign_registration'):
            registration=Path(entry['campaign_registration'])
            save(out/'source_guard.json',{'registration':str(registration),'sha256':sha(registration)})
        env = make(entry["model_id"], perception_camera=perception_camera, scene_spec=entry.get('scene'))
        obs, _ = env.reset(seed=entry["seed"])
        for _ in range(80):
            env._step_action(torch.tensor(env.initial_q[None], device=env.device))
        if 'observation' in entry:
            save(out/'state.json',snapshot(env))
            save(out/'observation_request.json',entry['observation'])
            print(entry['id'],'planning wrist observation',flush=True)
            worker('planner.py',out,planner_gpu,out/'observation_planner.log',observe=True)
            preparation=json.loads((out/'observation_plan_result.json').read_text())
            if not preparation['success']:
                raise RuntimeError(preparation.get('reason','observation_planning_failed'))
            execution=Execution(env,out)
            path=preparation['paths']['observe']
            execution.segment('observe',path['q'],path['times'],goal_world=entry['observation']['goal_world'])
            result['observation_executed']=True
            from scene_tools import visibility
            view=visibility(env)
            save(out/'visibility.json',view)
            if not view['admitted_view']:
                raise RuntimeError('registered_wrist_view_not_admitted')
            if entry['layout']=='S5_occluded' and not .20<=view['occlusion_fraction']<=.40:
                raise RuntimeError('registered_occlusion_outside_range')
        obs = env.get_obs()
        state = snapshot(env, obs)
        save(out / "state.json", state)
        save(out / "initial_conditions.json", {"spawn": env.spawn, "base_pose": BASE_POSE,
                                              "initial_q": env.initial_q.tolist(), "object_bbox": env.object_bbox})
        camera = out / "camera"; camera.mkdir()
        data = obs["sensor_data"][env.perception_camera]
        rgb = data["rgb"].cpu().numpy()[0]
        depth = data["depth"].cpu().numpy()[0, :, :, 0]
        seg = data["segmentation"].cpu().numpy()[0, :, :, 0]
        mask = seg == env.obj._objs[0].per_scene_id
        if mask.sum() < 50:
            raise RuntimeError("Target has fewer than 50 visible pixels")
        Image.fromarray(rgb).save(camera / "color.png")
        Image.fromarray(depth.astype(np.uint16)).save(camera / "depth.png")
        Image.fromarray(mask.astype(np.uint8)).save(camera / "mask.png")
        imageio.imwrite(out / "initial_overview.png", env.render().cpu().numpy()[0])
        k = np.array(state["intrinsics"])
        save(camera / "params.json", {"depth_scale": .001, "intrinsics": {
            "width": 640, "height": 480, "fx": k[0, 0], "fy": k[1, 1], "ppx": k[0, 2], "ppy": k[1, 2]}})
        save(out / "capture.json", {"mask_pixels": int(mask.sum()), "median_target_depth_mm": float(np.median(depth[mask])),
            "camera": env.perception_camera, "camera_alignment": env.camera_alignment,
            "target_touches_image_border": bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any()),
            "mask_source": "simulator instance segmentation", "fresh_rgbd": True})
        if entry.get('observation_only'):
            result['reason']='observation_only_preflight'
            return result
        print(entry["id"], "fresh ZeroGrasp", flush=True)
        if worker("neural.py", out, neural_gpu, out / "neural.log"):
            raise RuntimeError("zerograsp_failed")
        print(entry["id"], "cuRobo planning", flush=True)
        worker("planner.py", out, planner_gpu, out / "planner.log")
        plan = json.loads((out / "plan_result.json").read_text())
        if not plan["success"]:
            result["reason"] = plan["reason"]
            return result
        result["planning_success"] = True
        if execution is None:
            execution = Execution(env, out)
        result["physical_attempt"] = True
        selected = plan["selected"]
        path = plan["paths"]["pregrasp"]
        execution.segment("pregrasp", path["q"], path["times"], goal_world=selected["pregrasp_world"])
        save(out / "pregrasp_state.json", snapshot(env))
        worker("planner.py", out, planner_gpu, out / "approach_planner.log", approach=True)
        approach = json.loads((out / "approach_plan_result.json").read_text())
        if not approach["success"]:
            raise RuntimeError(approach["reason"])
        path = approach["paths"]["approach"]
        execution.segment("approach", path["q"], path["times"], goal_world=selected["grasp_world"])
        q = selected["q_grasp"]
        execution.segment("close", [q, q], [0, 3], finger=0., goal_world=selected["grasp_world"])
        result["bilateral_fingers"] = sorted(execution.bilateral)
        if len(execution.bilateral) != 2:
            raise RuntimeError("no_bilateral_target_contact")
        save(out / "closed_state.json", snapshot(env))
        print(entry["id"], "measured closure lift planning", flush=True)
        worker("planner.py", out, planner_gpu, out / "lift_planner.log", lift=True)
        lift = json.loads((out / "lift_plan_result.json").read_text())
        if not lift["success"]:
            raise RuntimeError(lift["reason"])
        path = lift["paths"]["lift"]
        execution.segment("lift", path["q"], path["times"], finger=0., goal_world=selected["lift_world"])
        q = path["q"][-1]
        execution.segment("hold", [q, q], [0, 3], finger=0., goal_world=selected["lift_world"])
        result.update(minimum_hold_lift_m=min(execution.heights)-execution.initial_z,
                      hold_seconds=len(execution.heights)/120,
                      final_lift_m=execution.heights[-1]-execution.initial_z)
        result["success"] = bool(result["minimum_hold_lift_m"] > .06 and result["hold_seconds"] >= 3)
        result["reason"] = "success" if result["success"] else "insufficient_hold_height"
    except Exception as exc:
        result.update(reason=str(exc), error=traceback.format_exc())
        print(result["error"], flush=True)
    finally:
        if execution:
            execution.finish()
            result.update(stages=execution.stages, forbidden_contact_events=len(execution.forbidden))
        if env:
            save(out / "final_state.json", snapshot(env))
            if env.scene_spec is not None:
                first=json.loads((out/'state.json').read_text())['objects'] if (out/'state.json').exists() else {}
                result['non_target_displacement_m']={n:float(np.linalg.norm(a.pose.p[0].cpu().numpy()-np.asarray(first[n]['pose'][:3])))
                                                     for n,a in env.objects.items() if n!='target' and n in first}
            env.close()
        result["wall_seconds"] = time.monotonic()-started
        save(out / "result.json", result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--neural-gpu", required=True)
    p.add_argument("--planner-gpu", required=True)
    p.add_argument("--camera", choices=["head", "right_wrist"], default=PERCEPTION_CAMERA)
    p.add_argument("--count", type=int, choices=range(1, 11), default=5,
                   help="Total examples to register (default: 5)")
    p.add_argument("--limit", type=int, default=10)
    args = p.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    if (out / "STOP").exists():
        raise RuntimeError("Batch is stopped: " + (out / "STOP").read_text().strip())
    reg_path = out / "registration.json"
    models = ["005_tomato_soup_can", "007_tuna_fish_can", "013_apple", "014_lemon", "025_mug"]
    schedule = [{"id": f"{i:02d}_{model}", "model_id": model, "seed": 20261116 + i}
                for i, model in enumerate([m for m in models for _ in range(2)])][:args.count]
    sources = {str(p): sha(p) for p in CODE.glob("*.py")}
    dependency_files = [ARX / "scripts/curobo_arx/optimize_path.py",
                        ARX / "scripts/curobo_arx/environment.sh",
                        ARX / "scripts/curobo_arx/grasp_execute.py",
                        ARX / "servers/zerograsp/coolas_zerograsp/grasp.py",
                        ARX / "arx-sim/config/controllers.yaml",
                        Path(META["source"]), ROOT / "assets/arx/manifest.json",
                        INITIAL_SOURCE, CONTROLLERS_SOURCE, RUNTIME_SOURCE]
    dependency_files += list((ARX / "scripts/curobo_arx/static_pick/reference/model").glob("arx.*"))
    dependency_files += list((ARX / "scripts/curobo_arx/static_pick/reference/model_closed").glob("arx.*"))
    package = ARX / "servers/zerograsp/vendor/zerograsp_mainline_minimal"
    dependency_files += list(package.rglob("*.py")) + list((package / "configs").glob("*.yaml"))
    dependencies = {str(p): sha(p) for p in dependency_files}
    for name, digest in META["files"].items():
        assert sha(ROOT / "assets/arx" / name) == digest, "Exported robot asset changed: " + name
    if not reg_path.exists():
        save(reg_path, {"schedule": schedule, "attempt_budget": len(schedule), "simulator": "ManiSkill 3.0.1 / SAPIEN 3.0.3",
            "gpu": {"simulator": os.environ.get("CUDA_VISIBLE_DEVICES"), "neural": args.neural_gpu, "planner": args.planner_gpu},
            "source_hashes": sources, "dependency_hashes": dependencies,
            "checkpoint_sha256": sha(CHECKPOINT), "robot_manifest_sha256": sha(ROOT / "assets/arx/manifest.json"),
            "sources": "Current COOLAS-ARX USD, controllers, ZeroGrasp mainline/checkpoint, frozen cuRobo model and optimize_path",
            "scope": "New ManiSkill tabletop placement benchmark, not a historical Isaac Sim regression or renderer/physics equivalence claim",
            "success": {"bilateral_target_contact": True, "min_continuous_hold_lift_m": .06, "hold_seconds": 3,
                        "arm_joint_tolerance_rad": .005, "forbidden_robot_contact_events": 0},
            "self_collision_enabled": False, "mask": "Ground-truth instance mask; no LangSAM", "robot": META,
            "initial_state_and_camera_alignment": alignment_for_camera(args.camera)})
        save(out / "scene_geometry.json", {"table_height_m": TABLE_HEIGHT, "base_pose": BASE_POSE,
                                           "cuboids": SCENE_CUBOIDS, "recording_camera": RECORDING_VIEW})
        historical_policy = ARX / "runs/research_contract_20261019/storage_policy.json"
        if historical_policy.is_file():
            shutil.copy2(historical_policy, out / "storage_policy.json")
        else:
            save(out / "storage_policy.json", {"minimum_free_bytes": 1073741824,
                "source": str(ARX / "AGENTS.md"), "source_sha256": sha(ARX / "AGENTS.md"),
                "historical_snapshot_missing": str(historical_policy),
                "historical_snapshot_verified": False})
        (out / "source").mkdir()
        for p in CODE.glob("*.py"):
            shutil.copy2(p, out / "source" / p.name)
    else:
        reg = json.loads(reg_path.read_text())
        if reg["schedule"] != schedule:
            raise RuntimeError("Schedule differs from registration; use the registered --count")
        if reg["initial_state_and_camera_alignment"]["camera_role"] != args.camera:
            raise RuntimeError("Perception camera differs from registration")
        if reg["source_hashes"] != sources or reg.get("dependency_hashes") != dependencies or reg["checkpoint_sha256"] != sha(CHECKPOINT):
            raise RuntimeError("Source/dependency changed since registration; use a new batch output")
    finished = []
    launched = 0
    for index, entry in enumerate(schedule):
        amendment = out / "budget_amendment.json"
        budget = (int(json.loads(amendment.read_text())["effective_attempt_budget"])
                  if amendment.exists() else len(schedule))
        if not 0 <= budget <= len(schedule):
            raise ValueError("Budget amendment must reduce the registered count")
        if index >= budget or (out / "STOP").exists():
            break
        path = out / entry["id"]
        if (path / "result.json").exists():
            result = json.loads((path / "result.json").read_text())
        elif path.exists():
            raise RuntimeError(f"Interrupted trial {path}; preserve unknown outcome, do not replay")
        elif launched < args.limit:
            assert shutil.disk_usage(out).free > 1073741824
            result = trial(entry, path, args.neural_gpu, args.planner_gpu, args.camera)
            launched += 1
        else:
            continue
        finished.append(result)
        summary = {"registered": budget, "original_registered": len(schedule), "completed": len(finished),
            "planning_success": sum(r["planning_success"] for r in finished),
            "physical_attempts": sum(r["physical_attempt"] for r in finished),
            "successes": sum(r["success"] for r in finished), "results": finished}
        save(out / "summary.json", summary)
        print(json.dumps({k: v for k, v in summary.items() if k != "results"}), flush=True)


if __name__ == "__main__":
    main()
