"""Experimental registered preparation-only ARX diagnostic harness."""
"""Experimental H017 harness; do not run before separate registration."""
"""Reproduce tabletop pose challenges with explicit ARX execution policies.

Uses the live ARX diagnostic mailbox and existing, preloaded objects. Default
trials get fresh ZeroGrasp inference. Explicit paired scenarios reuse a saved
initial proposal for controlled comparison. No gain changes or PRM fallback.
The default baseline is unchanged. Visual feedback selects a separate executor.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import imageio_ffmpeg
from client import Client

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "runs/arx_grasp"


def write(path, data):
    path.write_text(json.dumps(data, indent=2))


def encode(folder):
    info = json.loads((folder / "frames/frames.json").read_text())
    if info.get("error"):
        raise RuntimeError(info["error"])
    ranges = []
    for path in folder.glob("*_execution.json"):
        e = json.loads(path.read_text())
        end = e["final"]["sim_time"]
        ranges.append((end - e["sim_duration_s"] - .4, end + .5))
    frames = [(i, f) for i, f in enumerate(info["frames"])
              if not ranges or any(a <= f["sim_time"] <= b for a, b in ranges)]
    if not ranges:
        frames = frames[:15]
    if len(frames) < 3:
        raise RuntimeError("Too few viewport frames")
    lines = []
    for j, (i, f) in enumerate(frames):
        path = folder / "frames" / f["file"]
        if not path.exists():
            raise RuntimeError(f"Missing captured frame: {path}")
        duration = (frames[j + 1][1]["wall_time"] - f["wall_time"]
                    if j + 1 < len(frames) and frames[j + 1][0] == i + 1 else .2)
        lines.extend([f"file '{path}'", f"duration {max(.001, duration)}"])
    lines.append(f"file '{path}'")
    manifest = folder / "motion_frames.txt"
    manifest.write_text("\n".join(lines) + "\n")
    video = folder / "motion.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error",
                    "-f", "concat", "-safe", "0", "-i", str(manifest), "-an",
                    "-vf", "fps=5", "-c:v", "libx264", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)], check=True)
    return {"video": str(video), "camera_path": info["camera_path"],
            "camera_matrix": info["camera_matrix"], "frames": len(frames),
            "editing": "Only planning waits removed; original wall-time frame durations; no overlays/audio"}


def classify(result, folder):
    if result.get("success"):
        return "success"
    error = result.get("error", "")
    if "Trial exceeded" in error:
        return ("candidate_search_timeout" if (folder / "candidate_diagnostics.json").exists()
                and not (folder / "selected_grasp.json").exists() else "trial_timeout")
    if "No bilateral target contact" in error:
        return "closure_without_bilateral_contact"
    if "Unexpected contact" in error:
        return "unexpected_physical_contact"
    if "ring preshape" in error.lower():
        return "gripper_preshape_rejection"
    if "feedback alignment" in error.lower():
        return "feedback_alignment_rejection"
    if result.get("object_held") is False and (folder / "hold_execution.json").exists():
        return "failed_lift_or_hold"
    if "converge" in error:
        return "execution_convergence_failure"
    if "No ZeroGrasp candidate" in error:
        return "no_reachable_collision_free_candidate"
    if "IK failed" in error:
        return "inverse_kinematics_failure"
    if error in ("Frozen closed-gripper goals invalid", "Frozen open-gripper goals invalid"):
        return "candidate_goal_collision_rejection"
    if "No visible target cloud" in error:
        return "visual_feedback_observation_failure"
    if "gripper" in error.lower() and ("validated model range" in error or "projection tolerance" in error):
        return "measured_gripper_model_state_failure"
    if "optim" in error.lower() or "trajectory" in error.lower():
        return "trajectory_planning_or_validation_failure"
    if "planning failed" in error:
        return "path_planning_failure"
    return "infrastructure_or_input_failure"


def run(args):
    os.chdir(ROOT)
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.scenarios).read_text())
    scenarios = [s for s in config if not args.names or s["name"] in args.names]
    if not scenarios:
        raise ValueError("No matching scenarios")
    catalog = {a["name"]: a for a in json.loads((LIVE / "tabletop_catalog.json").read_text())}
    client = Client(ROOT / "runs/arx_curobo/mailbox")
    setup = output / "initial_setup.json"
    if not setup.exists():
        shutil.copy2(ROOT / "runs/tabletop_trials/initial_setup.json", setup)
    saved_scenarios = output / "scenarios.json"
    known = json.loads(saved_scenarios.read_text()) if saved_scenarios.exists() else []
    merged = {s["name"]: s for s in known}
    merged.update({s["name"]: s for s in config})
    write(saved_scenarios, list(merged.values()))
    session = str(time.time_ns())
    baseline = {"planner": "prm_trajopt", "bound_weight_multiplier": 1,
                "execution_policy": args.policy,
                "fresh_zerograsp": not any(s.get("paired_baseline") for s in scenarios),
                "initial_proposal_protocol": "Fresh inference unless scenario pairs a saved baseline proposal; feedback observations are always fresh",
                "seed": 0, "curobo_gpu": args.gpu,
                "trial_timeout_seconds": args.timeout,
                "physics_or_controller_changes": False, "self_collision_enabled": False,
                "success": "Bilateral finger contacts, accepted stage convergence, object >= 6 cm above initial throughout final hold",
                "scene_variation": "Only target pose; original size, mass and materials",
                "source_sha256": {}}
    source = output / "source_snapshot" / session
    source.mkdir(parents=True, exist_ok=False)
    for path in [*Path("scripts/curobo_arx").glob("*.py"), *Path("scripts/curobo_arx").glob("*.sh"),
                 Path("arx-sim/config/runtime.yaml"), Path("arx-sim/config/controllers.yaml")]:
        destination = source / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        baseline["source_sha256"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    baseline_path = output / f"baseline_{session}.json"
    write(baseline_path, baseline)
    if not (output / "baseline.json").exists():
        write(output / "baseline.json", baseline)
    results_path = output / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    for scenario in scenarios:
        name = scenario["name"]
        folder = output / name
        folder.mkdir(exist_ok=False)
        result = {"scenario": scenario, "success": False, "started_unix": time.time(),
                  "baseline_snapshot": str(baseline_path), "execution_policy": args.policy}
        print("START", name, flush=True)
        try:
            write(folder / "scenario.json", scenario)
            for pattern in ("*_optimization.json", "*_command.json", "*_execution.json", "*_settling.json", "*_prm_seed.json", "feedback_history*_goal.json"):
                for stale in LIVE.glob(pattern):
                    stale.unlink()
            for stale in ("selected_grasp.json", "closed_snapshot.json", "candidate_diagnostics.json",
                          "trajopt_effective_config.json", "fixed_trial_state.json", "input_report.json",
                          "visual_feedback_report.json", "selected_grasp_initial.json"):
                (LIVE / stale).unlink(missing_ok=True)
            for stale in ("research_variant.json", "ik_diagnostics.json", "sphere_fit.json", "selected_grasp_candidate.json", "goal_check_report.json", "shape_policy.json", "selected_grasp_original.json", "geometry_policy.json", "tracking_robustness.json", "tracking_feedback.json", "thin_fit.json", "thin_policy.json", "thin_candidate_diagnostics.json", "ring_fit.json", "ring_policy.json", "ring_candidate_diagnostics.json", "ring_preshape_report.json", "ring_preshape_snapshot.json", "execution_initial_snapshot.json"):
                (LIVE / stale).unlink(missing_ok=True)
            (LIVE / "gripper_model_projection.json").unlink(missing_ok=True)
            if (LIVE / "feedback").exists():
                shutil.rmtree(LIVE / "feedback")
            if (LIVE / "zerograsp").exists():
                (LIVE / "zerograsp").rename(LIVE / f"zerograsp_previous_{time.time_ns()}")
            write(LIVE / "grasp_report.json", {"success": False, "status": "input_pending"})
            guard = os.environ.get("ARX_RESEARCH_GUARD_FILE")
            if guard:
                from research_guard import check
                check(client, guard, name, 'before_setup')
            target = scenario.get("target", "test_object")
            asset = catalog[scenario.get("asset", "bbq_sauce_bottle")]
            write(LIVE / "active_asset.json", asset)
            if guard:
                from research_guard import invariant_digest
                prepared = client.request('grasp_research_setup', scenario=scenario, timeout=60)
                write(folder / 'atomic_setup.json', prepared)
                evidence = prepared['fingerprint']
                expected_guard = json.loads(Path(guard).read_text())
                if invariant_digest(evidence) != expected_guard['invariant_sha256']:
                    raise RuntimeError('Research configuration guard failed after atomic setup')
                write(folder / 'initial_snapshot.json', prepared['snapshot'])
                write(folder / 'initial_fingerprint.json', evidence)
                write(LIVE / 'capture_snapshot.json', prepared['snapshot'])
                from research_layout import compare_layout
                reference = scenario.get('layout_reference')
                if reference:
                    control = (output / reference).resolve()
                    expected_setup = json.loads((control / 'atomic_setup.json').read_text())
                    expected = json.loads((control / 'initial_fingerprint.json').read_text())
                    diagnostic = compare_layout(expected['scene_state'], evidence['scene_state'])
                    write(folder / 'post_settle_layout_diagnostic.json', diagnostic)
                    reset_equal = prepared['initial_conditions'] == expected_setup['initial_conditions']
                    layout_check = {'pass': reset_equal, 'exact_declared_reset_equal': reset_equal,
                        'scope': 'Exact joint/object reset inputs and zero velocities; post-3s contact dynamics reported separately, not a configuration change',
                        'post_settle_state_equal_within_diagnostic_tolerances': diagnostic['pass']}
                    write(folder / 'initial_layout_check.json', layout_check)
                    if not reset_equal:
                        raise RuntimeError('Frozen declared initial layout mismatch')
            else:
                shutil.copy2(setup, LIVE / "capture_snapshot.json")
                client.request("grasp_reset_setup")
                for obj, position in [("blue_cylinder", [4.33, 1.79, .76]),
                                      ("green_sphere", [4.46, 1.96, .76]),
                                      ("red_cube", [4.22, 1.75, .73])]:
                    client.request("place_object", name=obj, position=position)
                placement = scenario["placement"] if target == "test_object" else {"position": [10, 10, .4]}
                client.request("grasp_asset_load", asset_path=asset["container_path"], size_m=asset["size_m"], **placement)
                if target != "test_object":
                    client.request("place_object", name=target, **scenario["placement"])
                time.sleep(3)
                write(folder / "initial_snapshot.json", client.request("snapshot"))
            write(folder / "scenario.json", scenario)
            env = {k: v for k, v in os.environ.items() if not k.startswith("GRASP_")}
            env.update(GRASP_PLANNER="prm_trajopt", GRASP_BOUND_WEIGHT_MULTIPLIER="1",
                       GRASP_TARGET=target, CUROBO_GPU=str(args.gpu), GPU="1",
                       GRASP_VIDEO_DIR="/workspace/COOLAS-ARX/" + str(folder.relative_to(ROOT) / "frames"))
            if args.policy == "research_depth":
                if not scenario.get("paired_baseline"):
                    raise ValueError("Research depth policy requires a paired proposal")
                depth_shift = float(scenario["research_depth_shift_m"])
                if not -.012 <= depth_shift <= 0:
                    raise ValueError("Depth shift outside isolated H001 search space")
                env["GRASP_RESEARCH_DEPTH_SHIFT_M"] = str(depth_shift)
                env["GRASP_RESEARCH_HYPOTHESIS"] = scenario.get("hypothesis_id", "depth_ablation")
            if args.policy == "sphere_refinement":
                height = float(scenario.get("sphere_pad_height_m", .005))
                if height not in (.005, .010):
                    raise ValueError("Sphere pad height outside preregistered search space")
                env["GRASP_SPHERE_PAD_HEIGHT_M"] = str(height)
            if scenario.get("paired_baseline"):
                paired = (output / scenario["paired_baseline"]).resolve()
                proposal = paired / "selected_grasp.json"
                if not proposal.is_file():
                    if not scenario.get('fresh_if_no_paired_proposal'):
                        raise ValueError("Paired baseline did not save an initial proposal")
                    result['input_protocol'] = 'Fresh end-to-end inference; control produced no proposal'
                    paired = None
                paired_scene = json.loads(((paired or (output / scenario['paired_baseline']).resolve()) / "scenario.json").read_text())
                if (paired_scene["placement"] != scenario["placement"]
                        or paired_scene.get("target", "test_object") != target
                        or paired_scene.get("asset") != scenario.get("asset")):
                    raise ValueError("Paired initial proposal target/placement mismatch")
                if paired:
                    env["GRASP_FIXED_SELECTION"] = str(proposal)
                    result["paired_initial_proposal"] = str(proposal)
                    result['input_protocol'] = 'Same freshly generated control ZeroGrasp proposal; fresh RGB-D refinement observation'
                    shutil.copytree(paired / "zerograsp", LIVE / "zerograsp")
            if args.policy == 'geometry_candidates' and scenario.get('paired_baseline'):
                control = (output / scenario['paired_baseline']).resolve()
                env['GRASP_REPLAY_REFERENCE'] = str(control)
                if not (control / 'zerograsp/object_1.grasp.npy').is_file():
                    raise RuntimeError('Control did not archive a valid frozen network candidate input')
                if not (LIVE / 'zerograsp').exists():
                    shutil.copytree(control / 'zerograsp', LIVE / 'zerograsp')
                env['GRASP_REUSE_NETWORK_INPUT'] = '1'
                result['frozen_prediction_sha256'] = hashlib.sha256((control / 'zerograsp/object_1.grasp.npy').read_bytes()).hexdigest()
                result['candidate_prediction_sha256'] = hashlib.sha256((LIVE / 'zerograsp/object_1.grasp.npy').read_bytes()).hexdigest()
                if result['frozen_prediction_sha256'] != result['candidate_prediction_sha256']:
                    raise RuntimeError('Frozen network candidate tensor mismatch')
                result['input_protocol'] = 'Same frozen fresh-control ZeroGrasp candidate tensor; fresh RGB-D geometry; original control pose available only if selected'
            with (folder / "run.log").open("w") as log:
                script = {"baseline": "run_grasp.sh", "visual_feedback": "run_feedback_grasp.sh",
                          "research_depth": "run_depth_grasp.sh",
                          "sphere_refinement": "run_sphere_grasp.sh",
                          "shape_conditioned": "run_shape_grasp.sh",
                          "geometry_candidates": "run_contact_drag.sh",
                          "visual_feedback_feasible": "run_feedback_feasible_grasp.sh",
                          "visual_feedback_envelope": "run_feedback_envelope_grasp.sh"}[args.policy]
                process = subprocess.Popen(["bash", "scripts/curobo_arx/" + script], env=env,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = process.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=15)
                    raise RuntimeError(f"Trial exceeded {args.timeout} seconds")
            result.update(json.loads((LIVE / "grasp_report.json").read_text()))
            result["process_exit_code"] = code
            if code and not result.get("error"):
                result["error"] = "Pipeline failed before grasp report; see run.log"
        except Exception as exc:
            result.update(success=False, error=str(exc))
        finally:
            try:
                stopped = client.request("grasp_video_stop", timeout=30)
                if not stopped.get("frames"):
                    client.request("grasp_video_start", directory="/workspace/COOLAS-ARX/" + str(folder.relative_to(ROOT) / "frames"))
                    time.sleep(3)
                    client.request("grasp_video_stop")
                time.sleep(1)
                write(folder / "final_snapshot.json", client.request("snapshot", timeout=30))
                if os.environ.get("ARX_RESEARCH_GUARD_FILE"):
                    evidence = check(client, os.environ['ARX_RESEARCH_GUARD_FILE'], name, 'after_trial')
                    write(folder / 'final_fingerprint.json', evidence)
            except Exception as exc:
                result["recording_error"] = str(exc)
                if "Research configuration guard failed" in str(exc):
                    result.update(success=False, error=str(exc), configuration_guard_failed=True)
            # Archive only this attempt's defined pipeline artifacts. Do not copy
            # old gain/solver diagnostics or overwrite initial/final evidence.
            pipeline_files = {"grasp_report.json", "capture_snapshot.json", "camera_world_optical.json",
                              "input_report.json", "selected_grasp.json", "closed_snapshot.json",
                              "candidate_diagnostics.json", "trajopt_effective_config.json", "fixed_trial_state.json",
                              "fk_report.json", "scene_sources.json", "active_asset.json", "sphere_fit.json",
                              "shape_policy.json", "geometry_policy.json", "tracking_robustness.json", "tracking_feedback.json", "replay_observation_check.json", "thin_fit.json", "thin_policy.json", "thin_candidate_diagnostics.json", "ring_fit.json", "ring_policy.json", "ring_candidate_diagnostics.json", "ring_preshape_report.json", "ring_preshape_snapshot.json", "execution_initial_snapshot.json", "selected_grasp_original.json", "selected_grasp_candidate.json",
                              "preparation_error.json", "preparation_report.json", "preparation_final_snapshot.json", "drag_geometry.json", "drag_start_check.json", "drag_waypoints.json", "drag_release_snapshot.json", "contact_drag_cartesian_check.json", "goal_check_report.json", "research_variant.json", "ik_diagnostics.json"}
            copied = []
            for path in LIVE.glob("*.json"):
                if path.stat().st_mtime < result["started_unix"]:continue
                if path.name in pipeline_files or any(path.match(pattern) for pattern in
                    ("*_optimization.json", "*_command.json", "*_execution.json", "*_settling.json", "*_prm_seed.json", "feedback_history*_goal.json")):
                    shutil.copy2(path, folder / path.name)
                    copied.append(path.name)
            write(folder / 'archive_manifest.json', {'pipeline_files': copied,
                'scope': 'Explicit pipeline artifacts; legacy scratch gain/solver JSON excluded'})
            for subdir in ("camera", "zerograsp", "feedback"):
                if (LIVE / subdir).exists():
                    shutil.copytree(LIVE / subdir, folder / subdir, dirs_exist_ok=True)
            # Immutable model/world evidence for later read-only diagnosis.
            model_evidence=folder/'model_evidence';model_evidence.mkdir()
            for filename in ('scene.yml','scene_sources.json','snapshot.json'):
                if (LIVE/filename).is_file() and (LIVE/filename).stat().st_mtime>=result['started_unix']:shutil.copy2(LIVE/filename,model_evidence/filename)
            for subdir in ('model','model_closed','meshes'):
                if (LIVE/subdir).is_dir():
                    for f in (LIVE/subdir).rglob('*'):
                        if f.is_file() and f.stat().st_mtime>=result['started_unix']:
                            archive_target=model_evidence/subdir/f.relative_to(LIVE/subdir);archive_target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,archive_target)
            if (client.root/'scene_meshes.json').is_file() and (client.root/'scene_meshes.json').stat().st_mtime>=result['started_unix']:shutil.copy2(client.root/'scene_meshes.json',model_evidence/'scene_meshes.json')
            write(model_evidence/'provenance.json',{'phase':'End of attempt; open model may contain measured finger STATE after ordinary overhead preshape; closed model may contain measured closing state after bilateral gate',
                'scope':'Only artifacts written after attempt start copied; old unused mesh/model files excluded. No model or physical configuration changed',
                'sha256':{str(f.relative_to(model_evidence)):hashlib.sha256(f.read_bytes()).hexdigest() for f in model_evidence.rglob('*') if f.is_file()}})
            try:
                result["recording"] = encode(folder)
            except Exception as exc:
                result["video_error"] = str(exc)
            if result.get('recording'):
                try:
                    from archive_recorded_frames import archive
                    result['lossless_raw_recording']=archive(folder,remove_source=True)
                except Exception as exc:
                    # Archive validates every pixel before removing any PNG.
                    # Failures retain original PNGs; grasp metrics are separate.
                    result['lossless_archive_error']=repr(exc)
            result["outcome"] = classify(result, folder)
            result["finished_unix"] = time.time()
            result["folder"] = str(folder)
            write(folder / "result.json", result)
            results.append(result)
            write(results_path, results)
            print("RESULT", name, json.dumps({k: result.get(k) for k in ("success", "outcome", "error", "lift_m", "video_error")}), flush=True)
        if result["outcome"] == "infrastructure_or_input_failure":
            raise RuntimeError("Stop and inspect infrastructure/input failure before continuing")
    print("BATCH_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--names", nargs="*")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=720)
    parser.add_argument("--policy", choices=("baseline", "visual_feedback", "visual_feedback_feasible", "visual_feedback_envelope", "research_depth", "sphere_refinement", "shape_conditioned", "geometry_candidates"), default="baseline")
    run(parser.parse_args())
