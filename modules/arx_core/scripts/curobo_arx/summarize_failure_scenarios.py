"""Summarize measured ARX grasp outcomes without changing success criteria."""
import argparse
import json
from pathlib import Path
import shutil


def summarize(output):
    rows = []
    for result in json.loads((output / "results.json").read_text()):
        folder = Path(result["folder"])
        scenario = result["scenario"]
        target = scenario.get("target", "test_object")
        initial = json.loads((folder / "capture_snapshot.json").read_text())
        start = [initial["objects"][target]["world"][i][3] for i in range(3)]
        target_path = initial["objects"][target]["path"]
        limits = [initial["joint_limits"][initial["joint_names"].index(n)] for n in initial["right_arm"]]
        row = {"name": scenario["name"], "success": result["success"],
               "outcome": result["outcome"], "error": result.get("error"),
               "target": target, "asset": scenario.get("asset", target),
               "requested_pose": scenario["placement"], "actual_initial_position_m": start,
               "folder": str(folder), "video": result.get("recording", {}).get("video"),
               "stages": {}, "optimization": {}}
        maximum = 0.0
        for stage in ("pregrasp", "approach_mid", "approach", "close", "lift", "hold"):
            path = folder / f"{stage}_execution.json"
            if path.exists():
                execution = json.loads(path.read_text())
                records = execution["records"]
                heights = [r["objects"][target][2][3] - start[2] for r in records]
                contacts = [e for e in execution["contacts"] if e["num_contacts"] > 0 and target_path in e["a"] + e["b"]]
                external = [e for e in execution["contacts"] if e["num_contacts"] > 0
                            and not ("/World/Robot/" in e["a"] and "/World/Robot/" in e["b"])]
                unwanted = [e for e in external if not (target_path in e["a"] + e["b"]
                            and any(n in e["a"] + e["b"] for n in ("right_link27", "right_link28")))]
                fingers = sorted({n for e in contacts for n in ("right_link27", "right_link28") if n in e["a"] + e["b"]})
                maximum = max(maximum, *[max(0., lo - q, q - hi) for r in records for q, (lo, hi) in zip(r["position"], limits)])
                final_xyz = [records[-1]["objects"][target][i][3] for i in range(3)]
                row["stages"][stage] = {
                    "object_min_rise_m": min(heights), "object_max_rise_m": max(heights),
                    "object_final_rise_m": heights[-1], "final_position_m": final_xyz,
                    "final_gripper_joints_m": records[-1]["gripper"],
                    "contact_fingers_during_stage": fingers,
                    "contact_event_count": len(contacts),
                    "unexpected_contact_events": len(unwanted),
                    "sim_seconds": execution["sim_duration_s"],
                    "wall_seconds": execution["wall_duration_s"],
                    "realtime_ratio": execution["sim_duration_s"] / execution["wall_duration_s"],
                }
                meta_path = folder / "frames/frames.json"
                if meta_path.exists():
                    frames = json.loads(meta_path.read_text())["frames"]
                    nearest = min(frames, key=lambda f: abs(f["sim_time"] - execution["final"]["sim_time"]))
                    shutil.copy2(folder / "frames" / nearest["file"], folder / f"{stage}_end.png")
                    if stage == "lift":
                        peak_record = max(records, key=lambda r: r["objects"][target][2][3])
                        peak_time = execution["final"]["sim_time"] - execution["sim_duration_s"] + peak_record["time"]
                        peak_frame = min(frames, key=lambda f: abs(f["sim_time"] - peak_time))
                        shutil.copy2(folder / "frames" / peak_frame["file"], folder / "peak_lift.png")
            optimized = folder / f"{stage}_optimization.json"
            if optimized.exists():
                values = json.loads(optimized.read_text())
                row["optimization"][stage] = {k: values.get(k) for k in
                    ("success", "attempts", "max_limit_exceed_rad", "invalid_samples", "planning_wall_seconds")}
        row["measured_joint_limit_max_exceed_rad"] = maximum
        # The pipeline's final assertion mentions both hold and convergence.
        # Use its explicit object_held flag and measured height to disambiguate.
        if result.get("object_held") is False and "hold" in row["stages"]:
            peak = row["stages"].get("lift", {}).get("object_max_rise_m", 0.)
            row["outcome"] = "lifted_then_slipped" if peak > .02 else "failed_lift_or_hold"
            row["all_stage_convergence_accepted"] = all(v["settled"] for v in result.get("stage_convergence", {}).values())
        candidate_path = folder / "candidate_diagnostics.json"
        if candidate_path.exists():
            row["candidates"] = json.loads(candidate_path.read_text())
        rows.append(row)
    report = {"trials": len(rows), "successes": sum(r["success"] for r in rows),
              "interpretation": "Challenge discovery and any repeats are selected, not an unbiased benchmark success-rate estimate.",
              "rows": rows}
    (output / "report.json").write_text(json.dumps(report, indent=2))
    for row in rows:
        print(row["name"], row["outcome"], row["error"] or "",
              "hold rise:", row["stages"].get("hold", {}).get("object_min_rise_m"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    summarize(parser.parse_args().output.resolve())
