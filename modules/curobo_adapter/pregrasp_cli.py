"""Opt-in offline H-entry planner. Does not replace the existing planner CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np

from adapter import ARXPlanner, save, flush_planning_artifacts
from planner_cli import build_frozen_models, SelectorAdapter, load_candidates
from entry_selector import select_pregrasp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-ik-queries", type=int, default=256)
    args = parser.parse_args()
    source = args.input.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rows, metadata = load_candidates(source)
    state = json.loads((source / "state.json").read_text())
    if len(rows) == 0:
        save(out / "plan_result.json", {"success": False, "reason": "no_postprocessed_candidates",
             "planning_scope": "pregrasp_entry_only", "full_grasp_validated": False,
             "physical_execution": False})
        return
    build_frozen_models(state, out / "input")
    planner = ARXPlanner(out / "input", out / "model")
    try:
        wrapped = SelectorAdapter(planner, out / "requests.jsonl", args.max_ik_queries)
        result = select_pregrasp(wrapped, state, rows[np.argsort(-rows[:, 0], kind="stable")],
                                 np.asarray(state["camera_world_optical"]))
        result.update(wall_seconds=time.monotonic()-started,
            candidate_input={"stage":metadata["candidate_stage"], "count":len(rows),
                             "raw_fallback":False, "source":str(source)},
            backend="isolated cuRobo v0.9.0; explicit pregrasp-entry experiment")
        save(out / "plan_result.json", result)
        print(json.dumps({"entry_feasible":result["success"], "out":str(out),
                          "full_grasp_validated":False}), flush=True)
    except Exception:
        save(out / "plan_result.json", {"success":False,"reason":"planner_error",
            "error":traceback.format_exc(),"planning_scope":"pregrasp_entry_only",
            "full_grasp_validated":False,"physical_execution":False})
        raise
    finally:
        try:
            flush_planning_artifacts()
        finally:
            planner.planner.destroy()


if __name__ == "__main__":
    main()
