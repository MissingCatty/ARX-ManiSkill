"""Plan only the open-gripper transit to an original native pregrasp H."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "maniskill/planning_common"))
from grasp_selector import decode_candidate


def select_pregrasp(
    planner: Any,
    state: dict,
    ordered: np.ndarray,
    camera: np.ndarray,
    solutions: dict[int, Any] | None = None,
    max_path_attempts: int | None = None,
) -> dict:
    """Use native score order and preserve every check on the executed H path."""
    start = planner.measured(state)
    rejected = {"pregrasp_ik": 0, "pregrasp_path": 0, "not_queried": 0}
    path_attempts = 0
    for index, row in enumerate(ordered):
        grasp, pregrasp, lift, contact = decode_candidate(row, camera)
        if solutions is not None and index not in solutions:
            rejected["not_queried"] += 1
            continue
        qh = planner.ik(pregrasp, start) if solutions is None else solutions.get(index)
        if qh is None:
            rejected["pregrasp_ik"] += 1
            continue
        if max_path_attempts is not None and path_attempts >= max_path_attempts:
            return {"success": False, "reason": "registered_path_budget_exhausted",
                    "rejected": rejected, "path_attempts": path_attempts,
                    "planning_scope": "pregrasp_entry_only", "full_grasp_validated": False}
        path_attempts += 1
        path = planner.path(start, qh)
        if path is None:
            rejected["pregrasp_path"] += 1
            continue
        # Original optimizer, endpoint/limit/dense collision validation and
        # time-stretch procedure. No PRM-only fallback after optimizer failure.
        optimized = planner.optimize("pregrasp", path, 6.0)
        return {
            "success": True, "entry_feasible": True,
            "planning_scope": "pregrasp_entry_only", "full_grasp_validated": False,
            "physical_execution": False, "paths": {"pregrasp": optimized},
            "selected": {
                "index": index, "score": float(row[0]), "network_row": row.tolist(),
                "pregrasp_world": pregrasp.tolist(), "q_pre": qh.tolist(),
                "grasp_world": grasp.tolist(), "lift_world": lift.tolist(),
                "target_contact_world": contact.tolist(),
                "q_grasp": None, "q_lift": None,
                "future_goals_role": "network metadata only; not executed or validated",
                "source": "ZeroGrasp native postprocessed row; original H conversion",
                "height_adjustment_m": 0.0, "additional_orientation_search": False,
            },
            "rejected": rejected, "path_attempts": path_attempts,
            "geometric_prefilters": False,
        }
    return {"success": False, "reason": "no_reachable_native_pregrasp",
            "planning_scope": "pregrasp_entry_only", "full_grasp_validated": False,
            "physical_execution": False, "rejected": rejected,
            "path_attempts": path_attempts, "count": len(ordered)}
