"""Summarize preserved model results without changing gates or pending outcomes."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE / "research/20261117_01"


def load_records(path: Path) -> dict[int, dict]:
    return {r["index"]: r for r in map(json.loads, path.read_text().splitlines())}


def main() -> None:
    scope = json.loads((ROOT / "scope.json").read_text())
    oldroot = Path(scope["old_result_root"])
    result = {"cases": [], "screens": {}, "physical_trials": 0}
    arms = [p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith(("screen_", "expanded_", "known_"))]
    for case in scope["cases"]:
        old = load_records(oldroot / case["id"] / "calls.jsonl")
        directory = ROOT / "full_official32" / case["id"]
        row = {"id": case["id"], "control": case["control"], "registered": len(case["indices"]),
               "old_accepted": sum(old[i]["gate"] == "accepted" for i in case["indices"]),
               "new_complete": (directory / "result.json").exists()}
        if row["new_complete"]:
            new = load_records(directory / "calls.jsonl")
            assert set(new) == set(case["indices"])
            row.update(new_accepted=sum(r["gate"] == "accepted" for r in new.values()),
                       recovered_indices=[i for i in new if new[i]["gate"] == "accepted" and old[i]["gate"] != "accepted"],
                       lost_indices=[i for i in new if new[i]["gate"] != "accepted" and old[i]["gate"] == "accepted"],
                       new_pose_converged=sum(r["converged_seeds"] > 0 for r in new.values()))
        result["cases"].append(row)
    for arm in sorted(arms):
        summary = arm / "result.json"
        entry = {"complete": summary.exists(), "cases": {}}
        for directory in sorted(arm.iterdir()):
            if not directory.is_dir() or not (directory / "result.json").exists():
                continue
            records = load_records(directory / "calls.jsonl")
            entry["cases"][directory.name] = {"queries": len(records),
                "accepted_indices": [i for i, r in records.items()
                                     if r.get("gate") == "accepted" or r.get("accepted") is True],
                "pose_converged_indices": ([i for i, r in records.items() if r["converged_seeds"] > 0]
                                             if all("converged_seeds" in r for r in records.values()) else None)}
        result["screens"][arm.name] = entry
    request_files = list(ROOT.rglob("requests.jsonl"))
    result["native_requests_including_integration"] = sum(len(p.read_text().splitlines()) for p in request_files)
    result["native_receipts_including_integration"] = sum(
        len((p.parent / "calls.jsonl").read_text().splitlines())
        for p in request_files if (p.parent / "calls.jsonl").exists())
    result["completed_main_candidates"] = sum(c["registered"] for c in result["cases"] if c["new_complete"])
    result["main_accepted"] = sum(c.get("new_accepted", 0) for c in result["cases"])
    result["failed_case_candidates_recovered"] = sum(c.get("new_accepted", 0) for c in result["cases"] if not c["control"])
    (ROOT / "comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ["cases", "screens"]}))


if __name__ == "__main__":
    main()
