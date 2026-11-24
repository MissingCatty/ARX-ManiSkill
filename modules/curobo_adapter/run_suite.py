"""Bounded serial model queries in an isolated environment; never run simulation."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=["full", "screen", "expanded", "known"], required=True)
    parser.add_argument("--variant", default="official32")
    args = parser.parse_args()
    root = args.root.resolve()
    scope = json.loads((root / "scope.json").read_text())
    registration = json.loads((root / "execution_registration.json").read_text())
    out = root / (args.phase + "_" + args.variant)
    out.mkdir(exist_ok=False)
    known = json.loads((root / "known_goals.json").read_text()) if args.phase == "known" else None
    results = []
    for case in scope["cases"]:
        if known is not None and case["id"] not in known:
            continue
        for path, expected in {**scope["protected_legacy_hashes"], **scope["registered_input_hashes"],
                               **registration["source_hashes"]}.items():
            assert sha(Path(path)) == expected, "Protected file changed: " + path
        assert subprocess.check_output(["git", "-C", str(HERE / "source"), "rev-parse", "HEAD"],
                                        text=True).strip() == scope["official_commit"]
        assert not subprocess.check_output(["git", "-C", str(HERE / "source"), "diff", "--name-only"], text=True).strip()
        age = time.time() - datetime.datetime.fromisoformat(scope["registered_at"]).timestamp()
        assert age < scope["maximum_wall_seconds"], "Registered wall budget exhausted"
        requests = sum(len(p.read_text().splitlines()) for p in root.rglob("requests.jsonl"))
        next_calls = 8 if known is not None else len(case["screen_indices"] if args.phase == "screen" else case["indices"])
        assert requests + next_calls <= scope["maximum_native_ik_calls_including_integration"]
        gpu = registration["gpu_uuid"]
        memory = int(subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-gpu=memory.free",
                                              "--format=csv,noheader,nounits"], text=True).strip())
        assert memory >= scope["minimum_free_mib_before_child"], f"Insufficient free GPU memory: {memory} MiB"
        directory = out / case["id"]
        indices = case["screen_indices"] if args.phase == "screen" else case["indices"]
        command = [sys.executable, str(HERE / "probe.py"), "--case", case["id"], "--variant", args.variant,
                   "--out", str(directory), "--indices", ",".join(map(str, indices))]
        if known is not None:
            command += ["--known-goals", str(root / "known_goals.json")]
        print("START", args.phase, args.variant, case["id"], "queries", next_calls, flush=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONDONTWRITEBYTECODE="1")
        with (out / (case["id"] + ".log")).open("x") as log:
            run = subprocess.run(command, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600)
        if run.returncode:
            raise RuntimeError(f"Child failed; preserve partial folder and stop: {case['id']}, exit {run.returncode}")
        result = json.loads((directory / "result.json").read_text())
        assert len((directory / "requests.jsonl").read_text().splitlines()) == result["candidate_count"]
        assert len((directory / "calls.jsonl").read_text().splitlines()) == result["candidate_count"]
        results.append(result)
        (out / "progress.json").write_text(json.dumps({"complete_cases": len(results), "results": results}, indent=2))
        print("DONE", case["id"], result["gates"], flush=True)
    summary = {"phase": args.phase, "variant": args.variant, "completed_cases": len(results),
               "native_calls": sum(r["candidate_count"] for r in results),
               "accepted": sum(r["gates"].get("accepted", 0) for r in results), "results": results,
               "physical_trials": 0}
    (out / "result.json").write_text(json.dumps(summary, indent=2))
    print("COMPLETE", args.phase, args.variant, summary["native_calls"], summary["accepted"], flush=True)


if __name__ == "__main__":
    main()
