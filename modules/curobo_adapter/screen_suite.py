"""Reuse loaded CUDA modules while rebuilding each declared solver configuration."""
from __future__ import annotations

import argparse
import contextlib
import datetime
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch

from adapter import HERE, digest
import probe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=["screen", "known"], default="screen")
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--revision", default="")
    parser.add_argument("--execution-registration", default="execution_registration.json")
    parser.add_argument("--screen-registration", default="screen_registration.json")
    args = parser.parse_args()
    root = args.root.resolve()
    scope = json.loads((root / "scope.json").read_text())
    registration = json.loads((root / args.execution_registration).read_text())
    screen_reg = json.loads((root / args.screen_registration).read_text())
    expected_variants = screen_reg.get("phase_variants", {}).get(args.phase, screen_reg["variants"])
    assert args.variants == expected_variants, "Use the registered configuration order"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == registration["gpu_uuid"]
    known = json.loads((root / "known_goals.json").read_text()) if args.phase == "known" else None
    for variant in args.variants:
        label = args.phase + ("_" + args.revision if args.revision else "") + "_" + variant
        out = root / label
        out.mkdir(exist_ok=False)
        results = []
        for case in scope["cases"]:
            if known is not None and case["id"] not in known:
                continue
            for path, expected in {**scope["protected_legacy_hashes"], **scope["registered_input_hashes"],
                                  **registration["source_hashes"], **screen_reg["source_hashes"]}.items():
                assert digest(Path(path)) == expected, "Protected source/input changed: " + path
            assert not subprocess.check_output(["git", "-C", str(HERE / "source"), "diff", "--name-only"], text=True).strip()
            assert time.time() - datetime.datetime.fromisoformat(scope["registered_at"]).timestamp() < scope["maximum_wall_seconds"]
            indices = case["screen_indices"]
            count = 8 if known is not None else len(indices)
            requests = sum(len(p.read_text().splitlines()) for p in root.rglob("requests.jsonl"))
            assert requests + count <= scope["maximum_native_ik_calls_including_integration"]
            memory = int(subprocess.check_output(["nvidia-smi", "-i", registration["gpu_uuid"],
                "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
            assert memory >= scope["minimum_free_mib_before_child"], "Insufficient free GPU memory"
            argv = ["probe.py", "--case", case["id"], "--variant", variant, "--out", str(out / case["id"]),
                    "--indices", ",".join(map(str, indices))]
            if known is not None:
                argv += ["--known-goals", str(root / "known_goals.json")]
            print("START", args.phase, variant, case["id"], count, flush=True)
            previous = sys.argv
            try:
                sys.argv = argv
                with (out / (case["id"] + ".log")).open("x") as log:
                    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                        try:
                            probe.main()
                        except Exception:
                            traceback.print_exc()
                            raise
            finally:
                sys.argv = previous
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            directory = out / case["id"]
            result = json.loads((directory / "result.json").read_text())
            assert len((directory / "requests.jsonl").read_text().splitlines()) == count
            assert len((directory / "calls.jsonl").read_text().splitlines()) == count
            results.append(result)
            (out / "progress.json").write_text(json.dumps({"results": results}, indent=2))
            print("DONE", variant, case["id"], result["gates"], flush=True)
        summary = {"phase": args.phase, "variant": variant, "completed_cases": len(results),
                   "native_calls": sum(r["candidate_count"] for r in results),
                   "accepted": sum(r["gates"].get("accepted", 0) for r in results),
                   "results": results, "physical_trials": 0}
        (out / "result.json").write_text(json.dumps(summary, indent=2))
        print("COMPLETE", args.phase, variant, summary["accepted"], flush=True)


if __name__ == "__main__":
    main()
