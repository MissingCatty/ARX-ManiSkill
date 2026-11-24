#!/usr/bin/env python3
"""Compare M0 and 3/4/5-view M5-table collision-world benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Iterable

import numpy as np


DEFAULT_M0_ROOT = Path("maniskill_curobo_real/runs/pickclutter_full_depth_m0_seed1_200")
DEFAULT_VIEW3_ROOT = Path(
    "maniskill_curobo_real/runs/pickclutter_m5_instance_esdf_table_seed1_200"
)
DEFAULT_VIEW4_ROOT = Path(
    "maniskill_curobo_real/runs/pickclutter_m5_instance_esdf_table_4view_seed1_200"
)
DEFAULT_VIEW5_ROOT = Path(
    "maniskill_curobo_real/runs/pickclutter_m5_instance_esdf_table_5view_seed1_200"
)
BASE_STAGE = "m5_multiview_instance_voxel_esdf_table"
M0_PRIOR_MEAN_RUNTIME_SEC = 6.40


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=200)
    parser.add_argument("--m0-root", type=Path, default=DEFAULT_M0_ROOT)
    parser.add_argument("--view3-root", type=Path, default=DEFAULT_VIEW3_ROOT)
    parser.add_argument("--view4-root", type=Path, default=DEFAULT_VIEW4_ROOT)
    parser.add_argument("--view5-root", type=Path, default=DEFAULT_VIEW5_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    roots = {
        "m0": args.m0_root.expanduser().resolve(),
        "view3": args.view3_root.expanduser().resolve(),
        "view4": args.view4_root.expanduser().resolve(),
        "view5": args.view5_root.expanduser().resolve(),
    }
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else roots["view5"] / "analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(int(args.seed_start), int(args.seed_end) + 1))

    records = {
        "m0": load_jsonl_records(roots["m0"] / "records.jsonl", seeds),
        "view3": load_m5_summary(roots["view3"], view_count=3, seeds=seeds),
        "view4": load_m5_summary(roots["view4"], view_count=4, seeds=seeds),
        "view5": load_m5_summary(roots["view5"], view_count=5, seeds=seeds),
    }
    metrics = build_metrics(records, seeds)
    rows = build_rows(records, seeds)

    csv_path = output_dir / "m0_vs_m5_3view_4view_5view_per_seed.csv"
    write_csv(csv_path, rows)
    summary_path = output_dir / "m0_vs_m5_3view_4view_5view_summary.json"
    summary_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path = output_dir / "m0_vs_m5_3view_4view_5view_report.md"
    report_path.write_text(render_report(metrics, csv_path), encoding="utf-8")
    print(
        json.dumps(
            {"report": str(report_path), "summary": str(summary_path), "csv": str(csv_path)},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def load_jsonl_records(path: Path, seeds: list[int]) -> dict[int, dict]:
    loaded: dict[int, dict] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                loaded[int(record["seed"])] = normalize_record(record)
    return fill_missing(loaded, seeds, source=path)


def load_m5_summary(root: Path, *, view_count: int, seeds: list[int]) -> dict[int, dict]:
    suffix = "" if view_count == 3 else f"_{view_count}view"
    path = root / f"{BASE_STAGE}{suffix}_summary.json"
    loaded: dict[int, dict] = {}
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        for record in payload.get("records", []):
            loaded[int(record["seed"])] = normalize_record(record)
    return fill_missing(loaded, seeds, source=path)


def normalize_record(record: dict) -> dict:
    normalized = dict(record)
    normalized["success"] = bool(record.get("object_lift_success"))
    normalized["outcome"] = str(
        record.get("outcome") or record.get("status") or "unknown"
    )
    runtime = record.get("total_runtime_sec")
    if runtime is None:
        runtime = record.get("end_to_end_runtime_sec")
    normalized["runtime_sec"] = runtime
    command = record.get("command") or {}
    instance_scene = record.get("instance_scene") or {}
    execution_runtime = record.get("execution_runtime_sec", command.get("runtime_sec"))
    scene_runtime = record.get("scene_build_runtime_sec", instance_scene.get("runtime_sec"))
    capture_runtime = record.get("capture_runtime_sec")
    if (
        capture_runtime is None
        and runtime is not None
        and execution_runtime is not None
        and scene_runtime is not None
    ):
        capture_runtime = max(0.0, float(runtime) - float(execution_runtime) - float(scene_runtime))
    normalized["capture_runtime_sec"] = capture_runtime
    normalized["scene_build_runtime_sec"] = scene_runtime
    normalized["execution_runtime_sec"] = execution_runtime
    return normalized


def fill_missing(loaded: dict[int, dict], seeds: list[int], *, source: Path) -> dict[int, dict]:
    for seed in seeds:
        loaded.setdefault(
            seed,
            {
                "seed": seed,
                "status": "not_run",
                "outcome": "not_run",
                "success": False,
                "runtime_sec": None,
                "source": str(source),
            },
        )
    return loaded


def build_metrics(records: dict[str, dict[int, dict]], seeds: list[int]) -> dict:
    labels = {
        "m0": "M0 ManiSkill truth",
        "view3": "M5-table 3-view",
        "view4": "M5-table 4-view",
        "view5": "M5-table 5-view",
    }
    metrics = {
        "seed_start": min(seeds),
        "seed_end": max(seeds),
        "requested": len(seeds),
        "stages": {},
        "comparisons": {},
    }
    for key, stage_records in records.items():
        values = [stage_records[seed] for seed in seeds]
        valid = [r for r in values if r.get("status") != "missing_zerograsp_candidate" and r.get("outcome") not in {"missing_zerograsp_candidate", "not_run"}]
        successes = [r for r in valid if r.get("success")]
        failures = [r for r in valid if not r.get("success")]
        runtimes = [float(r["runtime_sec"]) for r in valid if r.get("runtime_sec") is not None]
        runtime = runtime_stats(runtimes)
        if key == "m0" and runtime["mean_sec"] is None:
            runtime["mean_sec"] = M0_PRIOR_MEAN_RUNTIME_SEC
            runtime["source"] = "prior seed1-200 benchmark"
        else:
            runtime["source"] = "per-seed records" if runtimes else "unavailable"
        failure_counts = Counter(str(r.get("outcome")) for r in failures)
        timing_breakdown = {
            name: runtime_stats(
                [
                    float(r[field])
                    for r in valid
                    if r.get(field) is not None
                ]
            )
            for name, field in (
                ("capture_and_overhead", "capture_runtime_sec"),
                ("scene_build", "scene_build_runtime_sec"),
                ("planning_and_execution", "execution_runtime_sec"),
            )
        }
        metrics["stages"][key] = {
            "label": labels[key],
            "records_found": sum(1 for r in values if r.get("outcome") != "not_run"),
            "valid": len(valid),
            "successes": len(successes),
            "failures": len(failures),
            "success_rate": len(successes) / len(valid) if valid else None,
            "all_outcomes": dict(Counter(str(r.get("outcome")) for r in values)),
            "failure_distribution": {
                outcome: {
                    "count": count,
                    "share_of_failures": count / len(failures) if failures else None,
                }
                for outcome, count in sorted(failure_counts.items())
            },
            "runtime": runtime,
            "timing_breakdown": timing_breakdown,
        }

    for key in ("view3", "view4", "view5"):
        metrics["comparisons"][f"{key}_vs_m0"] = compare_stages(
            records[key], records["m0"], seeds, metrics["stages"][key], metrics["stages"]["m0"]
        )
    for key in ("view4", "view5"):
        metrics["comparisons"][f"{key}_vs_view3"] = compare_stages(
            records[key], records["view3"], seeds, metrics["stages"][key], metrics["stages"]["view3"]
        )
    return metrics


def runtime_stats(values: list[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "mean_sec": None,
            "median_sec": None,
            "p95_sec": None,
            "min_sec": None,
            "max_sec": None,
        }
    return {
        "count": len(values),
        "mean_sec": mean(values),
        "median_sec": median(values),
        "p95_sec": float(np.percentile(values, 95)),
        "min_sec": min(values),
        "max_sec": max(values),
    }


def compare_stages(
    current: dict[int, dict],
    baseline: dict[int, dict],
    seeds: list[int],
    current_metrics: dict,
    baseline_metrics: dict,
) -> dict:
    comparable = [
        seed
        for seed in seeds
        if current[seed].get("outcome") not in {"not_run", "missing_zerograsp_candidate"}
        and baseline[seed].get("outcome") not in {"not_run", "missing_zerograsp_candidate"}
    ]
    improved = [seed for seed in comparable if current[seed].get("success") and not baseline[seed].get("success")]
    regressed = [seed for seed in comparable if baseline[seed].get("success") and not current[seed].get("success")]
    current_rate = current_metrics["success_rate"]
    baseline_rate = baseline_metrics["success_rate"]
    current_mean = current_metrics["runtime"]["mean_sec"]
    baseline_mean = baseline_metrics["runtime"]["mean_sec"]
    return {
        "comparable_seeds": len(comparable),
        "success_rate_delta_pp": (
            100.0 * (current_rate - baseline_rate)
            if current_rate is not None and baseline_rate is not None
            else None
        ),
        "success_count_delta": (
            current_metrics["successes"] - baseline_metrics["successes"]
            if comparable
            else None
        ),
        "improved_seeds": improved,
        "regressed_seeds": regressed,
        "net_improvement": len(improved) - len(regressed),
        "mean_runtime_delta_sec": (
            current_mean - baseline_mean
            if current_mean is not None and baseline_mean is not None
            else None
        ),
        "mean_runtime_ratio": (
            current_mean / baseline_mean
            if current_mean is not None and baseline_mean not in (None, 0)
            else None
        ),
    }


def build_rows(records: dict[str, dict[int, dict]], seeds: list[int]) -> list[dict]:
    rows = []
    for seed in seeds:
        row: dict = {"seed": seed}
        for key in ("m0", "view3", "view4", "view5"):
            record = records[key][seed]
            row[f"{key}_success"] = bool(record.get("success"))
            row[f"{key}_outcome"] = record.get("outcome")
            row[f"{key}_runtime_sec"] = record.get("runtime_sec")
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def render_report(metrics: dict, csv_path: Path) -> str:
    lines = [
        "# M0 vs M5-table 3/4/5-view Benchmark",
        "",
        f"Seeds: {metrics['seed_start']}-{metrics['seed_end']}",
        "",
        "## Success And Runtime",
        "",
        "| Method | Found | Valid | Lift success | Rate | Mean | Median | P95 | Min | Max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("m0", "view3", "view4", "view5"):
        stage = metrics["stages"][key]
        runtime = stage["runtime"]
        rate = stage["success_rate"]
        lines.append(
            f"| {stage['label']} | {stage['records_found']} | {stage['valid']} | "
            f"{stage['successes']} | {format_percent(rate)} | {format_seconds(runtime['mean_sec'])} | "
            f"{format_seconds(runtime['median_sec'])} | {format_seconds(runtime['p95_sec'])} | "
            f"{format_seconds(runtime['min_sec'])} | {format_seconds(runtime['max_sec'])} |"
        )
    lines.extend(
        [
            "",
            "## Runtime Breakdown",
            "",
            "| Method | Capture/overhead mean | ESDF build mean | Planning/execution mean |",
            "|---|---:|---:|---:|",
        ]
    )
    for key in ("m0", "view3", "view4", "view5"):
        stage = metrics["stages"][key]
        breakdown = stage["timing_breakdown"]
        lines.append(
            f"| {stage['label']} | {format_seconds(breakdown['capture_and_overhead']['mean_sec'])} | "
            f"{format_seconds(breakdown['scene_build']['mean_sec'])} | "
            f"{format_seconds(breakdown['planning_and_execution']['mean_sec'])} |"
        )
    lines.extend(["", "## Failure Distribution", ""])
    for key in ("m0", "view3", "view4", "view5"):
        stage = metrics["stages"][key]
        lines.extend(
            [
                f"### {stage['label']}",
                "",
                "| Failure reason | Count | Share of failures |",
                "|---|---:|---:|",
            ]
        )
        if not stage["failure_distribution"]:
            lines.append("| - | 0 | - |")
        for reason, item in stage["failure_distribution"].items():
            lines.append(
                f"| {reason} | {item['count']} | {format_percent(item['share_of_failures'])} |"
            )
        lines.append("")
    lines.extend(["## Relative Performance", ""])
    for name, comparison in metrics["comparisons"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Success-rate delta: {format_pp(comparison['success_rate_delta_pp'])}",
                f"- Success-count delta: {format_signed_int(comparison['success_count_delta'])}",
                f"- Improved / regressed: {len(comparison['improved_seeds'])} / {len(comparison['regressed_seeds'])}",
                f"- Mean runtime delta: {format_seconds(comparison['mean_runtime_delta_sec'])}",
                f"- Mean runtime ratio: {format_ratio(comparison['mean_runtime_ratio'])}",
                f"- Improved seeds: {format_seed_list(comparison['improved_seeds'])}",
                f"- Regressed seeds: {format_seed_list(comparison['regressed_seeds'])}",
                "",
            ]
        )
    lines.extend(["## Per-Seed Results", "", f"CSV: `{csv_path}`", ""])
    return "\n".join(lines)


def format_percent(value: float | None) -> str:
    return "-" if value is None else f"{100.0 * value:.2f}%"


def format_pp(value: float | None) -> str:
    return "-" if value is None else f"{value:+.2f} pp"


def format_seconds(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}s"


def format_ratio(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}x"


def format_signed_int(value: int | None) -> str:
    return "-" if value is None else f"{value:+d}"


def format_seed_list(seeds: list[int]) -> str:
    return ", ".join(str(seed) for seed in seeds) if seeds else "-"


if __name__ == "__main__":
    raise SystemExit(main())
