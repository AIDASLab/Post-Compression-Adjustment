#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Analyze loss histories written by strategy_train.py."""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List


STRATEGIES = [
    "full_finetune",
    "router_top128_expert_finetune",
    "only_router_finetune",
    "router_top8_expert_finetune",
    "router_top16_expert_finetune",
    "router_top50_expert_finetune",
    "only_router_kd",
    "direct_router_logit_matching",
    "router_top8_expert_kd",
    "router_top16_expert_kd",
    "router_top50_expert_kd",
    "router_top128_expert_kd",
    "full_kd",
]


def read_step_losses(path: Path) -> List[Dict[str, float]]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("event") != "step":
                continue
            loss = float(obj["loss"])
            if math.isfinite(loss):
                rows.append(
                    {
                        "step": float(obj.get("global_step", len(rows) + 1)),
                        "loss": loss,
                    }
                )
    return rows


def mean(values: List[float]) -> float:
    return sum(values) / max(1, len(values))


def linear_slope(points: List[Dict[str, float]]) -> float:
    if len(points) < 2:
        return 0.0
    xs = [point["step"] for point in points]
    ys = [point["loss"] for point in points]
    x_mean = mean(xs)
    y_mean = mean(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom


def read_contamination_metrics(base_dir: Path, ratio: str, strategy: str) -> Dict[str, object]:
    metrics_path = base_dir / "logs" / "metrics" / f"{ratio}__{strategy}.json"
    empty = {
        "gpu_metrics_json": str(metrics_path),
        "contamination_status": "not_recorded",
        "contamination_detected": "",
        "contamination_num_foreign_pids": "",
        "contamination_foreign_processes": [],
        "avg_own_process_gpu_memory_gb": "",
        "avg_foreign_process_gpu_memory_gb": "",
        "peak_foreign_process_gpu_memory_gb": "",
    }
    if not metrics_path.exists():
        return empty
    try:
        with open(metrics_path, "r", encoding="utf-8") as handle:
            metrics = json.load(handle)
    except Exception as exc:
        empty["contamination_status"] = "unknown"
        empty["process_monitor_errors"] = [f"failed to read metrics JSON: {exc}"]
        return empty
    return {
        "gpu_metrics_json": str(metrics_path),
        "contamination_status": metrics.get("contamination_status") or "not_recorded",
        "contamination_detected": metrics.get("contamination_detected", ""),
        "contamination_num_foreign_pids": metrics.get("contamination_num_foreign_pids", ""),
        "contamination_foreign_processes": metrics.get("contamination_foreign_processes", []),
        "avg_own_process_gpu_memory_gb": metrics.get("avg_own_process_gpu_memory_gb", ""),
        "avg_foreign_process_gpu_memory_gb": metrics.get("avg_foreign_process_gpu_memory_gb", ""),
        "peak_foreign_process_gpu_memory_gb": metrics.get("peak_foreign_process_gpu_memory_gb", ""),
    }


def format_foreign_processes(value: object) -> str:
    if not value:
        return ""
    if not isinstance(value, list):
        return str(value)
    items = []
    for proc in value:
        if isinstance(proc, dict):
            pid = proc.get("pid", "")
            name = proc.get("process_name", "")
            items.append(f"{pid}:{name}" if name else str(pid))
        else:
            items.append(str(proc))
    return "; ".join(items)


def analyze(rows: List[Dict[str, float]], min_relative_improvement: float, tail_slope_tolerance: float) -> Dict[str, object]:
    if len(rows) < 5:
        return {
            "status": "insufficient_logs",
            "num_logged_steps": len(rows),
            "converged": False,
        }
    losses = [row["loss"] for row in rows]
    window = max(3, len(rows) // 5)
    first_mean = mean(losses[:window])
    last_mean = mean(losses[-window:])
    relative_improvement = (first_mean - last_mean) / max(abs(first_mean), 1e-8)
    tail_points = rows[-max(window, len(rows) // 2):]
    slope = linear_slope(tail_points)
    normalized_tail_slope = slope / max(abs(last_mean), 1e-8)
    loss_worsened = last_mean > first_mean * 1.01
    converged = (
        not loss_worsened
        and (
            relative_improvement >= min_relative_improvement
            or abs(normalized_tail_slope) <= tail_slope_tolerance
            or normalized_tail_slope < 0.0
        )
    )
    return {
        "status": "ok" if converged else "needs_review",
        "converged": converged,
        "num_logged_steps": len(rows),
        "first_window_mean": first_mean,
        "last_window_mean": last_mean,
        "relative_improvement": relative_improvement,
        "tail_slope_per_logged_step": slope,
        "normalized_tail_slope": normalized_tail_slope,
        "min_loss": min(losses),
        "max_loss": max(losses),
        "final_loss": losses[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check convergence from strategy loss logs.")
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--ratio", required=True)
    parser.add_argument("--min_relative_improvement", type=float, default=0.02)
    parser.add_argument("--tail_slope_tolerance", type=float, default=0.01)
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    report = {
        "ratio": args.ratio,
        "criteria": {
            "min_relative_improvement": args.min_relative_improvement,
            "tail_slope_tolerance": args.tail_slope_tolerance,
        },
        "strategies": {},
    }
    for strategy in STRATEGIES:
        loss_path = base_dir / args.ratio / strategy / "loss_history.jsonl"
        rows = read_step_losses(loss_path)
        strategy_report = analyze(
            rows,
            min_relative_improvement=args.min_relative_improvement,
            tail_slope_tolerance=args.tail_slope_tolerance,
        )
        strategy_report["loss_history"] = str(loss_path)
        strategy_report.update(read_contamination_metrics(base_dir, args.ratio, strategy))
        report["strategies"][strategy] = strategy_report

    report["all_converged"] = all(
        item.get("converged", False) for item in report["strategies"].values()
    )
    report["all_runs_without_detected_contamination"] = all(
        item.get("contamination_status") != "invalid"
        for item in report["strategies"].values()
    )
    report["all_contamination_checks_valid"] = all(
        item.get("contamination_status") == "valid"
        for item in report["strategies"].values()
    )

    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / f"convergence_{args.ratio}.json"
    csv_path = reports_dir / f"convergence_{args.ratio}.csv"
    md_path = reports_dir / f"convergence_{args.ratio}.md"

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    fields = [
        "strategy",
        "status",
        "converged",
        "num_logged_steps",
        "first_window_mean",
        "last_window_mean",
        "relative_improvement",
        "normalized_tail_slope",
        "final_loss",
        "contamination_status",
        "contamination_detected",
        "contamination_num_foreign_pids",
        "contamination_foreign_processes",
        "avg_own_process_gpu_memory_gb",
        "avg_foreign_process_gpu_memory_gb",
        "peak_foreign_process_gpu_memory_gb",
        "gpu_metrics_json",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for strategy, metrics in report["strategies"].items():
            row = {"strategy": strategy}
            row.update({field: metrics.get(field) for field in fields if field != "strategy"})
            for field, value in list(row.items()):
                if isinstance(value, (list, dict)):
                    row[field] = json.dumps(value, ensure_ascii=False)
            writer.writerow(row)

    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(f"# Convergence Report: {args.ratio}\n\n")
        handle.write("| strategy | status | final loss | relative improvement | tail slope norm | contamination | foreign PIDs |\n")
        handle.write("|---|---:|---:|---:|---:|---|---|\n")
        for strategy, metrics in report["strategies"].items():
            handle.write(
                f"| {strategy} | {metrics.get('status')} | "
                f"{metrics.get('final_loss', float('nan')):.6g} | "
                f"{metrics.get('relative_improvement', float('nan')):.4f} | "
                f"{metrics.get('normalized_tail_slope', float('nan')):.4g} | "
                f"{metrics.get('contamination_status')} | "
                f"{format_foreign_processes(metrics.get('contamination_foreign_processes'))} |\n"
            )

    print(f"[Convergence] json={json_path}", flush=True)
    print(f"[Convergence] csv={csv_path}", flush=True)
    print(f"[Convergence] markdown={md_path}", flush=True)
    if not report["all_converged"]:
        print("[Convergence] Some strategies need review.", flush=True)
    if not report["all_runs_without_detected_contamination"]:
        print("[Convergence] GPU contamination detected in at least one strategy.", flush=True)


if __name__ == "__main__":
    main()
