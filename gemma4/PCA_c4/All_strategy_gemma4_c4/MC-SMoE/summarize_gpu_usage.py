#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summarize GPU usage metrics across ratios and strategies."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


RATIO_ORDER = ["ratio_75", "ratio_625", "ratio_50"]


def collect_metrics(base_dir: Path) -> List[Dict[str, object]]:
    metrics_dir = base_dir / "logs" / "metrics"
    rows: List[Dict[str, object]] = []
    if not metrics_dir.exists():
        return rows
    for path in sorted(metrics_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        data["metrics_json"] = str(path)
        rows.append(data)
    return rows


def ratio_sort_key(row: Dict[str, object]) -> int:
    ratio = str(row.get("ratio", ""))
    return RATIO_ORDER.index(ratio) if ratio in RATIO_ORDER else 999


def contamination_status(row: Dict[str, object]) -> str:
    return str(row.get("contamination_status") or "not_recorded")


def format_list_or_dict(value: object) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return "" if value is None else str(value)


def format_foreign_processes(row: Dict[str, object]) -> str:
    processes = row.get("contamination_foreign_processes") or []
    if not isinstance(processes, list):
        return str(processes)
    items = []
    for proc in processes:
        if isinstance(proc, dict):
            pid = proc.get("pid", "")
            name = proc.get("process_name", "")
            items.append(f"{pid}:{name}" if name else str(pid))
        else:
            items.append(str(proc))
    return "; ".join(items)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Gemma4 MC-SMoE GPU usage metrics.")
    parser.add_argument("--base_dir", required=True)
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    rows = collect_metrics(base_dir)
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    csv_path = reports_dir / "gpu_usage_summary.csv"
    json_path = reports_dir / "gpu_usage_summary.json"
    md_path = reports_dir / "gpu_usage_ranking.md"
    contamination_md_path = reports_dir / "gpu_contamination_report.md"

    fields = [
        "ratio",
        "strategy",
        "measurement_scope",
        "assigned_gpu_ids",
        "num_gpus",
        "training_loop_seconds",
        "measured_loop_hours",
        "sum_avg_gpu_power_watt",
        "avg_gpu_utilization_percent",
        "avg_total_gpu_memory_gb",
        "gpu_energy_kwh",
        "effective_gpu_hours",
        "gpu_memory_gb_hours",
        "contamination_status",
        "contamination_detected",
        "contamination_num_foreign_pids",
        "contamination_foreign_processes",
        "avg_own_process_gpu_memory_gb",
        "avg_foreign_process_gpu_memory_gb",
        "peak_foreign_process_gpu_memory_gb",
        "process_monitor_samples",
        "process_monitor_rows",
        "process_monitor_csv",
        "num_monitor_samples",
        "gpu_log_csv",
        "metrics_json",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (ratio_sort_key(item), str(item.get("strategy", "")))):
            out = {field: row.get(field) for field in fields}
            if isinstance(out.get("assigned_gpu_ids"), list):
                out["assigned_gpu_ids"] = ",".join(out["assigned_gpu_ids"])
            for field, value in list(out.items()):
                if isinstance(value, (list, dict)):
                    out[field] = json.dumps(value, ensure_ascii=False)
            writer.writerow(out)

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)

    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("# GPU Usage Ranking\n\n")
        handle.write("Primary ranking is ascending `gpu_energy_kwh` within each ratio; memory GB-hours are reported alongside it.\n\n")
        for ratio in RATIO_ORDER:
            ratio_rows = [row for row in rows if row.get("ratio") == ratio]
            if not ratio_rows:
                continue
            ratio_rows = sorted(ratio_rows, key=lambda row: float(row.get("gpu_energy_kwh", 1e99)))
            handle.write(f"## {ratio}\n\n")
            handle.write("| rank | strategy | gpu_energy_kwh | effective_gpu_hours | gpu_memory_gb_hours | avg memory GB | avg util % | loop sec | GPUs | contamination | foreign PIDs |\n")
            handle.write("|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|\n")
            for rank, row in enumerate(ratio_rows, start=1):
                gpu_ids = row.get("assigned_gpu_ids", [])
                if isinstance(gpu_ids, list):
                    gpu_ids = ",".join(gpu_ids)
                handle.write(
                    f"| {rank} | {row.get('strategy')} | "
                    f"{float(row.get('gpu_energy_kwh', 0.0)):.6f} | "
                    f"{float(row.get('effective_gpu_hours', 0.0)):.6f} | "
                    f"{float(row.get('gpu_memory_gb_hours', 0.0)):.6f} | "
                    f"{float(row.get('avg_total_gpu_memory_gb', 0.0)):.2f} | "
                    f"{float(row.get('avg_gpu_utilization_percent', 0.0)):.2f} | "
                    f"{float(row.get('training_loop_seconds', 0.0)):.2f} | {gpu_ids} | "
                    f"{contamination_status(row)} | {format_foreign_processes(row)} |\n"
                )
            handle.write("\n")

    with open(contamination_md_path, "w", encoding="utf-8") as handle:
        handle.write("# GPU Contamination Report\n\n")
        handle.write("A row is `invalid` when at least one GPU compute process outside this run's process tree was observed during measured segments. `not_recorded` means the metric JSON was produced before process-level monitoring was added.\n\n")
        for ratio in RATIO_ORDER:
            ratio_rows = [row for row in rows if row.get("ratio") == ratio]
            if not ratio_rows:
                continue
            ratio_rows = sorted(ratio_rows, key=lambda row: str(row.get("strategy", "")))
            handle.write(f"## {ratio}\n\n")
            handle.write("| strategy | status | foreign processes | avg own process GB | avg foreign process GB | peak foreign process GB | process samples | process log |\n")
            handle.write("|---|---|---|---:|---:|---:|---:|---|\n")
            for row in ratio_rows:
                process_log = row.get("process_monitor_csv", "")
                handle.write(
                    f"| {row.get('strategy')} | {contamination_status(row)} | "
                    f"{format_foreign_processes(row)} | "
                    f"{float(row.get('avg_own_process_gpu_memory_gb', 0.0) or 0.0):.2f} | "
                    f"{float(row.get('avg_foreign_process_gpu_memory_gb', 0.0) or 0.0):.2f} | "
                    f"{float(row.get('peak_foreign_process_gpu_memory_gb', 0.0) or 0.0):.2f} | "
                    f"{int(row.get('process_monitor_samples', 0) or 0)} | {process_log} |\n"
                )
            handle.write("\n")

    print(f"[GPU Summary] csv={csv_path}", flush=True)
    print(f"[GPU Summary] json={json_path}", flush=True)
    print(f"[GPU Summary] markdown={md_path}", flush=True)
    print(f"[GPU Summary] contamination={contamination_md_path}", flush=True)


if __name__ == "__main__":
    main()
