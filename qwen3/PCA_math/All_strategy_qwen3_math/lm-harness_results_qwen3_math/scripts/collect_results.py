#!/usr/bin/env python3
"""Collect result.json files into one CSV/JSON summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from eval_categories import METHODS


EXP_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize MoE evaluation results.")
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Default: results_<METHOD>/summary_<METHOD>",
    )
    return parser.parse_args()


def metric_rows(method: str, result_path: Path, data: dict[str, Any]) -> list[dict[str, Any]]:
    rel = result_path.relative_to(EXP_ROOT)
    _, ratio, strategy, category, _ = rel.parts
    rows = []
    for task, metrics in data.items():
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                rows.append(
                    {
                        "method": method,
                        "ratio": ratio,
                        "strategy": strategy,
                        "category": category,
                        "task": task,
                        "metric": metric_name,
                        "value": value,
                    }
                )
    return rows


def main() -> int:
    args = parse_args()
    result_root = EXP_ROOT / f"results_{args.method}"
    if not result_root.is_dir():
        raise FileNotFoundError(result_root)

    rows: list[dict[str, Any]] = []
    for result_path in sorted(result_root.glob("ratio_*/*/*/result.json")):
        with result_path.open("r", encoding="utf-8") as handle:
            rows.extend(metric_rows(args.method, result_path, json.load(handle)))

    prefix = (
        Path(args.output_prefix)
        if args.output_prefix
        else result_root / f"summary_{args.method}"
    )
    if not prefix.is_absolute():
        prefix = EXP_ROOT / prefix
    if not prefix.resolve().is_relative_to(EXP_ROOT.resolve()):
        raise ValueError(f"output prefix must stay under {EXP_ROOT}: {prefix}")
    prefix.parent.mkdir(parents=True, exist_ok=True)

    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["method", "ratio", "strategy", "category", "task", "metric", "value"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} metric rows")
    print(json_path)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
