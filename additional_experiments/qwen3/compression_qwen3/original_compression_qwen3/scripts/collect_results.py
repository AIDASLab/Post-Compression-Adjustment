#!/usr/bin/env python3
"""Collect completed result.json files into summary files under original_compression."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", required=True)
    return parser.parse_args()


def metric_scalar(metrics: dict[str, Any]) -> tuple[str, Any] | None:
    preferred = [
        "exact_match,flexible-extract",
        "exact_match,strict-match",
        "acc,none",
        "acc_norm,none",
        "pass@1,create_test",
        "pass@1,base",
    ]
    for key in preferred:
        if key in metrics and isinstance(metrics[key], (int, float)):
            return key, metrics[key]
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            return key, value
    return None


def main() -> int:
    args = parse_args()
    rows = []
    summary_dir = ROOT / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    for method in args.methods:
        result_root = ROOT / f"results_{method}"
        if not result_root.exists():
            continue
        for result_path in sorted(result_root.glob("*/*/*/result.json")):
            ratio, variant, category = result_path.relative_to(result_root).parts[:3]
            data = json.loads(result_path.read_text(encoding="utf-8"))
            for task, metrics in data.items():
                if not isinstance(metrics, dict):
                    continue
                selected = metric_scalar(metrics)
                rows.append(
                    {
                        "method": method,
                        "ratio": ratio,
                        "variant": variant,
                        "category": category,
                        "task": task,
                        "metric": selected[0] if selected else "",
                        "value": selected[1] if selected else "",
                        "result_path": str(result_path),
                    }
                )
    json_path = summary_dir / "summary.json"
    tsv_path = summary_dir / "summary.tsv"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with tsv_path.open("w", encoding="utf-8") as handle:
        handle.write("method\tratio\tvariant\tcategory\ttask\tmetric\tvalue\tresult_path\n")
        for row in rows:
            handle.write("\t".join(str(row[key]) for key in [
                "method",
                "ratio",
                "variant",
                "category",
                "task",
                "metric",
                "value",
                "result_path",
            ]))
            handle.write("\n")
    print(f"rows={len(rows)}")
    print(f"summary_json={json_path}")
    print(f"summary_tsv={tsv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
