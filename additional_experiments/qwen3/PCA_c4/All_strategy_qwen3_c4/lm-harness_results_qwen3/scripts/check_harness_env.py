#!/usr/bin/env python3
"""CPU-only environment preflight for the final qwen3 PCA_c4 vLLM runner."""

from __future__ import annotations

import argparse
import json
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    from packaging.version import Version
except Exception:  # pragma: no cover
    Version = None  # type: ignore[assignment]

from eval_categories import METHODS


EXP_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", EXP_ROOT.parent))
HARNESS_DIR = Path(os.environ.get("HARNESS_DIR", ""))


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def parse_version(text: str):
    if Version is None:
        return tuple(int(part) for part in text.split(".")[:3] if part.isdigit())
    return Version(text)


def is_less_than(version_text: str | None, required: str) -> bool:
    if version_text is None:
        return True
    return parse_version(version_text) < parse_version(required)


def scan_model_configs(method: str) -> list[dict[str, object]]:
    method_root = MODEL_ROOT / method
    scanned: list[dict[str, object]] = []
    for config_path in sorted(method_root.glob("ratio_*/*/config.json")):
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                cfg = json.load(handle)
        except Exception as exc:
            scanned.append({"path": str(config_path), "error": repr(exc)})
            continue

        model_path = config_path.parent
        rel = model_path.relative_to(method_root)
        architectures = cfg.get("architectures") or []
        model_type = str(cfg.get("model_type", ""))
        auto_map = cfg.get("auto_map") or {}
        is_moe = "moe" in model_type.lower() or any(
            "moe" in str(arch).lower() for arch in architectures
        )
        scanned.append(
            {
                "path": str(model_path),
                "ratio": rel.parts[0],
                "strategy": rel.parts[1],
                "architectures": architectures,
                "model_type": model_type,
                "has_auto_map": bool(auto_map),
                "is_moe": is_moe,
            }
        )
    return scanned


def main() -> int:
    parser = argparse.ArgumentParser(description="Check harness/vLLM environment without touching GPUs.")
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(HARNESS_DIR))
    try:
        import lm_eval  # noqa: PLC0415
    except Exception as exc:
        print(f"FATAL: failed to import lm_eval from {HARNESS_DIR}: {exc!r}", file=sys.stderr)
        return 3

    versions = {
        "python": sys.version.split()[0],
        "vllm": package_version("vllm"),
        "transformers": package_version("transformers"),
        "lm_eval": package_version("lm_eval"),
        "torch": package_version("torch"),
    }
    models = scan_model_configs(args.method)
    custom_moe_models = [
        item for item in models if item.get("is_moe") and item.get("has_auto_map")
    ]

    payload = {
        "versions": versions,
        "lm_eval_module": getattr(lm_eval, "__file__", ""),
        "num_model_configs": len(models),
        "num_custom_moe_model_configs": len(custom_moe_models),
        "custom_moe_examples": custom_moe_models[:5],
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"python={versions['python']}")
        print(f"vllm={versions['vllm']}")
        print(f"transformers={versions['transformers']}")
        print(f"lm_eval={versions['lm_eval']} module={payload['lm_eval_module']}")
        print(f"torch={versions['torch']}")
        print(f"model_configs={len(models)} custom_moe_model_configs={len(custom_moe_models)}")
        for item in custom_moe_models[:5]:
            print(
                "custom_moe_example="
                f"{item['ratio']}/{item['strategy']} "
                f"arch={item['architectures']} model_type={item['model_type']}"
            )

    if versions["vllm"] is None:
        print("FATAL: vLLM is not installed in this Python environment.", file=sys.stderr)
        return 3
    if versions["transformers"] is None:
        print("FATAL: transformers is not installed in this Python environment.", file=sys.stderr)
        return 3
    if custom_moe_models and is_less_than(versions["transformers"], "5.5.1"):
        print(
            "FATAL: Custom MoE checkpoints in this method use trust_remote_code/auto_map. "
            "vLLM falls back to its Transformers MoE backend for these models, and the installed "
            f"transformers={versions['transformers']} is too old. Install transformers>=5.5.1 "
            "or set SKIP_PREFLIGHT=1 if you intentionally want to bypass this check.",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
