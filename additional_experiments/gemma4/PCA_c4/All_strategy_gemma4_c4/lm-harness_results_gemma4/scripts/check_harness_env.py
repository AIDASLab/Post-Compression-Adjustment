#!/usr/bin/env python3
"""CPU-only environment preflight for the final Gemma4 lm-harness runner."""

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
ADDITIONAL_STRATEGY_ROOT = Path(os.environ.get("MODEL_ROOT", EXP_ROOT.parent))
HARNESS_DIR = Path(os.environ.get("HARNESS_DIR", ""))
INPUT_ROOTS = {
    "AIMER": ADDITIONAL_STRATEGY_ROOT / "AIMER",
    "MC-SMoE": ADDITIONAL_STRATEGY_ROOT / "MC-SMoE",
}


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
    method_root = INPUT_ROOTS[method]
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
        text_cfg = cfg.get("text_config") or {}
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
                "text_model_type": text_cfg.get("model_type"),
                "text_num_experts": text_cfg.get("num_experts"),
                "text_top_k_experts": text_cfg.get("top_k_experts"),
                "has_auto_map": bool(auto_map),
                "has_processor_config": (model_path / "processor_config.json").is_file(),
                "is_moe": is_moe,
            }
        )
    return scanned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check harness/vLLM environment without touching GPUs.")
    parser.add_argument("methods", nargs="*", choices=METHODS)
    parser.add_argument("--method", action="append", choices=METHODS, default=[])
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sys.dont_write_bytecode = True
    requested = list(args.methods) + list(args.method)
    methods = requested or list(METHODS)
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
    payload: dict[str, object] = {
        "versions": versions,
        "lm_eval_module": getattr(lm_eval, "__file__", ""),
        "methods": {},
    }
    fatal = False

    for method in methods:
        method_root = INPUT_ROOTS[method]
        models = scan_model_configs(method) if method_root.is_dir() else []
        custom_moe_models = [
            item
            for item in models
            if item.get("is_moe") and item.get("has_auto_map")
        ]
        missing_processor = [
            item
            for item in models
            if not item.get("has_processor_config")
        ]
        payload["methods"][method] = {
            "method_root": str(method_root),
            "method_root_exists": method_root.is_dir(),
            "num_model_configs": len(models),
            "num_custom_moe_model_configs": len(custom_moe_models),
            "num_missing_processor_config": len(missing_processor),
            "custom_moe_examples": custom_moe_models[:5],
            "missing_processor_examples": missing_processor[:5],
        }
        if not method_root.is_dir() or not models:
            fatal = True

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"python={versions['python']}")
        print(f"vllm={versions['vllm']}")
        print(f"transformers={versions['transformers']}")
        print(f"lm_eval={versions['lm_eval']} module={payload['lm_eval_module']}")
        print(f"torch={versions['torch']}")
        for method in methods:
            info = payload["methods"][method]  # type: ignore[index]
            print(
                f"method={method} root={info['method_root']} "
                f"configs={info['num_model_configs']} "
                f"custom_moe={info['num_custom_moe_model_configs']} "
                f"missing_processor={info['num_missing_processor_config']}"
            )
            for item in info["custom_moe_examples"]:  # type: ignore[index]
                print(
                    "  custom_moe_example="
                    f"{item['ratio']}/{item['strategy']} arch={item['architectures']} "
                    f"model_type={item['model_type']}"
                )
            for item in info["missing_processor_examples"]:  # type: ignore[index]
                print(f"  mirror_will_add_processor={item['ratio']}/{item['strategy']}")

    if versions["vllm"] is None:
        print("FATAL: vLLM is not installed in this Python environment.", file=sys.stderr)
        return 3
    if versions["transformers"] is None:
        print("FATAL: transformers is not installed in this Python environment.", file=sys.stderr)
        return 3
    if is_less_than(versions["transformers"], "5.5.1"):
        print(
            "FATAL: Gemma4 vLLM Transformers backend needs a compatible Transformers v5 "
            f"release; found transformers={versions['transformers']}.",
            file=sys.stderr,
        )
        return 3
    if fatal:
        print("FATAL: one or more requested method folders are missing model configs.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
