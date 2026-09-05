#!/usr/bin/env python3
"""CPU-only environment and registry check for Gemma4 original_compression evaluation."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HARNESS_DIR = Path(os.environ.get("HARNESS_DIR", ""))
COMPRESSED_MODELS_ROOT = Path(os.environ.get("COMPRESSED_MODELS_ROOT", ROOT.parent))
REGISTRY_PATH = ROOT / "model_registry.json"


def version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("methods", nargs="*")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(HARNESS_DIR))
    import torch  # noqa: PLC0415
    import lm_eval  # noqa: PLC0415

    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    methods = args.methods or list(registry)
    print(f"python={sys.version.split()[0]}")
    print(f"torch={torch.__version__}")
    print(f"vllm={version('vllm')}")
    print(f"transformers={version('transformers')}")
    print(f"lm_eval={version('lm_eval')} module={getattr(lm_eval, '__file__', '')}")
    total = 0
    for method in methods:
        entries = registry.get(method)
        if entries is None:
            print(f"invalid_method={method}")
            continue
        print(f"method={method} entries={len(entries)}")
        for entry in entries:
            total += 1
            ref = entry["model_ref"]
            status = "hf_id"
            if entry.get("ref_type") == "local_path":
                ref_path = Path(ref)
                if not ref_path.is_absolute():
                    ref_path = COMPRESSED_MODELS_ROOT / ref_path
                cfg = ref_path / "config.json"
                status = "ok" if cfg.is_file() else "missing_config"
                if cfg.is_file():
                    data = json.loads(cfg.read_text(encoding="utf-8"))
                    text_cfg = data.get("text_config") or {}
                    status += (
                        f" arch={data.get('architectures')}"
                        f" model_type={data.get('model_type')}"
                        f" text_num_experts={text_cfg.get('num_experts')}"
                        f" physical={text_cfg.get('physical_num_experts', data.get('physical_num_experts'))}"
                    )
            print(f"  {entry['ratio']}/{entry['variant']} {entry['name']} {status}")
    print(f"registry_entries_checked={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
