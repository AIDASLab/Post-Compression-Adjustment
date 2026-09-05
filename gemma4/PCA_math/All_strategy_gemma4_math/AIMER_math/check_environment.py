#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from pathlib import Path


def module_version(name):
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except Exception as exc:
        return f"unavailable: {exc!r}"


def main():
    root = Path(__file__).resolve().parent
    info = {
        "workspace_root": str(root),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"),
        "pip_constraint": os.environ.get("PIP_CONSTRAINT"),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "hf_home": os.environ.get("HF_HOME"),
        "huggingface_hub_cache_env": os.environ.get("HUGGINGFACE_HUB_CACHE"),
        "transformers_cache_env": os.environ.get("TRANSFORMERS_CACHE"),
        "torch": module_version("torch"),
        "transformers": module_version("transformers"),
        "accelerate": module_version("accelerate"),
        "safetensors": module_version("safetensors"),
    }
    try:
        from huggingface_hub import constants
        info["hf_hub_cache_resolved"] = str(constants.HF_HUB_CACHE)
    except Exception as exc:
        info["hf_hub_cache_resolved"] = f"unavailable: {exc!r}"
    try:
        from transformers import AutoModelForImageTextToText
        info["has_auto_model_for_image_text_to_text"] = AutoModelForImageTextToText is not None
    except Exception as exc:
        info["has_auto_model_for_image_text_to_text"] = False
        info["auto_model_for_image_text_to_text_error"] = repr(exc)
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration
        info["has_native_gemma4"] = Gemma4ForConditionalGeneration is not None
    except Exception as exc:
        info["has_native_gemma4"] = False
        info["native_gemma4_error"] = repr(exc)

    root_str = str(root)
    cache_values = [
        info.get("hf_home") or "",
        info.get("huggingface_hub_cache_env") or "",
        info.get("transformers_cache_env") or "",
        info.get("hf_hub_cache_resolved") or "",
    ]
    info["cache_inside_workspace"] = any(value.startswith(root_str) for value in cache_values if value)
    info["ok"] = (
        info["python_no_user_site"] == "1"
        and not info["pip_constraint"]
        and not info["pythonpath"]
        and info["has_auto_model_for_image_text_to_text"]
        and info["has_native_gemma4"]
        and not info["cache_inside_workspace"]
    )
    print(json.dumps(info, indent=2, ensure_ascii=False))
    if not info["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
