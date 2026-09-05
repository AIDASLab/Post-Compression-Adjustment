"""Gemma4 additional-strategy evaluation Python startup hooks."""

from __future__ import annotations

import os

if os.environ.get("GEMMA4_STRATEGY_PATCH_VLLM_GEMMA4") == "1":
    from vllm_gemma4_transformers_patch import apply

    apply()
