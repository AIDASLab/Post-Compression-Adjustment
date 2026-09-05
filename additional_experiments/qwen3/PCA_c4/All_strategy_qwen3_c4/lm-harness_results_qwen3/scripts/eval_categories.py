"""Shared category/task configuration for the reported behavioral proxies."""

from __future__ import annotations


SEED = 42

DEFAULT_CATEGORY_CONFIG = {
    "gpu_memory_utilization": 0.8,
    "model_args": {},
    "gen_kwargs": None,
    "confirm_run_unsafe_code": False,
}

TASK_CATEGORIES = {
    "ifeval": ["ifeval"],
    "truthfulqa_mc1": ["truthfulqa_mc1"],
    "toxigen": ["toxigen"],
    "wmdp": ["wmdp"],
    "crows_pairs_english": ["crows_pairs_english"],
    "winogender": ["winogender_all"],
}

CATEGORIES = {
    category: {
        "tasks": tasks,
        **DEFAULT_CATEGORY_CONFIG,
    }
    for category, tasks in TASK_CATEGORIES.items()
}

CATEGORIES["safety_alignment"] = {
    "tasks": [task for tasks in TASK_CATEGORIES.values() for task in tasks],
    **DEFAULT_CATEGORY_CONFIG,
}

METHODS = ("HC-SMoE", "REAP")
