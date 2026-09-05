"""Shared category/task configuration for original-vs-compressed model evaluation."""

from __future__ import annotations


SEED = 42

CATEGORIES = {
    "reasoning_math": {
        "tasks": [
            "gsm8k",
            "gsm8k_platinum",
            "minerva_math",
            "bbh_fewshot",
            "bbh_zeroshot",
            "coqa",
        ],
        "gpu_memory_utilization": 0.9,
        "model_args": {},
        "gen_kwargs": None,
        "confirm_run_unsafe_code": False,
    },
    "MC": {
        "tasks": [
            "medqa_4options",
            "medmcqa",
            "openbookqa",
            "arc_easy",
            "winogrande",
            "hellaswag",
            "arc_challenge",
            "piqa",
            "mmlu",
        ],
        "gpu_memory_utilization": 0.8,
        "model_args": {},
        "gen_kwargs": None,
        "confirm_run_unsafe_code": False,
    },
    "cot": {
        "tasks": [
            "gsm8k_cot",
            "gsm8k_platinum_cot",
            "gsm8k_cot_zeroshot",
            "gsm8k_platinum_cot_zeroshot",
            "bbh_cot_fewshot",
            "bbh_cot_zeroshot",
        ],
        "gpu_memory_utilization": 0.9,
        "model_args": {},
        "gen_kwargs": None,
        "confirm_run_unsafe_code": False,
    },
    "code": {
        "tasks": [
            "humaneval_instruct",
            "mbpp_instruct",
            "humaneval",
            "mbpp",
            "mbpp_plus",
            "mbpp_plus_instruct",
            "humaneval_plus",
            "humaneval_64",
            "humaneval_64_instruct",
        ],
        "gpu_memory_utilization": 0.9,
        "model_args": {},
        "gen_kwargs": None,
        "confirm_run_unsafe_code": True,
    },
    "AIME": {
        "tasks": [
            "aime",
            "aime24",
            "aime25",
        ],
        "gpu_memory_utilization": 0.9,
        "model_args": {
            "enable_thinking": False,
        },
        "gen_kwargs": {
            "max_gen_toks": 2780,
            "do_sample": False,
            "temperature": 0.0,
        },
        "confirm_run_unsafe_code": False,
    },
}

METHODS = (
    "original",
    "HC-SMoE",
    "REAP",
)
