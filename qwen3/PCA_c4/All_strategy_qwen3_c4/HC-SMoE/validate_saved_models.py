#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Load generated HC-SMoE models and verify structure plus a short inference."""

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


STRATEGIES = [
    "full_finetune",
    "router_top128_expert_finetune",
    "only_router_finetune",
    "router_top8_expert_finetune",
    "router_top16_expert_finetune",
    "router_top50_expert_finetune",
    "only_router_kd",
    "direct_router_logit_matching",
    "router_top8_expert_kd",
    "router_top16_expert_kd",
    "router_top50_expert_kd",
    "router_top128_expert_kd",
    "full_kd",
]

EXPECTED_PHYSICAL = {
    "ratio_50": 64,
    "ratio_625": 80,
    "ratio_75": 96,
}


def first_param_device(model) -> torch.device:
    for param in model.parameters():
        return param.device
    return torch.device("cpu")


def inspect_hcsmoe_structure(model) -> Dict[str, object]:
    sparse_layers: List[Dict[str, object]] = []
    for name, module in model.named_modules():
        if not hasattr(module, "experts") or not hasattr(module, "gate"):
            continue
        if not hasattr(module, "logical_to_physical"):
            continue
        sparse_layers.append(
            {
                "name": name,
                "logical_num_experts": int(getattr(module, "logical_num_experts", getattr(module, "num_experts", -1))),
                "physical_num_experts": int(getattr(module, "physical_num_experts", len(module.experts))),
                "num_experts": int(getattr(module, "num_experts", -1)),
                "top_k": int(getattr(module, "top_k", -1)),
                "mapping_len": int(module.logical_to_physical.numel()),
            }
        )
    return {
        "num_sparse_layers": len(sparse_layers),
        "physical_num_experts_unique": sorted({layer["physical_num_experts"] for layer in sparse_layers}),
        "logical_num_experts_unique": sorted({layer["logical_num_experts"] for layer in sparse_layers}),
        "top_k_unique": sorted({layer["top_k"] for layer in sparse_layers}),
        "mapping_len_unique": sorted({layer["mapping_len"] for layer in sparse_layers}),
        "sparse_layers": sparse_layers,
    }


def validate_one(model_dir: Path, expected_physical: int, prompt: str, max_new_tokens: int) -> Dict[str, object]:
    started = time.time()
    result: Dict[str, object] = {
        "model_dir": str(model_dir),
        "ok": False,
        "error": None,
    }
    if not model_dir.exists():
        result["error"] = "missing output directory"
        return result
    if not (model_dir / "config.json").exists():
        result["error"] = "missing config.json"
        return result

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
        )
        model.eval()
        structure = inspect_hcsmoe_structure(model)
        config = model.config.to_dict()
        structure_ok = (
            config.get("model_type") == "hcsmoe_qwen3_moe"
            and structure["physical_num_experts_unique"] == [expected_physical]
            and structure["logical_num_experts_unique"] == [128]
            and structure["top_k_unique"] == [8]
            and structure["mapping_len_unique"] == [128]
            and structure["num_sparse_layers"] > 0
        )

        device = first_param_device(model)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        result.update(
            {
                "ok": bool(structure_ok and len(decoded) > 0),
                "config_model_type": config.get("model_type"),
                "physical_num_experts": expected_physical,
                "structure": structure,
                "generated_preview": decoded[:500],
                "elapsed_seconds": time.time() - started,
            }
        )
        del model
        del tokenizer
    except Exception as exc:
        result["error"] = repr(exc)
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate saved HC-SMoE strategy checkpoints.")
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--ratio", required=True, choices=sorted(EXPECTED_PHYSICAL))
    parser.add_argument("--expected_physical_experts", type=int, default=None)
    parser.add_argument("--prompt", default="Tell me about large language models in one sentence.")
    parser.add_argument("--max_new_tokens", type=int, default=24)
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    expected = args.expected_physical_experts or EXPECTED_PHYSICAL[args.ratio]
    report = {
        "ratio": args.ratio,
        "expected_physical_experts": expected,
        "prompt": args.prompt,
        "strategies": {},
    }
    for strategy in STRATEGIES:
        print(f"[Validate] {args.ratio}/{strategy}", flush=True)
        report["strategies"][strategy] = validate_one(
            model_dir=base_dir / args.ratio / strategy,
            expected_physical=expected,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
        )

    report["ok"] = all(item.get("ok") for item in report["strategies"].values())
    output_dir = base_dir / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.ratio}_validation.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(f"[Validate] report={output_path} ok={report['ok']}", flush=True)
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
