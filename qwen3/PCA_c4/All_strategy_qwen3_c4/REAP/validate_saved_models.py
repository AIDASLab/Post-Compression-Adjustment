#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Load generated REAP strategy checkpoints and verify pruned MoE structure."""

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


STRATEGIES = [
    "router_top50_expert_kd",
    "router_top128_expert_kd",
    "full_kd",
    "only_router_finetune",
    "router_top8_expert_finetune",
    "router_top16_expert_finetune",
    "router_top50_expert_finetune",
    "router_top128_expert_finetune",
    "full_finetune",
    "only_router_kd",
    "router_top8_expert_kd",
    "router_top16_expert_kd",
    "direct_router_logit_matching",
]

EXPECTED_RETAINED = {
    "ratio_50": 64,
    "ratio_625": 80,
    "ratio_75": 96,
}
EXPECTED_PRUNED = {
    "ratio_50": 64,
    "ratio_625": 48,
    "ratio_75": 32,
}
EXPECTED_ORIGINAL_EXPERTS = 128


def first_param_device(model) -> torch.device:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for key in (
            "model.embed_tokens",
            "transformer.wte",
            "embed_tokens",
            "model.tok_embeddings",
        ):
            if key in device_map:
                value = device_map[key]
                if isinstance(value, int):
                    return torch.device(f"cuda:{value}")
                if isinstance(value, str) and value.startswith("cuda"):
                    return torch.device(value)
                if value == "cpu":
                    return torch.device("cpu")
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cpu")


def collect_moe_layers(model) -> List[Tuple[int, object, object, object]]:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return []
    moe_layers = []
    for layer_idx, layer in enumerate(layers):
        moe = getattr(layer, "mlp", None)
        experts = getattr(moe, "experts", None) if moe is not None else None
        gate = getattr(moe, "gate", None) if moe is not None else None
        if experts is not None and gate is not None:
            moe_layers.append((layer_idx, moe, experts, gate))
    return moe_layers


def load_json_map(path: Path) -> Dict[int, List[int]]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {int(key): [int(value) for value in values] for key, values in raw.items()}


def inspect_reap_structure(model, expected_retained: int) -> Dict[str, object]:
    moe_layers = collect_moe_layers(model)
    sparse_layers: List[Dict[str, object]] = []
    for layer_idx, moe, experts, gate in moe_layers:
        gate_weight_rows = tuple(gate.weight.shape)[0]
        sparse_layers.append(
            {
                "layer_index": layer_idx,
                "expert_count": len(experts),
                "moe_num_experts": int(getattr(moe, "num_experts", -1)),
                "gate_out_features": int(getattr(gate, "out_features", -1)),
                "gate_weight_rows": int(gate_weight_rows),
            }
        )
    return {
        "num_sparse_layers": len(sparse_layers),
        "expert_count_unique": sorted({layer["expert_count"] for layer in sparse_layers}),
        "moe_num_experts_unique": sorted({layer["moe_num_experts"] for layer in sparse_layers}),
        "gate_out_features_unique": sorted({layer["gate_out_features"] for layer in sparse_layers}),
        "gate_weight_rows_unique": sorted({layer["gate_weight_rows"] for layer in sparse_layers}),
        "expected_retained_experts": expected_retained,
        "sparse_layers": sparse_layers,
    }


def check_index_maps(model_dir: Path, layer_indices: List[int], expected_retained: int, expected_pruned: int) -> Dict[str, object]:
    retained_path = model_dir / "retained_expert_indices.json"
    pruned_path = model_dir / "pruned_expert_indices.json"
    result = {
        "retained_map": str(retained_path),
        "pruned_map": str(pruned_path),
        "ok": False,
        "error": None,
    }
    if not retained_path.is_file():
        result["error"] = "missing retained_expert_indices.json"
        return result
    if not pruned_path.is_file():
        result["error"] = "missing pruned_expert_indices.json"
        return result

    retained = load_json_map(retained_path)
    pruned = load_json_map(pruned_path)
    for layer_idx in layer_indices:
        if layer_idx not in retained:
            result["error"] = f"retained map missing layer {layer_idx}"
            return result
        if layer_idx not in pruned:
            result["error"] = f"pruned map missing layer {layer_idx}"
            return result
        retained_indices = retained[layer_idx]
        pruned_indices = pruned[layer_idx]
        if len(retained_indices) != expected_retained:
            result["error"] = f"layer {layer_idx} retained length {len(retained_indices)} != {expected_retained}"
            return result
        if len(pruned_indices) != expected_pruned:
            result["error"] = f"layer {layer_idx} pruned length {len(pruned_indices)} != {expected_pruned}"
            return result
        if len(set(retained_indices)) != len(retained_indices):
            result["error"] = f"layer {layer_idx} retained map contains duplicates"
            return result
        if len(set(pruned_indices)) != len(pruned_indices):
            result["error"] = f"layer {layer_idx} pruned map contains duplicates"
            return result
        retained_set = set(retained_indices)
        pruned_set = set(pruned_indices)
        if retained_set & pruned_set:
            result["error"] = f"layer {layer_idx} retained/pruned maps overlap"
            return result
        if len(retained_set | pruned_set) != EXPECTED_ORIGINAL_EXPERTS:
            result["error"] = f"layer {layer_idx} maps do not cover {EXPECTED_ORIGINAL_EXPERTS} original experts"
            return result
        invalid = [idx for idx in retained_indices + pruned_indices if idx < 0 or idx >= EXPECTED_ORIGINAL_EXPERTS]
        if invalid:
            result["error"] = f"layer {layer_idx} has invalid original expert indices: {invalid[:10]}"
            return result
    result["ok"] = True
    return result


def check_router_forward(model, tokenizer, prompt: str, expected_retained: int) -> Dict[str, object]:
    input_device = first_param_device(model)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(input_device) for key, value in inputs.items()}
    with torch.no_grad():
        try:
            outputs = model(
                **inputs,
                use_cache=False,
                output_router_logits=True,
                return_dict=True,
                logits_to_keep=1,
            )
        except TypeError:
            outputs = model(
                **inputs,
                use_cache=False,
                output_router_logits=True,
                return_dict=True,
            )
    router_logits = getattr(outputs, "router_logits", None)
    if router_logits is None:
        return {"ok": False, "error": "forward did not return router_logits"}
    dims = sorted({int(tensor.shape[-1]) for tensor in router_logits if tensor is not None})
    return {"ok": dims == [expected_retained], "router_logit_dims": dims}


def validate_one(model_dir: Path, expected_retained: int, expected_pruned: int, prompt: str, max_new_tokens: int) -> Dict[str, object]:
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
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            "device_map": "auto" if torch.cuda.is_available() else None,
            "low_cpu_mem_usage": True,
            "local_files_only": True,
        }
        model = AutoModelForCausalLM.from_pretrained(model_dir, **load_kwargs)
        model.eval()
        config = model.config.to_dict()
        structure = inspect_reap_structure(model, expected_retained)
        layer_indices = [layer["layer_index"] for layer in structure["sparse_layers"]]
        index_maps = check_index_maps(model_dir, layer_indices, expected_retained, expected_pruned)
        router_check = check_router_forward(model, tokenizer, prompt, expected_retained)

        structure_ok = (
            config.get("model_type") == "qwen3_moe"
            and config.get("num_experts") == expected_retained
            and config.get("num_experts_per_tok") == 8
            and structure["expert_count_unique"] == [expected_retained]
            and structure["gate_weight_rows_unique"] == [expected_retained]
            and structure["num_sparse_layers"] > 0
        )

        device = first_param_device(model)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        result.update(
            {
                "ok": bool(structure_ok and index_maps.get("ok") and router_check.get("ok") and len(decoded.strip()) > 0),
                "config_model_type": config.get("model_type"),
                "config_num_experts": config.get("num_experts"),
                "expected_retained_experts": expected_retained,
                "expected_pruned_experts": expected_pruned,
                "structure": structure,
                "index_maps": index_maps,
                "router_check": router_check,
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
    parser = argparse.ArgumentParser(description="Validate saved REAP strategy checkpoints.")
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--ratio", required=True, choices=sorted(EXPECTED_RETAINED))
    parser.add_argument("--expected_retained_experts", type=int, default=None)
    parser.add_argument("--expected_pruned_experts", type=int, default=None)
    parser.add_argument("--prompt", default="Tell me about large language models in one sentence.")
    parser.add_argument("--max_new_tokens", type=int, default=24)
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    expected_retained = args.expected_retained_experts or EXPECTED_RETAINED[args.ratio]
    expected_pruned = args.expected_pruned_experts or EXPECTED_PRUNED[args.ratio]
    report = {
        "ratio": args.ratio,
        "expected_retained_experts": expected_retained,
        "expected_pruned_experts": expected_pruned,
        "prompt": args.prompt,
        "strategies": {},
    }
    for strategy in STRATEGIES:
        print(f"[Validate] {args.ratio}/{strategy}", flush=True)
        report["strategies"][strategy] = validate_one(
            model_dir=base_dir / args.ratio / strategy,
            expected_retained=expected_retained,
            expected_pruned=expected_pruned,
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
