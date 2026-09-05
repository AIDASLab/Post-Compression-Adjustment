#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Load generated Gemma4 AIMER strategy checkpoints and verify pruned MoE structure."""

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoConfig, AutoTokenizer

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None

try:
    from transformers import AutoModelForCausalLM
except ImportError:
    AutoModelForCausalLM = None


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
EXPECTED_TOP_K = 8


def first_param_device(model) -> torch.device:
    try:
        emb = model.get_input_embeddings()
        if emb is not None:
            for param in emb.parameters():
                if param.device.type != "meta":
                    return param.device
    except Exception:
        pass
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cpu")


def get_model_loader():
    if AutoModelForImageTextToText is not None:
        return AutoModelForImageTextToText
    if AutoModelForCausalLM is not None:
        return AutoModelForCausalLM
    raise ImportError("Transformers must provide AutoModelForImageTextToText or AutoModelForCausalLM.")


def load_model(model_dir: Path, torch_dtype: str, device_map: str):
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch_dtype != "float32" else torch.float32
    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "local_files_only": True,
    }
    if torch.cuda.is_available() and device_map:
        kwargs["device_map"] = device_map
    return get_model_loader().from_pretrained(model_dir, **kwargs)


def prepare_chat_inputs(tokenizer, prompt: str) -> Dict[str, torch.Tensor]:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
    except TypeError as exc:
        raise TypeError(
            "Gemma4 tokenizer apply_chat_template must accept enable_thinking=False for this experiment."
        ) from exc


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def resolve_language_layers(model) -> Tuple[str, torch.nn.ModuleList]:
    candidates = [
        ("model.language_model", getattr(getattr(model, "model", None), "language_model", None)),
        ("language_model", getattr(model, "language_model", None)),
        ("model.model", getattr(getattr(model, "model", None), "model", None)),
        ("model", getattr(model, "model", None)),
    ]
    for prefix, module in candidates:
        layers = getattr(module, "layers", None)
        if isinstance(layers, torch.nn.ModuleList) and len(layers) > 0:
            return prefix, layers
    raise RuntimeError("Could not find Gemma4 language model layers.")


def load_json_map(path: Path) -> Dict[int, List[int]]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {int(key): [int(value) for value in values] for key, values in raw.items()}


def inspect_structure(model, expected_retained: int) -> Dict[str, object]:
    layer_prefix, layers = resolve_language_layers(model)
    sparse_layers: List[Dict[str, object]] = []
    for layer_idx, layer in enumerate(layers):
        if not (getattr(layer, "enable_moe_block", False) and hasattr(layer, "router") and hasattr(layer, "experts")):
            continue
        router = layer.router
        experts = layer.experts
        router_proj = getattr(router, "proj", None)
        gate_up_proj = getattr(experts, "gate_up_proj", None)
        down_proj = getattr(experts, "down_proj", None)
        per_expert_scale = getattr(router, "per_expert_scale", None)
        if router_proj is None or gate_up_proj is None or down_proj is None:
            continue
        entry = {
            "layer_index": layer_idx,
            "layer_prefix": f"{layer_prefix}.layers.{layer_idx}",
            "router_out_features": int(getattr(router_proj, "out_features", router_proj.weight.shape[0])),
            "router_weight_rows": int(router_proj.weight.shape[0]),
            "router_scale_len": int(per_expert_scale.numel()) if isinstance(per_expert_scale, torch.Tensor) else None,
            "expert_num_experts": int(getattr(experts, "num_experts", -1)),
            "gate_up_experts": int(gate_up_proj.shape[0]),
            "down_experts": int(down_proj.shape[0]),
            "has_shared_mlp": all(hasattr(layer.mlp, name) for name in ("gate_proj", "up_proj", "down_proj")),
        }
        entry["ok"] = bool(
            entry["router_out_features"] == expected_retained
            and entry["router_weight_rows"] == expected_retained
            and entry["router_scale_len"] == expected_retained
            and entry["expert_num_experts"] == expected_retained
            and entry["gate_up_experts"] == expected_retained
            and entry["down_experts"] == expected_retained
            and entry["has_shared_mlp"]
        )
        sparse_layers.append(entry)
    return {
        "num_sparse_layers": len(sparse_layers),
        "sparse_layers_ok": all(layer["ok"] for layer in sparse_layers) and bool(sparse_layers),
        "layer_indices": [layer["layer_index"] for layer in sparse_layers],
        "router_out_features_unique": sorted({layer["router_out_features"] for layer in sparse_layers}),
        "router_scale_len_unique": sorted({layer["router_scale_len"] for layer in sparse_layers}),
        "gate_up_experts_unique": sorted({layer["gate_up_experts"] for layer in sparse_layers}),
        "down_experts_unique": sorted({layer["down_experts"] for layer in sparse_layers}),
        "expected_retained_experts": expected_retained,
        "sparse_layers": sparse_layers,
    }


def check_index_maps(model_dir: Path, layer_indices: List[int], expected_retained: int, expected_pruned: int) -> Dict[str, object]:
    retained_path = model_dir / "retained_expert_indices.json"
    pruned_path = model_dir / "pruned_experts.json"
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
        result["error"] = "missing pruned_experts.json"
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
    _, layers = resolve_language_layers(model)
    first_sparse_idx = None
    for layer_idx, layer in enumerate(layers):
        if getattr(layer, "enable_moe_block", False) and hasattr(layer, "router") and hasattr(layer.router, "proj"):
            first_sparse_idx = layer_idx
            break
    if first_sparse_idx is None:
        return {"ok": False, "error": "No Gemma4 sparse MoE layer found."}

    captured = {}
    handle = layers[first_sparse_idx].router.proj.register_forward_hook(
        lambda _module, _inputs, output: captured.setdefault("router_logits", output.detach())
    )
    try:
        inputs = prepare_chat_inputs(tokenizer, prompt)
        input_device = first_param_device(model)
        inputs = {key: value.to(input_device) for key, value in inputs.items()}
        with torch.no_grad():
            model(**inputs, use_cache=False)
    finally:
        handle.remove()
    router_shape = tuple(captured["router_logits"].shape) if "router_logits" in captured else None
    return {
        "ok": bool(router_shape and router_shape[-1] == expected_retained),
        "first_sparse_layer_index": first_sparse_idx,
        "first_router_shape": router_shape,
    }


def validate_one(
    model_dir: Path,
    expected_retained: int,
    expected_pruned: int,
    prompt: str,
    max_new_tokens: int,
    run_generate: bool,
) -> Dict[str, object]:
    started = time.time()
    result: Dict[str, object] = {"model_dir": str(model_dir), "ok": False, "error": None}
    if not model_dir.exists():
        result["error"] = "missing output directory"
        return result
    if not (model_dir / "config.json").exists():
        result["error"] = "missing config.json"
        return result

    try:
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
        text_config = getattr(config, "text_config", config)
        config_num_experts = int(getattr(text_config, "num_experts"))
        config_top_k = int(getattr(text_config, "top_k_experts"))
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = load_model(model_dir, torch_dtype="bfloat16", device_map="auto")
        model.eval()
        structure = inspect_structure(model, expected_retained=expected_retained)
        index_maps = check_index_maps(model_dir, structure["layer_indices"], expected_retained, expected_pruned)
        router_check = check_router_forward(model, tokenizer, prompt, expected_retained)
        config_ok = (
            config.model_type == "gemma4"
            and bool(getattr(text_config, "enable_moe_block", False))
            and config_num_experts == expected_retained
            and config_top_k == EXPECTED_TOP_K
        )

        generated_preview = ""
        if run_generate and max_new_tokens > 0:
            inputs = prepare_chat_inputs(tokenizer, prompt)
            input_device = first_param_device(model)
            inputs = {key: value.to(input_device) for key, value in inputs.items()}
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            generated_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
            generated_preview = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        result.update(
            {
                "ok": bool(config_ok and structure["sparse_layers_ok"] and index_maps.get("ok") and router_check.get("ok")),
                "config_model_type": config.model_type,
                "config_num_experts": config_num_experts,
                "config_top_k_experts": config_top_k,
                "expected_retained_experts": expected_retained,
                "expected_pruned_experts": expected_pruned,
                "structure": structure,
                "index_maps": index_maps,
                "router_check": router_check,
                "generated_preview": generated_preview[:500],
                "total_params": count_parameters(model),
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
    parser = argparse.ArgumentParser(description="Validate saved Gemma4 AIMER strategy checkpoints.")
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--ratio", required=True, choices=sorted(EXPECTED_RETAINED))
    parser.add_argument("--expected_retained_experts", type=int, default=None)
    parser.add_argument("--expected_pruned_experts", type=int, default=None)
    parser.add_argument("--prompt", default="Explain mixture-of-experts models in one sentence.")
    parser.add_argument("--max_new_tokens", type=int, default=24)
    parser.add_argument("--skip_generate", action="store_true")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    expected_retained = args.expected_retained_experts or EXPECTED_RETAINED[args.ratio]
    expected_pruned = args.expected_pruned_experts or EXPECTED_PRUNED[args.ratio]
    report = {
        "ratio": args.ratio,
        "expected_retained_experts": expected_retained,
        "expected_pruned_experts": expected_pruned,
        "prompt": args.prompt,
        "enable_thinking": False,
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
            run_generate=not args.skip_generate,
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
