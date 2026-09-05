#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Dict, List

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

EXPECTED_PHYSICAL = {
    "ratio_50": 64,
    "ratio_625": 80,
    "ratio_75": 96,
}
EXPECTED_LOGICAL = 128


def first_param_device(model) -> torch.device:
    try:
        emb = model.get_input_embeddings()
        if emb is not None:
            for param in emb.parameters():
                return param.device
    except Exception:
        pass
    for param in model.parameters():
        return param.device
    return torch.device("cpu")


def load_model(model_dir: Path, torch_dtype: str, device_map: str):
    model_cls = AutoModelForImageTextToText or AutoModelForCausalLM
    if model_cls is None:
        raise ImportError("Transformers must provide AutoModelForImageTextToText or AutoModelForCausalLM.")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch_dtype != "float32" else torch.float32
    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available() and device_map:
        kwargs["device_map"] = device_map
    return model_cls.from_pretrained(model_dir, **kwargs)


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


def count_parameters(model, only_trainable: bool = False) -> int:
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def inspect_structure(model, expected_physical: int, expected_logical: int) -> Dict[str, object]:
    sparse_layers: List[Dict[str, object]] = []
    language_layers = model.model.language_model.layers
    for layer_idx, layer in enumerate(language_layers):
        if not (getattr(layer, "enable_moe_block", False) and hasattr(layer, "router") and hasattr(layer, "experts")):
            continue
        experts = layer.experts
        mapping = experts.expert_id_mapping.detach().cpu()
        entry = {
            "layer_index": layer_idx,
            "router_out_features": int(layer.router.proj.out_features),
            "router_scale_len": int(layer.router.per_expert_scale.numel()),
            "physical_gate_up_experts": int(experts.gate_up_proj.shape[0]),
            "physical_down_experts": int(experts.down_proj.shape[0]),
            "mapping_len": int(mapping.numel()),
            "mapping_min": int(mapping.min().item()),
            "mapping_max": int(mapping.max().item()),
            "has_shared_mlp": all(hasattr(layer.mlp, name) for name in ("gate_proj", "up_proj", "down_proj")),
        }
        layer_ok = (
            entry["router_out_features"] == expected_logical
            and entry["router_scale_len"] == expected_logical
            and entry["physical_gate_up_experts"] == expected_physical
            and entry["physical_down_experts"] == expected_physical
            and entry["mapping_len"] == expected_logical
            and entry["mapping_min"] >= 0
            and entry["mapping_max"] < expected_physical
            and entry["has_shared_mlp"]
        )
        entry["ok"] = bool(layer_ok)
        sparse_layers.append(entry)
    return {
        "num_sparse_layers": len(sparse_layers),
        "sparse_layers_ok": all(layer["ok"] for layer in sparse_layers) and bool(sparse_layers),
        "physical_gate_up_unique": sorted({layer["physical_gate_up_experts"] for layer in sparse_layers}),
        "router_out_features_unique": sorted({layer["router_out_features"] for layer in sparse_layers}),
        "mapping_len_unique": sorted({layer["mapping_len"] for layer in sparse_layers}),
        "sparse_layers": sparse_layers,
    }


def validate_one(model_dir: Path, expected_physical: int, prompt: str, max_new_tokens: int, run_generate: bool) -> Dict[str, object]:
    started = time.time()
    result: Dict[str, object] = {"model_dir": str(model_dir), "ok": False, "error": None}
    if not model_dir.exists():
        result["error"] = "missing output directory"
        return result
    if not (model_dir / "config.json").exists():
        result["error"] = "missing config.json"
        return result

    try:
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        text_config = getattr(config, "text_config", config)
        logical = int(getattr(text_config, "logical_num_experts", getattr(text_config, "num_experts")))
        physical = int(getattr(text_config, "physical_num_experts", getattr(text_config, "num_experts")))
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = load_model(model_dir, torch_dtype="bfloat16", device_map="auto")
        model.eval()
        structure = inspect_structure(model, expected_physical=expected_physical, expected_logical=EXPECTED_LOGICAL)
        config_ok = (
            config.model_type == "gemma4_moe_compressed"
            and logical == EXPECTED_LOGICAL
            and physical == expected_physical
            and int(getattr(text_config, "num_experts")) == EXPECTED_LOGICAL
        )

        inputs = prepare_chat_inputs(tokenizer, prompt)
        input_device = first_param_device(model)
        inputs = {key: value.to(input_device) for key, value in inputs.items()}
        sparse_layers = [layer["layer_index"] for layer in structure["sparse_layers"]]
        if not sparse_layers:
            raise ValueError("No compressed Gemma4 MoE layers were found.")
        captured = {}
        first_sparse = sparse_layers[0]
        handle = model.model.language_model.layers[first_sparse].router.proj.register_forward_hook(
            lambda _module, _inputs, output: captured.setdefault("router_logits", output.detach())
        )
        try:
            with torch.no_grad():
                model(**inputs, use_cache=False)
        finally:
            handle.remove()
        router_shape = tuple(captured["router_logits"].shape) if "router_logits" in captured else None
        router_ok = bool(router_shape and router_shape[-1] == EXPECTED_LOGICAL)

        generated_preview = ""
        if run_generate and max_new_tokens > 0:
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
                "ok": bool(config_ok and structure["sparse_layers_ok"] and router_ok),
                "config_model_type": config.model_type,
                "logical_num_experts": logical,
                "physical_num_experts": physical,
                "expected_physical_num_experts": expected_physical,
                "structure": structure,
                "first_router_shape": router_shape,
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
    parser = argparse.ArgumentParser(description="Validate saved Gemma4 MC-SMoE strategy checkpoints.")
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--ratio", required=True, choices=sorted(EXPECTED_PHYSICAL))
    parser.add_argument("--expected_physical_experts", type=int, default=None)
    parser.add_argument("--prompt", default="Explain mixture-of-experts models in one sentence.")
    parser.add_argument("--max_new_tokens", type=int, default=24)
    parser.add_argument("--skip_generate", action="store_true")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    expected = args.expected_physical_experts or EXPECTED_PHYSICAL[args.ratio]
    report = {
        "ratio": args.ratio,
        "expected_physical_experts": expected,
        "prompt": args.prompt,
        "enable_thinking": False,
        "strategies": {},
    }
    for strategy in STRATEGIES:
        print(f"[Validate] {args.ratio}/{strategy}", flush=True)
        report["strategies"][strategy] = validate_one(
            model_dir=base_dir / args.ratio / strategy,
            expected_physical=expected,
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
