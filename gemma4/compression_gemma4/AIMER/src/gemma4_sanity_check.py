from __future__ import annotations

import argparse
import json
import logging
import pathlib
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

try:
    from transformers import AutoModelForImageTextToText
except ImportError:  # pragma: no cover - depends on transformers version
    AutoModelForImageTextToText = None


LOGGER = logging.getLogger("gemma4_sanity_check")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def resolve_torch_dtype(dtype_name: str):
    name = str(dtype_name).strip().lower()
    if name == "auto":
        return "auto"
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[name]


def get_text_config(config: Any) -> Any:
    get_text = getattr(config, "get_text_config", None)
    if callable(get_text):
        return get_text()
    return getattr(config, "text_config", config)


def resolve_decoder_layers(model: torch.nn.Module) -> tuple[str, torch.nn.ModuleList]:
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
    raise RuntimeError("Could not find Gemma4 text decoder layers.")


def get_routed_counts(layer: torch.nn.Module) -> dict[str, Any] | None:
    router = getattr(layer, "router", None)
    experts = getattr(layer, "experts", None)
    router_proj = getattr(router, "proj", None)
    router_weight = getattr(router_proj, "weight", None)
    gate_up_proj = getattr(experts, "gate_up_proj", None)
    down_proj = getattr(experts, "down_proj", None)
    per_expert_scale = getattr(router, "per_expert_scale", None)
    if not isinstance(router_weight, torch.Tensor):
        return None
    if not isinstance(gate_up_proj, torch.Tensor):
        return None
    if not isinstance(down_proj, torch.Tensor):
        return None
    return {
        "router": int(router_weight.shape[0]),
        "gate_up": int(gate_up_proj.shape[0]),
        "down": int(down_proj.shape[0]),
        "per_expert_scale": int(per_expert_scale.shape[0]) if isinstance(per_expert_scale, torch.Tensor) else None,
        "gate_up_shape": list(gate_up_proj.shape),
        "down_shape": list(down_proj.shape),
        "router_shape": list(router_weight.shape),
        "shared_expert_present": bool(hasattr(layer, "mlp")),
    }


def load_json(path: pathlib.Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: pathlib.Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def get_input_device(model: torch.nn.Module) -> torch.device:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        preferred = [
            "model.language_model.embed_tokens",
            "model.embed_tokens",
            "language_model.embed_tokens",
            "model.model.embed_tokens",
            "model",
        ]
        for key in preferred:
            if key in device_map:
                dev = device_map[key]
                if isinstance(dev, int):
                    return torch.device(f"cuda:{dev}")
                if isinstance(dev, str):
                    return torch.device(dev)
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cpu")


def load_model(args: argparse.Namespace):
    dtype = resolve_torch_dtype(args.torch_dtype)
    device_map = None if str(args.device_map).strip().lower() == "none" else args.device_map
    kwargs = {
        "device_map": device_map,
        "dtype": dtype,
        "trust_remote_code": bool(args.trust_remote_code),
        "local_files_only": bool(args.local_files_only),
        "low_cpu_mem_usage": True,
    }
    if args.offload_folder:
        pathlib.Path(args.offload_folder).mkdir(parents=True, exist_ok=True)
        kwargs["offload_folder"] = args.offload_folder
        kwargs["offload_state_dict"] = True

    if args.model_class == "causal-lm":
        auto_cls = AutoModelForCausalLM
    elif args.model_class == "image-text-to-text":
        if AutoModelForImageTextToText is None:
            raise RuntimeError("AutoModelForImageTextToText is unavailable in this transformers build.")
        auto_cls = AutoModelForImageTextToText
    else:
        raise ValueError(f"Unsupported --model-class: {args.model_class}")

    try:
        return auto_cls.from_pretrained(str(args.pruned_dir), **kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        return auto_cls.from_pretrained(str(args.pruned_dir), **kwargs)


def load_processor_or_tokenizer(pruned_dir: pathlib.Path, trust_remote_code: bool, local_files_only: bool):
    try:
        return AutoProcessor.from_pretrained(
            pruned_dir,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
    except Exception as exc:  # pragma: no cover - model/version dependent
        LOGGER.warning("AutoProcessor load failed; falling back to tokenizer: %s", exc)
        return AutoTokenizer.from_pretrained(
            pruned_dir,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )


def build_text_inputs(processor: Any, prompt: str):
    messages_string = [{"role": "user", "content": prompt}]
    messages_mm = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    for messages in (messages_string, messages_mm):
        try:
            return processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            )
        except TypeError:
            try:
                text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return processor(text=text, return_tensors="pt")
            except Exception:
                continue
        except Exception:
            continue

    return processor(prompt, return_tensors="pt")


def decode_new_tokens(processor: Any, output_ids: torch.Tensor, input_len: int) -> str:
    new_ids = output_ids[input_len:]
    if hasattr(processor, "decode"):
        return processor.decode(new_ids, skip_special_tokens=True)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer.decode(new_ids, skip_special_tokens=True)
    return ""


@torch.inference_mode()
def run_generation_check(model: torch.nn.Module, processor: Any, prompt: str, max_new_tokens: int) -> str:
    inputs = build_text_inputs(processor, prompt)
    input_device = get_input_device(model)
    moved = {}
    for key, value in inputs.items():
        moved[key] = value.to(input_device) if isinstance(value, torch.Tensor) else value
    input_len = int(moved["input_ids"].shape[-1])
    outputs = model.generate(
        **moved,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    return decode_new_tokens(processor, outputs[0].detach().cpu(), input_len)


def parse_args(default_pruned_dir: str | None = None, default_keep_experts: int | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity check a pruned Gemma4 AIMER checkpoint.")
    parser.add_argument("--pruned-dir", default=default_pruned_dir, required=default_pruned_dir is None)
    parser.add_argument("--expected-keep-experts", type=int, default=default_keep_experts, required=default_keep_experts is None)
    parser.add_argument("--expected-original-experts", type=int, default=128)
    parser.add_argument("--expected-layers", type=int, default=30)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", type=str2bool, default=True)
    parser.add_argument("--local-files-only", type=str2bool, default=True)
    parser.add_argument("--model-class", choices=["causal-lm", "image-text-to-text"], default="causal-lm")
    parser.add_argument("--offload-folder", default="")
    parser.add_argument("--prompt", default="Answer in one short sentence: What is 2+2?")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    return parser.parse_args()


def main(default_pruned_dir: str | None = None, default_keep_experts: int | None = None) -> None:
    args = parse_args(default_pruned_dir, default_keep_experts)
    pruned_dir = pathlib.Path(args.pruned_dir).resolve()
    if not pruned_dir.is_dir():
        raise FileNotFoundError(f"Pruned model directory not found: {pruned_dir}")

    retained_path = pruned_dir / "retained_expert_indices.json"
    pruned_path = pruned_dir / "pruned_experts.json"
    mapping_path = pruned_dir / "expert_mapping.json"
    for path in (retained_path, pruned_path, mapping_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing pruning metadata: {path}")

    retained = load_json(retained_path)
    pruned = load_json(pruned_path)
    mapping = load_json(mapping_path)

    expected_keep = int(args.expected_keep_experts)
    expected_pruned = int(args.expected_original_experts - expected_keep)
    for layer in range(int(args.expected_layers)):
        key = str(layer)
        if key not in retained:
            raise RuntimeError(f"retained_expert_indices.json missing layer {layer}.")
        if len(retained[key]) != expected_keep:
            raise RuntimeError(f"Layer {layer} retained count mismatch: {len(retained[key])} != {expected_keep}")
        if len(pruned.get(key, [])) != expected_pruned:
            raise RuntimeError(f"Layer {layer} pruned count mismatch: {len(pruned.get(key, []))} != {expected_pruned}")
        ids = retained[key] + pruned.get(key, [])
        if sorted(ids) != list(range(int(args.expected_original_experts))):
            raise RuntimeError(f"Layer {layer} retained/pruned ids do not cover original expert ids.")
        original_to_pruned = mapping[key]["original_expert_id_to_pruned_model_expert_id"]
        for new_id, original_id in enumerate(retained[key]):
            if int(original_to_pruned[str(original_id)]) != new_id:
                raise RuntimeError(f"Layer {layer} mapping mismatch for original expert {original_id}.")

    LOGGER.info("Loading pruned model from %s", pruned_dir)
    model = load_model(args)
    model.eval()

    text_config = get_text_config(model.config)
    config_num_experts = int(getattr(text_config, "num_experts"))
    if config_num_experts != expected_keep:
        raise RuntimeError(f"Config num_experts mismatch: {config_num_experts} != {expected_keep}")
    top_k = int(getattr(text_config, "top_k_experts", 0))
    if top_k <= 0 or top_k > expected_keep:
        raise RuntimeError(f"Invalid top_k_experts after pruning: {top_k}")

    layer_prefix, layers = resolve_decoder_layers(model)
    layer_reports = []
    for layer_idx, layer in enumerate(layers):
        counts = get_routed_counts(layer)
        if counts is None:
            continue
        for field in ("router", "gate_up", "down", "per_expert_scale"):
            if counts[field] != expected_keep:
                raise RuntimeError(f"Layer {layer_idx} {field} count mismatch: {counts[field]} != {expected_keep}")
        layer_reports.append({"layer": layer_idx, "layer_path": f"{layer_prefix}.layers.{layer_idx}", **counts})

    if len(layer_reports) != int(args.expected_layers):
        raise RuntimeError(f"Detected {len(layer_reports)} routed MoE layers, expected {args.expected_layers}")

    total_params = int(sum(param.numel() for param in model.parameters()))
    trainable_params = int(sum(param.numel() for param in model.parameters() if param.requires_grad))

    processor = load_processor_or_tokenizer(
        pruned_dir,
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=bool(args.local_files_only),
    )
    generated_text = run_generation_check(
        model=model,
        processor=processor,
        prompt=args.prompt,
        max_new_tokens=int(args.max_new_tokens),
    )

    report = {
        "pruned_dir": str(pruned_dir),
        "status": "ok",
        "expected_original_experts_per_layer": int(args.expected_original_experts),
        "expected_keep_experts_per_layer": expected_keep,
        "expected_pruned_experts_per_layer": expected_pruned,
        "config_num_experts": config_num_experts,
        "config_top_k_experts": top_k,
        "num_routed_moe_layers": len(layer_reports),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "layer_reports": layer_reports,
        "generation_prompt": args.prompt,
        "generated_text": generated_text,
        "enable_thinking": False,
    }
    dump_json(pruned_dir / "sanity_check_report.json", report)

    print("=== Gemma4 AIMER sanity check OK ===")
    print(f"pruned_dir={pruned_dir}")
    print(f"routed_moe_layers={len(layer_reports)}")
    print(f"experts_per_layer={expected_keep}")
    print(f"total_params={total_params}")
    print(f"generated_text={generated_text!r}")


if __name__ == "__main__":
    main()
