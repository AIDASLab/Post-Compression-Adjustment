from __future__ import annotations

import argparse
import csv
import json
import logging
import pathlib
import random
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

try:
    from transformers import AutoModelForImageTextToText
except ImportError:  # pragma: no cover - depends on transformers version
    AutoModelForImageTextToText = None


LOGGER = logging.getLogger("gemma4_aimer_prune")
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    raise RuntimeError(
        "Could not find Gemma4 text decoder layers. Expected one of "
        "model.language_model.layers or model.layers."
    )


def tensor_to_metric_f32(tensor: torch.Tensor, metric_device: str) -> torch.Tensor:
    detached = tensor.detach()
    if metric_device == "cpu":
        return detached.to(device="cpu", dtype=torch.float32)
    if metric_device == "auto":
        return detached.to(dtype=torch.float32)
    raise ValueError("--metric-device must be 'auto' or 'cpu'.")


def get_routed_expert_bundle(layer: torch.nn.Module) -> dict[str, Any] | None:
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
    if gate_up_proj.ndim < 2 or down_proj.ndim < 2:
        return None

    routed_count = int(gate_up_proj.shape[0])
    if int(down_proj.shape[0]) != routed_count or int(router_weight.shape[0]) != routed_count:
        raise RuntimeError(
            "Gemma4 routed expert count mismatch: "
            f"router={tuple(router_weight.shape)}, "
            f"gate_up={tuple(gate_up_proj.shape)}, down={tuple(down_proj.shape)}"
        )

    return {
        "router": router,
        "router_proj": router_proj,
        "router_weight": router_weight,
        "per_expert_scale": per_expert_scale,
        "experts": experts,
        "gate_up_proj": gate_up_proj,
        "down_proj": down_proj,
        "num_experts": routed_count,
    }


def compute_aimer_scores(
    *,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    metric_device: str,
) -> list[float]:
    scores: list[float] = []
    num_experts = int(gate_up_proj.shape[0])

    for expert_id in range(num_experts):
        gate_up = tensor_to_metric_f32(gate_up_proj[expert_id], metric_device)
        down = tensor_to_metric_f32(down_proj[expert_id], metric_device)
        abs_sum = gate_up.abs().sum() + down.abs().sum()
        l2_sq = gate_up.square().sum() + down.square().sum()
        numel = int(gate_up.numel() + down.numel())
        if numel <= 0 or float(l2_sq.item()) <= 0.0:
            score = 0.0
        else:
            score_tensor = (abs_sum / numel) / torch.sqrt(l2_sq / numel)
            score = float(score_tensor.item())
        scores.append(score)
        del gate_up, down

    return scores


def select_pruned_experts(
    *,
    scores: list[float],
    keep_experts: int,
    prune_highest_score: bool,
) -> tuple[list[int], list[int], list[int]]:
    original_count = len(scores)
    if keep_experts <= 0 or keep_experts > original_count:
        raise ValueError(f"Invalid keep_experts={keep_experts} for {original_count} experts.")

    if prune_highest_score:
        ranked = sorted(range(original_count), key=lambda idx: (-scores[idx], idx))
    else:
        ranked = sorted(range(original_count), key=lambda idx: (scores[idx], idx))

    prune_count = original_count - keep_experts
    pruned = sorted(ranked[:prune_count])
    kept = sorted(idx for idx in range(original_count) if idx not in set(pruned))
    return kept, pruned, ranked


def slice_parameter_dim0(param: torch.nn.Parameter, retain_indices: list[int]) -> torch.nn.Parameter:
    index = torch.tensor(retain_indices, dtype=torch.long, device=param.device)
    sliced = param.data.index_select(0, index).clone()
    return torch.nn.Parameter(sliced, requires_grad=param.requires_grad)


def prune_gemma4_layer_in_place(
    *,
    bundle: dict[str, Any],
    retain_indices: list[int],
) -> int:
    kept = len(retain_indices)
    router = bundle["router"]
    router_proj = bundle["router_proj"]
    experts = bundle["experts"]

    with torch.no_grad():
        router_proj.weight = slice_parameter_dim0(router_proj.weight, retain_indices)
        if getattr(router_proj, "bias", None) is not None:
            router_proj.bias = slice_parameter_dim0(router_proj.bias, retain_indices)
        if hasattr(router_proj, "out_features"):
            router_proj.out_features = kept

        per_expert_scale = bundle.get("per_expert_scale")
        if isinstance(per_expert_scale, torch.nn.Parameter):
            router.per_expert_scale = slice_parameter_dim0(per_expert_scale, retain_indices)
        elif isinstance(per_expert_scale, torch.Tensor):
            index = torch.tensor(retain_indices, dtype=torch.long, device=per_expert_scale.device)
            router.per_expert_scale = per_expert_scale.index_select(0, index).clone()

        experts.gate_up_proj = slice_parameter_dim0(bundle["gate_up_proj"], retain_indices)
        experts.down_proj = slice_parameter_dim0(bundle["down_proj"], retain_indices)

    if hasattr(experts, "num_experts"):
        experts.num_experts = kept
    if hasattr(router, "config") and hasattr(router.config, "num_experts"):
        router.config.num_experts = kept
    if hasattr(router, "config") and hasattr(router.config, "top_k_experts"):
        router.config.top_k_experts = min(int(router.config.top_k_experts), kept)

    return kept


def update_gemma4_config(model: torch.nn.Module, kept_experts: int) -> None:
    text_config = get_text_config(model.config)
    if hasattr(text_config, "num_experts"):
        text_config.num_experts = int(kept_experts)
    if hasattr(text_config, "top_k_experts"):
        text_config.top_k_experts = min(int(text_config.top_k_experts), int(kept_experts))
    if hasattr(model.config, "text_config"):
        model.config.text_config = text_config


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def dump_json(path: pathlib.Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_model(args: argparse.Namespace):
    dtype = resolve_torch_dtype(args.torch_dtype)
    device_map = None if str(args.device_map).strip().lower() == "none" else args.device_map
    model_kwargs = {
        "device_map": device_map,
        "dtype": dtype,
        "trust_remote_code": bool(args.trust_remote_code),
        "local_files_only": bool(args.local_files_only),
        "low_cpu_mem_usage": True,
    }
    if args.offload_folder:
        pathlib.Path(args.offload_folder).mkdir(parents=True, exist_ok=True)
        model_kwargs["offload_folder"] = args.offload_folder
        model_kwargs["offload_state_dict"] = True

    model_class = args.model_class
    if model_class == "causal-lm":
        auto_cls = AutoModelForCausalLM
    elif model_class == "image-text-to-text":
        if AutoModelForImageTextToText is None:
            raise RuntimeError("AutoModelForImageTextToText is unavailable in this transformers build.")
        auto_cls = AutoModelForImageTextToText
    else:
        raise ValueError(f"Unsupported --model-class: {model_class}")

    try:
        return auto_cls.from_pretrained(args.model_name, **model_kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
        return auto_cls.from_pretrained(args.model_name, **model_kwargs)


def save_processor_and_tokenizer(args: argparse.Namespace, output_dir: pathlib.Path) -> None:
    processor = None
    tokenizer = None

    try:
        processor = AutoProcessor.from_pretrained(
            args.model_name,
            trust_remote_code=bool(args.trust_remote_code),
            local_files_only=bool(args.local_files_only),
        )
        processor.save_pretrained(output_dir)
        tokenizer = getattr(processor, "tokenizer", None)
    except Exception as exc:  # pragma: no cover - model/version dependent
        LOGGER.warning("AutoProcessor save failed; falling back to tokenizer only: %s", exc)

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            trust_remote_code=bool(args.trust_remote_code),
            local_files_only=bool(args.local_files_only),
        )
    tokenizer.save_pretrained(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gemma4 AIMER expert pruning. Uses only pretrained expert weights."
    )
    parser.add_argument("--model-name", default="google/gemma-4-26B-A4B-it")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--keep-experts", type=int, required=True)
    parser.add_argument("--keep-ratio", type=float, required=True)
    parser.add_argument("--expected-original-experts", type=int, default=128)
    parser.add_argument("--expected-layers", type=int, default=30)
    parser.add_argument("--metric", default="aimer")
    parser.add_argument("--metric-device", choices=["auto", "cpu"], default="auto")
    parser.add_argument("--prune-highest-score", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", type=str2bool, default=True)
    parser.add_argument("--local-files-only", type=str2bool, default=False)
    parser.add_argument("--model-class", choices=["causal-lm", "image-text-to-text"], default="causal-lm")
    parser.add_argument("--offload-folder", default="")
    parser.add_argument("--max-shard-size", default="5GB")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.time()
    if args.metric.strip().lower() != "aimer":
        raise ValueError("Only --metric aimer is supported.")

    expected_keep = int(round(args.expected_original_experts * args.keep_ratio))
    if expected_keep != int(args.keep_experts):
        raise ValueError(
            f"keep_ratio={args.keep_ratio} implies {expected_keep} experts, "
            f"but --keep-experts={args.keep_experts}."
        )
    if args.keep_experts <= 0 or args.keep_experts > args.expected_original_experts:
        raise ValueError("--keep-experts must be in (0, expected_original_experts].")

    pruning_ratio = 1.0 - float(args.keep_ratio)
    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    LOGGER.info("Loading Gemma4 model: %s", args.model_name)
    model = load_model(args)
    model.eval()

    text_config = get_text_config(model.config)
    model_type = getattr(model.config, "model_type", "")
    text_model_type = getattr(text_config, "model_type", "")
    if model_type != "gemma4" and text_model_type != "gemma4_text":
        raise RuntimeError(
            f"Expected Gemma4 config, got model_type={model_type!r}, "
            f"text_model_type={text_model_type!r}."
        )
    if not bool(getattr(text_config, "enable_moe_block", False)):
        raise RuntimeError("Gemma4 text_config.enable_moe_block is not enabled.")

    layer_prefix, layers = resolve_decoder_layers(model)
    LOGGER.info("Resolved decoder layers at %s (%d layers).", layer_prefix, len(layers))

    score_rows: list[dict[str, Any]] = []
    score_table_rows: list[dict[str, Any]] = []
    plan_rows: list[dict[str, Any]] = []
    retained_map: dict[str, list[int]] = {}
    pruned_map: dict[str, list[int]] = {}
    expert_mapping: dict[str, dict[str, Any]] = {}
    kept_counts: list[int] = []
    routed_layer_indices: list[int] = []

    for layer_idx, layer in enumerate(layers):
        bundle = get_routed_expert_bundle(layer)
        if bundle is None:
            continue

        num_experts = int(bundle["num_experts"])
        if num_experts != int(args.expected_original_experts):
            raise RuntimeError(
                f"Layer {layer_idx} has {num_experts} routed experts; "
                f"expected {args.expected_original_experts}. This script should be run on "
                "the original unpruned checkpoint."
            )

        layer_path = f"{layer_prefix}.layers.{layer_idx}"
        routed_layer_indices.append(layer_idx)
        LOGGER.info("Scoring layer %d routed experts from %s.experts", layer_idx, layer_path)

        scores = compute_aimer_scores(
            gate_up_proj=bundle["gate_up_proj"],
            down_proj=bundle["down_proj"],
            metric_device=args.metric_device,
        )
        kept_original_ids, pruned_original_ids, ranked_ids = select_pruned_experts(
            scores=scores,
            keep_experts=int(args.keep_experts),
            prune_highest_score=bool(args.prune_highest_score),
        )

        retained_map[str(layer_idx)] = kept_original_ids
        pruned_map[str(layer_idx)] = pruned_original_ids
        original_to_pruned = {str(orig): new for new, orig in enumerate(kept_original_ids)}
        pruned_to_original = {str(new): orig for new, orig in enumerate(kept_original_ids)}
        expert_mapping[str(layer_idx)] = {
            "layer": layer_idx,
            "layer_path": layer_path,
            "router_path": f"{layer_path}.router",
            "routed_experts_path": f"{layer_path}.experts",
            "shared_expert_path": f"{layer_path}.mlp",
            "kept_original_expert_ids": kept_original_ids,
            "pruned_original_expert_ids": pruned_original_ids,
            "original_expert_id_to_pruned_model_expert_id": original_to_pruned,
            "pruned_model_expert_id_to_original_expert_id": pruned_to_original,
            "ranked_original_expert_ids": ranked_ids,
            "prune_highest_score": bool(args.prune_highest_score),
        }

        pruned_set = set(pruned_original_ids)
        ranked_position = {expert_id: rank for rank, expert_id in enumerate(ranked_ids, start=1)}
        for expert_id, score in enumerate(scores):
            score_rows.append(
                {
                    "layer": layer_idx,
                    "expert_id": expert_id,
                    "aimer_score": score,
                    "calib_free_score": score,
                    "is_pruned": int(expert_id in pruned_set),
                }
            )
            score_table_rows.append(
                {
                    "layer": layer_idx,
                    "expert_id": expert_id,
                    "aimer_score": score,
                    "calib_free_score": score,
                    "score_rank_in_layer": ranked_position[expert_id],
                    "is_pruned": int(expert_id in pruned_set),
                }
            )

        kept = prune_gemma4_layer_in_place(bundle=bundle, retain_indices=kept_original_ids)
        kept_counts.append(kept)
        plan_rows.append(
            {
                "layer": layer_idx,
                "total_original_routed_experts": int(args.expected_original_experts),
                "kept_routed_experts": int(kept),
                "pruned_routed_experts": int(args.expected_original_experts - kept),
                "keep_ratio": float(args.keep_ratio),
                "pruning_ratio": float(pruning_ratio),
                "shared_expert_preserved": bool(hasattr(layer, "mlp")),
            }
        )

    if len(routed_layer_indices) != int(args.expected_layers):
        raise RuntimeError(
            f"Detected {len(routed_layer_indices)} Gemma4 MoE layers, "
            f"expected {args.expected_layers}. Layers: {routed_layer_indices}"
        )
    if sorted(set(kept_counts)) != [int(args.keep_experts)]:
        raise RuntimeError(f"Non-uniform kept expert counts after pruning: {sorted(set(kept_counts))}")

    update_gemma4_config(model, int(args.keep_experts))

    metadata = {
        "model_name": args.model_name,
        "model_type": model_type,
        "text_model_type": text_model_type,
        "method": "AIMER",
        "metric": "aimer",
        "calibration_free": True,
        "calibration_dataset_used_for_scoring": False,
        "weight_only_scoring": True,
        "aimer_score_definition": "(mean(abs(w))) / rms(w) over each routed expert gate/up/down weights",
        "prune_highest_score": bool(args.prune_highest_score),
        "keep_ratio": float(args.keep_ratio),
        "pruning_ratio": float(pruning_ratio),
        "original_routed_experts_per_layer": int(args.expected_original_experts),
        "kept_routed_experts_per_layer": int(args.keep_experts),
        "pruned_routed_experts_per_layer": int(args.expected_original_experts - args.keep_experts),
        "routed_moe_layers": routed_layer_indices,
        "num_routed_moe_layers": len(routed_layer_indices),
        "shared_expert_handling": "Gemma4 dense layer.mlp is preserved and is not part of AIMER pruning.",
        "seed": int(args.seed),
        "torch_dtype": str(args.torch_dtype),
        "metric_device": str(args.metric_device),
        "elapsed_seconds_before_save": round(time.time() - started_at, 3),
    }

    LOGGER.info("Saving pruned model to %s", output_dir)
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    save_processor_and_tokenizer(args, output_dir)

    dump_json(output_dir / "retained_expert_indices.json", retained_map)
    dump_json(output_dir / "pruned_experts.json", pruned_map)
    dump_json(output_dir / "expert_mapping.json", expert_mapping)
    dump_json(
        output_dir / "pruning_plan.json",
        {
            "plan": plan_rows,
            "metric": "aimer",
            "uniform_pruning": True,
            "uniform_pruned_per_layer": int(args.expected_original_experts - args.keep_experts),
            "uniform_kept_per_layer": int(args.keep_experts),
        },
    )
    metadata["elapsed_seconds_total"] = round(time.time() - started_at, 3)
    dump_json(output_dir / "calib_free_metadata.json", metadata)
    dump_json(output_dir / "gemma4_aimer_metadata.json", metadata)
    write_csv(output_dir / "calib_free_scores.csv", score_rows)
    write_csv(output_dir / "calib_free_score_table.csv", score_table_rows)

    LOGGER.info("Saved Gemma4 AIMER pruned model at %s", output_dir)
    print(f"PRUNED_MODEL_DIR={output_dir}")


if __name__ == "__main__":
    main()
