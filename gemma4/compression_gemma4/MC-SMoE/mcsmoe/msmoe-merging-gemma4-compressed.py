import os
import sys
from pathlib import Path
from typing import List, Optional, Union

import torch
from fire import Fire
from transformers import AutoConfig, AutoTokenizer

from configuration_gemma4_moe_compressed import Gemma4MoeCompressedConfig
sys.path.insert(0, str(Path(__file__).resolve().parent / "merging"))
from grouping_gemma4_compressed import (
    ExpertsGrouperForCompressedGemma4,
    apply_compressed_config_to_model,
    build_mapping_artifact,
    compress_gemma4_moe_in_place,
    copy_remote_code_files,
    get_c4_jsonl_dataloader,
    patch_saved_config_json,
    write_mapping_artifacts,
)


def _parse_layers(layers: Optional[Union[str, List[int], int]]) -> Optional[List[int]]:
    if layers is None:
        return None
    if isinstance(layers, str):
        if len(layers.strip()) == 0:
            return []
        return [int(x) for x in layers.split(",")]
    if isinstance(layers, int):
        return [layers]
    return list(layers)


def _parse_torch_dtype(torch_dtype: str):
    if torch_dtype == "auto":
        return "auto"
    if not hasattr(torch, torch_dtype):
        raise ValueError(f"Unknown torch dtype: {torch_dtype}")
    return getattr(torch, torch_dtype)


def _load_gemma4_model(model_name_or_path: str, torch_dtype: str, device_map: str, attn_implementation: Optional[str]):
    try:
        from transformers import AutoModelForImageTextToText
    except ImportError as exc:
        raise ImportError(
            "Gemma4 loading requires a Transformers version with AutoModelForImageTextToText "
            "and Gemma4 support."
        ) from exc

    kwargs = dict(
        torch_dtype=_parse_torch_dtype(torch_dtype),
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    return AutoModelForImageTextToText.from_pretrained(model_name_or_path, **kwargs)


def _load_tokenizer(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def merge_gemma4_moe_compressed(
    model_name_or_path: str = "google/gemma-4-26B-A4B-it",
    calibration_dataset_path: str = os.environ.get("C4_DATASET_PATH", ""),
    physical_num_experts: int = 80,
    expected_logical_num_experts: int = 128,
    ratio_label: str = "62_5",
    similarity_base: str = "router-logits",
    merging_layers: Optional[Union[str, List[int], int]] = None,
    block_size: int = 512,
    batch_size: int = 1,
    subset_ratio: float = 0.01,
    max_calibration_records: Optional[int] = None,
    output_dir: Optional[str] = None,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
    attn_implementation: Optional[str] = None,
    safe_serialization: bool = True,
    max_shard_size: str = "5GB",
):
    if similarity_base != "router-logits":
        raise ValueError("This Gemma4 compressed merge entrypoint currently implements router-logits grouping.")
    if output_dir is None:
        output_dir = os.path.abspath(f"gemma4-26b-a4b-it-mcsmoe-{ratio_label}")
    os.makedirs(output_dir, exist_ok=True)

    print("[MC-SMoE] Loading Gemma4 config:", model_name_or_path)
    original_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    text_config = getattr(original_config, "text_config", original_config)
    logical_num_experts = int(getattr(text_config, "num_experts"))
    print("[MC-SMoE] Confirmed Gemma4 text_config.num_experts:", logical_num_experts)
    if logical_num_experts != expected_logical_num_experts:
        raise ValueError(
            f"Expected Gemma4 logical routed experts/layer={expected_logical_num_experts}, "
            f"but config has {logical_num_experts}."
        )
    if not getattr(text_config, "enable_moe_block", False):
        raise ValueError("Gemma4 text_config.enable_moe_block is false; no routed experts to merge.")
    if physical_num_experts <= 0 or physical_num_experts > logical_num_experts:
        raise ValueError(
            f"physical_num_experts must be in [1, {logical_num_experts}], got {physical_num_experts}."
        )

    print("[MC-SMoE] Loading tokenizer:", model_name_or_path)
    tokenizer = _load_tokenizer(model_name_or_path)

    print("[MC-SMoE] Loading Gemma4 model:", model_name_or_path)
    model = _load_gemma4_model(
        model_name_or_path=model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
    )
    model.eval()

    dataloader_for_merging = get_c4_jsonl_dataloader(
        tokenizer=tokenizer,
        dataset_path=calibration_dataset_path,
        block_size=block_size,
        batch_size=batch_size,
        subset_ratio=subset_ratio,
        max_records=max_calibration_records,
    )

    print(
        f"[MC-SMoE] Expert merge only: logical experts={logical_num_experts}, "
        f"physical experts/layer={physical_num_experts}, ratio={ratio_label}"
    )
    print(f"[MC-SMoE] Calibration dataset: {calibration_dataset_path}")

    grouper = ExpertsGrouperForCompressedGemma4(config=model.config, model=model)
    grouper.compute_routing_statistics(model, dataloader_for_merging)

    merging_layers = _parse_layers(merging_layers)
    if merging_layers is None:
        merging_layers = grouper.sparse_layer_indices
    else:
        merging_layers = [idx for idx in merging_layers if idx in grouper.sparse_layer_indices]
    if len(merging_layers) == 0:
        raise ValueError("[MC-SMoE] No Gemma4 MoE layers selected for merging.")
    if set(merging_layers) != set(grouper.sparse_layer_indices):
        raise ValueError(
            "[MC-SMoE] Partial-layer Gemma4 physical compression is not supported here. "
            "Compress all routed MoE layers so every layer has the requested physical expert count."
        )

    grouper.group_experts_per_layer(
        physical_num_experts=physical_num_experts,
        merging_layers=merging_layers,
    )

    mapping_artifact = build_mapping_artifact(
        grouper=grouper,
        physical_num_experts=physical_num_experts,
        source_model_name_or_path=model_name_or_path,
        similarity_base=similarity_base,
        calibration_dataset=calibration_dataset_path,
        subset_ratio=subset_ratio,
        block_size=block_size,
        batch_size=batch_size,
    )

    compressed_config = Gemma4MoeCompressedConfig(
        **original_config.to_dict(),
        logical_num_experts=logical_num_experts,
        physical_num_experts=physical_num_experts,
        expert_id_mappings=mapping_artifact["config_expert_id_mappings"],
        expert_id_mapping_path="expert_mapping.json",
        compression_ratio=physical_num_experts / logical_num_experts,
        source_model_name_or_path=model_name_or_path,
        mcsmoe_metadata={
            "ratio_label": ratio_label,
            "method": "M-SMoE expert merge only",
            "similarity_base": similarity_base,
            "merge_weighting": "usage_frequency",
            "block_size": block_size,
            "batch_size": batch_size,
            "subset_ratio": subset_ratio,
            "max_calibration_records": max_calibration_records,
            "enable_thinking": False,
        },
    )
    apply_compressed_config_to_model(model, compressed_config)

    model = compress_gemma4_moe_in_place(
        model=model,
        grouper=grouper,
        physical_num_experts=physical_num_experts,
        merging_layers=merging_layers,
    )

    print("[MC-SMoE] Number of parameters after physical expert compression:", model.num_parameters())
    print("[MC-SMoE] Saving compressed model to:", output_dir)
    model.save_pretrained(
        output_dir,
        safe_serialization=safe_serialization,
        max_shard_size=max_shard_size,
    )
    tokenizer.save_pretrained(output_dir)
    write_mapping_artifacts(output_dir, mapping_artifact, grouper)
    copy_remote_code_files(output_dir)
    patch_saved_config_json(output_dir, compressed_config, mapping_artifact)

    print("[MC-SMoE] Saved expert mapping:", os.path.join(output_dir, "expert_mapping.json"))
    print("[MC-SMoE] Done.")


if __name__ == "__main__":
    Fire(merge_gemma4_moe_compressed)
