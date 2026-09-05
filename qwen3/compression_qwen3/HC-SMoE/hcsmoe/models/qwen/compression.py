import json
import os
import shutil
from copy import deepcopy
from pathlib import Path
from types import MethodType
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file


REPO_ROOT = Path(__file__).resolve().parents[3]
CUSTOM_CODE_FILES = (
    "configuration_hcsmoe_qwen3.py",
    "modeling_hcsmoe_qwen3.py",
)


def _ensure_inside_repo(path: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    root = REPO_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(
            f"Refusing to write outside {root}. Got output path: {resolved}"
        )
    return resolved


def _parse_layer_idx(ffn_name: str) -> int:
    # Expected form: model.layers.{idx}.mlp
    parts = ffn_name.split(".")
    if len(parts) < 4 or parts[0] != "model" or parts[1] != "layers":
        raise ValueError(f"Unexpected MoE layer name: {ffn_name}")
    return int(parts[2])


def _remap_group_labels(group_labels: torch.Tensor) -> torch.Tensor:
    labels = group_labels.detach().cpu().long()
    unique_labels = torch.unique(labels, sorted=True).tolist()
    label_to_physical = {int(label): i for i, label in enumerate(unique_labels)}
    return torch.tensor([label_to_physical[int(label)] for label in labels], dtype=torch.long)


def compressed_qwen3moe_forward(self, hidden_states: torch.Tensor):
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)

    router_logits = self.gate(hidden_states)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    if self.norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)

    logical_to_physical = self.logical_to_physical.to(selected_experts.device)
    selected_physical = logical_to_physical[selected_experts]
    if selected_physical.min().item() < 0 or selected_physical.max().item() >= self.physical_num_experts:
        raise IndexError("Mapped physical expert id is out of range.")

    token_physical_weights = torch.zeros(
        hidden_states.shape[0],
        self.physical_num_experts,
        dtype=routing_weights.dtype,
        device=hidden_states.device,
    )
    token_physical_weights.scatter_add_(1, selected_physical, routing_weights)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    active_physical = torch.nonzero(token_physical_weights.sum(dim=0) > 0, as_tuple=False).flatten()
    for physical_idx in active_physical.tolist():
        top_x = torch.nonzero(token_physical_weights[:, physical_idx] > 0, as_tuple=False).flatten()
        current_state = hidden_states[top_x]
        current_hidden_states = self.experts[physical_idx](current_state)
        current_hidden_states = current_hidden_states * token_physical_weights[top_x, physical_idx, None]
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


def compress_physical_experts_(
    model,
    group_state_dict: Dict[str, torch.Tensor],
    expected_physical_num_experts: Optional[int] = None,
    base_model_name: Optional[str] = None,
) -> Dict[str, Any]:
    logical_num_experts = int(getattr(model.config, "num_experts"))
    metadata: Dict[str, Any] = {
        "base_model_name_or_path": base_model_name,
        "logical_num_experts": logical_num_experts,
        "physical_num_experts": None,
        "num_experts_per_tok": int(getattr(model.config, "num_experts_per_tok")),
        "layers": {},
    }

    for ffn_name, group_labels in sorted(group_state_dict.items(), key=lambda item: _parse_layer_idx(item[0])):
        layer_idx = _parse_layer_idx(ffn_name)
        mlp = model.model.layers[layer_idx].mlp
        if not hasattr(mlp, "experts") or not isinstance(mlp.experts, nn.ModuleList):
            raise TypeError(
                "Expected Qwen3MoeSparseMoeBlock with a ModuleList named 'experts'. "
                "Use transformers==4.51.0 for Qwen3-30B-A3B-Instruct-2507."
            )

        original_to_physical = _remap_group_labels(group_labels)
        physical_num_experts = int(original_to_physical.max().item()) + 1
        if expected_physical_num_experts is not None and physical_num_experts != expected_physical_num_experts:
            raise ValueError(
                f"{ffn_name} has {physical_num_experts} physical groups, "
                f"expected {expected_physical_num_experts}."
            )
        if original_to_physical.numel() != logical_num_experts:
            raise ValueError(
                f"{ffn_name} mapping length is {original_to_physical.numel()}, "
                f"expected {logical_num_experts}."
            )

        new_experts: List[nn.Module] = []
        layer_groups: List[Dict[str, Any]] = []
        for physical_id in range(physical_num_experts):
            members = torch.where(original_to_physical == physical_id)[0].tolist()
            if not members:
                raise ValueError(f"{ffn_name} physical group {physical_id} is empty.")
            representative = int(members[0])
            new_experts.append(deepcopy(mlp.experts[representative]))
            layer_groups.append(
                {
                    "physical_expert_id": physical_id,
                    "representative_original_expert_id": representative,
                    "original_expert_ids": [int(member) for member in members],
                }
            )

        mlp.experts = nn.ModuleList(new_experts)
        mlp.logical_num_experts = logical_num_experts
        mlp.physical_num_experts = physical_num_experts
        mlp.num_experts = logical_num_experts
        mapping_device = mlp.gate.weight.device
        mapping_tensor = original_to_physical.to(device=mapping_device)
        if "logical_to_physical" in mlp._buffers:
            mlp._buffers["logical_to_physical"] = mapping_tensor
        else:
            mlp.register_buffer("logical_to_physical", mapping_tensor, persistent=False)
        mlp.forward = MethodType(compressed_qwen3moe_forward, mlp)

        metadata["layers"][str(layer_idx)] = {
            "ffn_name": ffn_name,
            "original_to_physical": [int(x) for x in original_to_physical.tolist()],
            "groups": layer_groups,
        }
        metadata["physical_num_experts"] = physical_num_experts

    if metadata["physical_num_experts"] is None:
        raise ValueError("No MoE layers were compressed.")
    return metadata


def _copy_custom_code(output_dir: Path) -> None:
    src_dir = Path(__file__).resolve().parent
    for filename in CUSTOM_CODE_FILES:
        shutil.copy2(src_dir / filename, output_dir / filename)


def _patch_config_json(output_dir: Path, metadata: Dict[str, Any]) -> None:
    config_path = output_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    layer_mapping = {
        layer_idx: layer_info["original_to_physical"]
        for layer_idx, layer_info in metadata["layers"].items()
    }
    config.update(
        {
            "model_type": "hcsmoe_qwen3_moe",
            "architectures": ["HCQwen3MoeForCausalLM"],
            "auto_map": {
                "AutoConfig": "configuration_hcsmoe_qwen3.HCQwen3MoeConfig",
                "AutoModel": "modeling_hcsmoe_qwen3.HCQwen3MoeModel",
                "AutoModelForCausalLM": "modeling_hcsmoe_qwen3.HCQwen3MoeForCausalLM",
            },
            "logical_num_experts": metadata["logical_num_experts"],
            "physical_num_experts": metadata["physical_num_experts"],
            "num_experts": metadata["logical_num_experts"],
            "expert_id_mapping": layer_mapping,
            "hcsmoe_expert_mapping_file": "expert_mapping.json",
            "hcsmoe_base_model_name_or_path": metadata.get("base_model_name_or_path"),
            "hcsmoe_compression": {
                "method": "HC-SMoE",
                "logical_num_experts": metadata["logical_num_experts"],
                "physical_num_experts": metadata["physical_num_experts"],
            },
        }
    )

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")


def _parse_shard_size(size: str) -> int:
    size = str(size).strip().upper()
    units = (
        ("GIB", 1024**3),
        ("GB", 1000**3),
        ("MIB", 1024**2),
        ("MB", 1000**2),
        ("KIB", 1024),
        ("KB", 1000),
    )
    for suffix, multiplier in units:
        if size.endswith(suffix):
            return int(float(size[: -len(suffix)]) * multiplier)
    return int(size)


def _cleanup_old_checkpoint_files(output_dir: Path) -> None:
    patterns = (
        "model.safetensors",
        "model.safetensors.index.json",
        "model-*.safetensors",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "pytorch_model-*.bin",
    )
    for pattern in patterns:
        for path in output_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def _save_state_dict_safetensors_sharded(model, output_dir: Path, max_shard_size: str) -> None:
    max_shard_bytes = _parse_shard_size(max_shard_size)
    state_dict = model.state_dict()
    named_tensors = [(name, tensor.detach()) for name, tensor in state_dict.items()]
    total_size = sum(tensor.numel() * tensor.element_size() for _, tensor in named_tensors)

    shards: List[List[Any]] = []
    current: List[Any] = []
    current_size = 0
    for name, tensor in named_tensors:
        tensor_size = tensor.numel() * tensor.element_size()
        if current and current_size + tensor_size > max_shard_bytes:
            shards.append(current)
            current = []
            current_size = 0
        current.append((name, tensor))
        current_size += tensor_size
    if current:
        shards.append(current)

    weight_map = {}
    num_shards = len(shards)
    for shard_idx, shard in enumerate(shards, start=1):
        if num_shards == 1:
            filename = "model.safetensors"
        else:
            filename = f"model-{shard_idx:05d}-of-{num_shards:05d}.safetensors"
        shard_tensors = {
            name: tensor.cpu().contiguous()
            for name, tensor in shard
        }
        save_file(shard_tensors, output_dir / filename, metadata={"format": "pt"})
        for name, _ in shard:
            weight_map[name] = filename
        del shard_tensors

    if num_shards > 1:
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map,
        }
        with (output_dir / "model.safetensors.index.json").open("w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, sort_keys=True)
            f.write("\n")


def save_compressed_qwen3_model(
    model,
    tokenizer,
    output_path: str,
    metadata: Dict[str, Any],
    max_shard_size: str = "5GB",
) -> None:
    output_dir = _ensure_inside_repo(output_path)
    os.makedirs(output_dir, exist_ok=True)

    model.config.logical_num_experts = metadata["logical_num_experts"]
    model.config.physical_num_experts = metadata["physical_num_experts"]
    model.config.expert_id_mapping = {
        layer_idx: layer_info["original_to_physical"]
        for layer_idx, layer_info in metadata["layers"].items()
    }
    model.config.architectures = ["HCQwen3MoeForCausalLM"]
    model.config.auto_map = {
        "AutoConfig": "configuration_hcsmoe_qwen3.HCQwen3MoeConfig",
        "AutoModel": "modeling_hcsmoe_qwen3.HCQwen3MoeModel",
        "AutoModelForCausalLM": "modeling_hcsmoe_qwen3.HCQwen3MoeForCausalLM",
    }

    _cleanup_old_checkpoint_files(output_dir)
    model.config.save_pretrained(output_dir)
    _save_state_dict_safetensors_sharded(model, output_dir, max_shard_size=max_shard_size)
    tokenizer.save_pretrained(output_dir)

    with (output_dir / "expert_mapping.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")

    _copy_custom_code(output_dir)
    _patch_config_json(output_dir, metadata)
