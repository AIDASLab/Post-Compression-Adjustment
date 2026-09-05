import json
import os
import shutil
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from configuration_gemma4_moe_compressed import Gemma4MoeCompressedConfig
from modeling_gemma4_moe_compressed import Gemma4CompressedTextExperts

FP32_EPS = 1e-7


def _text_config(config):
    return getattr(config, "text_config", config)


def _language_layers(model):
    try:
        return model.model.language_model.layers
    except AttributeError as exc:
        raise AttributeError("Expected Gemma4 model.model.language_model.layers to exist.") from exc


def _resolve_sparse_layer_indices(config, model=None) -> List[int]:
    if model is not None:
        return [
            layer_idx
            for layer_idx, layer in enumerate(_language_layers(model))
            if getattr(layer, "enable_moe_block", False) and hasattr(layer, "router") and hasattr(layer, "experts")
        ]

    text_cfg = _text_config(config)
    if not getattr(text_cfg, "enable_moe_block", False):
        return []
    return list(range(text_cfg.num_hidden_layers))


class C4JsonlTokenBlockDataset(IterableDataset):
    def __init__(
        self,
        dataset_path: str,
        tokenizer,
        block_size: int,
        subset_ratio: float = 0.01,
        max_records: Optional[int] = None,
        line_count: Optional[int] = None,
    ):
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.subset_ratio = subset_ratio
        self.max_records = max_records
        self.line_count = line_count

    def _iter_texts(self) -> Iterable[str]:
        limit = self.max_records
        if limit is None and self.subset_ratio is not None and self.subset_ratio < 1.0:
            if self.line_count is None:
                with open(self.dataset_path, "r", encoding="utf-8") as f:
                    self.line_count = sum(1 for _ in f)
            limit = max(1, int(self.line_count * self.subset_ratio))

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if limit is not None and idx >= limit:
                    break
                if not line.strip():
                    continue
                record = json.loads(line)
                text = record.get("text", "")
                if text:
                    yield text

    def __iter__(self):
        token_buffer: List[int] = []
        for text in self._iter_texts():
            token_buffer.extend(self.tokenizer(text, truncation=False)["input_ids"])
            while len(token_buffer) >= self.block_size:
                block = token_buffer[: self.block_size]
                token_buffer = token_buffer[self.block_size :]
                input_ids = torch.tensor(block, dtype=torch.long)
                attention_mask = torch.ones_like(input_ids)
                yield {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": input_ids.clone(),
                }


def get_c4_jsonl_dataloader(
    tokenizer,
    dataset_path: str,
    block_size: int,
    batch_size: int,
    subset_ratio: float = 0.01,
    max_records: Optional[int] = None,
) -> DataLoader:
    dataset = C4JsonlTokenBlockDataset(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        block_size=block_size,
        subset_ratio=subset_ratio,
        max_records=max_records,
    )
    return DataLoader(dataset, batch_size=batch_size)


class ExpertsGrouperForCompressedGemma4:
    def __init__(self, config, model=None):
        text_cfg = _text_config(config)
        self.logical_num_experts = int(text_cfg.num_experts)
        self.d_model = text_cfg.hidden_size
        self.sparse_layer_indices = _resolve_sparse_layer_indices(config=config, model=model)
        if len(self.sparse_layer_indices) == 0:
            raise ValueError("[MC-SMoE] No Gemma4 MoE layers detected.")
        self._similarity_state_dict: Dict[str, torch.Tensor] = {}
        self._usage_frequency_state_dict: Dict[str, torch.Tensor] = {}
        self._group_state_dict: Dict[str, torch.LongTensor] = {}
        self._dominant_experts: Dict[str, List[int]] = {}
        self.reset_all()

    def reset_all(self):
        for layer_idx in self.sparse_layer_indices:
            ffn_name = f"model.language_model.layers.{layer_idx}.experts"
            self._similarity_state_dict[ffn_name] = torch.eye(self.logical_num_experts, device="cpu")
            self._usage_frequency_state_dict[ffn_name] = torch.zeros(self.logical_num_experts, device="cpu")
            self._group_state_dict[ffn_name] = torch.arange(self.logical_num_experts, device="cpu")

    def group_state_dict(self) -> Dict[str, torch.LongTensor]:
        return deepcopy(self._group_state_dict)

    def usage_frequency_state_dict(self) -> Dict[str, torch.Tensor]:
        return deepcopy(self._usage_frequency_state_dict)

    def dominant_experts(self) -> Dict[str, List[int]]:
        return deepcopy(self._dominant_experts)

    @torch.no_grad()
    def compute_routing_statistics(self, model, dataloader: DataLoader):
        model.eval()
        layers = _language_layers(model)
        gram_acc = {
            f"model.language_model.layers.{layer_idx}.experts": torch.zeros(
                (self.logical_num_experts, self.logical_num_experts),
                dtype=torch.float64,
                device="cpu",
            )
            for layer_idx in self.sparse_layer_indices
        }
        norm_acc = {
            f"model.language_model.layers.{layer_idx}.experts": torch.zeros(
                self.logical_num_experts,
                dtype=torch.float64,
                device="cpu",
            )
            for layer_idx in self.sparse_layer_indices
        }
        captured_logits = {}
        captured_topk = {}
        handles = []

        for layer_idx in self.sparse_layer_indices:
            layer = layers[layer_idx]

            def capture_logits(_module, _inputs, output, layer_idx=layer_idx):
                captured_logits[layer_idx] = output.detach()

            def capture_topk(_module, _inputs, output, layer_idx=layer_idx):
                captured_topk[layer_idx] = output[2].detach()

            handles.append(layer.router.proj.register_forward_hook(capture_logits))
            handles.append(layer.router.register_forward_hook(capture_topk))

        input_device = model.get_input_embeddings().weight.device
        try:
            for batch in tqdm(dataloader, desc="[MC-SMoE] Collecting Gemma4 router statistics"):
                batch = {k: v.to(input_device) for k, v in batch.items()}
                batch.pop("labels", None)
                captured_logits.clear()
                captured_topk.clear()
                model(**batch, use_cache=False)

                missing = [idx for idx in self.sparse_layer_indices if idx not in captured_logits or idx not in captured_topk]
                if missing:
                    raise ValueError(f"[MC-SMoE] Missing Gemma4 router captures for layers: {missing}")

                for layer_idx in self.sparse_layer_indices:
                    ffn_name = f"model.language_model.layers.{layer_idx}.experts"
                    selected = captured_topk[layer_idx].reshape(-1)
                    unique, counts = torch.unique(selected, return_counts=True)
                    self._usage_frequency_state_dict[ffn_name][unique.cpu()] += counts.cpu()

                    logits = captured_logits[layer_idx].reshape(-1, self.logical_num_experts).float()
                    gram_acc[ffn_name] += torch.matmul(logits.transpose(0, 1), logits).double().cpu()
                    norm_acc[ffn_name] += torch.sum(logits * logits, dim=0).double().cpu()
        finally:
            for handle in handles:
                handle.remove()

        for ffn_name in self._usage_frequency_state_dict:
            usage = self._usage_frequency_state_dict[ffn_name]
            denom = torch.sum(usage)
            if denom > 0:
                self._usage_frequency_state_dict[ffn_name] = usage / denom

            denom_matrix = torch.sqrt(norm_acc[ffn_name][:, None] * norm_acc[ffn_name][None, :]).clamp_min(FP32_EPS)
            cosine = (gram_acc[ffn_name] / denom_matrix).clamp(min=-1.0, max=1.0)
            self._similarity_state_dict[ffn_name] = ((cosine + 1.0) / 2.0).float()
            self._similarity_state_dict[ffn_name].fill_diagonal_(1.0)

    def group_experts_per_layer(self, physical_num_experts: int, merging_layers: List[int]) -> Dict[str, List[int]]:
        if physical_num_experts <= 0 or physical_num_experts > self.logical_num_experts:
            raise ValueError(
                f"[MC-SMoE] physical_num_experts must be in [1, {self.logical_num_experts}], "
                f"got {physical_num_experts}."
            )

        for layer_idx in tqdm(
            self.sparse_layer_indices,
            desc=f"[MC-SMoE] Grouping Gemma4 experts into exactly {physical_num_experts} physical experts/layer",
        ):
            ffn_name = f"model.language_model.layers.{layer_idx}.experts"
            if layer_idx not in merging_layers:
                self._group_state_dict[ffn_name] = torch.arange(self.logical_num_experts, device="cpu")
                self._dominant_experts[ffn_name] = list(range(self.logical_num_experts))
                continue

            usage = self._usage_frequency_state_dict[ffn_name]
            indices_sorted_by_usage = torch.argsort(usage, descending=True)
            dominant = indices_sorted_by_usage[:physical_num_experts]
            self._dominant_experts[ffn_name] = dominant.tolist()

            for physical_idx, original_idx in enumerate(dominant.tolist()):
                self._group_state_dict[ffn_name][original_idx] = physical_idx

            similarity = self._similarity_state_dict[ffn_name]
            for original_idx in indices_sorted_by_usage[physical_num_experts:].tolist():
                nearest_dominant_offset = torch.argmax(similarity[original_idx, dominant]).item()
                self._group_state_dict[ffn_name][original_idx] = nearest_dominant_offset

        return self.dominant_experts()

    def save_group_state_dict(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self._group_state_dict, os.path.join(save_dir, "group_state_dict.pt"))


@torch.no_grad()
def _merge_expert_tensor(
    target_tensor: torch.Tensor,
    source_tensor: torch.Tensor,
    expert_indices: List[int],
    usage_frequencies: torch.Tensor,
):
    denom = sum(float(usage_frequencies[idx]) for idx in expert_indices)
    if denom <= 0:
        weights = {idx: 1.0 / float(len(expert_indices)) for idx in expert_indices}
    else:
        weights = {idx: float(usage_frequencies[idx]) / denom for idx in expert_indices}

    accumulator = torch.zeros_like(target_tensor, dtype=torch.float32, device=target_tensor.device)
    for expert_idx in expert_indices:
        accumulator.add_(source_tensor[expert_idx].float().to(target_tensor.device), alpha=weights[expert_idx])
    target_tensor.copy_(accumulator.to(dtype=target_tensor.dtype))


def apply_compressed_config_to_model(model, compressed_config: Gemma4MoeCompressedConfig):
    model.config = compressed_config
    model.model.config = compressed_config
    model.model.language_model.config = compressed_config.text_config
    for layer in _language_layers(model):
        layer.config = compressed_config.text_config
        if hasattr(layer, "router"):
            layer.router.config = compressed_config.text_config


@torch.no_grad()
def compress_gemma4_moe_in_place(
    model,
    grouper: ExpertsGrouperForCompressedGemma4,
    physical_num_experts: int,
    merging_layers: Optional[List[int]] = None,
):
    if merging_layers is None:
        merging_layers = list(grouper.sparse_layer_indices)

    usage_frequency_dict = grouper.usage_frequency_state_dict()
    group_labels_dict = grouper.group_state_dict()
    layers = _language_layers(model)

    for layer_idx in tqdm(
        grouper.sparse_layer_indices,
        desc="[MC-SMoE] Replacing Gemma4 routed expert tensors with compressed physical experts",
    ):
        if layer_idx not in merging_layers:
            continue

        ffn_name = f"model.language_model.layers.{layer_idx}.experts"
        layer = layers[layer_idx]
        old_experts = layer.experts
        if not hasattr(old_experts, "gate_up_proj") or not hasattr(old_experts, "down_proj"):
            raise TypeError(f"[MC-SMoE] Layer {layer_idx} experts do not look like Gemma4TextExperts.")

        compressed_experts = Gemma4CompressedTextExperts(model.config.text_config, layer_idx=layer_idx)
        compressed_experts.to(device=old_experts.gate_up_proj.device, dtype=old_experts.gate_up_proj.dtype)

        group_labels = group_labels_dict[ffn_name]
        usage_frequencies = usage_frequency_dict[ffn_name]
        for physical_idx in range(physical_num_experts):
            expert_indices = torch.where(group_labels == physical_idx)[0].tolist()
            if len(expert_indices) == 0:
                raise ValueError(f"[MC-SMoE] Empty physical expert {physical_idx} in {ffn_name}.")
            _merge_expert_tensor(
                target_tensor=compressed_experts.gate_up_proj[physical_idx],
                source_tensor=old_experts.gate_up_proj,
                expert_indices=expert_indices,
                usage_frequencies=usage_frequencies,
            )
            _merge_expert_tensor(
                target_tensor=compressed_experts.down_proj[physical_idx],
                source_tensor=old_experts.down_proj,
                expert_indices=expert_indices,
                usage_frequencies=usage_frequencies,
            )

        layer.experts = compressed_experts
        del old_experts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return model


def build_mapping_artifact(
    grouper: ExpertsGrouperForCompressedGemma4,
    physical_num_experts: int,
    source_model_name_or_path: str,
    similarity_base: str,
    calibration_dataset: str,
    subset_ratio: float,
    block_size: int,
    batch_size: int,
) -> OrderedDict:
    group_labels_dict = grouper.group_state_dict()
    usage_frequency_dict = grouper.usage_frequency_state_dict()
    dominant_experts = grouper.dominant_experts()
    layers = OrderedDict()
    config_mappings = OrderedDict()

    for layer_idx in grouper.sparse_layer_indices:
        ffn_name = f"model.language_model.layers.{layer_idx}.experts"
        mapping = group_labels_dict[ffn_name].tolist()
        config_mappings[str(layer_idx)] = mapping
        physical_to_original = OrderedDict()
        for physical_idx in range(physical_num_experts):
            physical_to_original[str(physical_idx)] = [
                original_idx for original_idx, mapped_idx in enumerate(mapping) if mapped_idx == physical_idx
            ]

        layers[ffn_name] = OrderedDict(
            layer_idx=layer_idx,
            original_to_physical=mapping,
            physical_to_original=physical_to_original,
            dominant_original_experts=dominant_experts[ffn_name],
            usage_frequency=[float(x) for x in usage_frequency_dict[ffn_name].tolist()],
        )

    return OrderedDict(
        format_version=1,
        method="M-SMoE expert merge only",
        source_model_name_or_path=source_model_name_or_path,
        logical_num_experts=grouper.logical_num_experts,
        physical_num_experts=physical_num_experts,
        similarity_base=similarity_base,
        merge_weighting="usage_frequency",
        calibration_dataset=calibration_dataset,
        subset_ratio=subset_ratio,
        block_size=block_size,
        batch_size=batch_size,
        config_expert_id_mappings=config_mappings,
        layers=layers,
    )


def write_mapping_artifacts(output_dir: str, mapping_artifact: OrderedDict, grouper: ExpertsGrouperForCompressedGemma4):
    os.makedirs(output_dir, exist_ok=True)
    mapping_json_path = os.path.join(output_dir, "expert_mapping.json")
    with open(mapping_json_path, "w", encoding="utf-8") as f:
        json.dump(mapping_artifact, f, indent=2)

    torch.save(
        {
            key: torch.tensor(value, dtype=torch.long)
            for key, value in mapping_artifact["config_expert_id_mappings"].items()
        },
        os.path.join(output_dir, "expert_id_mappings.pt"),
    )
    grouper.save_group_state_dict(output_dir)


def copy_remote_code_files(output_dir: str):
    repo_root = Path(__file__).resolve().parents[2]
    for filename in ("configuration_gemma4_moe_compressed.py", "modeling_gemma4_moe_compressed.py"):
        shutil.copyfile(repo_root / filename, Path(output_dir) / filename)


def patch_saved_config_json(
    output_dir: str,
    compressed_config: Gemma4MoeCompressedConfig,
    mapping_artifact: OrderedDict,
):
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config_json = json.load(f)

    config_json.update(
        {
            "architectures": ["Gemma4MoeCompressedForConditionalGeneration"],
            "auto_map": {
                "AutoConfig": "configuration_gemma4_moe_compressed.Gemma4MoeCompressedConfig",
                "AutoModel": "modeling_gemma4_moe_compressed.Gemma4MoeCompressedModel",
                "AutoModelForCausalLM": "modeling_gemma4_moe_compressed.Gemma4MoeCompressedForConditionalGeneration",
                "AutoModelForImageTextToText": "modeling_gemma4_moe_compressed.Gemma4MoeCompressedForConditionalGeneration",
            },
            "model_type": Gemma4MoeCompressedConfig.model_type,
            "logical_num_experts": compressed_config.logical_num_experts,
            "physical_num_experts": compressed_config.physical_num_experts,
            "expert_id_mappings": mapping_artifact["config_expert_id_mappings"],
            "expert_id_mapping_path": "expert_mapping.json",
            "compression_ratio": compressed_config.compression_ratio,
            "source_model_name_or_path": compressed_config.source_model_name_or_path,
            "mcsmoe_metadata": compressed_config.mcsmoe_metadata,
        }
    )
    text_config = config_json.setdefault("text_config", {})
    text_config.update(
        {
            "num_experts": compressed_config.logical_num_experts,
            "logical_num_experts": compressed_config.logical_num_experts,
            "physical_num_experts": compressed_config.physical_num_experts,
            "expert_id_mappings": mapping_artifact["config_expert_id_mappings"],
            "expert_id_mapping_path": "expert_mapping.json",
            "compression_ratio": compressed_config.compression_ratio,
        }
    )

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_json, f, indent=2)
