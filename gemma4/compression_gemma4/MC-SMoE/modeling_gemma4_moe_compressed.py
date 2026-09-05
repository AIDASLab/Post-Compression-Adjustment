from typing import Optional

import torch
from torch import nn

try:
    from transformers import AutoModel
    from transformers.activations import ACT2FN
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4ForConditionalGeneration,
        Gemma4Model,
        Gemma4MultimodalEmbedder,
        Gemma4PreTrainedModel,
        Gemma4RMSNorm,
        Gemma4TextAttention,
        Gemma4TextDecoderLayer,
        Gemma4TextExperts,
        Gemma4TextMLP,
        Gemma4TextModel,
        Gemma4TextRotaryEmbedding,
        Gemma4TextRouter,
        Gemma4TextScaledWordEmbedding,
    )
except ImportError as exc:  # pragma: no cover - depends on the runtime transformers version.
    raise ImportError(
        "Gemma4 compressed remote code requires a Transformers build with "
        "transformers.models.gemma4.modeling_gemma4. Install a Gemma4-capable "
        "Transformers release and reload with trust_remote_code=True."
    ) from exc

try:
    from .configuration_gemma4_moe_compressed import Gemma4MoeCompressedConfig
except ImportError:
    from configuration_gemma4_moe_compressed import Gemma4MoeCompressedConfig


def _lookup_layer_mapping(config, layer_idx: Optional[int]) -> torch.LongTensor:
    logical_num_experts = int(getattr(config, "logical_num_experts", config.num_experts))
    physical_num_experts = int(getattr(config, "physical_num_experts", logical_num_experts))
    mappings = getattr(config, "expert_id_mappings", None) or {}

    mapping = None
    if layer_idx is not None:
        for key in (str(layer_idx), f"model.language_model.layers.{layer_idx}.experts", f"layers.{layer_idx}.experts"):
            if key in mappings:
                mapping = mappings[key]
                break
    if mapping is None and "default" in mappings:
        mapping = mappings["default"]

    if mapping is None:
        if physical_num_experts != logical_num_experts:
            raise ValueError(
                "Compressed Gemma4 MoE requires expert_id_mappings when "
                "physical_num_experts != logical_num_experts."
            )
        mapping = list(range(logical_num_experts))

    mapping_list = [int(value) for value in mapping]
    if len(mapping_list) != logical_num_experts:
        raise ValueError(
            f"Expert mapping for layer {layer_idx} has {len(mapping_list)} entries, "
            f"expected {logical_num_experts}."
        )
    if min(mapping_list) < 0 or max(mapping_list) >= physical_num_experts:
        raise ValueError(
            f"Expert mapping for layer {layer_idx} contains physical ids outside "
            f"[0, {physical_num_experts})."
        )
    return torch.tensor(mapping_list, dtype=torch.long)


class Gemma4CompressedTextExperts(Gemma4TextExperts):
    """Gemma4 routed experts with logical-to-physical dispatch mapping."""

    def __init__(self, config, layer_idx: Optional[int] = None):
        nn.Module.__init__(self)
        self.logical_num_experts = int(getattr(config, "logical_num_experts", config.num_experts))
        self.physical_num_experts = int(getattr(config, "physical_num_experts", self.logical_num_experts))
        self.num_experts = self.physical_num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.physical_num_experts, 2 * self.intermediate_dim, self.hidden_dim)
        )
        self.down_proj = nn.Parameter(torch.empty(self.physical_num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_activation]
        self.layer_idx = layer_idx
        self.register_buffer("expert_id_mapping", _lookup_layer_mapping(config, layer_idx), persistent=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        mapping = self.expert_id_mapping.to(top_k_index.device)
        physical_index = mapping[top_k_index]

        with torch.no_grad():
            physical_mask = torch.nn.functional.one_hot(physical_index, num_classes=self.physical_num_experts).to(
                top_k_weights.dtype
            )
            weights_by_physical = torch.sum(physical_mask * top_k_weights.unsqueeze(-1), dim=1)
            expert_hit = torch.greater(weights_by_physical.sum(dim=0), 0).nonzero(as_tuple=False).flatten()

        for expert_idx in expert_hit.tolist():
            token_idx = torch.where(weights_by_physical[:, expert_idx] != 0)[0]
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * weights_by_physical[token_idx, expert_idx, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class Gemma4CompressedTextDecoderLayer(Gemma4TextDecoderLayer):
    def __init__(self, config, layer_idx: int):
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = Gemma4TextAttention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma4TextMLP(config, layer_idx)
        self.input_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.register_buffer("layer_scalar", torch.ones(1))

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.act_fn = ACT2FN[config.hidden_activation]
            self.per_layer_input_gate = nn.Linear(self.hidden_size, self.hidden_size_per_layer_input, bias=False)
            self.per_layer_projection = nn.Linear(self.hidden_size_per_layer_input, self.hidden_size, bias=False)
            self.post_per_layer_input_norm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        self.enable_moe_block = config.enable_moe_block
        if self.enable_moe_block:
            self.router = Gemma4TextRouter(config)
            self.experts = Gemma4CompressedTextExperts(config, layer_idx=layer_idx)
            self.post_feedforward_layernorm_1 = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
            self.post_feedforward_layernorm_2 = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
            self.pre_feedforward_layernorm_2 = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)


class Gemma4MoeCompressedTextModel(Gemma4TextModel):
    config_class = Gemma4MoeCompressedConfig
    _no_split_modules = ["Gemma4CompressedTextDecoderLayer", "Gemma4TextDecoderLayer"]

    def __init__(self, config):
        Gemma4PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = Gemma4TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            embed_scale=self.config.hidden_size**0.5,
        )
        self.layers = nn.ModuleList(
            [Gemma4CompressedTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma4TextRotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.unique_layer_types = set(self.config.layer_types)

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.embed_tokens_per_layer = Gemma4TextScaledWordEmbedding(
                config.vocab_size_per_layer_input,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
                self.padding_idx,
                embed_scale=config.hidden_size_per_layer_input**0.5,
            )
            self.per_layer_input_scale = 2.0**-0.5
            self.per_layer_model_projection = nn.Linear(
                config.hidden_size,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
                bias=False,
            )
            self.per_layer_model_projection_scale = config.hidden_size**-0.5
            self.per_layer_projection_norm = Gemma4RMSNorm(config.hidden_size_per_layer_input, eps=config.rms_norm_eps)

        self._keys_to_ignore_on_load_unexpected = []
        for i, layer in enumerate(self.layers):
            if layer.self_attn.is_kv_shared_layer:
                self._keys_to_ignore_on_load_unexpected.extend(
                    [f"layers.{i}.self_attn.{name}" for name in ("k_proj", "v_proj", "k_norm", "v_norm")]
                )

        self.post_init()


class Gemma4MoeCompressedModel(Gemma4Model):
    config_class = Gemma4MoeCompressedConfig
    _no_split_modules = ["Gemma4CompressedTextDecoderLayer", "Gemma4TextDecoderLayer"]

    def __init__(self, config: Gemma4MoeCompressedConfig):
        Gemma4PreTrainedModel.__init__(self, config)
        self.vision_tower = AutoModel.from_config(config.vision_config) if config.vision_config is not None else None
        self.vocab_size = config.text_config.vocab_size
        self.language_model = Gemma4MoeCompressedTextModel(config.text_config)
        self.vocab_size_per_layer_input = config.text_config.vocab_size_per_layer_input
        self.audio_tower = AutoModel.from_config(config.audio_config) if config.audio_config is not None else None
        self.embed_vision = (
            Gemma4MultimodalEmbedder(config.vision_config, config.text_config)
            if config.vision_config is not None
            else None
        )
        self.embed_audio = (
            Gemma4MultimodalEmbedder(config.audio_config, config.text_config)
            if config.audio_config is not None
            else None
        )
        self.post_init()


class Gemma4MoeCompressedForConditionalGeneration(Gemma4ForConditionalGeneration):
    config_class = Gemma4MoeCompressedConfig
    _no_split_modules = ["Gemma4CompressedTextDecoderLayer", "Gemma4TextDecoderLayer"]

    def __init__(self, config: Gemma4MoeCompressedConfig):
        Gemma4PreTrainedModel.__init__(self, config)
        self.model = Gemma4MoeCompressedModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()


__all__ = [
    "Gemma4MoeCompressedConfig",
    "Gemma4MoeCompressedForConditionalGeneration",
    "Gemma4MoeCompressedModel",
    "Gemma4MoeCompressedTextModel",
    "Gemma4CompressedTextDecoderLayer",
    "Gemma4CompressedTextExperts",
]
