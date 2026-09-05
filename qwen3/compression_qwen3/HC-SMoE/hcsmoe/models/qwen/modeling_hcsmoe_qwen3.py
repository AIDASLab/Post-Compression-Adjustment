import torch
import torch.nn.functional as F
from torch import nn

from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeMLP,
    Qwen3MoeModel,
    Qwen3MoePreTrainedModel,
    Qwen3MoeRMSNorm,
    Qwen3MoeRotaryEmbedding,
)

from .configuration_hcsmoe_qwen3 import HCQwen3MoeConfig


def _mapping_for_layer(config: HCQwen3MoeConfig, layer_idx: int):
    mapping = getattr(config, "expert_id_mapping", None) or {}
    return mapping.get(str(layer_idx)) or mapping.get(f"model.layers.{layer_idx}.mlp")


class HCQwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config: HCQwen3MoeConfig, layer_idx: int):
        super().__init__()
        self.logical_num_experts = int(getattr(config, "logical_num_experts", config.num_experts))
        self.physical_num_experts = int(getattr(config, "physical_num_experts", self.logical_num_experts))
        self.num_experts = self.logical_num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        self.gate = nn.Linear(config.hidden_size, self.logical_num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                Qwen3MoeMLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(self.physical_num_experts)
            ]
        )

        mapping = _mapping_for_layer(config, layer_idx)
        if mapping is None:
            if self.logical_num_experts != self.physical_num_experts:
                raise ValueError(
                    f"Missing expert_id_mapping for layer {layer_idx}: "
                    f"logical={self.logical_num_experts}, physical={self.physical_num_experts}"
                )
            mapping = list(range(self.logical_num_experts))

        if len(mapping) != self.logical_num_experts:
            raise ValueError(
                f"Layer {layer_idx} mapping length is {len(mapping)}, "
                f"expected {self.logical_num_experts}."
            )
        mapping_tensor = torch.tensor(mapping, dtype=torch.long)
        if mapping_tensor.min().item() < 0 or mapping_tensor.max().item() >= self.physical_num_experts:
            raise ValueError(
                f"Layer {layer_idx} mapping contains physical ids outside "
                f"[0, {self.physical_num_experts})."
            )
        self.register_buffer("logical_to_physical", mapping_tensor, persistent=False)

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        selected_physical = self.logical_to_physical.to(selected_experts.device)[selected_experts]
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
            token_idx = torch.nonzero(token_physical_weights[:, physical_idx] > 0, as_tuple=False).flatten()
            current_state = hidden_states[token_idx]
            current_hidden_states = self.experts[physical_idx](current_state)
            current_hidden_states = current_hidden_states * token_physical_weights[token_idx, physical_idx, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


class HCQwen3MoeDecoderLayer(Qwen3MoeDecoderLayer):
    def __init__(self, config: HCQwen3MoeConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3MoeAttention(config, layer_idx)
        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = HCQwen3MoeSparseMoeBlock(config, layer_idx)
        else:
            self.mlp = Qwen3MoeMLP(config, intermediate_size=config.intermediate_size)
        self.input_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class HCQwen3MoeModel(Qwen3MoeModel):
    config_class = HCQwen3MoeConfig
    _no_split_modules = ["HCQwen3MoeDecoderLayer", "Qwen3MoeDecoderLayer"]

    def __init__(self, config: HCQwen3MoeConfig):
        Qwen3MoePreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [HCQwen3MoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3MoeRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()


class HCQwen3MoeForCausalLM(Qwen3MoeForCausalLM):
    config_class = HCQwen3MoeConfig
    _no_split_modules = ["HCQwen3MoeDecoderLayer", "Qwen3MoeDecoderLayer"]

    def __init__(self, config: HCQwen3MoeConfig):
        Qwen3MoePreTrainedModel.__init__(self, config)
        self.model = HCQwen3MoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.logical_num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.post_init()
