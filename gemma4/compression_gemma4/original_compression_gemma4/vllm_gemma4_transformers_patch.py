"""Runtime patches for Gemma4 compressed models under vLLM Transformers backend."""

from __future__ import annotations


def _is_gemma4_mixed_attention_config(config, text_config) -> bool:
    model_types = {
        str(getattr(config, "model_type", "")),
        str(getattr(text_config, "model_type", "")),
    }
    return (
        any("gemma4" in model_type for model_type in model_types)
        and hasattr(text_config, "layer_types")
        and hasattr(text_config, "global_head_dim")
        and hasattr(text_config, "head_dim")
    )


def apply() -> None:
    from vllm.model_executor.models.transformers import base as tf_base

    if getattr(tf_base.Base, "_origcomp_gemma4_mixed_attention_patch", False):
        return

    original_create_attention_instances = tf_base.Base.create_attention_instances

    def create_attention_instances(self):
        text_config = self.text_config
        if not _is_gemma4_mixed_attention_config(self.config, text_config):
            return original_create_attention_instances(self)

        num_heads = self.model_config.get_num_attention_heads(self.parallel_config)
        logits_soft_cap = getattr(text_config, "attn_logit_softcapping", None)

        is_encoder = lambda module: not getattr(module, "is_causal", True)
        has_encoder = lambda model: any(is_encoder(m) for m in model.modules())
        is_multimodal = lambda config: config != config.get_text_config()
        if has_encoder(self.model) and not is_multimodal(self.config):
            self.check_version("5.0.0", "encoder models support")
            attn_type = tf_base.AttentionType.ENCODER_ONLY
        else:
            attn_type = tf_base.AttentionType.DECODER

        pp_rank = self.pp_group.rank_in_group
        pp_size = self.pp_group.world_size
        start, end = tf_base.get_pp_indices(text_config.num_hidden_layers, pp_rank, pp_size)

        base_num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        tp_size = max(1, int(getattr(self.parallel_config, "tensor_parallel_size", 1)))
        attention_instances = {}
        for layer_idx in range(start, end):
            layer_type = text_config.layer_types[layer_idx]
            is_sliding = layer_type == "sliding_attention"
            layer_head_size = int(text_config.head_dim)
            layer_num_kv_heads = int(base_num_kv_heads)
            per_layer_sliding_window = None

            if is_sliding:
                per_layer_sliding_window = getattr(text_config, "sliding_window", None)
            else:
                global_head_dim = getattr(text_config, "global_head_dim", None)
                if global_head_dim:
                    layer_head_size = int(global_head_dim)
                if getattr(text_config, "attention_k_eq_v", False):
                    total_global_kv = int(getattr(text_config, "num_global_key_value_heads", layer_num_kv_heads))
                    layer_num_kv_heads = max(1, total_global_kv // tp_size)

            attn_cls = (
                tf_base.EncoderOnlyAttention
                if attn_type == tf_base.AttentionType.ENCODER_ONLY
                else tf_base.Attention
            )
            attention_instances[layer_idx] = attn_cls(
                num_heads=num_heads,
                head_size=layer_head_size,
                scale=layer_head_size**-0.5,
                num_kv_heads=layer_num_kv_heads,
                cache_config=self.cache_config,
                quant_config=self.quant_config,
                logits_soft_cap=logits_soft_cap,
                per_layer_sliding_window=per_layer_sliding_window,
                prefix=f"{layer_idx}.attn",
                attn_type=attn_type,
            )
        return attention_instances

    tf_base.Base.create_attention_instances = create_attention_instances
    tf_base.Base._origcomp_gemma4_mixed_attention_patch = True


__all__ = ["apply"]
