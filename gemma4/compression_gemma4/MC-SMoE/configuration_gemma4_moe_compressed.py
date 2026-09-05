try:
    from transformers import Gemma4Config
except ImportError as exc:  # pragma: no cover - depends on the runtime transformers version.
    raise ImportError(
        "Gemma4MoeCompressedConfig requires a Transformers build that provides Gemma4Config. "
        "Install a Gemma4-capable Transformers release before loading this compressed model."
    ) from exc


class Gemma4MoeCompressedConfig(Gemma4Config):
    """Gemma4 config with separate logical router IDs and physical experts.

    `text_config.num_experts` stays equal to the logical router output dimension
    so the router keeps scoring the original expert-id space. `physical_num_experts`
    controls the number of routed expert tensors stored in each layer.
    """

    model_type = "gemma4_moe_compressed"
    has_no_defaults_at_init = True

    def __init__(
        self,
        logical_num_experts=None,
        physical_num_experts=None,
        expert_id_mappings=None,
        expert_id_mapping_path=None,
        compression_ratio=None,
        source_model_name_or_path=None,
        mcsmoe_metadata=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        text_config = self.text_config
        if logical_num_experts is None:
            logical_num_experts = getattr(text_config, "num_experts", None)
        if logical_num_experts is None:
            raise ValueError("Gemma4 MoE compression requires text_config.num_experts.")
        if physical_num_experts is None:
            physical_num_experts = logical_num_experts

        self.logical_num_experts = int(logical_num_experts)
        self.physical_num_experts = int(physical_num_experts)
        self.expert_id_mappings = expert_id_mappings or {}
        self.expert_id_mapping_path = expert_id_mapping_path
        self.compression_ratio = compression_ratio
        self.source_model_name_or_path = source_model_name_or_path
        self.mcsmoe_metadata = mcsmoe_metadata or {}

        text_config.num_experts = self.logical_num_experts
        text_config.logical_num_experts = self.logical_num_experts
        text_config.physical_num_experts = self.physical_num_experts
        text_config.expert_id_mappings = self.expert_id_mappings
        text_config.expert_id_mapping_path = self.expert_id_mapping_path
        text_config.compression_ratio = self.compression_ratio


__all__ = ["Gemma4MoeCompressedConfig"]
