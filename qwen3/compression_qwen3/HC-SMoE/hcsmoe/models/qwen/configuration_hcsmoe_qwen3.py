try:
    from transformers import Qwen3MoeConfig
except ImportError:
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig


class HCQwen3MoeConfig(Qwen3MoeConfig):
    model_type = "hcsmoe_qwen3_moe"

    def __init__(
        self,
        logical_num_experts=None,
        physical_num_experts=None,
        expert_id_mapping=None,
        hcsmoe_expert_mapping_file=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.logical_num_experts = int(logical_num_experts or self.num_experts)
        self.physical_num_experts = int(physical_num_experts or self.logical_num_experts)
        self.expert_id_mapping = expert_id_mapping
        self.hcsmoe_expert_mapping_file = hcsmoe_expert_mapping_file
        self.num_experts = self.logical_num_experts
