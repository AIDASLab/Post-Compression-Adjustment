from typing import Dict

import torch
from fire import Fire
from transformers import AutoConfig, AutoTokenizer


def count_parameters(model, only_trainable: bool = False) -> int:
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def _load_model(model_dir: str, torch_dtype: str, device_map: str):
    try:
        from transformers import AutoModelForImageTextToText
    except ImportError as exc:
        raise ImportError(
            "Gemma4 compressed checks require a Transformers version with AutoModelForImageTextToText "
            "and Gemma4 support."
        ) from exc

    return AutoModelForImageTextToText.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )


def _prepare_chat_inputs(tokenizer, prompt: str) -> Dict[str, torch.Tensor]:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
    except TypeError as exc:
        raise TypeError(
            "Gemma4 tokenizer apply_chat_template must accept enable_thinking=False "
            "for this experiment."
        ) from exc


def check_model(
    model_dir: str,
    expected_physical_num_experts: int,
    expected_logical_num_experts: int = 128,
    prompt: str = "Explain mixture-of-experts models in one sentence.",
    max_new_tokens: int = 32,
    torch_dtype: str = "auto",
    device_map: str = "auto",
    run_generate: bool = True,
):
    print("[CHECK] Loading config:", model_dir)
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    text_config = getattr(config, "text_config", config)
    logical = int(getattr(text_config, "logical_num_experts", getattr(text_config, "num_experts")))
    physical = int(getattr(text_config, "physical_num_experts", getattr(text_config, "num_experts")))
    print(f"[CHECK] logical_num_experts={logical}, physical_num_experts={physical}")

    if logical != expected_logical_num_experts:
        raise ValueError(f"Expected logical experts {expected_logical_num_experts}, got {logical}.")
    if physical != expected_physical_num_experts:
        raise ValueError(f"Expected physical experts {expected_physical_num_experts}, got {physical}.")
    if int(getattr(text_config, "num_experts")) != expected_logical_num_experts:
        raise ValueError("text_config.num_experts must remain equal to logical_num_experts for router output.")

    print("[CHECK] Loading tokenizer/model with trust_remote_code=True")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = _load_model(model_dir=model_dir, torch_dtype=torch_dtype, device_map=device_map)
    model.eval()

    sparse_layers = []
    for layer_idx, layer in enumerate(model.model.language_model.layers):
        if not (getattr(layer, "enable_moe_block", False) and hasattr(layer, "router") and hasattr(layer, "experts")):
            continue
        sparse_layers.append(layer_idx)

        if layer.router.proj.out_features != expected_logical_num_experts:
            raise ValueError(
                f"Layer {layer_idx} router out_features={layer.router.proj.out_features}, "
                f"expected {expected_logical_num_experts}."
            )
        if layer.router.per_expert_scale.numel() != expected_logical_num_experts:
            raise ValueError(f"Layer {layer_idx} router per_expert_scale is not logical-sized.")

        experts = layer.experts
        if experts.gate_up_proj.shape[0] != expected_physical_num_experts:
            raise ValueError(
                f"Layer {layer_idx} has {experts.gate_up_proj.shape[0]} physical gate_up experts, "
                f"expected {expected_physical_num_experts}."
            )
        if experts.down_proj.shape[0] != expected_physical_num_experts:
            raise ValueError(
                f"Layer {layer_idx} has {experts.down_proj.shape[0]} physical down experts, "
                f"expected {expected_physical_num_experts}."
            )
        mapping = experts.expert_id_mapping.detach().cpu()
        if mapping.numel() != expected_logical_num_experts:
            raise ValueError(f"Layer {layer_idx} mapping length mismatch: {mapping.numel()}.")
        if mapping.min().item() < 0 or mapping.max().item() >= expected_physical_num_experts:
            raise ValueError(f"Layer {layer_idx} mapping contains out-of-range physical expert ids.")

        if not all(hasattr(layer.mlp, name) for name in ("gate_proj", "up_proj", "down_proj")):
            raise ValueError(f"Layer {layer_idx} shared Gemma4 mlp projections are missing.")

    if len(sparse_layers) == 0:
        raise ValueError("No compressed Gemma4 MoE layers were found.")
    print(f"[CHECK] Verified {len(sparse_layers)} sparse layers.")

    total_params = count_parameters(model)
    trainable_params = count_parameters(model, only_trainable=True)
    print("[CHECK] Total params:", f"{total_params:,}", f"({total_params / 1_000_000:.2f}M)")
    print("[CHECK] Trainable params:", f"{trainable_params:,}", f"({trainable_params / 1_000_000:.2f}M)")

    inputs = _prepare_chat_inputs(tokenizer, prompt)
    input_device = model.get_input_embeddings().weight.device
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

    captured = {}
    handle = model.model.language_model.layers[sparse_layers[0]].router.proj.register_forward_hook(
        lambda _module, _inputs, output: captured.setdefault("router_logits", output.detach())
    )
    try:
        with torch.no_grad():
            model(**inputs, use_cache=False)
    finally:
        handle.remove()

    if "router_logits" not in captured:
        raise ValueError("Did not capture Gemma4 router logits from the first sparse layer.")
    first_router_shape = tuple(captured["router_logits"].shape)
    print("[CHECK] First router logits shape:", first_router_shape)
    if first_router_shape[-1] != expected_logical_num_experts:
        raise ValueError(f"Router logits last dim should be {expected_logical_num_experts}.")

    if run_generate and max_new_tokens > 0:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generated_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
        print("[CHECK] Generated continuation:")
        if hasattr(tokenizer, "batch_decode"):
            print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
        else:
            print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))

    print("[CHECK] OK")


if __name__ == "__main__":
    Fire(check_model)
