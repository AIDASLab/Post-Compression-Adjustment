from fire import Fire


def _inspect_backend_model(model, expected_physical_num_experts: int, expected_logical_num_experts: int):
    candidate = model
    if hasattr(candidate, "model") and hasattr(candidate.model, "language_model"):
        candidate = candidate.model
    language_model = getattr(candidate, "language_model", None)
    if language_model is None and hasattr(candidate, "model"):
        language_model = getattr(candidate.model, "language_model", None)
    if language_model is None:
        raise ValueError(f"Could not locate Gemma4 language_model in vLLM backend model: {type(model)}")

    sparse_layers = []
    for layer_idx, layer in enumerate(language_model.layers):
        if not (getattr(layer, "enable_moe_block", False) and hasattr(layer, "router") and hasattr(layer, "experts")):
            continue
        sparse_layers.append(layer_idx)
        if layer.router.proj.out_features != expected_logical_num_experts:
            raise ValueError(f"Layer {layer_idx} router out_features mismatch: {layer.router.proj.out_features}")
        if layer.experts.gate_up_proj.shape[0] != expected_physical_num_experts:
            raise ValueError(f"Layer {layer_idx} physical expert count mismatch: {layer.experts.gate_up_proj.shape[0]}")
        mapping = layer.experts.expert_id_mapping.detach().cpu()
        if mapping.numel() != expected_logical_num_experts:
            raise ValueError(f"Layer {layer_idx} mapping length mismatch: {mapping.numel()}")
        if mapping.min().item() < 0 or mapping.max().item() >= expected_physical_num_experts:
            raise ValueError(f"Layer {layer_idx} mapping contains out-of-range ids.")
    if not sparse_layers:
        raise ValueError("No Gemma4 compressed sparse layers found in vLLM backend model.")
    print(f"[VLLM CHECK] Verified {len(sparse_layers)} sparse layers.")


def check_vllm(
    model_dir: str,
    expected_physical_num_experts: int,
    expected_logical_num_experts: int = 128,
    prompt: str = "Explain mixture-of-experts models in one sentence.",
    max_tokens: int = 32,
    dtype: str = "bfloat16",
    gpu_memory_utilization: float = 0.90,
):
    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise ImportError(
            "vLLM check requires vllm and a Gemma4-capable Transformers build. "
            "Run the standard HF check if vLLM is not installed."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )

    llm = LLM(
        model=model_dir,
        task="generate",
        trust_remote_code=True,
        model_impl="transformers",
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    llm.apply_model(lambda backend_model: _inspect_backend_model(backend_model, expected_physical_num_experts, expected_logical_num_experts))
    outputs = llm.generate([prompt_text], SamplingParams(max_tokens=max_tokens, temperature=0.0))
    print("[VLLM CHECK] Generated text:")
    print(outputs[0].outputs[0].text)
    print("[VLLM CHECK] OK")


if __name__ == "__main__":
    Fire(check_vllm)
