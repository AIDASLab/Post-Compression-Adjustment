import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
PRUNED_MODEL_DIR = (
    ROOT_DIR
    / "artifacts"
    / "Qwen3-30B-A3B-Instruct-2507"
    / "c4"
    / "pruned_models"
    / "reap-renorm_true-seed_42-0.375"
)
EXPECTED_ORIGINAL_EXPERTS = 128
EXPECTED_RETAINED_EXPERTS = 80
EXPECTED_PRUNED_EXPERTS = 48
PROMPT = "Tell me briefly what mixture-of-experts language models are."


def configure_runtime() -> None:
    os.chdir(ROOT_DIR)
    cache_root = ROOT_DIR / ".cache"
    for key in (
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE",
        "XDG_CACHE_HOME",
    ):
        os.environ.pop(key, None)
    defaults = {
        "TORCH_HOME": cache_root / "torch",
        "TORCH_EXTENSIONS_DIR": cache_root / "torch_extensions",
        "TRITON_CACHE_DIR": cache_root / "triton",
        "VLLM_CACHE_ROOT": cache_root / "vllm",
        "CUDA_CACHE_PATH": cache_root / "cuda",
        "PYTHONPYCACHEPREFIX": cache_root / "pycache",
        "TMPDIR": ROOT_DIR / "tmp",
        "TOKENIZERS_PARALLELISM": "false",
    }
    for key, value in defaults.items():
        os.environ[key] = str(value)
        if key != "TOKENIZERS_PARALLELISM":
            Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


configure_runtime()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def fail(message: str) -> None:
    print(f"[FAIL] {message}", file=sys.stderr)
    raise SystemExit(1)


def count_parameters(model, only_trainable: bool = False) -> int:
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_input_device(model) -> torch.device:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for key in (
            "model.embed_tokens",
            "transformer.wte",
            "embed_tokens",
            "model.tok_embeddings",
        ):
            if key in device_map:
                dev = device_map[key]
                if isinstance(dev, int):
                    return torch.device(f"cuda:{dev}")
                if isinstance(dev, str) and dev.startswith("cuda"):
                    return torch.device(dev)
                if dev == "cpu":
                    return torch.device("cpu")
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cpu")


def load_json_map(path: Path) -> dict[int, list[int]]:
    if not path.is_file():
        fail(f"Missing required index map: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): [int(x) for x in v] for k, v in raw.items()}


def load_pruned_model():
    kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
        "local_files_only": True,
    }
    try:
        return AutoModelForCausalLM.from_pretrained(
            PRUNED_MODEL_DIR,
            dtype=torch.bfloat16,
            **kwargs,
        )
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            PRUNED_MODEL_DIR,
            torch_dtype=torch.bfloat16,
            **kwargs,
        )


def collect_moe_layers(model):
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        fail("Could not find model.model.layers")

    moe_layers = []
    for layer_idx, layer in enumerate(layers):
        moe = getattr(layer, "mlp", None)
        experts = getattr(moe, "experts", None) if moe is not None else None
        gate = getattr(moe, "gate", None) if moe is not None else None
        if experts is not None and gate is not None:
            moe_layers.append((layer_idx, moe, experts, gate))
    if not moe_layers:
        fail("No MoE layers with mlp.experts and mlp.gate were found")
    return moe_layers


def check_model_structure(model) -> list[int]:
    config_experts = getattr(model.config, "num_experts", None)
    if config_experts != EXPECTED_RETAINED_EXPERTS:
        fail(
            f"model.config.num_experts={config_experts}, "
            f"expected {EXPECTED_RETAINED_EXPERTS}"
        )

    moe_layers = collect_moe_layers(model)
    layer_indices = []
    for layer_idx, moe, experts, gate in moe_layers:
        layer_indices.append(layer_idx)
        expert_count = len(experts)
        if expert_count != EXPECTED_RETAINED_EXPERTS:
            fail(
                f"layer {layer_idx} has {expert_count} experts, "
                f"expected {EXPECTED_RETAINED_EXPERTS}"
            )
        gate_out = getattr(gate, "out_features", None)
        if gate_out is not None and gate_out != EXPECTED_RETAINED_EXPERTS:
            fail(
                f"layer {layer_idx} gate.out_features={gate_out}, "
                f"expected {EXPECTED_RETAINED_EXPERTS}"
            )
        if tuple(gate.weight.shape)[0] != EXPECTED_RETAINED_EXPERTS:
            fail(
                f"layer {layer_idx} gate.weight rows={tuple(gate.weight.shape)[0]}, "
                f"expected {EXPECTED_RETAINED_EXPERTS}"
            )
        if hasattr(moe, "num_experts") and moe.num_experts != EXPECTED_RETAINED_EXPERTS:
            fail(
                f"layer {layer_idx} moe.num_experts={moe.num_experts}, "
                f"expected {EXPECTED_RETAINED_EXPERTS}"
            )
    return layer_indices


def check_index_maps(layer_indices: list[int]) -> None:
    retained = load_json_map(PRUNED_MODEL_DIR / "retained_expert_indices.json")
    pruned = load_json_map(PRUNED_MODEL_DIR / "pruned_expert_indices.json")

    missing_retained = sorted(set(layer_indices) - set(retained))
    missing_pruned = sorted(set(layer_indices) - set(pruned))
    if missing_retained:
        fail(f"retained_expert_indices.json missing layers: {missing_retained[:10]}")
    if missing_pruned:
        fail(f"pruned_expert_indices.json missing layers: {missing_pruned[:10]}")

    for layer_idx in layer_indices:
        retained_indices = retained[layer_idx]
        pruned_indices = pruned[layer_idx]
        if len(retained_indices) != EXPECTED_RETAINED_EXPERTS:
            fail(f"layer {layer_idx} retained map length is {len(retained_indices)}")
        if len(pruned_indices) != EXPECTED_PRUNED_EXPERTS:
            fail(f"layer {layer_idx} pruned map length is {len(pruned_indices)}")
        if len(set(retained_indices)) != len(retained_indices):
            fail(f"layer {layer_idx} retained map contains duplicates")
        if retained_indices != sorted(retained_indices):
            fail(f"layer {layer_idx} retained map is not sorted")
        invalid = [
            i
            for i in retained_indices + pruned_indices
            if i < 0 or i >= EXPECTED_ORIGINAL_EXPERTS
        ]
        if invalid:
            fail(f"layer {layer_idx} has invalid original expert indices: {invalid}")


def check_router_forward(model, tokenizer) -> None:
    input_device = get_input_device(model)
    inputs = tokenizer(PROMPT, return_tensors="pt")
    inputs = {k: v.to(input_device) for k, v in inputs.items()}
    with torch.inference_mode():
        try:
            outputs = model(
                **inputs,
                use_cache=False,
                output_router_logits=True,
                return_dict=True,
                logits_to_keep=1,
            )
        except TypeError:
            outputs = model(
                **inputs,
                use_cache=False,
                output_router_logits=True,
                return_dict=True,
            )
    router_logits = getattr(outputs, "router_logits", None)
    if router_logits is None:
        fail("Forward pass did not return router_logits")
    dims = sorted({int(t.shape[-1]) for t in router_logits if t is not None})
    if dims != [EXPECTED_RETAINED_EXPERTS]:
        fail(f"router_logits expert dims are {dims}, expected {[EXPECTED_RETAINED_EXPERTS]}")


def check_generation(model, tokenizer) -> None:
    input_device = get_input_device(model)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    inputs = tokenizer(PROMPT, return_tensors="pt")
    inputs = {k: v.to(input_device) for k, v in inputs.items()}
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if not text.strip():
        fail("Generation returned empty text")
    print("\n=== Smoke Generation ===")
    print(text[:800])


def main() -> None:
    print(f"[Check] pruned model dir: {PRUNED_MODEL_DIR}")
    if not PRUNED_MODEL_DIR.is_dir():
        fail(f"Pruned model directory does not exist: {PRUNED_MODEL_DIR}")
    if not list(PRUNED_MODEL_DIR.glob("*.safetensors")):
        fail("No safetensors files found in pruned model directory")

    tokenizer = AutoTokenizer.from_pretrained(
        PRUNED_MODEL_DIR,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = load_pruned_model()
    model.eval()

    total_params = count_parameters(model)
    trainable_params = count_parameters(model, only_trainable=True)
    print("\n=== Model Parameter Count ===")
    print(f"Total params: {total_params:,} ({total_params / 1_000_000:.2f}M)")
    print(f"Trainable params: {trainable_params:,} ({trainable_params / 1_000_000:.2f}M)")

    layer_indices = check_model_structure(model)
    check_index_maps(layer_indices)
    check_router_forward(model, tokenizer)
    check_generation(model, tokenizer)
    print(
        f"\n[OK] keep62_5 check passed: {len(layer_indices)} MoE layers, "
        f"{EXPECTED_RETAINED_EXPERTS}/{EXPECTED_ORIGINAL_EXPERTS} experts retained."
    )


if __name__ == "__main__":
    main()
