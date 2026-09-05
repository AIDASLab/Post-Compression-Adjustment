import json
import os
import re
from pathlib import Path
from typing import Dict, List


REPO_DIR = Path(__file__).resolve().parents[3]


def _set_repo_local_env() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1,2,3,4")
    os.environ.setdefault("PYTHONPATH", str(REPO_DIR))
    os.environ.setdefault("TMPDIR", str(REPO_DIR / "tmp"))
    os.environ.setdefault("MPLCONFIGDIR", str(REPO_DIR / ".cache" / "matplotlib"))


def _read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _checkpoint_keys(model_dir: Path) -> List[str]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = _read_json(index_path)
        return sorted(index["weight_map"].keys())

    safetensor_path = model_dir / "model.safetensors"
    if safetensor_path.exists():
        from safetensors import safe_open

        with safe_open(safetensor_path, framework="pt", device="cpu") as f:
            return sorted(f.keys())

    pytorch_path = model_dir / "pytorch_model.bin"
    if pytorch_path.exists():
        import torch

        return sorted(torch.load(pytorch_path, map_location="cpu").keys())

    raise FileNotFoundError(f"No checkpoint file found in {model_dir}")


def _check_checkpoint_expert_keys(model_dir: Path, expected_physical: int) -> None:
    keys = _checkpoint_keys(model_dir)
    expert_key_pattern = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.")
    per_layer = {}
    for key in keys:
        match = expert_key_pattern.match(key)
        if match is None:
            continue
        layer_idx = int(match.group(1))
        expert_idx = int(match.group(2))
        per_layer.setdefault(layer_idx, set()).add(expert_idx)

    if not per_layer:
        raise AssertionError("No MoE expert weights were found in the saved checkpoint.")

    bad_layers = {
        layer_idx: sorted(expert_ids)
        for layer_idx, expert_ids in per_layer.items()
        if len(expert_ids) != expected_physical or max(expert_ids) != expected_physical - 1
    }
    if bad_layers:
        preview = {layer: ids[:8] for layer, ids in list(bad_layers.items())[:3]}
        raise AssertionError(
            f"Checkpoint expert key count mismatch. Expected {expected_physical} physical experts/layer. "
            f"Preview: {preview}"
        )

    print(f"[CHECK] Checkpoint expert weights: {len(per_layer)} layers x {expected_physical} physical experts")


def _input_device(model):
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        for key in ("model.embed_tokens", "embed_tokens"):
            if key in model.hf_device_map:
                device = model.hf_device_map[key]
                if isinstance(device, int):
                    return f"cuda:{device}"
                return str(device)
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return "cpu"


def run_check(model_dir: str, expected_physical: int, ratio_label: str) -> None:
    _set_repo_local_env()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(model_dir).resolve()
    if model_path != REPO_DIR and REPO_DIR not in model_path.parents:
        raise ValueError(f"Refusing to check a model outside {REPO_DIR}: {model_path}")

    print(f"[CHECK] Ratio {ratio_label}")
    print(f"[CHECK] Model dir: {model_path}")

    config = _read_json(model_path / "config.json")
    mapping = _read_json(model_path / "expert_mapping.json")
    assert config["logical_num_experts"] == 128, config["logical_num_experts"]
    assert config["physical_num_experts"] == expected_physical, config["physical_num_experts"]
    assert mapping["logical_num_experts"] == 128, mapping["logical_num_experts"]
    assert mapping["physical_num_experts"] == expected_physical, mapping["physical_num_experts"]
    print("[CHECK] Config/mapping metadata is valid")

    _check_checkpoint_expert_keys(model_path, expected_physical)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    total_params = sum(param.numel() for param in model.parameters())
    print(f"[CHECK] Total parameters: {total_params:,} ({total_params / 1_000_000_000:.3f}B)")

    physical_counts = []
    router_dims = []
    mapping_lengths = []
    for layer in model.model.layers:
        mlp = layer.mlp
        if not hasattr(mlp, "experts"):
            continue
        physical_counts.append(len(mlp.experts))
        router_dims.append(mlp.gate.out_features)
        mapping_lengths.append(int(mlp.logical_to_physical.numel()))

    if not physical_counts:
        raise AssertionError("Loaded model has no sparse MoE layers.")
    if any(count != expected_physical for count in physical_counts):
        raise AssertionError(f"Loaded physical expert counts mismatch: {sorted(set(physical_counts))}")
    if any(dim != 128 for dim in router_dims):
        raise AssertionError(f"Router output dims must stay logical 128: {sorted(set(router_dims))}")
    if any(length != 128 for length in mapping_lengths):
        raise AssertionError(f"Mapping lengths must be 128: {sorted(set(mapping_lengths))}")
    print("[CHECK] Loaded model uses logical router=128 and compressed physical experts")

    prompt = "Tell me about large language models in one sentence."
    inputs = tokenizer(prompt, return_tensors="pt").to(_input_device(model))
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            use_cache=True,
        )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("[CHECK] Short generation succeeded:")
    print(text)
    print(f"[CHECK] Ratio {ratio_label} passed.")
