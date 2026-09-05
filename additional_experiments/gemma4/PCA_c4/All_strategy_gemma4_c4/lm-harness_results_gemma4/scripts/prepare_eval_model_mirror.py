#!/usr/bin/env python3
"""Create an lm-harness_results_gemma4-local model mirror for Gemma4 eval.

Input model directories are read-only. The mirror keeps large model/tokenizer
files as symlinks and only copies files that must be patched for vLLM loading.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from eval_categories import METHODS


EXP_ROOT = Path(__file__).resolve().parents[1]
ADDITIONAL_STRATEGY_ROOT = Path(os.environ.get("MODEL_ROOT", EXP_ROOT.parent))
MIRROR_ROOT = EXP_ROOT / "model_mirrors"
INPUT_ROOTS = {
    "AIMER": ADDITIONAL_STRATEGY_ROOT / "AIMER",
    "MC-SMoE": ADDITIONAL_STRATEGY_ROOT / "MC-SMoE",
}
PROCESSOR_TEMPLATE_CANDIDATES = (
    # MC-SMoE checkpoints are text-only and may not carry processor_config.json.
    # Reuse the processor metadata from the same Gemma4 PCA tree instead of the
    # older non-final experiment path.
    ADDITIONAL_STRATEGY_ROOT / "AIMER" / "ratio_50" / "direct_router_logit_matching" / "processor_config.json",
    ADDITIONAL_STRATEGY_ROOT / "AIMER" / "ratio_50" / "full_finetune" / "processor_config.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--source", required=True, type=Path)
    return parser.parse_args()


def resolved_inside(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    root_resolved = root.resolve()
    if not resolved.is_relative_to(root_resolved):
        raise ValueError(f"{label} must stay under {root_resolved}: {resolved}")
    return resolved


def rel_model_parts(method: str, source: Path) -> tuple[str, str]:
    method_root = INPUT_ROOTS[method].resolve()
    relative = source.resolve().relative_to(method_root)
    parts = relative.parts
    if len(parts) != 2:
        raise ValueError(f"expected <ratio>/<strategy> under {method_root}, got {relative}")
    return parts[0], parts[1]


def ensure_symlink(src: Path, dst: Path) -> None:
    src = src.resolve()
    if dst.is_symlink() and Path(dst.readlink()) == src:
        return
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(
            f"refusing to replace existing mirror path; inspect manually: {dst}"
        )
    dst.symlink_to(src)


def copy_text_file(src: Path, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tmp.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    tmp.replace(dst)


def patch_mcsmoe_json_config_file(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["architectures"] = ["TransformersForCausalLM"]
    text_config = data.get("text_config", {})
    top_k = text_config.get("top_k_experts")
    if top_k is not None:
        text_config.setdefault("num_experts_per_tok", top_k)
        text_config.setdefault("top_k", top_k)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    tmp.replace(path)


def patch_mcsmoe_modeling_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    marker = "_GEMMA4_STRATEGY_VLLM_HEAD_NORM_PATCH"
    if marker in text:
        return
    class_anchor = "def _lookup_layer_mapping(config, layer_idx: Optional[int]) -> torch.LongTensor:\n"
    class_patch = (
        "class Gemma4StrategyWeightlessHeadNorm(nn.Module):\n"
        "    # _GEMMA4_STRATEGY_VLLM_HEAD_NORM_PATCH: Gemma4 text v_norm is a\n"
        "    # weightless head-dim norm. vLLM's generic Transformers backend\n"
        "    # replaces weightless *RMSNorm modules using text hidden_size, which\n"
        "    # breaks v_norm for Gemma4 mixed attention layers.\n"
        "    def __init__(self, hidden_size: int, eps: float = 1e-6):\n"
        "        super().__init__()\n"
        "        self.hidden_size = int(hidden_size)\n"
        "        self.eps = eps\n"
        "\n"
        "    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:\n"
        "        input_dtype = hidden_states.dtype\n"
        "        hidden_states = hidden_states.to(torch.float32)\n"
        "        variance = hidden_states.pow(2).mean(-1, keepdim=True)\n"
        "        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)\n"
        "        return hidden_states.to(input_dtype)\n"
        "\n\n"
        + class_anchor
    )
    if class_anchor not in text:
        raise RuntimeError(f"failed to patch MC-SMoE modeling copy: class anchor not found in {path}")
    text = text.replace(class_anchor, class_patch, 1)

    attn_anchor = "        self.self_attn = Gemma4TextAttention(config=config, layer_idx=layer_idx)\n"
    attn_patch = (
        attn_anchor
        + "        self.self_attn.v_norm = Gemma4StrategyWeightlessHeadNorm(\n"
        + "            self.self_attn.head_dim, eps=config.rms_norm_eps\n"
        + "        )\n"
    )
    if attn_anchor not in text:
        raise RuntimeError(f"failed to patch MC-SMoE modeling copy: attention anchor not found in {path}")
    path.write_text(text.replace(attn_anchor, attn_patch, 1), encoding="utf-8")


def patch_mcsmoe_config_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    marker = "_GEMMA4_STRATEGY_VLLM_SUBCONFIG_CLASS_PATCH"
    if marker in text:
        return
    anchor = '    model_type = "gemma4_moe_compressed"\n'
    patch = (
        "    # _GEMMA4_STRATEGY_VLLM_SUBCONFIG_CLASS_PATCH: this text-only\n"
        "    # checkpoint has audio_config=None; vLLM's generic Transformers backend\n"
        "    # assumes every listed sub-config exists, so keep only non-null Gemma4\n"
        "    # sub-configs.\n"
        "    sub_configs = {\n"
        '        "text_config": Gemma4Config.sub_configs["text_config"],\n'
        '        "vision_config": Gemma4Config.sub_configs["vision_config"],\n'
        "    }\n"
        "    model_type = \"gemma4_moe_compressed\"\n"
    )
    if anchor not in text:
        raise RuntimeError(f"failed to patch MC-SMoE config copy: anchor not found in {path}")
    path.write_text(text.replace(anchor, patch, 1), encoding="utf-8")


def processor_template() -> Path:
    for candidate in PROCESSOR_TEMPLATE_CANDIDATES:
        if candidate.is_file():
            return candidate
    candidates = ", ".join(str(path) for path in PROCESSOR_TEMPLATE_CANDIDATES)
    raise FileNotFoundError(f"missing processor_config.json template; tried: {candidates}")


def copy_processor_template(dst: Path) -> None:
    copy_text_file(processor_template(), dst)


def prepare(method: str, source: Path) -> Path:
    method_root = INPUT_ROOTS[method]
    source = resolved_inside(source, method_root, "source model")
    if not (source / "config.json").is_file():
        raise FileNotFoundError(f"missing config.json: {source / 'config.json'}")

    ratio, strategy = rel_model_parts(method, source)
    target = MIRROR_ROOT / method / ratio / strategy
    target.mkdir(parents=True, exist_ok=True)

    for src in sorted(source.iterdir()):
        dst = target / src.name
        if method == "MC-SMoE" and src.name == "config.json" and src.is_file():
            copy_text_file(src, dst)
            patch_mcsmoe_json_config_file(dst)
        elif method == "MC-SMoE" and src.suffix == ".py" and src.is_file():
            copy_text_file(src, dst)
            if src.name == "configuration_gemma4_moe_compressed.py":
                patch_mcsmoe_config_file(dst)
            elif src.name == "modeling_gemma4_moe_compressed.py":
                patch_mcsmoe_modeling_file(dst)
        else:
            ensure_symlink(src, dst)

    if not (source / "processor_config.json").is_file():
        copy_processor_template(target / "processor_config.json")
    return target.resolve()


def main() -> int:
    args = parse_args()
    print(prepare(args.method, args.source))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
