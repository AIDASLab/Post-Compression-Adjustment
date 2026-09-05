#!/usr/bin/env python3
"""Build an evaluation manifest for Gemma4 original and compressed-only models."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from eval_categories import CATEGORIES, METHODS


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "model_registry.json"
MIRROR_ROOT = ROOT / "model_mirrors"
COMPRESSED_MODELS_ROOT = ROOT.parent
PROCESSOR_TEMPLATE = Path(
    COMPRESSED_MODELS_ROOT / "AIMER" / "gemma4_aimer_keep50" / "processor_config.json"
)
ALLOWED_LOCAL_INPUT_ROOTS = (
    COMPRESSED_MODELS_ROOT / "AIMER",
    COMPRESSED_MODELS_ROOT / "MC-SMoE",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--categories", nargs="+", required=True, choices=tuple(CATEGORIES))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--mirror-log", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_registry() -> dict[str, list[dict[str, Any]]]:
    with REGISTRY_PATH.open("r", encoding="utf-8") as handle:
        registry = json.load(handle)
    return {method: list(registry.get(method, [])) for method in METHODS}


def resolved_inside(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    root_resolved = root.resolve()
    if not resolved.is_relative_to(root_resolved):
        raise ValueError(f"{label} must stay under {root_resolved}: {resolved}")
    return resolved


def resolved_inside_any(path: Path, roots: tuple[Path, ...], label: str) -> Path:
    resolved = path.resolve()
    resolved_roots = tuple(root.resolve() for root in roots)
    if not any(resolved.is_relative_to(root) for root in resolved_roots):
        roots_text = ", ".join(str(root) for root in resolved_roots)
        raise ValueError(f"{label} must stay under one of [{roots_text}]: {resolved}")
    return resolved


def ensure_symlink(src: Path, dst: Path) -> None:
    if dst.is_symlink() and Path(dst.readlink()) == src:
        return
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.symlink_to(src)


def copy_python_file(src: Path, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tmp.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    tmp.replace(dst)


def copy_json_file(src: Path, dst: Path) -> None:
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
    marker = "_ORIGCOMP_VLLM_HEAD_NORM_PATCH"
    if marker in text:
        return
    class_anchor = "def _lookup_layer_mapping(config, layer_idx: Optional[int]) -> torch.LongTensor:\n"
    class_patch = (
        "class OrigCompGemma4WeightlessHeadNorm(nn.Module):\n"
        "    # _ORIGCOMP_VLLM_HEAD_NORM_PATCH: Gemma4 text v_norm is a weightless\n"
        "    # head-dim norm. vLLM's generic Transformers backend replaces *RMSNorm\n"
        "    # modules without weights using text hidden_size, which breaks v_norm.\n"
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
        + "        self.self_attn.v_norm = OrigCompGemma4WeightlessHeadNorm(\n"
        + "            self.self_attn.head_dim, eps=config.rms_norm_eps\n"
        + "        )\n"
    )
    if attn_anchor not in text:
        raise RuntimeError(f"failed to patch MC-SMoE modeling copy: attention anchor not found in {path}")
    path.write_text(text.replace(attn_anchor, attn_patch, 1), encoding="utf-8")


def patch_mcsmoe_config_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    marker = "_ORIGCOMP_VLLM_SUBCONFIG_CLASS_PATCH"
    if marker in text:
        return
    anchor = '    model_type = "gemma4_moe_compressed"\n'
    patch = (
        "    # _ORIGCOMP_VLLM_SUBCONFIG_CLASS_PATCH: this text-only checkpoint has\n"
        "    # audio_config=None; vLLM's generic Transformers backend assumes every\n"
        "    # listed sub-config exists, so keep only the non-null Gemma4 sub-configs.\n"
        "    sub_configs = {\n"
        '        "text_config": Gemma4Config.sub_configs["text_config"],\n'
        '        "vision_config": Gemma4Config.sub_configs["vision_config"],\n'
        "    }\n"
        "    model_type = \"gemma4_moe_compressed\"\n"
    )
    if anchor not in text:
        raise RuntimeError(f"failed to patch MC-SMoE config copy: anchor not found in {path}")
    path.write_text(text.replace(anchor, patch, 1), encoding="utf-8")


def copy_processor_template(dst: Path) -> None:
    if not PROCESSOR_TEMPLATE.is_file():
        raise FileNotFoundError(f"missing processor template: {PROCESSOR_TEMPLATE}")
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tmp.write_text(PROCESSOR_TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
    tmp.replace(dst)


def prepare_mirror(method: str, entry: dict[str, Any], source: Path) -> Path:
    ratio = entry["ratio"]
    variant = entry["variant"]
    target = MIRROR_ROOT / method / ratio / variant
    target.mkdir(parents=True, exist_ok=True)
    for src in sorted(source.iterdir()):
        dst = target / src.name
        if entry.get("mirror_python") and src.name == "config.json" and src.is_file():
            copy_json_file(src, dst)
            patch_mcsmoe_json_config_file(dst)
        elif entry.get("mirror_python") and src.suffix == ".py" and src.is_file():
            copy_python_file(src, dst)
            if src.name == "configuration_gemma4_moe_compressed.py":
                patch_mcsmoe_config_file(dst)
            elif src.name == "modeling_gemma4_moe_compressed.py":
                patch_mcsmoe_modeling_file(dst)
        else:
            ensure_symlink(src, dst)
    if not (source / "processor_config.json").is_file():
        copy_processor_template(target / "processor_config.json")
    return target.resolve()


def needs_mirror(entry: dict[str, Any], source: Path) -> bool:
    return bool(entry.get("mirror_python")) or not (source / "processor_config.json").is_file()


def model_ref_for(method: str, entry: dict[str, Any], mirror_log: Path) -> str:
    ref_type = entry.get("ref_type", "local_path")
    model_ref = str(entry["model_ref"])
    if ref_type == "hf_id":
        return model_ref
    if ref_type != "local_path":
        raise ValueError(f"unsupported ref_type={ref_type} for {method}/{entry.get('name')}")
    source_path = Path(model_ref)
    if not source_path.is_absolute():
        source_path = COMPRESSED_MODELS_ROOT / source_path
    source = resolved_inside_any(source_path, ALLOWED_LOCAL_INPUT_ROOTS, "local model input")
    if not (source / "config.json").is_file():
        raise FileNotFoundError(f"missing config.json: {source / 'config.json'}")
    if needs_mirror(entry, source):
        mirror = prepare_mirror(method, entry, source)
        with mirror_log.open("a", encoding="utf-8") as handle:
            handle.write(f"{source}\t{mirror}\n")
        return str(mirror)
    return str(source)


def result_dir_for(method: str, entry: dict[str, Any], category: str) -> Path:
    return ROOT / f"results_{method}" / entry["ratio"] / entry["variant"] / category


def log_file_for(method: str, entry: dict[str, Any], category: str) -> Path:
    return ROOT / "logs_eval" / method / entry["ratio"] / entry["variant"] / f"{category}.log"


def main() -> int:
    args = parse_args()
    manifest = resolved_inside(args.manifest, ROOT, "manifest")
    mirror_log = resolved_inside(args.mirror_log, ROOT, "mirror log")
    mirror_log.parent.mkdir(parents=True, exist_ok=True)
    mirror_log.write_text("", encoding="utf-8")

    requested = []
    for method in args.methods:
        if method == "all":
            requested.extend(METHODS)
        elif method in METHODS:
            requested.append(method)
        else:
            raise ValueError(f"invalid method: {method}")
    seen = set()
    methods = [method for method in requested if not (method in seen or seen.add(method))]

    registry = load_registry()
    rows = []
    for method in methods:
        entries = registry.get(method, [])
        if not entries:
            continue
        for entry in entries:
            model_ref = model_ref_for(method, entry, mirror_log)
            for category in args.categories:
                out_dir = result_dir_for(method, entry, category)
                log_file = log_file_for(method, entry, category)
                if (out_dir / "result.json").is_file() and not args.force:
                    continue
                rows.append(
                    {
                        "method": method,
                        "ratio": entry["ratio"],
                        "variant": entry["variant"],
                        "model_name": entry["name"],
                        "model_ref": model_ref,
                        "category": category,
                        "output_dir": str(out_dir),
                        "log_file": str(log_file),
                    }
                )

    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8") as handle:
        handle.write("method\tratio\tvariant\tmodel_name\tmodel_ref\tcategory\toutput_dir\tlog_file\n")
        for row in rows:
            handle.write("\t".join(str(row[key]) for key in [
                "method",
                "ratio",
                "variant",
                "model_name",
                "model_ref",
                "category",
                "output_dir",
                "log_file",
            ]))
            handle.write("\n")
    print(len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
