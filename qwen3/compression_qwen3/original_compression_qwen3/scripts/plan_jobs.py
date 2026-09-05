#!/usr/bin/env python3
"""Build an evaluation manifest for original and compressed-only models."""

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
ALLOWED_LOCAL_INPUT_ROOTS = (
    COMPRESSED_MODELS_ROOT / "HC-SMoE",
    COMPRESSED_MODELS_ROOT / "REAP",
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


def patch_hcsmoe_modeling(text: str) -> str:
    replacements = [
        (
            """        self.experts = nn.ModuleList(\n            [\n                Qwen3MoeMLP(config, intermediate_size=config.moe_intermediate_size)\n                for _ in range(self.physical_num_experts)\n            ]\n        )\n""",
            """        self.experts = nn.ModuleDict(\n            {\n                str(expert_idx): Qwen3MoeMLP(config, intermediate_size=config.moe_intermediate_size)\n                for expert_idx in range(self.physical_num_experts)\n            }\n        )\n""",
        ),
        (
            """        mapping_tensor = torch.tensor(mapping, dtype=torch.long)\n        if mapping_tensor.min().item() < 0 or mapping_tensor.max().item() >= self.physical_num_experts:\n            raise ValueError(\n                f\"Layer {layer_idx} mapping contains physical ids outside \"\n                f\"[0, {self.physical_num_experts}).\"\n            )\n        self.register_buffer(\"logical_to_physical\", mapping_tensor, persistent=False)\n""",
            """        if any(int(physical_id) < 0 or int(physical_id) >= self.physical_num_experts for physical_id in mapping):\n            raise ValueError(\n                f\"Layer {layer_idx} mapping contains physical ids outside \"\n                f\"[0, {self.physical_num_experts}).\"\n            )\n        mapping_tensor = torch.tensor(mapping, dtype=torch.long)\n        self.register_buffer(\"logical_to_physical\", mapping_tensor, persistent=False)\n""",
        ),
        (
            """            current_hidden_states = self.experts[physical_idx](current_state)\n""",
            """            current_hidden_states = self.experts[str(int(physical_idx))](current_state)\n""",
        ),
        (
            """        return final_hidden_states, router_logits\n""",
            """        return final_hidden_states\n""",
        ),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def copy_python_file(src: Path, dst: Path) -> None:
    text = src.read_text(encoding="utf-8")
    if src.name == "modeling_hcsmoe_qwen3.py":
        text = patch_hcsmoe_modeling(text)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(dst)


def prepare_mirror(method: str, entry: dict[str, Any], source: Path) -> Path:
    ratio = entry["ratio"]
    variant = entry["variant"]
    target = MIRROR_ROOT / method / ratio / variant
    target.mkdir(parents=True, exist_ok=True)
    for src in sorted(source.iterdir()):
        dst = target / src.name
        if entry.get("mirror_python") and src.suffix == ".py" and src.is_file():
            copy_python_file(src, dst)
        else:
            ensure_symlink(src, dst)
    return target.resolve()


def model_ref_for(method: str, entry: dict[str, Any], mirror_log: Path) -> str:
    ref_type = entry.get("ref_type", "local_path")
    model_ref = str(entry["model_ref"])
    if ref_type == "hf_id":
        return model_ref
    if ref_type != "local_path":
        raise ValueError(f"unsupported ref_type={ref_type} for {method}/{entry.get('name')}")
    source_path = Path(model_ref).expanduser()
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    source = resolved_inside_any(source_path, ALLOWED_LOCAL_INPUT_ROOTS, "local model input")
    if not (source / "config.json").is_file():
        raise FileNotFoundError(f"missing config.json: {source / 'config.json'}")
    if entry.get("mirror_python"):
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
