#!/usr/bin/env python3
"""Create an lm-harness_results-local model mirror for evaluation.

The source model directories are read-only inputs. This script mirrors a model
under lm-harness_results/model_mirrors by symlinking large/static files and copying
custom Python files that Transformers/vLLM loads with trust_remote_code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eval_categories import METHODS


EXP_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = EXP_ROOT.parent
MIRROR_ROOT = EXP_ROOT / "model_mirrors"


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
    method_root = (MODEL_ROOT / method).resolve()
    relative = source.resolve().relative_to(method_root)
    parts = relative.parts
    if len(parts) != 2:
        raise ValueError(f"expected <ratio>/<strategy> under {method_root}, got {relative}")
    return parts[0], parts[1]


def ensure_symlink(src: Path, dst: Path) -> None:
    if dst.is_symlink() and Path(dst.readlink()) == src:
        return
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(
            f"refusing to replace existing mirror path; inspect manually: {dst}"
        )
    dst.symlink_to(src)


def patch_hcsmoe_modeling(text: str) -> str:
    replacements = [
        (
            """        self.experts = nn.ModuleList(
            [
                Qwen3MoeMLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(self.physical_num_experts)
            ]
        )
""",
            """        # Keep state_dict keys as experts.<id>.* while avoiding vLLM's
        # Transformers MoE fallback replacing ModuleList experts with FusedMoE.
        self.experts = nn.ModuleDict(
            {
                str(expert_idx): Qwen3MoeMLP(config, intermediate_size=config.moe_intermediate_size)
                for expert_idx in range(self.physical_num_experts)
            }
        )
""",
        ),
        (
            """        mapping_tensor = torch.tensor(mapping, dtype=torch.long)
        if mapping_tensor.min().item() < 0 or mapping_tensor.max().item() >= self.physical_num_experts:
            raise ValueError(
                f"Layer {layer_idx} mapping contains physical ids outside "
                f"[0, {self.physical_num_experts})."
            )
        self.register_buffer("logical_to_physical", mapping_tensor, persistent=False)
""",
            """        if any(int(physical_id) < 0 or int(physical_id) >= self.physical_num_experts for physical_id in mapping):
            raise ValueError(
                f"Layer {layer_idx} mapping contains physical ids outside "
                f"[0, {self.physical_num_experts})."
            )
        mapping_tensor = torch.tensor(mapping, dtype=torch.long)
        self.register_buffer("logical_to_physical", mapping_tensor, persistent=False)
""",
        ),
        (
            """            current_hidden_states = self.experts[physical_idx](current_state)
""",
            """            current_hidden_states = self.experts[str(int(physical_idx))](current_state)
""",
        ),
        (
            """        return final_hidden_states, router_logits
""",
            """        return final_hidden_states
""",
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


def prepare(method: str, source: Path) -> Path:
    method_root = MODEL_ROOT / method
    source = resolved_inside(source, method_root, "source model")
    if not (source / "config.json").is_file():
        raise FileNotFoundError(f"missing config.json: {source / 'config.json'}")

    ratio, strategy = rel_model_parts(method, source)
    target = MIRROR_ROOT / method / ratio / strategy
    target.mkdir(parents=True, exist_ok=True)

    for src in sorted(source.iterdir()):
        dst = target / src.name
        if src.suffix == ".py" and src.is_file():
            copy_python_file(src, dst)
        else:
            ensure_symlink(src, dst)

    return target.resolve()


def main() -> int:
    args = parse_args()
    print(prepare(args.method, args.source))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
