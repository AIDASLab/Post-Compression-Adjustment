#!/usr/bin/env python3
"""Run one vLLM lm-evaluation-harness category for one Gemma4 strategy model."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval_categories import CATEGORIES, METHODS, SEED


EXP_ROOT = Path(__file__).resolve().parents[1]
ADDITIONAL_STRATEGY_ROOT = Path(os.environ.get("MODEL_ROOT", EXP_ROOT.parent))
DEFAULT_HARNESS_DIR = Path(os.environ.get("HARNESS_DIR", ""))
INPUT_ROOTS = {
    "AIMER": ADDITIONAL_STRATEGY_ROOT / "AIMER",
    "MC-SMoE": ADDITIONAL_STRATEGY_ROOT / "MC-SMoE",
}
ALLOWED_LOCAL_MODEL_ROOTS = tuple(INPUT_ROOTS.values()) + (EXP_ROOT / "model_mirrors",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--ratio", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--category", required=True, choices=tuple(CATEGORIES))
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--harness-dir", default=DEFAULT_HARNESS_DIR, type=Path)
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", default=4096, type=int)
    parser.add_argument("--gpu-memory-utilization", default=None, type=float)
    parser.add_argument("--model-impl", default=os.environ.get("VLLM_MODEL_IMPL"))
    parser.add_argument("--limit", default=None, type=float)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


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


def bool_to_harness(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def build_model_args(
    method: str,
    model_path: Path,
    category: str,
    seed: int,
    dtype: str,
    max_model_len: int,
    gpu_memory_utilization: float | None,
    model_impl: str | None,
) -> str:
    category_cfg = CATEGORIES[category]
    memory_util = (
        gpu_memory_utilization
        if gpu_memory_utilization is not None
        else category_cfg["gpu_memory_utilization"]
    )
    args: dict[str, Any] = {
        "pretrained": str(model_path),
        "dtype": dtype,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": memory_util,
        "max_model_len": max_model_len,
        "trust_remote_code": True,
        "seed": seed,
        "enable_thinking": False,
    }
    selected_model_impl = model_impl
    if method == "MC-SMoE" and selected_model_impl is None:
        selected_model_impl = "transformers"
    if selected_model_impl:
        args["model_impl"] = selected_model_impl
    if method == "MC-SMoE":
        args["enforce_eager"] = True
        args["attention_backend"] = "TRITON_ATTN"
    args.update(category_cfg["model_args"])
    return ",".join(f"{key}={bool_to_harness(value)}" for key, value in args.items())


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    return obj


def write_json(path: Path, payload: Any) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, ensure_ascii=False, indent=4)
        handle.write("\n")
    tmp_path.replace(path)


def pretty_print(results: dict[str, Any]) -> None:
    print("\n====== EVAL RESULTS (selected) ======")
    for task, metrics in results.get("results", {}).items():
        shown = {
            key: round(float(value), 4)
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        }
        print(f"{task:30s} -> {shown}")


def configure_code_eval_manager() -> None:
    if getattr(multiprocessing, "_gemma4_additional_strategy_tcp_manager_patched", False):
        return

    def tcp_manager():
        from multiprocessing.managers import SyncManager

        manager = SyncManager(
            address=("127.0.0.1", 0),
            authkey=multiprocessing.current_process().authkey,
        )
        manager.start()
        return manager

    multiprocessing.Manager = tcp_manager  # type: ignore[assignment]
    multiprocessing._gemma4_additional_strategy_tcp_manager_patched = True  # type: ignore[attr-defined]
    os.environ["GEMMA4_STRATEGY_CODE_EVAL_MANAGER"] = "tcp_loopback"


def apply_model_runtime_patches(method: str) -> None:
    if method != "MC-SMoE":
        return
    os.environ["GEMMA4_STRATEGY_PATCH_VLLM_GEMMA4"] = "1"
    root_text = str(EXP_ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from vllm_gemma4_transformers_patch import apply

    apply()


def configure_local_runtime(category: str) -> None:
    os.environ.setdefault("NUMEXPR_MAX_THREADS", "256")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "64")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(EXP_ROOT / ".cache" / "torchinductor"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(EXP_ROOT / ".cache" / "triton"))
    os.environ.setdefault("VLLM_CACHE_ROOT", str(EXP_ROOT / ".cache" / "vllm"))
    os.environ.setdefault("NLTK_DATA", str(EXP_ROOT / ".cache" / "nltk_data"))
    os.chdir(EXP_ROOT)
    os.environ["TMPDIR"] = "tmp"
    mp_tempdir = Path("tmp") / f"pymp-{os.getpid()}"
    mp_tempdir.mkdir(parents=True, exist_ok=True)
    multiprocessing.current_process()._config["tempdir"] = str(mp_tempdir)
    os.environ["VLLM_RPC_BASE_PATH"] = "tmp/vllm_rpc"
    os.environ["HF_MODULES_CACHE"] = str(EXP_ROOT / ".cache" / "huggingface" / "modules")
    for name in (
        "TMPDIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "VLLM_CACHE_ROOT",
        "VLLM_RPC_BASE_PATH",
        "HF_MODULES_CACHE",
        "NLTK_DATA",
    ):
        path = Path(os.environ[name])
        if not path.is_absolute():
            path = EXP_ROOT / path
        path.mkdir(parents=True, exist_ok=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if category == "code":
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"
        configure_code_eval_manager()


def category_uses_ifeval(tasks: list[str]) -> bool:
    return "ifeval" in tasks


def ensure_ifeval_nltk_data(tasks: list[str]) -> None:
    if not category_uses_ifeval(tasks):
        return
    import fcntl
    import nltk

    nltk_root = Path(os.environ["NLTK_DATA"])
    nltk_root.mkdir(parents=True, exist_ok=True)
    if str(nltk_root) not in nltk.data.path:
        nltk.data.path.insert(0, str(nltk_root))
    lock_path = nltk_root / ".punkt_tab.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", download_dir=str(nltk_root), quiet=False)
            nltk.data.find("tokenizers/punkt_tab")
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def main() -> int:
    args = parse_args()
    sys.dont_write_bytecode = True
    expected_result_root = EXP_ROOT / f"results_{args.method}_gemma4"
    output_dir = resolved_inside(args.output_dir, expected_result_root, "output dir")
    harness_dir = args.harness_dir.resolve()
    if not (harness_dir / "lm_eval").is_dir():
        raise FileNotFoundError(f"lm_eval package not found under {harness_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    result_path = output_dir / "result.json"
    run_info_path = output_dir / "run_info.json"
    if result_path.exists() and not args.force:
        print(f"SKIP existing result: {result_path}")
        return 0

    model_path = resolved_inside_any(args.model_path, ALLOWED_LOCAL_MODEL_ROOTS, "model path")
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"model config not found: {model_path / 'config.json'}")

    configure_local_runtime(args.category)
    apply_model_runtime_patches(args.method)
    random.seed(args.seed)
    sys.path.insert(0, str(harness_dir))

    from lm_eval import evaluator  # noqa: PLC0415


    category_cfg = CATEGORIES[args.category]
    tasks = category_cfg["tasks"]
    ensure_ifeval_nltk_data(tasks)
    gen_kwargs = category_cfg["gen_kwargs"]
    model_args = build_model_args(
        method=args.method,
        model_path=model_path,
        category=args.category,
        seed=args.seed,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        model_impl=args.model_impl,
    )

    print("============================= vLLM + harness start =========================================")
    print(f"method={args.method} ratio={args.ratio} strategy={args.strategy} category={args.category}")
    print(f"model_path={model_path}")
    print(f"output_dir={output_dir}")
    print(f"tasks={tasks}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
    print(f"TOKENIZERS_PARALLELISM={os.environ.get('TOKENIZERS_PARALLELISM', '')}")
    print(f"HF_ALLOW_CODE_EVAL={os.environ.get('HF_ALLOW_CODE_EVAL', '')}")
    print(f"HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE', '')}")
    print(f"HF_DATASETS_OFFLINE={os.environ.get('HF_DATASETS_OFFLINE', '')}")
    print(f"TRANSFORMERS_OFFLINE={os.environ.get('TRANSFORMERS_OFFLINE', '')}")
    print(f"HF_HOME={os.environ.get('HF_HOME', '')}")
    print(f"HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE', '')}")
    print(f"HF_DATASETS_CACHE={os.environ.get('HF_DATASETS_CACHE', '')}")
    print(f"HF_MODULES_CACHE={os.environ.get('HF_MODULES_CACHE', '')}")
    print(f"NLTK_DATA={os.environ.get('NLTK_DATA', '')}")
    print(f"TORCHINDUCTOR_CACHE_DIR={os.environ.get('TORCHINDUCTOR_CACHE_DIR', '')}")
    print(f"TRITON_CACHE_DIR={os.environ.get('TRITON_CACHE_DIR', '')}")
    print(f"VLLM_CACHE_ROOT={os.environ.get('VLLM_CACHE_ROOT', '')}")
    print(f"TMPDIR={os.environ.get('TMPDIR', '')}")
    print(f"MULTIPROCESSING_TMPDIR={multiprocessing.current_process()._config.get('tempdir', '')}")
    print(f"GEMMA4_STRATEGY_CODE_EVAL_MANAGER={os.environ.get('GEMMA4_STRATEGY_CODE_EVAL_MANAGER', '')}")
    print(f"VLLM_RPC_BASE_PATH={os.environ.get('VLLM_RPC_BASE_PATH', '')}")
    print(f"GEMMA4_STRATEGY_PATCH_VLLM_GEMMA4={os.environ.get('GEMMA4_STRATEGY_PATCH_VLLM_GEMMA4', '')}")
    print(f"vLLM model_args={model_args}")
    if gen_kwargs:
        print(f"gen_kwargs={gen_kwargs}")

    start_time = time.time()
    started_at = datetime.now(timezone.utc).isoformat()
    results = evaluator.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks=tasks,
        batch_size="auto",
        apply_chat_template=True,
        fewshot_as_multiturn=True,
        gen_kwargs=gen_kwargs,
        log_samples=True,
        limit=args.limit,
        confirm_run_unsafe_code=bool(category_cfg["confirm_run_unsafe_code"]),
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
        metadata={
            "method": args.method,
            "ratio": args.ratio,
            "strategy": args.strategy,
            "category": args.category,
            "enable_thinking": False,
        },
    )
    if results is None:
        raise RuntimeError("simple_evaluate returned None on this process")

    pretty_print(results)
    write_json(result_path, results["results"])
    finished_at = datetime.now(timezone.utc).isoformat()
    write_json(
        run_info_path,
        {
            "method": args.method,
            "ratio": args.ratio,
            "strategy": args.strategy,
            "category": args.category,
            "model_path": str(model_path),
            "harness_dir": str(harness_dir),
            "output_dir": str(output_dir),
            "tasks": tasks,
            "model_args": model_args,
            "gen_kwargs": gen_kwargs,
            "seed": args.seed,
            "enable_thinking": False,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "elapsed_seconds": round(time.time() - start_time, 3),
            "config": results.get("config"),
            "versions": results.get("versions"),
            "n-shot": results.get("n-shot"),
            "higher_is_better": results.get("higher_is_better"),
            "n-samples": results.get("n-samples"),
            "git_hash": results.get("git_hash"),
            "lm_eval_version": results.get("lm_eval_version"),
            "transformers_version": results.get("transformers_version"),
        },
    )
    print(f"Saved result: {result_path}")
    print(f"Saved run info: {run_info_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
