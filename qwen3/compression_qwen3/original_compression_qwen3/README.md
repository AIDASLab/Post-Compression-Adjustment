# Original and compressed Qwen3 evaluation

This directory preserves the pre-adjustment evaluation runner for the original
Qwen3 model and its HC-SMoE-merged and REAP-pruned checkpoints. Model locations
are registered in `model_registry.json` using paths relative to this directory,
so generated checkpoints remain in their original sibling method directories.

Set `HARNESS_DIR` to an lm-evaluation-harness `0.4.13.dev0` checkout. If unset,
the runner looks for `common/lm-evaluation-harness` at the repository root.

```bash
HARNESS_DIR=/path/to/lm-evaluation-harness \
  bash run_eval_method.sh original HC-SMoE REAP

HARNESS_DIR=/path/to/lm-evaluation-harness \
  bash run_eval_method.sh original HC-SMoE REAP --dry-run
```

The paper environment used lm-evaluation-harness 0.4.13.dev0, vLLM 0.19.1,
Transformers 5.8.1, PyTorch 2.10.0+cu128, and Python 3.12.12. The runner retains the original five
categories and vLLM settings: BF16, tensor parallel size 1, maximum model length
4096, automatic batching, chat templates, sample logging, and seed 42. `MC`
uses 0.8 GPU memory utilization and the other categories use 0.9. HC-SMoE uses
eager mode by default.

Results keep the original hierarchy
`results_<METHOD>/<ratio>/<variant>/<category>/result.json`; logs, manifests,
summaries, model mirrors, and runtime caches also remain under this directory.
The generated compressed model weights themselves are not included in the
repository.

Use `GPU_IDS` and `MAX_PARALLEL` to change scheduling, or `PYTHON_BIN` to select
another Python executable. Code-category evaluation explicitly enables unsafe
code execution and should be run only in an isolated environment.
