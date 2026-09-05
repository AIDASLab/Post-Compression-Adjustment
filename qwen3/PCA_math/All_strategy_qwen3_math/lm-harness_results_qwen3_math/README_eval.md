# Qwen3 math-adjusted model evaluation

This directory preserves the lm-evaluation-harness/vLLM evaluation runner for
the OpenR1-Math-adjusted Qwen3 HC-SMoE checkpoints. HC-SMoE is the only Qwen3
compression method included in the paper's math-calibration robustness study.

Set `HARNESS_DIR` to an lm-evaluation-harness checkout. If it is unset, the
runner looks for `common/lm-evaluation-harness` at the repository root. The
paper environment used lm-evaluation-harness 0.4.13.dev0, vLLM 0.19.1,
Transformers 5.8.1, PyTorch 2.10.0+cu128, and Python 3.12.12.

```bash
HARNESS_DIR=/path/to/lm-evaluation-harness \
  bash run_eval_method.sh HC-SMoE_math

HARNESS_DIR=/path/to/lm-evaluation-harness \
  GPU_IDS=0,1 MAX_PARALLEL=2 \
  bash run_eval_method.sh HC-SMoE_math --categories reasoning_math,MC
```

The retained categories are `reasoning_math`, `MC`, `cot`, `code`, and `AIME`.
Evaluation uses vLLM, BF16, tensor parallel size 1, maximum model length 4096,
automatic batching, chat templates, multiturn few-shot formatting, sample
logging, and seed 42. GPU memory utilization is 0.8 for `MC` and 0.9 otherwise;
AIME disables thinking and uses greedy generation with 2,780 maximum tokens.

Results are written to
`results_HC-SMoE_math/<ratio>/<strategy>/<category>/result.json`. Use
`--dry-run`, `--force`, `--categories`, or `--limit` as needed. `PYTHON_BIN`
overrides Python selection; otherwise the active environment's `python` is
used.
