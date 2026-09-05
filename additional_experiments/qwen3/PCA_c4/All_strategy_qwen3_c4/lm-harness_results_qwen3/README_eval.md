# Qwen3 PCA-c4 Behavioral-Proxy Evaluation

This folder preserves the original Qwen3 adjusted-checkpoint evaluation
layout. The default run covers the six proxies listed in the repository-level
`additional_experiments/README.md`.

Run from this directory:

```bash
HARNESS_DIR=/path/to/lm-evaluation-harness \
MODEL_ROOT=/path/to/All_strategy_qwen3_c4 \
bash run_eval_method.sh HC-SMoE --dry-run
```

The example uses `--dry-run` to print and validate the planned jobs. Remove
that option to launch the evaluations.
`--categories safety_alignment` is a convenience alias that launches the six
proxy tasks separately; it does not produce a combined safety score.

Results are written under
`results_<METHOD>/<ratio>/<strategy>/<proxy>/result.json`. Checkpoint weights
are external and are not included here.
