# Qwen3 Original vs Compressed Behavioral-Proxy Evaluation

This folder preserves the original Qwen3 original/compressed-only evaluation
layout. Its default task allowlist contains the six paper-reported behavioral
proxies.

Run from this directory:

```bash
HARNESS_DIR=/path/to/lm-evaluation-harness \
COMPRESSED_MODELS_ROOT=/path/to/qwen3/compression_qwen3 \
bash run_eval_method.sh original HC-SMoE REAP --dry-run
```

The example uses `--dry-run` to print and validate the planned jobs. Remove
that option to launch the evaluations.
`--categories safety_alignment` is a convenience alias that launches the six
proxy tasks separately; it does not produce a combined safety score.

Results are written under
`results_<METHOD>/<ratio>/<variant>/<proxy>/result.json`. Local model paths in
`model_registry.json` are relative to `COMPRESSED_MODELS_ROOT`.
