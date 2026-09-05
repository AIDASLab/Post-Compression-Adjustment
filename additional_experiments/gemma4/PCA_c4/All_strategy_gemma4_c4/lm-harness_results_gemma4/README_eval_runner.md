# Gemma4 PCA-c4 Behavioral-Proxy Evaluation

This folder preserves the original Gemma4 adjusted-checkpoint evaluation
layout. The default run covers the six proxies listed in the repository-level
`additional_experiments/README.md`.

Run `hf auth login` if an evaluated Gemma4 checkpoint still needs gated files
from Hugging Face Hub.

Run from this directory:

```bash
HARNESS_DIR=/path/to/lm-evaluation-harness \
MODEL_ROOT=/path/to/All_strategy_gemma4_c4 \
bash run_eval_method.sh AIMER MC-SMoE --dry-run
```

The example uses `--dry-run` to print and validate the planned jobs. Remove
that option to launch the evaluations.
`--categories safety_alignment` is a convenience alias that launches the six
proxy tasks separately; it does not produce a combined safety score.

The runner prepends this directory to `PYTHONPATH`, causing
`sitecustomize.py` to apply `vllm_gemma4_transformers_patch.py` before the
evaluation imports vLLM. This local compatibility layer is required for the
bundled Gemma4 custom-model code; it leaves the lm-evaluation-harness
`0.4.13.dev0` checkout itself unmodified. Run evaluations through
`run_eval_method.sh` so the patch is loaded automatically.

Results are written under
`results_<METHOD>_gemma4/<ratio>/<strategy>/<proxy>/result.json`. Checkpoint
weights are external and are not included here.
