# Gemma4 Original vs Compressed Behavioral-Proxy Evaluation

This folder preserves the original Gemma4 original/compressed-only evaluation
layout. Its default task allowlist contains the six paper-reported behavioral
proxies.

Run `hf auth login` first when the gated Gemma4 base checkpoint is not already
available in the local Hugging Face cache.

Run from this directory:

```bash
HARNESS_DIR=/path/to/lm-evaluation-harness \
COMPRESSED_MODELS_ROOT=/path/to/gemma4/compression_gemma4 \
bash run_eval_method.sh original AIMER MC-SMoE --dry-run
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
`results_<METHOD>/<ratio>/<variant>/<proxy>/result.json`. Local model paths in
`model_registry.json` are relative to `COMPRESSED_MODELS_ROOT`.
