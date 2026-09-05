# Gemma4 Original/Compressed Evaluation

This folder contains the Gemma4 original-vs-compressed-only lm-evaluation-harness runner.
Generated logs, mirrors, summaries, runtime caches, and results stay under this directory; Hugging Face model/dataset caches stay at their default location.
The runner looks for `common/lm-evaluation-harness` from the repository root by
default. Set `HARNESS_DIR=/path/to/lm-evaluation-harness` to use another clean
checkout. The paper used lm-evaluation-harness version `0.4.13.dev0`.
Run `hf auth login` first when the gated Gemma4 base checkpoint is not already
available in the local Hugging Face cache.

Run all currently registered models:

```bash
env -u HF_HUB_OFFLINE -u HF_DATASETS_OFFLINE -u TRANSFORMERS_OFFLINE \
  bash run_eval_method.sh original MC-SMoE AIMER --force 2>&1 | tee log_gemma4_original_compression_all.txt
```

Dry-run without launching GPU jobs:

```bash
bash run_eval_method.sh original MC-SMoE AIMER --dry-run
```

MC-SMoE uses `model_impl=transformers` by default because its custom Gemma4 compressed model code was validated with the vLLM Transformers backend.
All categories force `enable_thinking=False` in vLLM model arguments.

The runner prepends this directory to `PYTHONPATH`, causing
`sitecustomize.py` to apply `vllm_gemma4_transformers_patch.py` before the
evaluation imports vLLM. This local compatibility layer supports the bundled
Gemma4 custom-model code while leaving the lm-evaluation-harness `0.4.13.dev0`
checkout unmodified. Run evaluations through `run_eval_method.sh` so the patch
is loaded automatically.
