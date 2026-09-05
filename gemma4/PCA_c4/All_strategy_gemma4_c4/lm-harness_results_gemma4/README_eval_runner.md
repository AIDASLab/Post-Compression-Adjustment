# Final Gemma4 Evaluation Runner

This directory contains wrapper scripts for evaluating compressed Gemma4 MoE
models with an unmodified `lm-evaluation-harness` checkout. By default the
runner looks for it at `common/lm-evaluation-harness` from the repository root;
set `HARNESS_DIR=/path/to/lm-evaluation-harness` for any other location. The
paper environment used lm-evaluation-harness version `0.4.13.dev0`.
Run `hf auth login` if an evaluated Gemma4 checkpoint still needs gated files
from Hugging Face Hub.

The runner prepends this directory to `PYTHONPATH`, causing
`sitecustomize.py` to apply `vllm_gemma4_transformers_patch.py` before the
evaluation imports vLLM. This local compatibility layer supports the bundled
Gemma4 custom-model code while leaving the lm-evaluation-harness `0.4.13.dev0`
checkout unmodified. Run evaluations through `run_eval_method.sh` so the patch
is loaded automatically.

Run from this directory:

```bash
ENV_NAME=beyond-moe-gemma4
conda activate "$ENV_NAME"
bash run_eval_method.sh MC-SMoE
```

Supported method names:

- `AIMER`
- `MC-SMoE`
- `all`

Output layout:

```text
results_<METHOD>_gemma4/<ratio>/<strategy>/<category>/result.json
results_<METHOD>_gemma4/<ratio>/<strategy>/<category>/run_info.json
```

The five default categories are `reasoning_math`, `MC`, `cot`, `code`, and
`AIME`. Each category is launched as one vLLM job with `tensor_parallel_size=1`.
The scheduler uses `GPU_IDS=0,1,2,3,4,5,6,7` and runs up to eight jobs in
parallel by default.

Important runtime choices:

- `enable_thinking=False` is forced in vLLM `model_args`.
- `apply_chat_template=True` and `fewshot_as_multiturn=True` are used.
- Random seeds are fixed to `42` for Python, lm-eval random, NumPy, PyTorch,
  and few-shot sampling.
- `HF_MODULES_CACHE`, TorchInductor, Triton, vLLM, temp, and vLLM RPC paths are
  forced under this `lm-harness_results_gemma4` directory.
- Hugging Face model and dataset caches are not overridden by default. Set
  `USE_EXP_LOCAL_CACHE=1` if you intentionally want them under
  `lm-harness_results_gemma4/.cache`.

Useful options:

```bash
bash run_eval_method.sh MC-SMoE --dry-run
bash run_eval_method.sh AIMER --categories code,AIME
bash run_eval_method.sh MC-SMoE --force
GPU_IDS=0,1 MAX_PARALLEL=2 bash run_eval_method.sh AIMER
```
