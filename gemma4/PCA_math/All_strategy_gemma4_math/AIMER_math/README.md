# Gemma4 AIMER post-compression adjustment on OpenR1-Math

This directory preserves the math-domain robustness experiment for Gemma4
AIMER. The Camera Ready experiment uses the first 1,024 examples from
`open-r1/OpenR1-Math-220k`; `strategy_train.py` loads the dataset through the
Hugging Face `datasets` package.

Only AIMER was run for the Gemma4 math-domain robustness setting. No Gemma4
M-SMoE math adjustment is part of the paper experiment.

## Inputs and layout

- Teacher: `google/gemma-4-26B-A4B-it` by default.
- Students: AIMER checkpoints at 50%, 62.5%, and 75% expert retention.
- Calibration identifier: `open-r1/OpenR1-Math-220k` by default.
- Outputs: `ratio_<retention>/<strategy>/`, matching the original experiment.

Set `AIMER_COMPRESSED_ROOT` if the compressed checkpoints are not under the
same-layout `gemma4/compression_gemma4/AIMER/` directory.

## Environment

The environment name is a local choice. This math-domain entry point requires
CUDA-enabled PyTorch, Gemma4-capable Transformers, Accelerate, Datasets,
Safetensors, and SentencePiece. A minimal environment can be prepared as
follows; `common/README.md` documents the complete evaluation environment.

```bash
ENV_NAME=beyond-moe-gemma4
conda create -n "$ENV_NAME" python=3.12 -y
conda activate "$ENV_NAME"
# Install the CUDA-enabled PyTorch build appropriate for the host first.
python -m pip install "transformers==5.8.1" accelerate datasets safetensors sentencepiece
hf auth login
```

Authentication is required when the gated `google/gemma-4-26B-A4B-it`
checkpoint is not already available in the local Hugging Face cache.

## Run

```bash
cd gemma4/PCA_math/All_strategy_gemma4_math/AIMER_math
export CONDA_ENV_NAME="${ENV_NAME:-beyond-moe-gemma4}"
bash run_all_aimer_gemma4_strategies.sh

# Or run one ratio or one strategy.
bash scripts/run_ratio_75.sh
bash scripts/run_strategy.sh ratio_75 only_router_kd
```

The schedule covers the paper's 13 adjustment strategies. Defaults are one
epoch, batch size 2, gradient accumulation 4, sequence length 512, learning
rate `5e-5`, no weight decay, warmup ratio 0.03, temperature 1.0, 1,024
calibration examples, and seed 42. The wrappers preserve the original parallel
schedule and telemetry flow while allowing overrides through
`scripts/common.sh`.

Generated weights are intentionally not committed. The existing
`../lm-harness_results_gemma4_math/` runner evaluates saved checkpoints.
