# Gemma4 AIMER post-compression adjustment on C4

This directory preserves the experiment layout used for the Gemma4 AIMER runs.
`strategy_train.py` contains the original training implementation, the shell
wrappers in `scripts/` schedule the paper's adjustment strategies, and each run
writes to `ratio_<retention>/<strategy>/` below this directory.

## Inputs

- Teacher: `google/gemma-4-26B-A4B-it` by default.
- Students: the three AIMER checkpoints produced under
  `gemma4/compression_gemma4/AIMER/`.
- Calibration data: the local C4 JSONL shard documented in `common/README.md`.
  The dataset is not included in this repository.

The paper setting reads the first 3,000 non-empty C4 examples. Set the dataset
explicitly before launching:

```bash
cd gemma4/PCA_c4/All_strategy_gemma4_c4/AIMER
export DATASET_PATH=/path/to/c4-train.00000-of-01024.json
```

If compressed checkpoints live somewhere else, set `AIMER_COMPRESSED_ROOT` to
the directory containing `gemma4_aimer_keep50`, `gemma4_aimer_keep62_5`, and
`gemma4_aimer_keep75`.

## Paper strategies

The launch schedule contains the 13 strategies reported in the paper:

- `full_finetune`, `full_kd`
- `only_router_finetune`, `only_router_kd`
- `router_top8_expert_finetune`, `router_top8_expert_kd`
- `router_top16_expert_finetune`, `router_top16_expert_kd`
- `router_top50_expert_finetune`, `router_top50_expert_kd`
- `router_top128_expert_finetune`, `router_top128_expert_kd`
- `direct_router_logit_matching`

The retained-expert settings are 64, 80, and 96 experts per layer, represented
by `ratio_50`, `ratio_625`, and `ratio_75` respectively.

## Environment

The environment name is a local choice. The adjustment entry point requires
CUDA-enabled PyTorch, Gemma4-capable Transformers, Accelerate, Safetensors,
and SentencePiece. A minimal environment can be prepared as follows;
`common/README.md` documents the complete evaluation environment.

```bash
ENV_NAME=beyond-moe-gemma4
conda create -n "$ENV_NAME" python=3.12 -y
conda activate "$ENV_NAME"
# Install the CUDA-enabled PyTorch build appropriate for the host first.
python -m pip install "transformers==5.8.1" accelerate safetensors sentencepiece
hf auth login
```

Authentication is required when the gated `google/gemma-4-26B-A4B-it`
checkpoint is not already available in the local Hugging Face cache.

## Run

The wrappers use the active conda environment by default. Set
`CONDA_ENV_NAME` only when the launcher should activate a different
environment through `CONDA_ROOT` or the `conda` found on `PATH`.

```bash
# Reproduce all ratios in the historical order: 75%, 62.5%, then 50%.
export CONDA_ENV_NAME="${ENV_NAME:-beyond-moe-gemma4}"
bash run_all_aimer_gemma4_strategies.sh

# Run one ratio or one strategy.
bash scripts/run_ratio_50.sh
bash scripts/run_strategy.sh ratio_50 only_router_finetune
```

Defaults reproduce the paper configuration: one epoch, batch size 2, gradient
accumulation 4, sequence length 512, learning rate `5e-5`, linear warmup ratio
0.03, no weight decay, gradient norm 1.0, temperature 1.0, seed 42, and 3,000
calibration examples. All values remain overrideable through
`scripts/common.sh`. Historical GPU assignments are retained and can be
overridden with variables such as `GPUS_ROUTER_FINETUNE`, `GPUS_FULL_KD`, and
`VALIDATION_GPU`.

## Outputs

Training checkpoints, tokenizer files, logs, GPU telemetry, convergence CSVs,
and validation reports use the original hierarchy. Model weights are
intentionally not committed to this repository. Benchmark evaluation remains
in `../lm-harness_results_gemma4/`.
