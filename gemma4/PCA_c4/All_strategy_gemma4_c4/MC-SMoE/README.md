# Gemma4 M-SMoE post-compression adjustment on C4

The directory name `MC-SMoE` is retained from the original experiment tree so
that archived commands and result paths remain reproducible. The Gemma4 method
reported in the paper uses only the routing-guided M-SMoE expert-merge stage;
it does not apply the later low-rank MC-SMoE compression stage.

`strategy_train.py` contains the original adjustment implementation,
`scripts/` keeps the original scheduling architecture, and runs are written to
`ratio_<retention>/<strategy>/`.

## Inputs

- Teacher: `google/gemma-4-26B-A4B-it` by default.
- Students: the 50%, 62.5%, and 75% expert-retention checkpoints produced in
  `gemma4/compression_gemma4/MC-SMoE/`.
- Calibration data: the local C4 JSONL shard documented in `common/README.md`.
  The dataset is not included in this repository.

```bash
cd gemma4/PCA_c4/All_strategy_gemma4_c4/MC-SMoE
export DATASET_PATH=/path/to/c4-train.00000-of-01024.json
```

Set `MCSMOE_COMPRESSED_ROOT` if the checkpoints are outside the same-layout
compression directory.

## Paper strategies

The schedule contains the 13 paper strategies: full-model fine-tuning and KD,
router-only fine-tuning and KD, top-8/top-16/top-50/all-expert fine-tuning and
KD, and direct router-logit matching.

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

```bash
export CONDA_ENV_NAME="${ENV_NAME:-beyond-moe-gemma4}"
bash run_all_mcsmoe_gemma4_strategies.sh
bash scripts/run_ratio_625.sh
bash scripts/run_strategy.sh ratio_625 direct_router_logit_matching
```

The wrappers discover Conda portably and keep the historical ratio order,
parallel stages, hyperparameters, GPU telemetry, validation, and output names.
Defaults use one epoch, batch size 2, gradient accumulation 4, length 512,
learning rate `5e-5`, 3,000 C4 examples, and seed 42. See
`scripts/common.sh` for overrides, including `GPUS_DRLM`, `GPUS_TOP16_KD`,
and `VALIDATION_GPU`.

Generated artifacts remain under the original `ratio_*`, `logs/`, and
`reports/` hierarchy. Model weights are intentionally not committed.
Evaluation remains in `../lm-harness_results_gemma4/`.
