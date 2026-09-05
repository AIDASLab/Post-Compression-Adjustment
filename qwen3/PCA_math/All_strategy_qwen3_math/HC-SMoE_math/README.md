# Post-compression adjustment: Qwen3 + HC-SMoE + OpenR1-Math

This directory preserves the OpenR1-Math robustness experiment for
HC-SMoE-compressed Qwen3 checkpoints. The paper uses the 50%, 62.5%, and 75%
retention ratios; the full launcher keeps the original order `ratio_75`,
`ratio_625`, then `ratio_50`. Qwen3 REAP was not part of this math-calibration
experiment and is therefore not included under `PCA_math`.

## Paper strategies

The launch schedule contains exactly the 13 paper-reported strategies:

- `full_finetune`, `full_kd`
- `only_router_finetune`, `only_router_kd`
- `router_top8_expert_finetune`, `router_top8_expert_kd`
- `router_top16_expert_finetune`, `router_top16_expert_kd`
- `router_top50_expert_finetune`, `router_top50_expert_kd`
- `router_top128_expert_finetune`, `router_top128_expert_kd`
- `direct_router_logit_matching`

`strategy_train.py` is kept as the original experiment implementation. The
public launch schedule and result/validation utilities are restricted to the
paper-reported strategies above.

## Environment

The environment name is a local choice. This math-domain entry point requires
CUDA-enabled PyTorch, Transformers, Accelerate, Datasets, Safetensors, and
SentencePiece. A minimal environment can be prepared as follows;
`common/README.md` documents the complete evaluation environment.

```bash
ENV_NAME=beyond-moe-qwen3
conda create -n "$ENV_NAME" python=3.12 -y
conda activate "$ENV_NAME"
# Install the CUDA-enabled PyTorch build appropriate for the host first.
python -m pip install "transformers==5.8.1" accelerate datasets safetensors sentencepiece
```

## Reproduction

The loader reads `open-r1/OpenR1-Math-220k`, configuration `all`, split
`train`, and takes the first 1,024 examples. It renders the `problem`,
`solution`, and `answer` fields, falling back to chat messages when needed.

```bash
export CONDA_ENV_NAME="${ENV_NAME:-beyond-moe-qwen3}"
bash run_all_hcsmoe_strategies.sh
```

Conda is discovered from the active shell or `PATH`; set `CONDA_ROOT` only
when its shell hook is not discoverable. Student checkpoints resolve to the
matching outputs under `qwen3/compression_qwen3/HC-SMoE` and may be overridden
with `QWEN3_HCSMOE_RATIO_50`, `QWEN3_HCSMOE_RATIO_625`, and
`QWEN3_HCSMOE_RATIO_75`.

The remaining settings match the C4 adjustment runs: one epoch, batch size 2,
gradient accumulation 4, maximum sequence length 512, learning rate `5e-5`,
weight decay 0, warmup ratio 0.03, maximum gradient norm 1, temperature 1,
logging every 10 steps, and seed 42. The teacher is
`Qwen/Qwen3-30B-A3B-Instruct-2507`.

Override the original scheduling slots with `GPU_PAIR_01`, `GPU_PAIR_23`,
`GPU_PAIR_45`, and `GPU_PAIR_67`, and the validator with `VALIDATION_GPU`.
Hugging Face caches default to the user's XDG cache location. Checkpoints are
generated under `ratio_*/<strategy>/` but are not included in this repository.
