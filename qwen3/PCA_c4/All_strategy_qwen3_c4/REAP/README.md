# Post-compression adjustment: Qwen3 + REAP + C4

This directory preserves the experiment layout used to adjust REAP-pruned
Qwen3 checkpoints at 50%, 62.5%, and 75% expert retention. The full launcher
runs ratios in the original order `ratio_75`, `ratio_625`, then `ratio_50` and
writes checkpoints, telemetry, validation reports, and convergence reports in
their original subdirectory hierarchy.

## Paper strategies

The launch schedule contains exactly the 13 adjustment strategies reported in
the paper:

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

The environment name is a local choice. The shared evaluation setup is
documented in `common/README.md`; the adjustment entry point itself requires
CUDA-enabled PyTorch, Transformers, Accelerate, Safetensors, and SentencePiece.
A minimal environment can be prepared as follows:

```bash
ENV_NAME=beyond-moe-qwen3
conda create -n "$ENV_NAME" python=3.12 -y
conda activate "$ENV_NAME"
# Install the CUDA-enabled PyTorch build appropriate for the host first.
python -m pip install "transformers==5.8.1" accelerate safetensors sentencepiece
```

## Reproduction

Use the uncompressed first English C4 training shard and a compatible conda
environment:

```bash
export DATASET_PATH=/path/to/c4-train.00000-of-01024.json
export CONDA_ENV_NAME="${ENV_NAME:-beyond-moe-qwen3}"
bash run_all_reap_strategies.sh
```

`C4_DATASET_PATH` is accepted as an alias for `DATASET_PATH`. Conda is
discovered from the active shell or `PATH`; set `CONDA_ROOT` only when its shell
hook is not discoverable. The default student paths resolve to the matching
outputs under `qwen3/compression_qwen3/REAP`. They can be overridden with
`QWEN3_REAP_RATIO_50`, `QWEN3_REAP_RATIO_625`, and `QWEN3_REAP_RATIO_75`.

The paper settings in `scripts/common.sh` are one epoch, batch size 2, gradient
accumulation 4, maximum sequence length 512, learning rate `5e-5`, weight decay
0, linear warmup ratio 0.03, maximum gradient norm 1, temperature 1, 3,000 C4
samples, logging every 10 steps, and seed 42. Gradient checkpointing is off by
default. The teacher is `Qwen/Qwen3-30B-A3B-Instruct-2507`.

The original four two-GPU scheduling slots are retained. Override their device
IDs with `GPU_PAIR_01`, `GPU_PAIR_23`, `GPU_PAIR_45`, and `GPU_PAIR_67`; set
`VALIDATION_GPU` for the post-save validator. For example:

```bash
GPU_PAIR_01=0,1 GPU_PAIR_23=2,3 GPU_PAIR_45=4,5 GPU_PAIR_67=6,7 \
  DATASET_PATH=/path/to/c4.json bash run_all_reap_strategies.sh
```

Hugging Face caches default to the user's XDG cache location and may be changed
with `HF_HOME` and `HF_HUB_CACHE`. Model checkpoints are produced under
`ratio_*/<strategy>/` at runtime but are not included in this repository.

For top-N adjustment, expert selection uses routed calibration counts. GPU
telemetry covers expert selection when applicable and the adjustment loop; it
does not include model/data loading, checkpoint saving, or post-save validation.
