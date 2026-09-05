# Gemma4 AIMER compression

This directory preserves the Gemma4 pruning code and wrapper layout used for
the paper, adapted from the [official AIMER repository](https://github.com/ZongfangLiu/AIMER).
The upstream AIMER implementation builds on infrastructure from the
[official REAP repository](https://github.com/CerebrasResearch/reap). AIMER
ranks experts from their weights and prunes the highest-scoring experts
uniformly across Gemma4's 30 routed MoE layers. It is calibration-free.

## Files

- `src/gemma4_aimer_prune.py`: pruning and Hugging Face checkpoint export.
- `src/gemma4_sanity_check.py`: structural and generation sanity checks.
- `scripts/prune_gemma4_keep*.sh`: the three paper retention ratios.
- `scripts/run_gemma4_aimer_all.sh`: parallel launcher for all ratios.
- `check_gemma4_keep*.py`: ratio-specific check wrappers.

The algorithm files are retained as used in the experiment. Release edits are
limited to making shell paths and GPU assignments portable.

## Environment

Create or activate an environment with a CUDA-compatible PyTorch build, then
install the local requirements:

```bash
ENV_NAME=beyond-moe-aimer
conda create -n "$ENV_NAME" python=3.12 -y
conda activate "$ENV_NAME"
# Install a CUDA-enabled PyTorch build appropriate for the host first.
python -m pip install -r requirements-gemma4-aimer.txt
hf auth login
```

Gemma4 support requires the Transformers version specified in the requirements
file. The requirements file deliberately does not select a PyTorch CUDA wheel;
install it separately for the host driver. `GEMMA4_AIMER_RUNBOOK.md` provides a
CUDA 12.8 example and an environment sanity check. Hugging Face authentication
is required when the gated base checkpoint is not already cached.

## Run

From the repository root:

```bash
cd gemma4/compression_gemma4/AIMER

# Run one ratio.
bash scripts/prune_gemma4_keep50.sh
bash scripts/prune_gemma4_keep62_5.sh
bash scripts/prune_gemma4_keep75.sh

# Or launch all three in parallel.
bash scripts/run_gemma4_aimer_all.sh
```

The wrappers keep the original output names:

- `gemma4_aimer_keep50` (64 of 128 experts per layer)
- `gemma4_aimer_keep62_5` (80 of 128)
- `gemma4_aimer_keep75` (96 of 128)

Set `OUTPUT_ROOT` to place these three directories outside the code checkout.
`PYTHON_BIN` selects an interpreter. For the parallel launcher, GPU defaults
0/1/2 can be overridden with `AIMER_GPU_50`, `AIMER_GPU_625`, and
`AIMER_GPU_75`.

Each output contains a complete Hugging Face checkpoint plus expert mappings,
score tables, pruning metadata, and a sanity-check report. Generated weights
are intentionally absent from this repository and must not be committed.

The retained third-party license is in `LICENSE`.
