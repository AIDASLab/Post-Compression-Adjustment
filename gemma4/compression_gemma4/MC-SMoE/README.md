# Gemma4 routing-guided expert merging

This Gemma4 implementation adapts the routing-guided M-SMoE expert-merging
stage from the [official MC-SMoE repository](https://github.com/UNITES-Lab/MC-SMoE).
The directory name `MC-SMoE` is preserved from the original experiment tree.
The Gemma4 compressor reported in the paper uses this expert-merge stage only;
the later low-rank MC-SMoE stage is not applied by these scripts.

## Files

- `mcsmoe/msmoe-merging-gemma4-compressed.py`: original merge entry point.
- `mcsmoe/merging/grouping_gemma4_compressed.py`: routing statistics, grouping,
  and expert merging.
- `configuration_gemma4_moe_compressed.py` and
  `modeling_gemma4_moe_compressed.py`: remote code saved with compressed
  checkpoints.
- `scripts/gemma4/run_gemma4_merge_*.sh`: ratio-specific wrappers.
- `check_gemma4_mcsmoe_*.py`: structural check wrappers.

The numerical grouping, merge, and remote-code implementations retain the
experiment logic. Portability edits are confined to wrappers, check-script
defaults, the entry point's C4 default, and removal of re-exports for unrelated
upstream modules that are not included here.

## Environment and data

```bash
ENV_NAME=beyond-moe-msmoe
conda create -n "$ENV_NAME" python=3.12 -y
conda activate "$ENV_NAME"
# Install a CUDA-enabled PyTorch build appropriate for the host first.
python -m pip install -r requirements.txt
hf auth login
```

Use Python 3.10 or newer so that the environment can install a
Gemma4-capable Transformers release; Python 3.12 is shown above to match the
paper's recorded Gemma4 stack. Installing PyTorch first prevents the unpinned
`torch` entry in `requirements.txt` from selecting a CPU-only or incompatible
CUDA build. Hugging Face authentication is required when the gated
`google/gemma-4-26B-A4B-it` checkpoint is not already cached.

Download the C4 JSONL shard documented in `common/README.md`; it is not
included. Every public wrapper requires its path explicitly:

```bash
export CALIBRATION_DATASET_PATH=/path/to/c4-train.00000-of-01024.json
```

`C4_DATASET_PATH` is accepted as an alias. The historical merge configuration
uses routing logits, `subset_ratio=0.01`, sequence length 512, batch size 1,
and the Gemma4 base checkpoint `google/gemma-4-26B-A4B-it`.

## Run

```bash
cd gemma4/compression_gemma4/MC-SMoE
export MCSMOE_CONDA_ENV="${ENV_NAME:-beyond-moe-msmoe}"
bash run_gemma4_merge_50.sh
bash run_gemma4_merge_62_5.sh
bash run_gemma4_merge_75.sh

# Or launch all three in parallel.
bash run_gemma4_all_parallel.sh
```

Output directory names remain unchanged:

- `gemma4-26b-a4b-it-mcsmoe-50` (64 physical experts per layer)
- `gemma4-26b-a4b-it-mcsmoe-62_5` (80 physical experts per layer)
- `gemma4-26b-a4b-it-mcsmoe-75` (96 physical experts per layer)

Set `OUTPUT_ROOT` to relocate those directories. Set `PYTHON_BIN` to choose an
interpreter. Parallel-launch GPU defaults can be overridden with
`MCSMOE_GPU_50`, `MCSMOE_GPU_625`, and `MCSMOE_GPU_75`.

Generated model weights are intentionally absent and must not be committed.
The retained third-party license is in `LICENSE`.
