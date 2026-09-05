# Gemma4 AIMER pruning runbook

Working directory:

```bash
cd /path/to/Post-Compression-Adjustment/gemma4/compression_gemma4/AIMER
```

Create or activate a dedicated Conda environment. The name below is only an
example and may be changed:

```bash
ENV_NAME=beyond-moe-aimer
conda create -n "$ENV_NAME" python=3.12 -y
conda activate "$ENV_NAME"
```

Before installing, clear inherited pip/Python constraints. This avoids an
inherited `PIP_CONSTRAINT` file, a stale `PYTHONPATH`, or user-site packages
silently changing what gets installed or imported:

```bash
unset PIP_CONSTRAINT
unset PYTHONPATH
export PYTHONNOUSERSITE=1
```

Check that the active interpreter belongs to the environment you selected:

```bash
which python
python -c "import sys; print(sys.executable)"
```

Every install command below also uses a one-shot clean environment, so it remains
safe even if the variables come back in a new shell.

Upgrade pip:

```bash
env -u PIP_CONSTRAINT -u PYTHONPATH PYTHONNOUSERSITE=1 \
  python -m pip install --upgrade pip
```

Install PyTorch for the machine CUDA version. For CUDA 12.8:

```bash
env -u PIP_CONSTRAINT -u PYTHONPATH PYTHONNOUSERSITE=1 \
  python -m pip install --upgrade torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128
```

Then install the pruning-only dependencies:

```bash
env -u PIP_CONSTRAINT -u PYTHONPATH PYTHONNOUSERSITE=1 \
  python -m pip install -r requirements-gemma4-aimer.txt
```

Do not run `pip install -e .` for this Gemma4 pruning path. The upstream AIMER
`pyproject.toml` is configured for the original evaluation package layout and
can fail editable metadata generation in this checkout. The Gemma4 scripts run
directly from this repository root and do not require editable installation.

Quick environment sanity check:

```bash
env -u PIP_CONSTRAINT -u PYTHONPATH PYTHONNOUSERSITE=1 \
  python - <<'PY'
import sys
import torch
import transformers

print("python:", sys.executable)
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
PY
```

If the Gemma checkpoint is gated, authenticate before running:

```bash
huggingface-cli login
```

Run all three pruning jobs in parallel:

```bash
bash scripts/run_gemma4_aimer_all.sh
```

Outputs are saved under:

```text
gemma4_aimer_keep50/
gemma4_aimer_keep62_5/
gemma4_aimer_keep75/
```

Each output directory contains the pruned Hugging Face checkpoint plus:

```text
retained_expert_indices.json
pruned_experts.json
expert_mapping.json
pruning_plan.json
calib_free_metadata.json
calib_free_scores.csv
calib_free_score_table.csv
sanity_check_report.json
```

The scripts use the standard Hugging Face cache unless the corresponding cache
environment variables are set by the caller. Use `OUTPUT_ROOT` to keep the
large generated checkpoints outside the repository checkout.
