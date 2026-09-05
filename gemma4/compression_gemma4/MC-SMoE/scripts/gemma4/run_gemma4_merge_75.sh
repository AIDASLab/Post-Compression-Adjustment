#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${OUTPUT_ROOT:-${ROOT}}/gemma4-26b-a4b-it-mcsmoe-75"
CALIBRATION_DATASET_PATH="${CALIBRATION_DATASET_PATH:-${C4_DATASET_PATH:-}}"

if [[ -z "${CALIBRATION_DATASET_PATH}" || ! -f "${CALIBRATION_DATASET_PATH}" ]]; then
  echo "Set CALIBRATION_DATASET_PATH (or C4_DATASET_PATH) to the downloaded C4 JSONL shard." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME:-${HOME}/.cache}/huggingface}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1

if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV:-}" = "base" ]; then
  if [[ -z "${MCSMOE_CONDA_ENV:-}" ]]; then
    echo "Activate a non-base conda environment or set MCSMOE_CONDA_ENV explicitly." >&2
    exit 1
  fi
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda was not found; activate '${MCSMOE_CONDA_ENV}' before running." >&2
    exit 1
  fi
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${MCSMOE_CONDA_ENV}"
fi

cd "${ROOT}"
mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN:-python}" -u mcsmoe/msmoe-merging-gemma4-compressed.py \
  --model_name_or_path="google/gemma-4-26B-A4B-it" \
  --calibration_dataset_path="${CALIBRATION_DATASET_PATH}" \
  --physical_num_experts=96 \
  --expected_logical_num_experts=128 \
  --ratio_label="75" \
  --similarity_base="router-logits" \
  --subset_ratio=0.01 \
  --block_size=512 \
  --batch_size=1 \
  --output_dir="${OUTPUT_DIR}" \
  2>&1 | tee "${OUTPUT_DIR}/merge.log"

MCSMOE_MODEL_DIR="${OUTPUT_DIR}" "${PYTHON_BIN:-python}" -u check_gemma4_mcsmoe_75.py \
  2>&1 | tee "${OUTPUT_DIR}/check.log"
