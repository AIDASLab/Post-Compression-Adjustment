#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_ROOT:-${WORK_DIR}}/gemma4_aimer_keep75"
mkdir -p "${OUTPUT_DIR}"
cd "${WORK_DIR}"

LOG_FILE="${OUTPUT_DIR}/run_keep75.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export TOKENIZERS_PARALLELISM=false
unset PIP_CONSTRAINT
unset PYTHONPATH
export PYTHONNOUSERSITE=1

"${PYTHON_BIN:-python}" src/gemma4_aimer_prune.py \
  --model-name "google/gemma-4-26B-A4B-it" \
  --output-dir "${OUTPUT_DIR}" \
  --keep-ratio 0.75 \
  --keep-experts 96 \
  --expected-original-experts 128 \
  --expected-layers 30 \
  --metric aimer \
  --metric-device auto \
  --prune-highest-score true \
  --seed 42 \
  --device-map auto \
  --torch-dtype bfloat16 \
  --trust-remote-code true \
  --local-files-only false \
  --model-class causal-lm \
  --offload-folder "${OUTPUT_DIR}/offload_prune"

"${PYTHON_BIN:-python}" check_gemma4_keep75.py \
  --pruned-dir "${OUTPUT_DIR}" \
  --device-map auto \
  --torch-dtype bfloat16 \
  --trust-remote-code true \
  --local-files-only true \
  --model-class causal-lm \
  --offload-folder "${OUTPUT_DIR}/offload_check"
