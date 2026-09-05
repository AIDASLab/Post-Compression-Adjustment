#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_DIR}}"
export OUTPUT_ROOT
cd "${WORK_DIR}"

mkdir -p \
  "${OUTPUT_ROOT}/gemma4_aimer_keep50" \
  "${OUTPUT_ROOT}/gemma4_aimer_keep62_5" \
  "${OUTPUT_ROOT}/gemma4_aimer_keep75"

echo "[launch] CUDA_VISIBLE_DEVICES=${AIMER_GPU_50:-0} -> keep50"
(
  export CUDA_VISIBLE_DEVICES="${AIMER_GPU_50:-0}"
  bash "${WORK_DIR}/scripts/prune_gemma4_keep50.sh"
) &
pid50=$!

echo "[launch] CUDA_VISIBLE_DEVICES=${AIMER_GPU_625:-1} -> keep62_5"
(
  export CUDA_VISIBLE_DEVICES="${AIMER_GPU_625:-1}"
  bash "${WORK_DIR}/scripts/prune_gemma4_keep62_5.sh"
) &
pid62=$!

echo "[launch] CUDA_VISIBLE_DEVICES=${AIMER_GPU_75:-2} -> keep75"
(
  export CUDA_VISIBLE_DEVICES="${AIMER_GPU_75:-2}"
  bash "${WORK_DIR}/scripts/prune_gemma4_keep75.sh"
) &
pid75=$!

status=0
wait "${pid50}" || status=1
wait "${pid62}" || status=1
wait "${pid75}" || status=1

if [[ "${status}" -ne 0 ]]; then
  echo "[done] one or more Gemma4 AIMER jobs failed"
  exit "${status}"
fi

echo "[done] all Gemma4 AIMER pruning and sanity-check jobs completed"
