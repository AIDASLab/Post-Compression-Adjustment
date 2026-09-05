#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}}"
export OUTPUT_ROOT
cd "${ROOT}"

mkdir -p \
  "${OUTPUT_ROOT}/gemma4-26b-a4b-it-mcsmoe-50" \
  "${OUTPUT_ROOT}/gemma4-26b-a4b-it-mcsmoe-62_5" \
  "${OUTPUT_ROOT}/gemma4-26b-a4b-it-mcsmoe-75"

echo "[MC-SMoE] Launching Gemma4 50 ratio on GPU ${MCSMOE_GPU_50:-0}"
CUDA_VISIBLE_DEVICES="${MCSMOE_GPU_50:-0}" bash scripts/gemma4/run_gemma4_merge_50.sh \
  > "${OUTPUT_ROOT}/gemma4-26b-a4b-it-mcsmoe-50/launcher.log" 2>&1 &
pid50=$!

echo "[MC-SMoE] Launching Gemma4 62.5 ratio on GPU ${MCSMOE_GPU_625:-1}"
CUDA_VISIBLE_DEVICES="${MCSMOE_GPU_625:-1}" bash scripts/gemma4/run_gemma4_merge_62_5.sh \
  > "${OUTPUT_ROOT}/gemma4-26b-a4b-it-mcsmoe-62_5/launcher.log" 2>&1 &
pid625=$!

echo "[MC-SMoE] Launching Gemma4 75 ratio on GPU ${MCSMOE_GPU_75:-2}"
CUDA_VISIBLE_DEVICES="${MCSMOE_GPU_75:-2}" bash scripts/gemma4/run_gemma4_merge_75.sh \
  > "${OUTPUT_ROOT}/gemma4-26b-a4b-it-mcsmoe-75/launcher.log" 2>&1 &
pid75=$!

status=0
wait "${pid50}" || status=1
wait "${pid625}" || status=1
wait "${pid75}" || status=1

if [ "${status}" -ne 0 ]; then
  echo "[MC-SMoE] At least one Gemma4 merge/check job failed. See each output folder's launcher.log, merge.log, and check.log."
  exit "${status}"
fi

echo "[MC-SMoE] All Gemma4 merge/check jobs completed."
