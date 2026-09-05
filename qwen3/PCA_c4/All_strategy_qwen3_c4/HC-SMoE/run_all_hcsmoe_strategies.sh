#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${ROOT_DIR}/logs"

echo "[All] HC-SMoE strategy recovery pipeline"
echo "[All] root=${ROOT_DIR}"
echo "[All] order: ratio_75 -> ratio_625 -> ratio_50"

for ratio in ratio_75 ratio_625 ratio_50; do
  echo "[All] Running ${ratio}"
  bash "${ROOT_DIR}/scripts/run_${ratio}.sh" 2>&1 | tee "${ROOT_DIR}/logs/${ratio}_pipeline.stdout.log"
done

source "${ROOT_DIR}/scripts/common.sh"
setup_env
python "${ROOT_DIR}/summarize_gpu_usage.py" --base_dir "$ROOT_DIR" \
  >"${ROOT_DIR}/logs/final_gpu_summary.stdout.log" 2>&1

echo "[All] Done. Reports are under ${ROOT_DIR}/reports"
