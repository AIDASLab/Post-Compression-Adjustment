#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

echo "[1/3] Running Qwen3 REAP C4 pruning: keep 50% experts"
bash "${ROOT_DIR}/run_qwen3_reap_c4_keep50.sh"

echo "[2/3] Running Qwen3 REAP C4 pruning: keep 62.5% experts"
bash "${ROOT_DIR}/run_qwen3_reap_c4_keep62_5.sh"

echo "[3/3] Running Qwen3 REAP C4 pruning: keep 75% experts"
bash "${ROOT_DIR}/run_qwen3_reap_c4_keep75.sh"

echo "All Qwen3 REAP C4 pruning runs and checks completed."
