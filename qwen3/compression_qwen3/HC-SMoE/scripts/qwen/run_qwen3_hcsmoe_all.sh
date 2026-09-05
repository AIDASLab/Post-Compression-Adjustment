#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_DIR}"

run_check() {
  local check_py="$1"
  local model_dir="$2"
  mkdir -p "${model_dir}/logs"
  python -u "${check_py}" 2>&1 | tee -a "${model_dir}/logs/check.log"
}

bash scripts/qwen/run_qwen3_hcsmoe_50p.sh
run_check check_qwen3_hcsmoe_50p.py "${REPO_DIR}/qwen3-30b-a3b-hcsmoe-50p-64e"

bash scripts/qwen/run_qwen3_hcsmoe_625p.sh
run_check check_qwen3_hcsmoe_625p.py "${REPO_DIR}/qwen3-30b-a3b-hcsmoe-625p-80e"

bash scripts/qwen/run_qwen3_hcsmoe_75p.sh
run_check check_qwen3_hcsmoe_75p.py "${REPO_DIR}/qwen3-30b-a3b-hcsmoe-75p-96e"

echo "[DONE] Saved and checked all compressed models under ${REPO_DIR}"
