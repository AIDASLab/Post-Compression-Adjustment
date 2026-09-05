#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
setup_env
source "${SCRIPT_DIR}/run_ratio_common.sh"
run_ratio_pipeline ratio_75
