#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QWEN3_DIR="$(cd "${ROOT_DIR}/../../.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"

TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
DATASET_PATH="${DATASET_PATH:-open-r1/OpenR1-Math-220k}"

NUM_EPOCHS="${NUM_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_LENGTH="${MAX_LENGTH:-512}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_CALIB_SAMPLES="${MAX_CALIB_SAMPLES:-1024}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SEED="${SEED:-42}"
MAX_MEMORY_PER_GPU_GB="${MAX_MEMORY_PER_GPU_GB:-110}"
MAX_SHARD_SIZE="${MAX_SHARD_SIZE:-10GB}"
GPU_LOG_INTERVAL="${GPU_LOG_INTERVAL:-1.0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DETERMINISTIC="${DETERMINISTIC:-1}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export PYTHONHASHSEED="$SEED"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-64}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-64}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS_OVERRIDE:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS_OVERRIDE:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS_OVERRIDE:-8}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME:-${HOME}/.cache}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

if [[ "${HF_HOME:-}" == "$ROOT_DIR"* ]]; then
  echo "[Env] HF_HOME must not point inside $ROOT_DIR" >&2
  exit 1
fi
if [[ "${HF_HUB_CACHE:-}" == "$ROOT_DIR"* ]]; then
  echo "[Env] HF_HUB_CACHE must not point inside $ROOT_DIR" >&2
  exit 1
fi
if [[ "${HUGGINGFACE_HUB_CACHE:-}" == "$ROOT_DIR"* ]]; then
  echo "[Env] HUGGINGFACE_HUB_CACHE must not point inside $ROOT_DIR" >&2
  exit 1
fi
if [[ "${TRANSFORMERS_CACHE:-}" == "$ROOT_DIR"* ]]; then
  echo "[Env] TRANSFORMERS_CACHE must not point inside $ROOT_DIR" >&2
  exit 1
fi

setup_env() {
  unset PIP_CONSTRAINT
  unset PYTHONPATH
  export PYTHONNOUSERSITE=1
  local target_env="${CONDA_ENV_NAME:-}"
  if [[ -z "$target_env" ]]; then
    if [[ -n "${CONDA_DEFAULT_ENV:-}" && "${CONDA_DEFAULT_ENV}" != "base" ]]; then
      target_env="${CONDA_DEFAULT_ENV}"
    else
      echo "[Env] Activate a non-base conda environment or set CONDA_ENV_NAME explicitly." >&2
      return 1
    fi
  fi
  if [[ "${CONDA_DEFAULT_ENV:-}" != "$target_env" ]]; then
    if [[ -n "$CONDA_ROOT" && -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
      source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    elif [[ "$(type -t conda 2>/dev/null || true)" != "function" ]]; then
      if ! command -v conda >/dev/null 2>&1; then
        echo "[Env] conda was not found. Activate '$target_env' first or set CONDA_ROOT." >&2
        return 1
      fi
      eval "$(conda shell.bash hook)"
    fi
    conda activate "$target_env"
  fi
  unset PIP_CONSTRAINT
  unset PYTHONPATH
  export PYTHONNOUSERSITE=1
}

student_model_for_ratio() {
  case "$1" in
    ratio_50)
      echo "${QWEN3_HCSMOE_RATIO_50:-${QWEN3_DIR}/compression_qwen3/HC-SMoE/qwen3-30b-a3b-hcsmoe-50p-64e}"
      ;;
    ratio_625)
      echo "${QWEN3_HCSMOE_RATIO_625:-${QWEN3_DIR}/compression_qwen3/HC-SMoE/qwen3-30b-a3b-hcsmoe-625p-80e}"
      ;;
    ratio_75)
      echo "${QWEN3_HCSMOE_RATIO_75:-${QWEN3_DIR}/compression_qwen3/HC-SMoE/qwen3-30b-a3b-hcsmoe-75p-96e}"
      ;;
    *)
      echo "Unknown ratio: $1" >&2
      return 1
      ;;
  esac
}

expected_physical_for_ratio() {
  case "$1" in
    ratio_50) echo "64" ;;
    ratio_625) echo "80" ;;
    ratio_75) echo "96" ;;
    *)
      echo "Unknown ratio: $1" >&2
      return 1
      ;;
  esac
}

gpus_for_strategy() {
  case "$1" in
    full_finetune) echo "${GPU_PAIR_45:-4,5}" ;;
    router_top128_expert_finetune) echo "${GPU_PAIR_67:-6,7}" ;;
    only_router_finetune) echo "${GPU_PAIR_01:-0,1}" ;;
    router_top8_expert_finetune) echo "${GPU_PAIR_23:-2,3}" ;;
    router_top16_expert_finetune) echo "${GPU_PAIR_45:-4,5}" ;;
    router_top50_expert_finetune) echo "${GPU_PAIR_67:-6,7}" ;;
    only_router_kd) echo "${GPU_PAIR_01:-0,1}" ;;
    direct_router_logit_matching) echo "${GPU_PAIR_23:-2,3}" ;;
    router_top8_expert_kd) echo "${GPU_PAIR_45:-4,5}" ;;
    router_top16_expert_kd) echo "${GPU_PAIR_01:-0,1}" ;;
    router_top50_expert_kd) echo "${GPU_PAIR_23:-2,3}" ;;
    router_top128_expert_kd) echo "${GPU_PAIR_45:-4,5}" ;;
    full_kd) echo "${GPU_PAIR_67:-6,7}" ;;
    *)
      echo "Unknown strategy: $1" >&2
      return 1
      ;;
  esac
}

python_common_flags() {
  local flags=()
  if [[ "$DETERMINISTIC" == "1" ]]; then
    flags+=(--deterministic)
  fi
  if [[ "$ALLOW_OVERWRITE" == "1" ]]; then
    flags+=(--allow_overwrite)
  fi
  if [[ "$GRADIENT_CHECKPOINTING" == "1" ]]; then
    flags+=(--gradient_checkpointing)
  fi
  printf '%s\n' "${flags[@]}"
}
