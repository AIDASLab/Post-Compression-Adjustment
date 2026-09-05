#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export NUMEXPR_MAX_THREADS=64
export NUMEXPR_NUM_THREADS=64
export REAP_LOCAL_FILES_ONLY="${REAP_LOCAL_FILES_ONLY:-false}"
unset HF_HOME
unset HF_HUB_CACHE
unset HUGGINGFACE_HUB_CACHE
unset TRANSFORMERS_CACHE
unset XDG_CACHE_HOME
export HF_DATASETS_CACHE="${ROOT_DIR}/.cache/huggingface/datasets"

export TORCH_HOME="${ROOT_DIR}/.cache/torch"
export TORCH_EXTENSIONS_DIR="${ROOT_DIR}/.cache/torch_extensions"
export TRITON_CACHE_DIR="${ROOT_DIR}/.cache/triton"
export VLLM_CACHE_ROOT="${ROOT_DIR}/.cache/vllm"
export CUDA_CACHE_PATH="${ROOT_DIR}/.cache/cuda"
export PYTHONPYCACHEPREFIX="${ROOT_DIR}/.cache/pycache"
export TMPDIR="${ROOT_DIR}/tmp"

mkdir -p \
  "${TORCH_HOME}" \
  "${TORCH_EXTENSIONS_DIR}" \
  "${TRITON_CACHE_DIR}" \
  "${VLLM_CACHE_ROOT}" \
  "${CUDA_CACHE_PATH}" \
  "${HF_DATASETS_CACHE}" \
  "${PYTHONPYCACHEPREFIX}" \
  "${TMPDIR}" \
  "${ROOT_DIR}/logs"

MODEL_NAME="Qwen/Qwen3-30B-A3B-Instruct-2507"
PRUNING_METHOD="reap"
SEED=42
COMPRESSION_RATIO=0.5
DATASET_NAME="allenai/c4"
BATCH_SIZE=1
NUM_BATCHES=1024
OUTPUT_FILE_NAME="observations_${NUM_BATCHES}_cosine-seed_${SEED}.pt"
SERVER_LOG_FILE_NAME="logs/pruning-cli-0-keep50.log"

python src/reap/prune.py \
  --model-name "${MODEL_NAME}" \
  --dataset-name "${DATASET_NAME}" \
  --compression-ratio "${COMPRESSION_RATIO}" \
  --prune-method "${PRUNING_METHOD}" \
  --profile false \
  --vllm_port 8000 \
  --server-log-file-name "${SERVER_LOG_FILE_NAME}" \
  --do-eval false \
  --distance_measure cosine \
  --seed "${SEED}" \
  --output_file_name "${OUTPUT_FILE_NAME}" \
  --singleton_super_experts false \
  --singleton_outlier_experts false \
  --batch_size "${BATCH_SIZE}" \
  --batches_per_category "${NUM_BATCHES}" \
  --record_pruning_metrics_only true

python test_pruned_qwen_keep50.py
