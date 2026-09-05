#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6}"
: "${C4_DATASET_PATH:?Set C4_DATASET_PATH to the local uncompressed C4 JSON file}"
export C4_DATASET_PATH

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export TMPDIR="${REPO_DIR}/tmp"
export WANDB_DIR="${REPO_DIR}/.cache/wandb"
export WANDB_CACHE_DIR="${REPO_DIR}/.cache/wandb"
export MPLCONFIGDIR="${REPO_DIR}/.cache/matplotlib"
export HF_DATASETS_CACHE="${REPO_DIR}/.cache/huggingface/datasets"
unset HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE

MODEL_NAME="Qwen/Qwen3-30B-A3B-Instruct-2507"
OUT_DIR="${REPO_DIR}/qwen3-30b-a3b-hcsmoe-625p-80e"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${TMPDIR}" "${WANDB_DIR}" "${MPLCONFIGDIR}" "${HF_DATASETS_CACHE}" "${LOG_DIR}"

TASK="rte"
NUM_AVG_GROUPS=80
DOMINANT="no"
SIM_BASE="expert-output"
MERGE_METHOD="zipit"
MODE="normal"
N_SENTENCES=8
TRAIN_BS=2
START_LAYER=0
GROUP_LIMIT=4
DATA_LIMIT=1000000
PARTITION=8
CLUSTER_METHOD="hierarchical"
LINKAGE_METHOD="average"
STOP_METRIC="silhouette"
INGREDIENT="act"

LOG_FILE="${LOG_DIR}/merge.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[RUN] HC-SMoE Qwen3-30B 62.5% experts: 128 -> 80 physical experts/layer"
echo "  REPO_DIR=${REPO_DIR}"
echo "  OUT_DIR=${OUT_DIR}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

python -u hcsmoe/merging-qwen.py \
  --task "${TASK}" \
  --model_name "${MODEL_NAME}" \
  --num_average_groups "${NUM_AVG_GROUPS}" \
  --dominant "${DOMINANT}" \
  --similarity_base "${SIM_BASE}" \
  --merge "${MERGE_METHOD}" \
  --mode "${MODE}" \
  --n_sentences "${N_SENTENCES}" \
  --train_batch_size "${TRAIN_BS}" \
  --start_layer "${START_LAYER}" \
  --group_limit "${GROUP_LIMIT}" \
  --data_limit "${DATA_LIMIT}" \
  --partition "${PARTITION}" \
  --cluster "${CLUSTER_METHOD}" \
  --linkage "${LINKAGE_METHOD}" \
  --hierarchical_stopping_metric "${STOP_METRIC}" \
  --ingredient "${INGREDIENT}" \
  --output_path "${OUT_DIR}"
