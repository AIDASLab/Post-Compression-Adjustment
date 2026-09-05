#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
setup_env

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <ratio_50|ratio_625|ratio_75> <strategy>" >&2
  exit 2
fi

RATIO="$1"
STRATEGY="$2"
STUDENT_MODEL="$(student_model_for_ratio "$RATIO")"
GPUS="$(gpus_for_strategy "$STRATEGY")"
OUTPUT_DIR="${ROOT_DIR}/${RATIO}/${STRATEGY}"
GPU_LOG_CSV="${ROOT_DIR}/logs/gpu_usage/${RATIO}/${STRATEGY}.csv"
GPU_METRICS_JSON="${ROOT_DIR}/logs/metrics/${RATIO}__${STRATEGY}.json"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$GPU_LOG_CSV")"
mkdir -p "$(dirname "$GPU_METRICS_JSON")"

export CUDA_VISIBLE_DEVICES="$GPUS"
export ASSIGNED_GPU_IDS="$GPUS"

COMMON_FLAGS=()
while IFS= read -r flag; do
  [[ -n "$flag" ]] && COMMON_FLAGS+=("$flag")
done < <(python_common_flags)

ARGS=(
  "${ROOT_DIR}/strategy_train.py"
  --strategy "$STRATEGY"
  --ratio "$RATIO"
  --workspace_root "$ROOT_DIR"
  --student_model_name_or_path "$STUDENT_MODEL"
  --model_name_or_path "$STUDENT_MODEL"
  --calib_dataset_path "$DATASET_PATH"
  --output_dir "$OUTPUT_DIR"
  --num_epochs "$NUM_EPOCHS"
  --batch_size "$BATCH_SIZE"
  --max_length "$MAX_LENGTH"
  --learning_rate "$LEARNING_RATE"
  --weight_decay "$WEIGHT_DECAY"
  --warmup_ratio "$WARMUP_RATIO"
  --max_grad_norm "$MAX_GRAD_NORM"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --temperature "$TEMPERATURE"
  --max_calib_samples "$MAX_CALIB_SAMPLES"
  --logging_steps "$LOGGING_STEPS"
  --seed "$SEED"
  --num_workers "$NUM_WORKERS"
  --max_memory_per_gpu_gb "$MAX_MEMORY_PER_GPU_GB"
  --max_shard_size "$MAX_SHARD_SIZE"
  --assigned_gpu_ids "$GPUS"
  --gpu_log_csv "$GPU_LOG_CSV"
  --gpu_metrics_json "$GPU_METRICS_JSON"
  --gpu_log_interval "$GPU_LOG_INTERVAL"
)

if [[ "$STRATEGY" == "direct_router_logit_matching" || "$STRATEGY" == "only_router_kd" || "$STRATEGY" == "router_top8_expert_kd" || "$STRATEGY" == "router_top16_expert_kd" || "$STRATEGY" == "router_top50_expert_kd" || "$STRATEGY" == "router_top128_expert_kd" || "$STRATEGY" == "full_kd" ]]; then
  ARGS+=(--teacher_model_name_or_path "$TEACHER_MODEL")
fi

case "$STRATEGY" in
  router_top8_expert_kd|router_top8_expert_finetune)
    ARGS+=(--experts_per_layer 8 --selection_top_k 8 --selection_logging_steps 20)
    ;;
  router_top16_expert_kd|router_top16_expert_finetune)
    ARGS+=(--experts_per_layer 16 --selection_top_k 8 --selection_logging_steps 20)
    ;;
  router_top50_expert_kd|router_top50_expert_finetune)
    ARGS+=(--experts_per_layer 50 --selection_top_k 8 --selection_logging_steps 20)
    ;;
esac

ARGS+=("${COMMON_FLAGS[@]}")

echo "[Run] ratio=$RATIO strategy=$STRATEGY gpus=$GPUS output=$OUTPUT_DIR"
python "${ARGS[@]}"
