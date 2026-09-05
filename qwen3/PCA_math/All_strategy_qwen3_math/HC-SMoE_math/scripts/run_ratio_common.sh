#!/usr/bin/env bash
set -euo pipefail

run_ratio_pipeline() {
  local ratio="$1"
  local expected_physical
  expected_physical="$(expected_physical_for_ratio "$ratio")"

  mkdir -p "${ROOT_DIR}/logs/${ratio}"
  echo "[Ratio] Starting ${ratio}"

  declare -A PIDS=()
  declare -A LOGS=()

  launch_strategy() {
    local strategy="$1"
    local log_file="${ROOT_DIR}/logs/${ratio}/${strategy}.stdout.log"
    LOGS["$strategy"]="$log_file"
    echo "[Stage] Launch ${ratio}/${strategy} log=${log_file}"
    bash "${ROOT_DIR}/scripts/run_strategy.sh" "$ratio" "$strategy" >"$log_file" 2>&1 &
    PIDS["$strategy"]=$!
  }

  wait_stage() {
    local failed=0
    local strategy
    for strategy in "$@"; do
      if ! wait "${PIDS[$strategy]}"; then
        echo "[Stage] FAILED ${ratio}/${strategy}. Last log lines:" >&2
        tail -n 80 "${LOGS[$strategy]}" >&2 || true
        failed=1
      else
        echo "[Stage] Done ${ratio}/${strategy}"
      fi
    done
    if [[ "$failed" -ne 0 ]]; then
      exit 1
    fi
  }

echo "[Stage] ${ratio} stage 1: full/top128 finetune"
launch_strategy full_finetune
launch_strategy router_top128_expert_finetune
wait_stage full_finetune router_top128_expert_finetune

echo "[Stage] ${ratio} stage 2: router/top8/top16/top50 finetune"
launch_strategy only_router_finetune
launch_strategy router_top8_expert_finetune
launch_strategy router_top16_expert_finetune
launch_strategy router_top50_expert_finetune
wait_stage only_router_finetune router_top8_expert_finetune router_top16_expert_finetune router_top50_expert_finetune

echo "[Stage] ${ratio} stage 3: router/DRLM/top128/full KD"
launch_strategy only_router_kd
launch_strategy direct_router_logit_matching
launch_strategy router_top128_expert_kd
launch_strategy full_kd
wait_stage only_router_kd direct_router_logit_matching router_top128_expert_kd full_kd

echo "[Stage] ${ratio} stage 4: top16/top50/top8 KD"
launch_strategy router_top16_expert_kd
launch_strategy router_top50_expert_kd
launch_strategy router_top8_expert_kd
wait_stage router_top16_expert_kd router_top50_expert_kd router_top8_expert_kd


  echo "[Check] Validating saved models for ${ratio}"
  if ! CUDA_VISIBLE_DEVICES="${VALIDATION_GPU:-0}" ASSIGNED_GPU_IDS="${VALIDATION_GPU:-0}" python "${ROOT_DIR}/validate_saved_models.py" \
    --base_dir "$ROOT_DIR" \
    --ratio "$ratio" \
    --expected_physical_experts "$expected_physical" \
    >"${ROOT_DIR}/logs/${ratio}/validation.stdout.log" 2>&1; then
    echo "[Check] Validation failed for ${ratio}. Last log lines:" >&2
    tail -n 80 "${ROOT_DIR}/logs/${ratio}/validation.stdout.log" >&2 || true
    exit 1
  fi

  echo "[Check] Analyzing convergence for ${ratio}"
  if ! python "${ROOT_DIR}/analyze_convergence.py" \
    --base_dir "$ROOT_DIR" \
    --ratio "$ratio" \
    >"${ROOT_DIR}/logs/${ratio}/convergence.stdout.log" 2>&1; then
    echo "[Check] Convergence analysis failed for ${ratio}. Last log lines:" >&2
    tail -n 80 "${ROOT_DIR}/logs/${ratio}/convergence.stdout.log" >&2 || true
    exit 1
  fi

  echo "[Check] Updating GPU usage summary"
  if ! python "${ROOT_DIR}/summarize_gpu_usage.py" \
    --base_dir "$ROOT_DIR" \
    >"${ROOT_DIR}/logs/${ratio}/gpu_summary.stdout.log" 2>&1; then
    echo "[Check] GPU usage summary failed for ${ratio}. Last log lines:" >&2
    tail -n 80 "${ROOT_DIR}/logs/${ratio}/gpu_summary.stdout.log" >&2 || true
    exit 1
  fi

  echo "[Ratio] Completed ${ratio}"
}
