#!/usr/bin/env bash

set -uo pipefail

EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${EXP_ROOT}"
ADDITIONAL_STRATEGY_ROOT="${MODEL_ROOT:-$(dirname "${EXP_ROOT}")}"
MODEL_ROOT="${ADDITIONAL_STRATEGY_ROOT}"
SCRIPT_PATH="${EXP_ROOT}/scripts/eval_category_vllm.py"
MIRROR_SCRIPT="${EXP_ROOT}/scripts/prepare_eval_model_mirror.py"
CHECK_SCRIPT="${EXP_ROOT}/scripts/check_harness_env.py"
COLLECT_SCRIPT="${EXP_ROOT}/scripts/collect_results.py"
HARNESS_DIR="${HARNESS_DIR:-}"
export MODEL_ROOT HARNESS_DIR

VALID_METHODS=("AIMER" "MC-SMoE" "all")
DEFAULT_CATEGORIES=("ifeval" "truthfulqa_mc1" "toxigen" "wmdp" "crows_pairs_english" "winogender")
VALID_CATEGORIES=("${DEFAULT_CATEGORIES[@]}" "safety_alignment")
VALID_STRATEGIES=(
  "full_finetune" "full_kd" "only_router_finetune" "only_router_kd"
  "router_top8_expert_finetune" "router_top8_expert_kd"
  "router_top16_expert_finetune" "router_top16_expert_kd"
  "router_top50_expert_finetune" "router_top50_expert_kd"
  "router_top128_expert_finetune" "router_top128_expert_kd"
  "direct_router_logit_matching"
)

usage() {
  cat <<'USAGE'
Usage:
  bash run_eval_method.sh <METHOD...> [--force] [--dry-run] [--categories c1,c2] [--limit N]

Examples:
  bash run_eval_method.sh MC-SMoE
  bash run_eval_method.sh AIMER MC-SMoE --force
  bash run_eval_method.sh MC-SMoE --categories safety_alignment
  bash run_eval_method.sh all --dry-run

Environment:
  GPU_IDS=0,1,2,3,4,5,6,7          GPUs to schedule, default all 8.
  MAX_PARALLEL=8                   Max concurrent jobs, default number of GPU_IDS.
  HARNESS_DIR=/path/to/harness      Required lm-evaluation-harness checkout.
  MODEL_ROOT=/path/to/checkpoints   Parent containing AIMER/ and MC-SMoE/.
  PYTHON_BIN=/path/to/python        Override Python command.
  VLLM_MODEL_IMPL=transformers      Optional global vLLM model_impl override.
  USE_EXP_LOCAL_CACHE=1             Put HF/XDG caches under this lm-harness_results_gemma4/.cache.
  SKIP_PREFLIGHT=1                  Skip CPU-only environment compatibility check.
  KEEP_PYTHONPATH=1                 Do not unset PYTHONPATH before running jobs.
  SKIP_MODEL_MIRROR=1               Evaluate original model dirs directly. Not recommended for MC-SMoE.
  VLLM_ENGINE_READY_TIMEOUT_S=1800  Engine startup wait timeout, default 1800.
USAGE
}

contains() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    if [[ "${item}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

method_dir_for() {
  case "$1" in
    AIMER) echo "${ADDITIONAL_STRATEGY_ROOT}/AIMER" ;;
    MC-SMoE) echo "${ADDITIONAL_STRATEGY_ROOT}/MC-SMoE" ;;
    *) return 1 ;;
  esac
}

method_label_for() {
  case "$1" in
    AIMER) echo "AIMER_gemma4" ;;
    MC-SMoE) echo "MC-SMoE_gemma4" ;;
    *) return 1 ;;
  esac
}

METHODS=()
FORCE=0
DRY_RUN=0
LIMIT_ARG=()
FORCE_ARG=()
CATEGORIES=("${DEFAULT_CATEGORIES[@]}")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --categories)
      if [[ $# -lt 2 ]]; then
        echo "--categories requires a comma-separated value" >&2
        exit 2
      fi
      IFS=',' read -r -a CATEGORIES <<< "$2"
      shift 2
      ;;
    --limit)
      if [[ $# -lt 2 ]]; then
        echo "--limit requires a numeric value" >&2
        exit 2
      fi
      LIMIT_ARG=(--limit "$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      METHODS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#METHODS[@]} -eq 0 ]]; then
  usage
  exit 2
fi

for method in "${METHODS[@]}"; do
  if ! contains "${method}" "${VALID_METHODS[@]}"; then
    echo "Invalid METHOD: ${method}" >&2
    usage
    exit 2
  fi
done

EXPANDED_METHODS=()
for method in "${METHODS[@]}"; do
  if [[ "${method}" == "all" ]]; then
    EXPANDED_METHODS+=("AIMER" "MC-SMoE")
  else
    EXPANDED_METHODS+=("${method}")
  fi
done
METHODS=()
for method in "${EXPANDED_METHODS[@]}"; do
  if ! contains "${method}" "${METHODS[@]}"; then
    METHODS+=("${method}")
  fi
done

for category in "${CATEGORIES[@]}"; do
  if ! contains "${category}" "${VALID_CATEGORIES[@]}"; then
    echo "Invalid category: ${category}" >&2
    exit 2
  fi
done

if [[ "${FORCE}" -eq 1 ]]; then
  FORCE_ARG=(--force)
fi

mkdir -p "${EXP_ROOT}/logs_eval" "${EXP_ROOT}/tmp" "${EXP_ROOT}/.cache"
if [[ ! -d "${HARNESS_DIR}/lm_eval" ]]; then
  echo "Missing lm-evaluation-harness package: ${HARNESS_DIR}/lm_eval" >&2
  exit 1
fi

export TMPDIR="tmp"
export NLTK_DATA="${EXP_ROOT}/.cache/nltk_data"
export VLLM_RPC_BASE_PATH="tmp/vllm_rpc"
mkdir -p "${VLLM_RPC_BASE_PATH}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export TOKENIZERS_PARALLELISM="false"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-256}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-64}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"
export TORCHINDUCTOR_CACHE_DIR="${EXP_ROOT}/.cache/torchinductor"
export TRITON_CACHE_DIR="${EXP_ROOT}/.cache/triton"
export VLLM_CACHE_ROOT="${EXP_ROOT}/.cache/vllm"
export HF_MODULES_CACHE="${EXP_ROOT}/.cache/huggingface/modules"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${VLLM_CACHE_ROOT}" "${HF_MODULES_CACHE}" "${NLTK_DATA}"

if [[ "${USE_EXP_LOCAL_CACHE:-0}" == "1" ]]; then
  export HF_HOME="${EXP_ROOT}/.cache/huggingface"
  export HF_HUB_CACHE="${HF_HOME}/hub"
  export HF_DATASETS_CACHE="${HF_HOME}/datasets"
  export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
  export HF_MODULES_CACHE="${HF_HOME}/modules"
  export XDG_CACHE_HOME="${EXP_ROOT}/.cache/xdg"
  mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_MODULES_CACHE}" "${XDG_CACHE_HOME}"
fi

if [[ "${KEEP_PYTHONPATH:-0}" != "1" ]]; then
  unset PYTHONPATH
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  read -r -a PY_CMD <<< "${PYTHON_BIN}"
elif command -v python >/dev/null 2>&1; then
  PY_CMD=(python)
else
  echo "No Python command found. Activate the intended environment or set PYTHON_BIN." >&2
  exit 1
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
METHOD_LABEL="$(IFS=_; echo "${METHODS[*]}")"
RUN_LOG_ROOT="${EXP_ROOT}/logs_eval/runs/${RUN_ID}_${METHOD_LABEL}"
mkdir -p "${RUN_LOG_ROOT}"
CACHE_LOG="${RUN_LOG_ROOT}/cache_policy.log"
MIRROR_LOG="${RUN_LOG_ROOT}/model_mirrors.log"
SKIPPED_LOG="${RUN_LOG_ROOT}/skipped.log"
MANIFEST="${RUN_LOG_ROOT}/jobs_manifest.tsv"
: > "${MIRROR_LOG}"
: > "${SKIPPED_LOG}"

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "exp_root=${EXP_ROOT}"
  echo "harness_dir=${HARNESS_DIR}"
  echo "methods=${METHODS[*]}"
  echo "categories=${CATEGORIES[*]}"
  echo "Default Hugging Face model/dataset caches are not overridden unless USE_EXP_LOCAL_CACHE=1."
  echo "HF dynamic module cache is forced under this lm-harness_results_gemma4 directory to avoid writing custom code cache outside it."
  echo "Torch/Triton/vLLM runtime caches are forced under this lm-harness_results_gemma4 directory."
  echo "enable_thinking=False is forced in model_args for every category."
  echo "Model mirrors symlink large files and copy only patched config/python/processor files under lm-harness_results_gemma4/model_mirrors."
  echo "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-}"
  echo "HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-}"
  echo "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-}"
  echo "HF_HOME=${HF_HOME:-}"
  echo "HF_HUB_CACHE=${HF_HUB_CACHE:-}"
  echo "HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-}"
  echo "TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-}"
  echo "XDG_CACHE_HOME=${XDG_CACHE_HOME:-}"
  echo "HF_MODULES_CACHE=${HF_MODULES_CACHE}"
  echo "NLTK_DATA=${NLTK_DATA}"
  echo "TMPDIR=${TMPDIR}"
  echo "VLLM_RPC_BASE_PATH=${VLLM_RPC_BASE_PATH}"
  echo "note=TMPDIR and VLLM_RPC_BASE_PATH intentionally use short relative paths under lm-harness_results_gemma4 because Unix socket paths are limited."
  echo "PYTHONNOUSERSITE=${PYTHONNOUSERSITE}"
  echo "PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE}"
  echo "TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR}"
  echo "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"
  echo "VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT}"
  echo "VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S}"
} > "${CACHE_LOG}"

GPU_IDS_TEXT="${GPU_IDS:-0,1,2,3,4,5,6,7}"
GPU_IDS_TEXT="${GPU_IDS_TEXT//,/ }"
read -r -a GPU_LIST <<< "${GPU_IDS_TEXT}"
if [[ ${#GPU_LIST[@]} -eq 0 ]]; then
  echo "GPU_IDS resolved to an empty list" >&2
  exit 2
fi
MAX_PARALLEL="${MAX_PARALLEL:-${#GPU_LIST[@]}}"
if (( MAX_PARALLEL < 1 )); then
  echo "MAX_PARALLEL must be >= 1" >&2
  exit 2
fi

JOB_METHODS=()
JOB_RATIOS=()
JOB_STRATEGIES=()
JOB_CATEGORIES=()
JOB_MODELS=()
JOB_OUTPUTS=()
JOB_LOGS=()

shopt -s nullglob
for method in "${METHODS[@]}"; do
  method_dir="$(method_dir_for "${method}")"
  method_label="$(method_label_for "${method}")"
  result_root="${EXP_ROOT}/results_${method_label}"
  log_root="${EXP_ROOT}/logs_eval/${method_label}"
  mkdir -p "${result_root}" "${log_root}"
  if [[ ! -d "${method_dir}" ]]; then
    echo "Missing method directory: ${method_dir}" >&2
    exit 1
  fi

  for ratio_dir in "${method_dir}"/ratio_*; do
    [[ -d "${ratio_dir}" ]] || continue
    ratio="$(basename "${ratio_dir}")"
    for strategy_dir in "${ratio_dir}"/*; do
      [[ -d "${strategy_dir}" ]] || continue
      strategy="$(basename "${strategy_dir}")"
      contains "${strategy}" "${VALID_STRATEGIES[@]}" || continue
      if [[ ! -f "${strategy_dir}/config.json" ]]; then
        echo "Skip non-model directory without config.json: ${strategy_dir}" | tee -a "${SKIPPED_LOG}"
        continue
      fi

      eval_model_dir="${strategy_dir}"
      if [[ "${SKIP_MODEL_MIRROR:-0}" != "1" ]]; then
        if ! eval_model_dir="$("${PY_CMD[@]}" "${MIRROR_SCRIPT}" --method "${method}" --source "${strategy_dir}")"; then
          echo "Failed to prepare eval model mirror: ${strategy_dir}" | tee -a "${SKIPPED_LOG}"
          continue
        fi
        echo "${strategy_dir} -> ${eval_model_dir}" >> "${MIRROR_LOG}"
      fi

      for category in "${CATEGORIES[@]}"; do
        out_dir="${result_root}/${ratio}/${strategy}/${category}"
        log_dir="${log_root}/${ratio}/${strategy}"
        log_file="${log_dir}/${category}.log"
        if [[ -f "${out_dir}/result.json" && "${FORCE}" -eq 0 ]]; then
          echo "Skip existing result: ${out_dir}/result.json" | tee -a "${SKIPPED_LOG}"
          continue
        fi
        JOB_METHODS+=("${method}")
        JOB_RATIOS+=("${ratio}")
        JOB_STRATEGIES+=("${strategy}")
        JOB_CATEGORIES+=("${category}")
        JOB_MODELS+=("${eval_model_dir}")
        JOB_OUTPUTS+=("${out_dir}")
        JOB_LOGS+=("${log_file}")
      done
    done
  done
done
shopt -u nullglob

TOTAL_JOBS="${#JOB_MODELS[@]}"
{
  printf "method\tratio\tstrategy\tcategory\tmodel_path\toutput_dir\tlog_file\n"
  for idx in "${!JOB_MODELS[@]}"; do
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${JOB_METHODS[$idx]}" \
      "${JOB_RATIOS[$idx]}" \
      "${JOB_STRATEGIES[$idx]}" \
      "${JOB_CATEGORIES[$idx]}" \
      "${JOB_MODELS[$idx]}" \
      "${JOB_OUTPUTS[$idx]}" \
      "${JOB_LOGS[$idx]}"
  done
} > "${MANIFEST}"

echo "Methods: ${METHODS[*]}"
echo "Jobs: ${TOTAL_JOBS}"
echo "GPUs: ${GPU_LIST[*]}"
echo "Max parallel: ${MAX_PARALLEL}"
echo "Manifest: ${MANIFEST}"
echo "Cache policy log: ${CACHE_LOG}"
echo "Mirror log: ${MIRROR_LOG}"

if [[ "${TOTAL_JOBS}" -eq 0 ]]; then
  echo "No jobs to run."
  exit 0
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "Dry run only. No evaluation jobs launched."
  exit 0
fi

PREFLIGHT_LOG="${RUN_LOG_ROOT}/preflight_env.log"
if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
  if ! "${PY_CMD[@]}" "${CHECK_SCRIPT}" "${METHODS[@]}" > "${PREFLIGHT_LOG}" 2>&1; then
    cat "${PREFLIGHT_LOG}" >&2
    echo "Preflight failed before launching GPU jobs. See: ${PREFLIGHT_LOG}" >&2
    echo "Set SKIP_PREFLIGHT=1 only if you intentionally want to bypass this check." >&2
    exit 3
  fi
fi

RUNNING_PIDS=()
RUNNING_GPUS=()
RUNNING_LABELS=()
AVAILABLE_GPUS=("${GPU_LIST[@]}")
NEXT_JOB=0
FAILURES=0

launch_job() {
  local idx="$1"
  local gpu="$2"
  local method="${JOB_METHODS[$idx]}"
  local ratio="${JOB_RATIOS[$idx]}"
  local strategy="${JOB_STRATEGIES[$idx]}"
  local category="${JOB_CATEGORIES[$idx]}"
  local model_path="${JOB_MODELS[$idx]}"
  local output_dir="${JOB_OUTPUTS[$idx]}"
  local log_file="${JOB_LOGS[$idx]}"
  local label="${method}/${ratio}/${strategy}/${category}"

  mkdir -p "${output_dir}" "$(dirname "${log_file}")"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    if [[ "${method}" == "MC-SMoE" ]]; then
      export GEMMA4_STRATEGY_PATCH_VLLM_GEMMA4="1"
      if [[ -n "${PYTHONPATH:-}" ]]; then
        export PYTHONPATH="${EXP_ROOT}:${PYTHONPATH}"
      else
        export PYTHONPATH="${EXP_ROOT}"
      fi
    else
      unset GEMMA4_STRATEGY_PATCH_VLLM_GEMMA4
    fi
    if [[ "${category}" == "code" ]]; then
      export HF_ALLOW_CODE_EVAL="1"
    else
      unset HF_ALLOW_CODE_EVAL
    fi
    echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "label=${label}"
    echo "gpu=${gpu}"
    echo "model_path=${model_path}"
    echo "output_dir=${output_dir}"
    echo "HF_ALLOW_CODE_EVAL=${HF_ALLOW_CODE_EVAL:-}"
    echo "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-}"
    echo "HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-}"
    echo "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-}"
    echo "TMPDIR=${TMPDIR}"
    echo "VLLM_RPC_BASE_PATH=${VLLM_RPC_BASE_PATH}"
    echo "VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S}"
    echo "GEMMA4_STRATEGY_PATCH_VLLM_GEMMA4=${GEMMA4_STRATEGY_PATCH_VLLM_GEMMA4:-}"
    "${PY_CMD[@]}" "${SCRIPT_PATH}" \
      --method "${method}" \
      --ratio "${ratio}" \
      --strategy "${strategy}" \
      --category "${category}" \
      --model-path "${model_path}" \
      --output-dir "${output_dir}" \
      --harness-dir "${HARNESS_DIR}" \
      "${LIMIT_ARG[@]}" \
      "${FORCE_ARG[@]}"
  ) > "${log_file}" 2>&1 &

  local pid="$!"
  RUNNING_PIDS+=("${pid}")
  RUNNING_GPUS+=("${gpu}")
  RUNNING_LABELS+=("${label}")
  echo "LAUNCH pid=${pid} gpu=${gpu} ${label}"
}

reap_one() {
  local finished_pid=""
  local status=0
  set +e
  wait -n -p finished_pid
  status="$?"
  set -u

  local idx=""
  for i in "${!RUNNING_PIDS[@]}"; do
    if [[ "${RUNNING_PIDS[$i]}" == "${finished_pid}" ]]; then
      idx="$i"
      break
    fi
  done

  if [[ -z "${idx}" ]]; then
    echo "WARN unable to match finished pid=${finished_pid} status=${status}" >&2
    if [[ "${status}" -ne 0 ]]; then
      FAILURES=$((FAILURES + 1))
    fi
    return 0
  fi

  local gpu="${RUNNING_GPUS[$idx]}"
  local label="${RUNNING_LABELS[$idx]}"
  if [[ "${status}" -eq 0 ]]; then
    echo "DONE pid=${finished_pid} gpu=${gpu} ${label}"
  else
    echo "FAIL pid=${finished_pid} gpu=${gpu} status=${status} ${label}" >&2
    FAILURES=$((FAILURES + 1))
  fi

  AVAILABLE_GPUS+=("${gpu}")
  unset 'RUNNING_PIDS[idx]'
  unset 'RUNNING_GPUS[idx]'
  unset 'RUNNING_LABELS[idx]'
  RUNNING_PIDS=("${RUNNING_PIDS[@]}")
  RUNNING_GPUS=("${RUNNING_GPUS[@]}")
  RUNNING_LABELS=("${RUNNING_LABELS[@]}")
}

while (( NEXT_JOB < TOTAL_JOBS || ${#RUNNING_PIDS[@]} > 0 )); do
  while (( NEXT_JOB < TOTAL_JOBS && ${#AVAILABLE_GPUS[@]} > 0 && ${#RUNNING_PIDS[@]} < MAX_PARALLEL )); do
    gpu="${AVAILABLE_GPUS[0]}"
    AVAILABLE_GPUS=("${AVAILABLE_GPUS[@]:1}")
    launch_job "${NEXT_JOB}" "${gpu}"
    NEXT_JOB=$((NEXT_JOB + 1))
  done

  if (( ${#RUNNING_PIDS[@]} > 0 )); then
    reap_one
  fi
done

echo "All scheduled jobs finished. failures=${FAILURES}"

if [[ "${FAILURES}" -ne 0 ]]; then
  exit 1
fi

for method in "${METHODS[@]}"; do
  "${PY_CMD[@]}" "${COLLECT_SCRIPT}" --method "${method}" \
    > "${RUN_LOG_ROOT}/collect_${method}.log" 2>&1
  echo "Summary log: ${RUN_LOG_ROOT}/collect_${method}.log"
done

exit 0
