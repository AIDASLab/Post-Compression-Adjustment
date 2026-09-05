#!/usr/bin/env bash

set -uo pipefail

EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT="$(dirname "${EXP_ROOT}")"
REPO_ROOT="$(cd "${EXP_ROOT}/../../../.." && pwd)"
SCRIPT_PATH="${EXP_ROOT}/scripts/eval_category_vllm.py"
HARNESS_DIR="${HARNESS_DIR:-${REPO_ROOT}/common/lm-evaluation-harness}"

VALID_METHODS=("HC-SMoE" "REAP")
DEFAULT_CATEGORIES=("reasoning_math" "MC" "cot" "code" "AIME")

usage() {
  cat <<'USAGE'
Usage:
  bash run_eval_method.sh <METHOD> [--force] [--dry-run] [--categories c1,c2] [--limit N]

Examples:
  bash run_eval_method.sh HC-SMoE
  bash run_eval_method.sh HC-SMoE --categories reasoning_math,MC
  bash run_eval_method.sh REAP --dry-run

Environment:
  GPU_IDS=0,1,2,3,4,5,6,7          GPUs to schedule, default all 8.
  MAX_PARALLEL=8                   Max concurrent jobs, default number of GPU_IDS.
  HARNESS_DIR=/path/to/harness      lm-evaluation-harness path, default final/common clone.
  PYTHON_BIN=/path/to/python        Override Python command.
  VLLM_MODEL_IMPL=transformers      Optional vLLM model_impl override for custom HF models.
  USE_EXP_LOCAL_CACHE=1             Put HF/XDG caches under lm-harness_results_qwen3/.cache.
  SKIP_PREFLIGHT=1                  Skip CPU-only environment compatibility check.
  KEEP_PYTHONPATH=1                 Do not unset PYTHONPATH before running jobs.
  SKIP_MODEL_MIRROR=1               Evaluate read-only model dirs directly.
  HC_SMOE_ENFORCE_EAGER=0           Disable HC-SMoE eager-mode fallback if needed.
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

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

METHOD="$1"
shift

if ! contains "${METHOD}" "${VALID_METHODS[@]}"; then
  echo "Invalid METHOD: ${METHOD}" >&2
  usage
  exit 2
fi

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
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ "${FORCE}" -eq 1 ]]; then
  FORCE_ARG=(--force)
fi

for category in "${CATEGORIES[@]}"; do
  if ! contains "${category}" "${DEFAULT_CATEGORIES[@]}"; then
    echo "Invalid category: ${category}" >&2
    exit 2
  fi
done

METHOD_DIR="${MODEL_ROOT}/${METHOD}"
RESULT_ROOT="${EXP_ROOT}/results_${METHOD}"
LOG_ROOT="${EXP_ROOT}/logs_eval/${METHOD}"
TMP_ROOT="${EXP_ROOT}/tmp"

if [[ ! -d "${METHOD_DIR}" ]]; then
  echo "Missing method directory: ${METHOD_DIR}" >&2
  exit 1
fi

if [[ ! -d "${HARNESS_DIR}/lm_eval" ]]; then
  echo "Missing lm-evaluation-harness package: ${HARNESS_DIR}/lm_eval" >&2
  exit 1
fi

cd "${EXP_ROOT}"
mkdir -p "${RESULT_ROOT}"
mkdir -p "${LOG_ROOT}"
mkdir -p "${TMP_ROOT}"
mkdir -p "${TMP_ROOT}/vllm_rpc"

export TMPDIR="tmp"
export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export TOKENIZERS_PARALLELISM="false"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-256}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-64}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"
export VLLM_RPC_BASE_PATH="${VLLM_RPC_BASE_PATH:-tmp/vllm_rpc}"
export TORCHINDUCTOR_CACHE_DIR="${EXP_ROOT}/.cache/torchinductor"
export TRITON_CACHE_DIR="${EXP_ROOT}/.cache/triton"
export TORCH_EXTENSIONS_DIR="${EXP_ROOT}/.cache/torch_extensions"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${EXP_ROOT}/.cache/huggingface/modules}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"
mkdir -p "${TRITON_CACHE_DIR}"
mkdir -p "${TORCH_EXTENSIONS_DIR}"
mkdir -p "${HF_MODULES_CACHE}"
if [[ "${KEEP_PYTHONPATH:-0}" != "1" ]]; then
  unset PYTHONPATH
fi

CACHE_LOG="${LOG_ROOT}/cache_policy.log"
{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "Default Hugging Face model/dataset caches are not overridden unless USE_EXP_LOCAL_CACHE=1."
  echo "HF dynamic module cache is forced under lm-harness_results_qwen3 to avoid writing custom code cache outside the work dir."
  echo "vLLM RPC, TMPDIR, torchinductor, Triton, and torch extension caches are forced under lm-harness_results_qwen3."
  echo "HF_HOME=${HF_HOME:-}"
  echo "HF_HUB_CACHE=${HF_HUB_CACHE:-}"
  echo "HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-}"
  echo "TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-}"
  echo "XDG_CACHE_HOME=${XDG_CACHE_HOME:-}"
  echo "HF_MODULES_CACHE=${HF_MODULES_CACHE}"
  echo "TMPDIR=${TMPDIR}"
  echo "PYTHONNOUSERSITE=${PYTHONNOUSERSITE}"
  echo "NUMEXPR_MAX_THREADS=${NUMEXPR_MAX_THREADS}"
  echo "NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS}"
  echo "TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR}"
  echo "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"
  echo "TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR}"
  echo "VLLM_RPC_BASE_PATH=${VLLM_RPC_BASE_PATH}"
  echo "VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S}"
  echo "HC_SMOE_ENFORCE_EAGER=${HC_SMOE_ENFORCE_EAGER:-<default:1>}"
  echo "HARNESS_DIR=${HARNESS_DIR}"
} > "${CACHE_LOG}"


if [[ "${USE_EXP_LOCAL_CACHE:-0}" == "1" ]]; then
  export HF_HOME="${EXP_ROOT}/.cache/huggingface"
  export HF_HUB_CACHE="${HF_HOME}/hub"
  export HF_DATASETS_CACHE="${HF_HOME}/datasets"
  export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
  export HF_MODULES_CACHE="${HF_HOME}/modules"
  export XDG_CACHE_HOME="${EXP_ROOT}/.cache/xdg"
  mkdir -p "${HF_HUB_CACHE}"
  mkdir -p "${HF_DATASETS_CACHE}"
  mkdir -p "${TRANSFORMERS_CACHE}"
  mkdir -p "${HF_MODULES_CACHE}"
  mkdir -p "${XDG_CACHE_HOME}"
  {
    echo "USE_EXP_LOCAL_CACHE=1"
    echo "HF_HOME=${HF_HOME}"
    echo "HF_HUB_CACHE=${HF_HUB_CACHE}"
    echo "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
    echo "TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}"
    echo "HF_MODULES_CACHE=${HF_MODULES_CACHE}"
    echo "XDG_CACHE_HOME=${XDG_CACHE_HOME}"
  } >> "${CACHE_LOG}"
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  read -r -a PY_CMD <<< "${PYTHON_BIN}"
elif command -v python >/dev/null 2>&1; then
  PY_CMD=(python)
else
  echo "No Python command found. Activate the intended environment or set PYTHON_BIN." >&2
  exit 1
fi

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

JOB_RATIOS=()
JOB_STRATEGIES=()
JOB_CATEGORIES=()
JOB_MODELS=()
JOB_OUTPUTS=()
JOB_LOGS=()
MIRROR_LOG="${LOG_ROOT}/model_mirrors.log"
: > "${MIRROR_LOG}"

shopt -s nullglob
for ratio_dir in "${METHOD_DIR}"/ratio_*; do
  [[ -d "${ratio_dir}" ]] || continue
  ratio="$(basename "${ratio_dir}")"
  for strategy_dir in "${ratio_dir}"/*; do
    [[ -d "${strategy_dir}" ]] || continue
    strategy="$(basename "${strategy_dir}")"
    if [[ ! -f "${strategy_dir}/config.json" ]]; then
      echo "Skip non-model directory without config.json: ${strategy_dir}" | tee -a "${LOG_ROOT}/skipped.log"
      continue
    fi
    eval_model_dir="${strategy_dir}"
    if [[ "${SKIP_MODEL_MIRROR:-0}" != "1" ]]; then
      if ! eval_model_dir="$("${PY_CMD[@]}" "${EXP_ROOT}/scripts/prepare_eval_model_mirror.py" --method "${METHOD}" --source "${strategy_dir}")"; then
        echo "Failed to prepare eval model mirror: ${strategy_dir}" | tee -a "${LOG_ROOT}/skipped.log"
        continue
      fi
      echo "${strategy_dir} -> ${eval_model_dir}" >> "${MIRROR_LOG}"
    fi
    for category in "${CATEGORIES[@]}"; do
      out_dir="${RESULT_ROOT}/${ratio}/${strategy}/${category}"
      log_dir="${LOG_ROOT}/${ratio}/${strategy}"
      log_file="${log_dir}/${category}.log"
      if [[ -f "${out_dir}/result.json" && "${FORCE}" -eq 0 ]]; then
        echo "Skip existing result: ${out_dir}/result.json" | tee -a "${LOG_ROOT}/skipped.log"
        continue
      fi
      JOB_RATIOS+=("${ratio}")
      JOB_STRATEGIES+=("${strategy}")
      JOB_CATEGORIES+=("${category}")
      JOB_MODELS+=("${eval_model_dir}")
      JOB_OUTPUTS+=("${out_dir}")
      JOB_LOGS+=("${log_file}")
    done
  done
done
shopt -u nullglob

TOTAL_JOBS="${#JOB_MODELS[@]}"
MANIFEST="${LOG_ROOT}/jobs_manifest.tsv"
{
  printf "method\tratio\tstrategy\tcategory\tmodel_path\toutput_dir\tlog_file\n"
  for idx in "${!JOB_MODELS[@]}"; do
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${METHOD}" \
      "${JOB_RATIOS[$idx]}" \
      "${JOB_STRATEGIES[$idx]}" \
      "${JOB_CATEGORIES[$idx]}" \
      "${JOB_MODELS[$idx]}" \
      "${JOB_OUTPUTS[$idx]}" \
      "${JOB_LOGS[$idx]}"
  done
} > "${MANIFEST}"

echo "Method: ${METHOD}"
echo "Jobs: ${TOTAL_JOBS}"
echo "GPUs: ${GPU_LIST[*]}"
echo "Max parallel: ${MAX_PARALLEL}"
echo "Manifest: ${MANIFEST}"
echo "Cache policy log: ${CACHE_LOG}"

if [[ "${TOTAL_JOBS}" -eq 0 ]]; then
  echo "No jobs to run."
  exit 0
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "Dry run only. No evaluation jobs launched."
  exit 0
fi

PREFLIGHT_LOG="${LOG_ROOT}/preflight_env.log"
if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
  if ! "${PY_CMD[@]}" "${EXP_ROOT}/scripts/check_harness_env.py" --method "${METHOD}" > "${PREFLIGHT_LOG}" 2>&1; then
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
  local ratio="${JOB_RATIOS[$idx]}"
  local strategy="${JOB_STRATEGIES[$idx]}"
  local category="${JOB_CATEGORIES[$idx]}"
  local model_path="${JOB_MODELS[$idx]}"
  local output_dir="${JOB_OUTPUTS[$idx]}"
  local log_file="${JOB_LOGS[$idx]}"
  local label="${METHOD}/${ratio}/${strategy}/${category}"

  mkdir -p "${output_dir}"
  mkdir -p "$(dirname "${log_file}")"

  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
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
    echo "NUMEXPR_MAX_THREADS=${NUMEXPR_MAX_THREADS}"
    echo "VLLM_RPC_BASE_PATH=${VLLM_RPC_BASE_PATH}"
    echo "VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S}"
    echo "HC_SMOE_ENFORCE_EAGER=${HC_SMOE_ENFORCE_EAGER:-<default:1>}"
    echo "HARNESS_DIR=${HARNESS_DIR}"
    "${PY_CMD[@]}" "${SCRIPT_PATH}" \
      --method "${METHOD}" \
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

"${PY_CMD[@]}" "${EXP_ROOT}/scripts/collect_results.py" --method "${METHOD}" \
  > "${LOG_ROOT}/collect_results.log" 2>&1

echo "Summary log: ${LOG_ROOT}/collect_results.log"
exit 0
