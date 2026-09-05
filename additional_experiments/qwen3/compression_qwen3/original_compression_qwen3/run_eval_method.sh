#!/usr/bin/env bash

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
EXP_ROOT="$(dirname "${ROOT}")"
SCRIPT_PATH="${ROOT}/scripts/eval_category_vllm.py"
PLAN_PATH="${ROOT}/scripts/plan_jobs.py"
HARNESS_DIR="${HARNESS_DIR:-}"
COMPRESSED_MODELS_ROOT="${COMPRESSED_MODELS_ROOT:-${EXP_ROOT}}"
export HARNESS_DIR COMPRESSED_MODELS_ROOT
VALID_METHODS=("original" "HC-SMoE" "REAP" "all")
DEFAULT_CATEGORIES=("ifeval" "truthfulqa_mc1" "toxigen" "wmdp" "crows_pairs_english" "winogender")
VALID_CATEGORIES=("${DEFAULT_CATEGORIES[@]}" "safety_alignment")

usage() {
  cat <<'USAGE'
Usage:
  bash run_eval_method.sh <METHOD...> [--force] [--dry-run] [--categories c1,c2] [--limit N]

Examples:
  bash run_eval_method.sh original HC-SMoE REAP
  bash run_eval_method.sh original HC-SMoE --categories ifeval
  bash run_eval_method.sh original HC-SMoE --categories safety_alignment
  bash run_eval_method.sh all --dry-run

Environment:
  GPU_IDS=0,1,2,3,4,5,6,7          GPUs to schedule, default all 8.
  MAX_PARALLEL=8                   Max concurrent jobs, default number of GPU_IDS.
  PYTHON_BIN=/path/to/python        Override Python command.
  HARNESS_DIR=/path/to/harness      Required lm-evaluation-harness checkout.
  COMPRESSED_MODELS_ROOT=/path      Parent containing HC-SMoE/ and REAP/.
  VLLM_MODEL_IMPL=transformers      Optional vLLM model_impl override.
  HC_SMOE_ENFORCE_EAGER=0           Disable HC-SMoE eager-mode fallback.
  VLLM_ENGINE_READY_TIMEOUT_S=1800  Engine startup wait timeout, default 1800.
  TMPDIR=tmp                       Forced short Python temp dir under original_compression.
  VLLM_RPC_BASE_PATH=tmp/vllm_rpc  Forced short vLLM IPC base under original_compression.
  HF model/dataset caches are not redirected; use the default Hugging Face cache.
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
    echo "Invalid method: ${method}" >&2
    usage
    exit 2
  fi
done

EXPANDED_METHODS=()
for method in "${METHODS[@]}"; do
  if [[ "${method}" == "all" ]]; then
    EXPANDED_METHODS+=("original" "HC-SMoE" "REAP")
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

mkdir -p "${ROOT}/logs_eval" "${ROOT}/tmp" "${ROOT}/.cache"
export NLTK_DATA="${ROOT}/.cache/nltk_data"
export TMPDIR="tmp"
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
export TORCHINDUCTOR_CACHE_DIR="${ROOT}/.cache/torchinductor"
export TRITON_CACHE_DIR="${ROOT}/.cache/triton"
export VLLM_CACHE_ROOT="${ROOT}/.cache/vllm"
export HF_MODULES_CACHE="${ROOT}/.cache/huggingface/modules"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${VLLM_CACHE_ROOT}" "${HF_MODULES_CACHE}" "${NLTK_DATA}"

if [[ "${USE_ORIGCOMP_LOCAL_CACHE:-0}" == "1" ]]; then
  echo "USE_ORIGCOMP_LOCAL_CACHE=1 is disabled here; leave Hugging Face model/dataset caches at their default location." >&2
  exit 2
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
LOG_ROOT="${ROOT}/logs_eval/runs/${RUN_ID}_${METHOD_LABEL}"
mkdir -p "${LOG_ROOT}"
CACHE_LOG="${LOG_ROOT}/cache_policy.log"
MANIFEST="${LOG_ROOT}/jobs_manifest.tsv"
MIRROR_LOG="${LOG_ROOT}/model_mirrors.log"

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "root=${ROOT}"
  echo "harness_dir=${HARNESS_DIR}"
  echo "compressed_models_root=${COMPRESSED_MODELS_ROOT}"
  echo "methods=${METHODS[*]}"
  echo "categories=${CATEGORIES[*]}"
  echo "Default Hugging Face model/dataset caches are not overridden by this script."
  echo "HF dynamic module cache is forced under original_compression to avoid writing custom code cache outside this directory."
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
  echo "note=TMPDIR and VLLM_RPC_BASE_PATH intentionally use short relative paths under original_compression because Unix socket paths are limited."
  echo "PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE}"
  echo "TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR}"
  echo "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"
  echo "VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT}"
  echo "VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S}"
  echo "HC_SMOE_ENFORCE_EAGER=${HC_SMOE_ENFORCE_EAGER:-<default:1>}"
} > "${CACHE_LOG}"

PLAN_ARGS=(--methods "${METHODS[@]}" --categories "${CATEGORIES[@]}" --manifest "${MANIFEST}" --mirror-log "${MIRROR_LOG}")
if [[ "${FORCE}" -eq 1 ]]; then
  PLAN_ARGS+=(--force)
fi

if ! TOTAL_JOBS="$("${PY_CMD[@]}" "${PLAN_PATH}" "${PLAN_ARGS[@]}")"; then
  echo "Failed to build job manifest" >&2
  exit 3
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

PREFLIGHT_LOG="${LOG_ROOT}/preflight_env.log"
if ! "${PY_CMD[@]}" "${ROOT}/scripts/check_env.py" "${METHODS[@]}" > "${PREFLIGHT_LOG}" 2>&1; then
  cat "${PREFLIGHT_LOG}" >&2
  echo "Preflight failed before launching GPU jobs. See: ${PREFLIGHT_LOG}" >&2
  exit 4
fi

JOB_METHODS=()
JOB_RATIOS=()
JOB_VARIANTS=()
JOB_MODEL_NAMES=()
JOB_MODEL_REFS=()
JOB_CATEGORIES=()
JOB_OUTPUTS=()
JOB_LOGS=()

while IFS=$'\t' read -r method ratio variant model_name model_ref category output_dir log_file; do
  [[ "${method}" == "method" ]] && continue
  JOB_METHODS+=("${method}")
  JOB_RATIOS+=("${ratio}")
  JOB_VARIANTS+=("${variant}")
  JOB_MODEL_NAMES+=("${model_name}")
  JOB_MODEL_REFS+=("${model_ref}")
  JOB_CATEGORIES+=("${category}")
  JOB_OUTPUTS+=("${output_dir}")
  JOB_LOGS+=("${log_file}")
done < "${MANIFEST}"

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
  local variant="${JOB_VARIANTS[$idx]}"
  local model_name="${JOB_MODEL_NAMES[$idx]}"
  local model_ref="${JOB_MODEL_REFS[$idx]}"
  local category="${JOB_CATEGORIES[$idx]}"
  local output_dir="${JOB_OUTPUTS[$idx]}"
  local log_file="${JOB_LOGS[$idx]}"
  local label="${method}/${ratio}/${variant}/${category}"

  mkdir -p "${output_dir}" "$(dirname "${log_file}")"
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
    echo "model_name=${model_name}"
    echo "model_ref=${model_ref}"
    echo "output_dir=${output_dir}"
    echo "HF_ALLOW_CODE_EVAL=${HF_ALLOW_CODE_EVAL:-}"
    echo "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-}"
    echo "HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-}"
    echo "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-}"
    echo "TMPDIR=${TMPDIR}"
    echo "VLLM_RPC_BASE_PATH=${VLLM_RPC_BASE_PATH}"
    echo "VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S}"
    echo "HC_SMOE_ENFORCE_EAGER=${HC_SMOE_ENFORCE_EAGER:-<default:1>}"
    "${PY_CMD[@]}" "${SCRIPT_PATH}" \
      --method "${method}" \
      --ratio "${ratio}" \
      --variant "${variant}" \
      --model-name "${model_name}" \
      --model-ref "${model_ref}" \
      --category "${category}" \
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

"${PY_CMD[@]}" "${ROOT}/scripts/collect_results.py" --methods "${METHODS[@]}" > "${LOG_ROOT}/collect_results.log" 2>&1
echo "Summary log: ${LOG_ROOT}/collect_results.log"
exit 0
