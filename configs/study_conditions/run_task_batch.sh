#!/usr/bin/env bash
# Batch-run all *.yml conditions for one study task, or all four tasks at once.
#
# Usage:
#   run_task_batch.sh <task|all> [smoke|one-turn|check] [--collaboration [true]] [--conditions a,b] [--jobs N] [--force] [--retry-failed] [--list-failed] [--no-wandb-upload]
#
# Features:
#   - Skips conditions that already have results (resume-friendly); use --force to re-run all.
#   - --retry-failed: re-run incomplete conditions even if partial actions.jsonl exists.
#   - --list-failed: print incomplete conditions and exit (no runs).
#   - Runs pending conditions in parallel (--jobs N, default: all conditions at once).
#   - task=maptask without --collaboration: default --conditions baseline,bandwidth_max_words_5 --jobs 1
#     (output dirs unchanged: experiments/study_conditions/maptask/<condition>/).
#   - After the batch finishes, uploads results to W&B as a directory artifact (disable with --no-wandb-upload).
#   - --collaboration: append prompts/collaboration_module.md to each agent's initial prompt;
#     results are written under <condition>_collab output folders.
#
# Examples:
#   ./configs/study_conditions/run_task_batch.sh all
#   ./configs/study_conditions/run_task_batch.sh shapefactory --retry-failed
#   ./configs/study_conditions/run_task_batch.sh all --list-failed
#   ./configs/study_conditions/run_task_batch.sh all smoke --collaboration
#   ./configs/study_conditions/shapefactory/run.sh --jobs 4 --collaboration true
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <task|all> [smoke] [--collaboration [true]] [--conditions slug1,slug2] [--jobs N] [--force] [--retry-failed] [--list-failed] [--no-wandb-upload]" >&2
  exit 2
fi

TASK="$1"
shift
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_BASE="${COLLABSIM_OUT_BASE:-$ROOT/experiments/study_conditions/${TASK}}"
cd "$ROOT"
HELPERS="$ROOT/configs/study_conditions/batch_helpers.py"

if [[ -f .env ]]; then
  set -a
  # shellcheck source=/dev/null
  source .env
  set +a
fi

CLI_EXTRA=()
MODE="full"
COLLABORATION=false
FORCE=false
RETRY_FAILED=false
LIST_FAILED=false
WANDB_UPLOAD=true
ENSURE_DIRS_ONLY=false
JOBS="${COLLABSIM_BATCH_JOBS:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-collabsim}"
CONDITIONS_CSV=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    smoke|one-turn|check)
      MODE="smoke"
      CLI_EXTRA=(--max-steps 10)
      shift
      ;;
    --collaboration)
      COLLABORATION=true
      shift
      if [[ $# -gt 0 && "$1" == true ]]; then
        shift
      elif [[ $# -gt 0 && "$1" == false ]]; then
        COLLABORATION=false
        shift
      fi
      ;;
    --collaboration=*)
      val="${1#*=}"
      if [[ "$val" == true || "$val" == 1 || "$val" == yes ]]; then
        COLLABORATION=true
      else
        COLLABORATION=false
      fi
      shift
      ;;
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    --jobs=*)
      JOBS="${1#*=}"
      shift
      ;;
    --conditions)
      CONDITIONS_CSV="$2"
      shift 2
      ;;
    --conditions=*)
      CONDITIONS_CSV="${1#*=}"
      shift
      ;;
    --force)
      FORCE=true
      shift
      ;;
    --retry-failed)
      RETRY_FAILED=true
      shift
      ;;
    --list-failed)
      LIST_FAILED=true
      shift
      ;;
    --no-wandb-upload)
      WANDB_UPLOAD=false
      shift
      ;;
    --ensure-dirs)
      ENSURE_DIRS_ONLY=true
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 <task|all> [smoke] [--collaboration [true]] [--conditions slug1,slug2] [--jobs N] [--force] [--retry-failed] [--list-failed] [--no-wandb-upload]" >&2
      exit 2
      ;;
  esac
done

list_failed_for_task() {
  local task="$1"
  local collab_flag=()
  if [[ "$COLLABORATION" == true ]]; then
    collab_flag=(--collaboration)
  fi
  uv run python "$HELPERS" list_failed --root "$ROOT" --task "$task" "${collab_flag[@]}"
}

upload_batch_results_to_wandb() {
  local upload_path="$1"
  local artifact_name="$2"
  local run_name="$3"
  local metadata_json="$4"

  if [[ "$WANDB_UPLOAD" != true ]]; then
    return 0
  fi
  if [[ ! -d "$upload_path" ]]; then
    echo "wandb_upload_skip missing_path=${upload_path}" | tee -a "${BATCH_LOG:-/dev/stderr}"
    return 0
  fi

  echo "wandb_upload_start path=${upload_path} artifact=${artifact_name} run=${run_name}" | tee -a "${BATCH_LOG:-/dev/stderr}"
  set +e
  uv run python "$HELPERS" upload_wandb \
    --path "$upload_path" \
    --project "$WANDB_PROJECT" \
    --run-name "$run_name" \
    --artifact-name "$artifact_name" \
    --metadata-json "$metadata_json" 2>&1 | tee -a "${BATCH_LOG:-/dev/stderr}"
  local upload_status="${PIPESTATUS[0]}"
  set -e
  if [[ "$upload_status" -ne 0 ]]; then
    echo "wandb_upload_failed exit=${upload_status}" | tee -a "${BATCH_LOG:-/dev/stderr}"
    return "$upload_status"
  fi
  return 0
}

if [[ "$LIST_FAILED" == true ]]; then
  if [[ "$TASK" == "all" ]]; then
    for task in shapefactory daytrader hidden_profile maptask; do
      echo "======== ${task} ========"
      list_failed_for_task "$task" || true
    done
  else
    list_failed_for_task "$TASK"
  fi
  exit 0
fi

if [[ "$TASK" == "all" ]]; then
  SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
  ALL_TASKS=(shapefactory daytrader hidden_profile maptask)
  ALL_ARGS=()
  if [[ "$MODE" == "smoke" ]]; then
    ALL_ARGS+=(smoke)
  fi
  if [[ "$COLLABORATION" == true ]]; then
    ALL_ARGS+=(--collaboration)
  fi
  if [[ "$FORCE" == true ]]; then
    ALL_ARGS+=(--force)
  fi
  if [[ "$RETRY_FAILED" == true ]]; then
    ALL_ARGS+=(--retry-failed)
  fi
  # Per-task uploads are disabled; the all-tasks batch uploads once at the end.
  ALL_ARGS+=(--no-wandb-upload)
  if [[ "$JOBS" -gt 0 ]]; then
    ALL_ARGS+=(--jobs "$JOBS")
  fi

  ALL_STAMP="$(date +%Y%m%d_%H%M%S)"
  if [[ "$MODE" == "smoke" ]]; then
    ALL_STAMP="${ALL_STAMP}_smoke10step"
  fi
  ALL_BATCH_LOG="$ROOT/experiments/study_conditions/_batch_all_${ALL_STAMP}.log"

  failures=0
  for task in "${ALL_TASKS[@]}"; do
    echo "======== ${task} ========"
    if ! BATCH_LOG="$ALL_BATCH_LOG" bash "$SCRIPT" "$task" "${ALL_ARGS[@]}"; then
      failures=$((failures + 1))
    fi
  done

  metadata_json="$(python3 - <<PY
import json
print(json.dumps({
    "batch_stamp": "${ALL_STAMP}",
    "task": "all",
    "mode": "${MODE}",
    "collaboration": "${COLLABORATION}",
    "task_failures": ${failures},
}))
PY
)"
  upload_batch_results_to_wandb \
    "$ROOT/experiments/study_conditions" \
    "study_conditions_${ALL_STAMP}" \
    "all_tasks_batch_${ALL_STAMP}" \
    "$metadata_json" || true

  echo "batch_all_end=${ALL_STAMP} task_failures=${failures} log=${ALL_BATCH_LOG}"
  exit "$failures"
fi

if [[ "$COLLABORATION" == true ]]; then
  CLI_EXTRA+=(--collaboration)
fi

# maptask default batch: two conditions, no collab, original output paths from YAML.
if [[ "$TASK" == "maptask" && "$COLLABORATION" != true && -z "$CONDITIONS_CSV" ]]; then
  CONDITIONS_CSV="baseline,bandwidth_max_words_5"
  if [[ "$JOBS" -le 0 ]]; then
    JOBS=1
  fi
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
if [[ "$MODE" == "smoke" ]]; then
  STAMP="${STAMP}_smoke10step"
fi

OUT_BASE="${COLLABSIM_OUT_BASE:-$ROOT/experiments/study_conditions/${TASK}}"
CFG_DIR="$ROOT/configs/study_conditions/${TASK}"

resolve_condition_output_dir() {
  local cfg="$1"
  local slug="$2"
  uv run python - "$cfg" "$ROOT" "$OUT_BASE" "$slug" <<'PY'
import sys
from pathlib import Path

import yaml

cfg_path = Path(sys.argv[1])
root = Path(sys.argv[2])
out_base = Path(sys.argv[3])
slug = sys.argv[4]

with cfg_path.open(encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}
logging_cfg = data.get("logging") if isinstance(data, dict) else {}
raw = logging_cfg.get("output_dir") if isinstance(logging_cfg, dict) else None
if isinstance(raw, str) and raw.strip():
    out_dir = Path(raw.strip())
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    print(out_dir.resolve())
else:
    print((out_base / slug).resolve())
PY
}

condition_out_dir() {
  local base="$1"
  local slug="$2"
  if [[ "$COLLABORATION" == true ]]; then
    echo "${base}/${slug}_collab"
  else
    echo "${base}/${slug}"
  fi
}

resolve_condition_out_dir() {
  local cfg="$1"
  local slug="$2"
  local out
  out="$(resolve_condition_output_dir "$cfg" "$slug")"
  if [[ "$COLLABORATION" == true ]]; then
    out="${out}_collab"
  fi
  echo "$out"
}

ensure_task_output_dirs() {
  shopt -s nullglob
  local cfg slug out
  mkdir -p "$OUT_BASE"
  for cfg in "$CFG_DIR"/*.yml; do
    slug="$(basename "$cfg" .yml)"
    out="$(resolve_condition_out_dir "$cfg" "$slug")"
    mkdir -p "$out"
  done
}

ensure_task_output_dirs
if [[ "$ENSURE_DIRS_ONLY" == true ]]; then
  exit 0
fi

condition_should_skip() {
  local out_dir="$1"
  local mode="$2"
  local skip_args=(should_skip "$out_dir" "$mode")
  if [[ "$RETRY_FAILED" == true ]]; then
    skip_args+=(--retry-failed)
  fi
  uv run python "$HELPERS" "${skip_args[@]}"
}

condition_in_filter() {
  local slug="$1"
  if [[ -z "$CONDITIONS_CSV" ]]; then
    return 0
  fi
  case ",${CONDITIONS_CSV}," in
    *,"${slug}",*) return 0 ;;
    *) return 1 ;;
  esac
}

run_one_condition() {
  local cfg="$1"
  local slug="$2"
  local out="$3"
  local runlog="$4"
  local status=0

  mkdir -p "$out"
  {
    echo ""
    echo "=== ${slug} ==="
    echo "config=${cfg}"
    echo "output_dir=${out}"
    echo "cli_extra=${CLI_EXTRA[*]:-}"
    echo "collaboration=${COLLABORATION}"
  } | tee -a "$runlog" "$BATCH_LOG"

  set +e
  uv run python -m src.cli "$cfg" \
    --run-id "${STAMP}_${slug}" \
    --output-dir "$out" \
    --print-actions \
    --wandb \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "${TASK}_$(basename "$out")_${STAMP}" \
    ${CLI_EXTRA+"${CLI_EXTRA[@]}"} 2>&1 | tee -a "$runlog"
  status="${PIPESTATUS[0]}"
  set -e

  if [[ "$status" -ne 0 ]]; then
    echo "FAILED ${slug} exit=${status}" | tee -a "$BATCH_LOG"
  else
    echo "OK ${slug}" | tee -a "$BATCH_LOG"
  fi
  echo "$status" > "${out}/.batch_exit_${STAMP}"
  return "$status"
}

shopt -s nullglob
configs=("$CFG_DIR"/*.yml)
if [[ ${#configs[@]} -eq 0 ]]; then
  echo "No configs in ${CFG_DIR}" >&2
  exit 1
fi

pending_cfgs=()
pending_slugs=()
pending_outs=()
skipped=0
BATCH_LOG=""
for cfg in "${configs[@]}"; do
  slug="$(basename "$cfg" .yml)"
  if ! condition_in_filter "$slug"; then
    continue
  fi
  out="$(resolve_condition_out_dir "$cfg" "$slug")"
  if [[ -z "$BATCH_LOG" ]]; then
    mkdir -p "$(dirname "$out")"
    BATCH_LOG="$(dirname "$out")/_batch_${STAMP}.log"
    {
      echo "batch_start=${STAMP}"
      echo "mode=${MODE}"
      echo "task=${TASK}"
      echo "collaboration=${COLLABORATION}"
      echo "conditions=${CONDITIONS_CSV:-all}"
      echo "retry_failed=${RETRY_FAILED}"
      echo "jobs=${JOBS}"
      echo "force=${FORCE}"
      echo "wandb_upload=${WANDB_UPLOAD}"
      echo "repo=${ROOT}"
    } | tee "$BATCH_LOG"
  fi
  if [[ "$FORCE" != true ]] && condition_should_skip "$out" "$MODE"; then
    echo "SKIP ${slug} (existing results in ${out})" | tee -a "$BATCH_LOG"
    skipped=$((skipped + 1))
    continue
  fi
  pending_cfgs+=("$cfg")
  pending_slugs+=("$slug")
  pending_outs+=("$out")
done

if [[ ${#pending_cfgs[@]} -eq 0 ]]; then
  if [[ -z "$BATCH_LOG" ]]; then
    mkdir -p "$OUT_BASE"
    BATCH_LOG="$OUT_BASE/_batch_${STAMP}.log"
    echo "batch_start=${STAMP} nothing_to_run skipped=${skipped}" | tee "$BATCH_LOG"
  else
    echo "batch_end=${STAMP} nothing_to_run skipped=${skipped} log=${BATCH_LOG}" | tee -a "$BATCH_LOG"
  fi
  metadata_json="$(python3 - <<PY
import json
print(json.dumps({
    "batch_stamp": "${STAMP}",
    "task": "${TASK}",
    "mode": "${MODE}",
    "collaboration": "${COLLABORATION}",
    "skipped": ${skipped},
    "pending": 0,
    "failures": 0,
}))
PY
)"
  upload_batch_results_to_wandb \
    "$OUT_BASE" \
    "${TASK}_study_conditions_${STAMP}" \
    "${TASK}_batch_${STAMP}" \
    "$metadata_json" || true
  exit 0
fi

if [[ "$JOBS" -le 0 ]]; then
  JOBS=${#pending_cfgs[@]}
fi

echo "pending=${#pending_cfgs[@]} skipped=${skipped} parallel_jobs=${JOBS}" | tee -a "$BATCH_LOG"

failures=0
for i in "${!pending_cfgs[@]}"; do
  cfg="${pending_cfgs[$i]}"
  slug="${pending_slugs[$i]}"
  out="${pending_outs[$i]}"
  runlog="${out}/run_${STAMP}.log"

  while (( $(jobs -r 2>/dev/null | wc -l | tr -d ' ') >= JOBS )); do
    sleep 0.25
  done

  run_one_condition "$cfg" "$slug" "$out" "$runlog" &
done
wait || true

for i in "${!pending_slugs[@]}"; do
  out="${pending_outs[$i]}"
  status_file="${out}/.batch_exit_${STAMP}"
  if [[ ! -f "$status_file" ]] || [[ "$(cat "$status_file")" != "0" ]]; then
    failures=$((failures + 1))
  fi
done

echo "batch_end=${STAMP} skipped=${skipped} failures=${failures} log=${BATCH_LOG}"

metadata_json="$(python3 - <<PY
import json
print(json.dumps({
    "batch_stamp": "${STAMP}",
    "task": "${TASK}",
    "mode": "${MODE}",
    "collaboration": "${COLLABORATION}",
    "skipped": ${skipped},
    "pending": ${#pending_cfgs[@]},
    "failures": ${failures},
}))
PY
)"
upload_batch_results_to_wandb \
  "$OUT_BASE" \
  "${TASK}_study_conditions_${STAMP}" \
  "${TASK}_batch_${STAMP}" \
  "$metadata_json" || true

if [[ "$failures" -gt 0 ]]; then
  exit 1
fi
