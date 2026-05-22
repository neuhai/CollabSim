#!/usr/bin/env bash
# Batch-run all *.yml conditions for one study task, or all four tasks at once.
#
# Usage:
#   run_task_batch.sh <task|all> [smoke|one-turn|check] [--collaboration [true]] [--jobs N] [--force]
#
# Features:
#   - Skips conditions that already have results (resume-friendly); use --force to re-run all.
#   - Runs pending conditions in parallel (--jobs N, default: all conditions at once).
#   - --collaboration: append prompts/collaboration_module.md to each agent's initial prompt.
#
# Examples:
#   ./configs/study_conditions/run_task_batch.sh all
#   ./configs/study_conditions/run_task_batch.sh shapefactory
#   ./configs/study_conditions/run_task_batch.sh all smoke --collaboration
#   ./configs/study_conditions/shapefactory/run.sh --jobs 4 --collaboration true
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <task|all> [smoke] [--collaboration [true]] [--jobs N] [--force]" >&2
  exit 2
fi

TASK="$1"
shift

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

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
JOBS="${COLLABSIM_BATCH_JOBS:-0}"

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
    --force)
      FORCE=true
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 <task|all> [smoke] [--collaboration [true]] [--jobs N] [--force]" >&2
      exit 2
      ;;
  esac
done

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
  if [[ "$JOBS" -gt 0 ]]; then
    ALL_ARGS+=(--jobs "$JOBS")
  fi

  failures=0
  for task in "${ALL_TASKS[@]}"; do
    echo "======== ${task} ========"
    if ! bash "$SCRIPT" "$task" "${ALL_ARGS[@]}"; then
      failures=$((failures + 1))
    fi
  done
  exit "$failures"
fi

if [[ "$COLLABORATION" == true ]]; then
  CLI_EXTRA+=(--collaboration)
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
if [[ "$MODE" == "smoke" ]]; then
  STAMP="${STAMP}_smoke10step"
fi

OUT_BASE="$ROOT/experiments/study_conditions/${TASK}"
CFG_DIR="$ROOT/configs/study_conditions/${TASK}"
mkdir -p "$OUT_BASE"
BATCH_LOG="$OUT_BASE/_batch_${STAMP}.log"

condition_has_results() {
  local out_dir="$1"
  local mode="$2"
  python3 - "$out_dir" "$mode" <<'PY'
import json
import os
import sys

out_dir, mode = sys.argv[1], sys.argv[2]
summary_path = os.path.join(out_dir, "run_summary.json")
if not os.path.isfile(summary_path):
    sys.exit(1)
try:
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
except (OSError, json.JSONDecodeError):
    sys.exit(1)
run_id = str(summary.get("run_id", ""))
is_smoke_run = "smoke" in run_id
if mode == "smoke":
    sys.exit(0 if is_smoke_run else 1)
if is_smoke_run:
    sys.exit(1)
if summary.get("complete") is True:
    sys.exit(0)
actions_path = os.path.join(out_dir, "actions.jsonl")
if os.path.isfile(actions_path) and os.path.getsize(actions_path) > 64:
    sys.exit(0)
sys.exit(1)
PY
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
    --wandb-project "collabsim" \
    --wandb-run-name "${TASK}_${slug}_${STAMP}" \
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

{
  echo "batch_start=${STAMP}"
  echo "mode=${MODE}"
  echo "task=${TASK}"
  echo "collaboration=${COLLABORATION}"
  echo "jobs=${JOBS}"
  echo "force=${FORCE}"
  echo "repo=${ROOT}"
} | tee "$BATCH_LOG"

shopt -s nullglob
configs=("$CFG_DIR"/*.yml)
if [[ ${#configs[@]} -eq 0 ]]; then
  echo "No configs in ${CFG_DIR}" | tee -a "$BATCH_LOG"
  exit 1
fi

pending_cfgs=()
pending_slugs=()
skipped=0
for cfg in "${configs[@]}"; do
  slug="$(basename "$cfg" .yml)"
  out="${OUT_BASE}/${slug}"
  if [[ "$FORCE" != true ]] && condition_has_results "$out" "$MODE"; then
    echo "SKIP ${slug} (existing results in ${out})" | tee -a "$BATCH_LOG"
    skipped=$((skipped + 1))
    continue
  fi
  pending_cfgs+=("$cfg")
  pending_slugs+=("$slug")
done

if [[ ${#pending_cfgs[@]} -eq 0 ]]; then
  echo "batch_end=${STAMP} nothing_to_run skipped=${skipped} log=${BATCH_LOG}"
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
  out="${OUT_BASE}/${slug}"
  runlog="${out}/run_${STAMP}.log"

  while (( $(jobs -r 2>/dev/null | wc -l | tr -d ' ') >= JOBS )); do
    sleep 0.25
  done

  run_one_condition "$cfg" "$slug" "$out" "$runlog" &
done
wait || true

for slug in "${pending_slugs[@]}"; do
  out="${OUT_BASE}/${slug}"
  status_file="${out}/.batch_exit_${STAMP}"
  if [[ ! -f "$status_file" ]] || [[ "$(cat "$status_file")" != "0" ]]; then
    failures=$((failures + 1))
  fi
done

echo "batch_end=${STAMP} skipped=${skipped} failures=${failures} log=${BATCH_LOG}"
if [[ "$failures" -gt 0 ]]; then
  exit 1
fi
