#!/usr/bin/env bash
# Optional: export COLLABSIM_MODEL_PROVIDER, COLLABSIM_MODEL_NAME, COLLABSIM_MODEL_TEMPERATURE
# to override every agent's model without editing YAML (see README).
# Usage:
#   ./run.sh                 Run all condition configs to completion.
#   ./run.sh smoke           Same configs but --max-steps 10 each (quick I/O check; see per-condition logs).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck source=/dev/null
  source .env
  set +a
fi

CLI_EXTRA=()
MODE="full"
case "${1:-}" in
  smoke|one-turn|check)
    MODE="smoke"
    CLI_EXTRA=(--max-steps 10)
    ;;
  ""|*)
    if [[ -n "${1:-}" ]]; then
      echo "Unknown option: $1" >&2
      echo "Usage: $0 [smoke|one-turn|check]" >&2
      exit 2
    fi
    ;;
esac

STAMP="$(date +%Y%m%d_%H%M%S)"
if [[ "$MODE" == "smoke" ]]; then
  STAMP="${STAMP}_smoke10step"
fi

TASK="shapefactory"
OUT_BASE="$ROOT/experiments/study_conditions/${TASK}"
mkdir -p "$OUT_BASE"
BATCH_LOG="$OUT_BASE/_batch_${STAMP}.log"

{
  echo "batch_start=${STAMP}"
  echo "mode=${MODE}"
  echo "repo=${ROOT}"
} | tee "$BATCH_LOG"

shopt -s nullglob
for cfg in "$ROOT/configs/study_conditions/${TASK}"/*.yml; do
  slug="$(basename "$cfg" .yml)"
  out="${OUT_BASE}/${slug}"
  mkdir -p "$out"
  runlog="${out}/run_${STAMP}.log"
  {
    echo ""
    echo "=== ${slug} ==="
    echo "config=${cfg}"
    echo "output_dir=${out}"
    echo "cli_extra=${CLI_EXTRA[*]:-}"
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
done

echo "batch_end=${STAMP} log=${BATCH_LOG}"
