#!/usr/bin/env bash
# Pilot batch 07 — hidden_profile: discussion_duration_sec 300→600; all 4 experiments in parallel.
# Results: experiments/p07_*/  |  Logs: experiments/p07_*/run.log

set -euo pipefail
cd "$(dirname "$0")/.."

set -a; source .env; set +a

START=$(date +%Y%m%d_%H%M%S)
echo "[${START}] Starting pilot_07 — 4 experiments in parallel"
echo "  Deployment : ${AZURE_OPENAI_DEPLOYMENT}"
echo ""

run_exp() {
  local name=$1 config=$2
  mkdir -p "experiments/${name}"
  python3 -m src.cli "${config}" \
    --run-id "${name}" \
    --output-dir "experiments/${name}" \
    --print-actions \
    > "experiments/${name}/run.log" 2>&1
  local exit_code=$?
  if [ $exit_code -eq 0 ]; then
    echo "[$(date +%H:%M:%S)] DONE    ${name}"
  else
    echo "[$(date +%H:%M:%S)] FAILED  ${name} (exit ${exit_code})"
  fi
  return $exit_code
}

run_exp p07_maptask         configs/maptask_example.yml                & PID_MAP=$!
run_exp p07_daytrader       configs/daytrader_example.yml              & PID_DAY=$!
run_exp p07_shapefactory    configs/shapefactory_time_mode_example.yml & PID_SF=$!
run_exp p07_hiddenprofile   configs/hidden_profile_azure.yml           & PID_HP=$!

echo "PIDs — maptask:${PID_MAP}  daytrader:${PID_DAY}  shapefactory:${PID_SF}  hiddenprofile:${PID_HP}"
echo ""

FAIL=0
wait $PID_MAP  || FAIL=$((FAIL+1))
wait $PID_DAY  || FAIL=$((FAIL+1))
wait $PID_SF   || FAIL=$((FAIL+1))
wait $PID_HP   || FAIL=$((FAIL+1))

echo ""
echo "[$(date +%H:%M:%S)] All done. Failures: ${FAIL}"
echo ""
echo "Results:"
for name in p07_maptask p07_daytrader p07_shapefactory p07_hiddenprofile; do
  dir="experiments/${name}"
  if [ -f "${dir}/metrics.json" ]; then
    echo "  ${name}: metrics.json ✓"
  else
    echo "  ${name}: metrics.json MISSING"
  fi
  if [ -f "${dir}/trace.jsonl" ]; then
    echo "  ${name}: trace.jsonl ✓"
  else
    echo "  ${name}: trace.jsonl MISSING"
  fi
done
exit $FAIL
