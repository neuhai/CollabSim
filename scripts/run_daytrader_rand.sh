#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

set -a; source .env; set +a

echo "[$(date +%H:%M:%S)] Starting 5 random-seed daytrader experiments in parallel"

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

run_exp daytrader_rand1 configs/daytrader_rand1.yml & PID1=$!
run_exp daytrader_rand2 configs/daytrader_rand2.yml & PID2=$!
run_exp daytrader_rand3 configs/daytrader_rand3.yml & PID3=$!
run_exp daytrader_rand4 configs/daytrader_rand4.yml & PID4=$!
run_exp daytrader_rand5 configs/daytrader_rand5.yml & PID5=$!

echo "PIDs: $PID1 $PID2 $PID3 $PID4 $PID5"

FAIL=0
wait $PID1 || FAIL=$((FAIL+1))
wait $PID2 || FAIL=$((FAIL+1))
wait $PID3 || FAIL=$((FAIL+1))
wait $PID4 || FAIL=$((FAIL+1))
wait $PID5 || FAIL=$((FAIL+1))

echo "[$(date +%H:%M:%S)] All done. Failures: ${FAIL}"
