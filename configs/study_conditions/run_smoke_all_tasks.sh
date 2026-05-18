#!/usr/bin/env bash
# Run smoke mode (10 controller steps per config) for all four study tasks in sequence.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
for script in \
  "$ROOT/configs/study_conditions/shapefactory/run.sh" \
  "$ROOT/configs/study_conditions/daytrader/run.sh" \
  "$ROOT/configs/study_conditions/hidden_profile/run.sh" \
  "$ROOT/configs/study_conditions/maptask/run.sh"
do
  echo "======== $(basename "$(dirname "$script")") smoke ========"
  bash "$script" smoke
done
