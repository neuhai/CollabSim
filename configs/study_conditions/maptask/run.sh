#!/usr/bin/env bash
# Optional: export COLLABSIM_MODEL_PROVIDER, COLLABSIM_MODEL_NAME, COLLABSIM_MODEL_TEMPERATURE
# to override every agent's model without editing YAML (see README).
#
# Usage:
#   ./run.sh                              Run all conditions (skip those with results; parallel).
#   ./run.sh smoke                        Quick 10-step smoke per condition.
#   ./run.sh --collaboration [true]       Append collaboration_module.md to agent prompts.
#   ./run.sh --jobs 4                     Limit parallel conditions (default: all at once).
#   ./run.sh --force                      Re-run even if results exist.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../run_task_batch.sh" maptask "$@"
