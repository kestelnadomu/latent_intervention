#!/usr/bin/env bash
# Twenty sequential fits by default. No reserved-audit or official-test evaluation.
set -euo pipefail
cd "$(dirname "$0")/.."
run_id="${1:-editor-clean-v1}"
if [[ ! "$run_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
  echo "Usage: bash scripts/run_hz_editor.sh [RUN_ID] [--tier 1|2] [--seed N] [--protocol YAML]" >&2
  exit 2
fi
if (( $# > 0 )); then shift; fi
session="editor-${run_id}"
command -v tmux >/dev/null
if tmux has-session -t "$session" 2>/dev/null; then
  echo "Session already exists: $session" >&2
  exit 1
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python -m src.latent_intervention.editor preflight --all --run-id "$run_id" "$@"
log_dir="reports/talent/editor_logs"
mkdir -p "$log_dir"
printf -v command '%q ' .venv/bin/python -u -m src.latent_intervention.editor train --all --run-id "$run_id" "$@"
printf -v logfile '%q' "$log_dir/$session.log"
tmux new-session -d -s "$session" -c "$PWD" "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; ${command} >> ${logfile} 2>&1"
echo "Detached session: $session; log: $log_dir/$session.log"
