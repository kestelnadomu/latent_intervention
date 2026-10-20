#!/usr/bin/env bash
# Four individual fits, sequentially in detached tmux. No audit/test evaluation.
set -euo pipefail
cd "$(dirname "$0")/.."
run_id="${1:-}"
if [[ ! "$run_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
  echo "Usage: bash scripts/run_hz_ot.sh RUN_ID [--tier 1|2] [--seed N] [--protocol YAML]" >&2
  exit 2
fi
shift
session="ot-${run_id}"
command -v tmux >/dev/null
if tmux has-session -t "$session" 2>/dev/null; then
  echo "Session already exists: $session" >&2
  exit 1
fi
.venv/bin/python -m src.latent_intervention.ot preflight --all --run-id "$run_id" "$@"
log_dir="reports/talent/ot_logs"
mkdir -p "$log_dir"
printf -v command '%q ' .venv/bin/python -u -m src.latent_intervention.ot train --all --run-id "$run_id" "$@"
printf -v logfile '%q' "$log_dir/$session.log"
tmux new-session -d -s "$session" -c "$PWD" "${command} >> ${logfile} 2>&1"
echo "Detached session: $session; log: $log_dir/$session.log"
