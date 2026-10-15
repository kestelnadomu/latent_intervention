#!/usr/bin/env bash
# Queue both dimensions in one detached session. Training requires a fresh output tag.
set -euo pipefail
cd "$(dirname "$0")/.."
stage="${1:-}"
tag="${2:-}"
if [[ "$stage" != train && "$stage" != evaluate ]]; then
  echo "Usage: bash scripts/run_hz_csym_project.sh train|evaluate OUTPUT_TAG" >&2
  exit 2
fi
if [[ ! "$tag" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
  echo "Provide a simple output tag (use c_sym_project to evaluate the published models)." >&2
  exit 2
fi
session="csym-${stage}-${tag}"
log_dir="reports/talent/csym_logs"
mkdir -p "$log_dir"
printf -v command '%q ' .venv/bin/python -m src.latent_intervention.csym "$stage" --all-dimensions --tag "$tag"
printf -v logfile '%q' "$log_dir/$session.log"
tmux new-session -d -s "$session" -c "$PWD" "${command} >> ${logfile} 2>&1"
echo "Detached session: $session; log: $log_dir/$session.log"
