#!/usr/bin/env bash
# Launch the prepared queue in tmux; closing the terminal will not stop it.
set -euo pipefail

queue_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
queue_run_id="${1:-nomic-dimensions-v1}"
if [[ ! "$queue_run_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]]; then
    printf 'Invalid run ID: %s\n' "$queue_run_id" >&2
    exit 1
fi
queue_session="nomic-dimensions"
queue_python="$queue_repo/.venv/bin/python"
queue_report_dir="$queue_repo/reports/talent/nomic_dimensions/$queue_run_id"
command -v tmux >/dev/null
test -x "$queue_python"
if tmux has-session -t "=$queue_session" 2>/dev/null; then
    printf 'tmux session %s already exists; inspect it before starting another run.\n' "$queue_session" >&2
    exit 1
fi
cd -- "$queue_repo"
# Read-only with respect to latents; checks the saved protocol and existing files.
"$queue_python" -m src.encoders.nomic_encoding --prepare --run-id "$queue_run_id" --threads 4
printf -v queue_command '%q ' env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$queue_python" -u -m src.encoders.nomic_encoding --run --run-id "$queue_run_id" --threads 4
printf -v queue_launcher_log '%q' "$queue_report_dir/launcher.log"
queue_command+=" >> $queue_launcher_log 2>&1"
tmux new-session -d -s "$queue_session" -c "$queue_repo" "$queue_command"
printf 'Started tmux session %s.\nLogs/status: %s\n' "$queue_session" "$queue_report_dir"
