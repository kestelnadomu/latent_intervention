#!/usr/bin/env bash
# Explicit opt-in: creates oracle TRAINING targets only, not any trained h_Z.
set -euo pipefail
oracle_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
oracle_run_id="${1:-oracle-targets-v1}"
if [[ ! "$oracle_run_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]]; then
    printf 'Invalid run ID: %s\n' "$oracle_run_id" >&2
    exit 1
fi
oracle_session="oracle-targets"
oracle_python="$oracle_repo/.venv-embeddinggemma/bin/python"
oracle_report_dir="$oracle_repo/reports/talent/oracle_targets/$oracle_run_id"
command -v tmux >/dev/null
test -x "$oracle_python"
if tmux has-session -t "=$oracle_session" 2>/dev/null; then
    printf 'tmux session %s already exists; inspect it before launching.\n' "$oracle_session" >&2
    exit 1
fi
cd -- "$oracle_repo"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$oracle_python" -m exp.encoding.oracle_targets --prepare --run-id "$oracle_run_id" --threads 4
printf -v oracle_command '%q ' env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    "$oracle_python" -u -m exp.encoding.oracle_targets --run --run-id "$oracle_run_id" --threads 4
printf -v oracle_log '%q' "$oracle_report_dir/launcher.log"
oracle_command+=" >> $oracle_log 2>&1"
tmux new-session -d -s "$oracle_session" -c "$oracle_repo" "$oracle_command"
printf 'Started tmux session %s.\nLogs/status: %s\n' "$oracle_session" "$oracle_report_dir"
