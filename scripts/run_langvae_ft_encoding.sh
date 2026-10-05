#!/usr/bin/env bash
# The checkpoint must be downloaded/verified and encoding.yaml generated first.
set -euo pipefail

encoding_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
encoding_run_id="${1:-langvae-ft-epoch24-v1}"
if [[ ! "$encoding_run_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]]; then
    printf 'Invalid run ID: %s\n' "$encoding_run_id" >&2
    exit 1
fi
encoding_session="langvae-ft-encoding"
encoding_python="$encoding_repo/.venv/bin/python"
encoding_report_dir="$encoding_repo/reports/talent/langvae_ft/encoding/$encoding_run_id"
command -v tmux >/dev/null
test -x "$encoding_python"
test -f "$encoding_repo/models/langvae_ft/encoding.yaml"
if tmux has-session -t "=$encoding_session" 2>/dev/null; then
    printf 'Session %s already exists; inspect it first.\n' "$encoding_session" >&2
    exit 1
fi
test ! -e "$encoding_repo/data/latents/talent/langvae_ft"
test ! -e "$encoding_report_dir/status.json"
mkdir -p -- "$encoding_report_dir"
printf -v encoding_command '%q ' env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    "$encoding_python" -u -m src.encoders.langvae_ft_encoding --run-id "$encoding_run_id" --threads 4
printf -v encoding_log '%q' "$encoding_report_dir/run.log"
encoding_command+=" >> $encoding_log 2>&1"
tmux new-session -d -s "$encoding_session" -c "$encoding_repo" "$encoding_command"
printf 'Started tmux session %s.\nLogs/status: %s\n' "$encoding_session" "$encoding_report_dir"
