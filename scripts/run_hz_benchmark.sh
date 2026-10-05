#!/usr/bin/env bash
# Explicit opt-in launch only; preparation never starts this script.
set -euo pipefail
hz_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
hz_run_id="${1:-hz-gemma-v1}"
hz_protocol="${2:-$hz_repo/configs/hz_benchmark.yaml}"
if [[ ! "$hz_run_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]]; then
    printf 'Invalid run ID: %s\n' "$hz_run_id" >&2
    exit 1
fi
hz_session="hz-benchmark"
hz_python="$hz_repo/.venv/bin/python"
hz_report="$hz_repo/reports/talent/hz_benchmarks/$hz_run_id"
command -v tmux >/dev/null
test -x "$hz_python"
if tmux has-session -t "=$hz_session" 2>/dev/null; then
    printf 'Session %s already exists; inspect it before launching again.\n' "$hz_session" >&2
    exit 1
fi
cd -- "$hz_repo"
"$hz_python" -m src.hz.hz_benchmark --check-ready --run-id "$hz_run_id" --protocol "$hz_protocol"
printf -v hz_command '%q ' env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    "$hz_python" -u -m src.hz.hz_benchmark --run --run-id "$hz_run_id" --protocol "$hz_protocol"
printf -v hz_log '%q' "$hz_report/run.log"
hz_command+=" >> $hz_log 2>&1"
tmux new-session -d -s "$hz_session" -c "$hz_repo" "$hz_command"
printf 'Started detached tmux session %s. Reports/status/logs: %s\n' "$hz_session" "$hz_report"
