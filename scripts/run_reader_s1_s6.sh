#!/usr/bin/env bash
# Only independent g: six state spaces, common search, five final seeds.
set -euo pipefail
reader_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
reader_run="${1:-g-independent-s1-s6-v1}"
if [[ ! "$reader_run" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]]; then
    printf 'Invalid run ID: %s\n' "$reader_run" >&2
    exit 1
fi
reader_session="reader-s1-s6"
reader_python="$reader_repo/.venv/bin/python"
reader_report="$reader_repo/reports/talent_motivation/reader_benchmarks/$reader_run"
command -v tmux >/dev/null
test -x "$reader_python"
if tmux has-session -t "=$reader_session" 2>/dev/null; then
    printf 'Session %s already exists.\n' "$reader_session" >&2
    exit 1
fi
cd -- "$reader_repo"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$reader_python" -m exp.benchmarks.reader.run preflight --run-id "$reader_run"
printf -v reader_command '%q ' env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    "$reader_python" -u -m exp.benchmarks.reader.run run --run-id "$reader_run"
printf -v reader_log '%q' "$reader_report/run.log"
reader_command+=" >> $reader_log 2>&1"
tmux new-session -d -s "$reader_session" -c "$reader_repo" "$reader_command"
printf 'Started %s. Logs/status: %s\n' "$reader_session" "$reader_report"
