#!/usr/bin/env bash
# All saved latent spaces, both g variants, fixed tuning budget and automatic report.
set -euo pipefail
benchmark_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
benchmark_run_id="${1:-g-all-encoders-v1}"
if [[ ! "$benchmark_run_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]]; then
    printf 'Invalid run ID: %s\n' "$benchmark_run_id" >&2
    exit 1
fi
benchmark_session="decoder-benchmark"
benchmark_python="$benchmark_repo/.venv/bin/python"
benchmark_report="$benchmark_repo/reports/talent/decoder_benchmarks/$benchmark_run_id"
command -v tmux >/dev/null
test -x "$benchmark_python"
if tmux has-session -t "=$benchmark_session" 2>/dev/null; then
    printf 'Session %s already exists; inspect it before launching another.\n' "$benchmark_session" >&2
    exit 1
fi
cd -- "$benchmark_repo"
"$benchmark_python" -m src.decoder_benchmark --prepare --run-id "$benchmark_run_id"
printf -v benchmark_command '%q ' env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    "$benchmark_python" -u -m src.decoder_benchmark --run --run-id "$benchmark_run_id" \
    --workers 4 --threads-per-worker 1
printf -v benchmark_log '%q' "$benchmark_report/run.log"
benchmark_command+=" >> $benchmark_log 2>&1"
tmux new-session -d -s "$benchmark_session" -c "$benchmark_repo" "$benchmark_command"
printf 'Started tmux session %s.\nReport/status/logs: %s\n' "$benchmark_session" "$benchmark_report"
