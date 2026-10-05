#!/usr/bin/env bash
# Explicit opt-in fit/validation pilot; detached from the calling terminal.
set -euo pipefail
hz_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
hz_protocol="${1:-$hz_repo/configs/hz_benchmark.yaml}"
hz_session="hz-pilot"
hz_python="$hz_repo/.venv/bin/python"
command -v tmux >/dev/null
test -x "$hz_python"
if tmux has-session -t "=$hz_session" 2>/dev/null; then
    printf 'Session %s already exists; inspect it before launching again.\n' "$hz_session" >&2
    exit 1
fi
cd -- "$hz_repo"
"$hz_python" -m exp.benchmarks.latent_intervention.pilot --check-ready --protocol "$hz_protocol"
hz_run_id="$($hz_python -c 'import sys; from exp.benchmarks.latent_intervention.matrix import read_settings; from exp.benchmarks.latent_intervention.pilot import pilot_config; print(pilot_config(read_settings(sys.argv[1]))["run_id"])' "$hz_protocol")"
hz_report="$hz_repo/reports/talent/hz_benchmarks/$hz_run_id"
printf -v hz_command '%q ' env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    "$hz_python" -u -m exp.benchmarks.latent_intervention.pilot --run --protocol "$hz_protocol"
printf -v hz_log '%q' "$hz_report/run.log"
hz_command+=" >> $hz_log 2>&1"
tmux new-session -d -s "$hz_session" -c "$hz_repo" "$hz_command"
printf 'Started detached tmux session %s. Pilot status and logs: %s\n' "$hz_session" "$hz_report"
