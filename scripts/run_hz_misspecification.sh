#!/usr/bin/env bash
# Prepare/fit only. Official-test evaluation is always a separate explicit command.
set -euo pipefail
misspec_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
misspec_run="${1:-hz-misspec-v1}"
misspec_mode="${2:-full}"
if [[ ! "$misspec_run" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || [[ ! "$misspec_mode" =~ ^(full|pilot)$ ]]; then
    printf 'Usage: bash scripts/run_hz_misspecification.sh [run-id] [full|pilot]\n' >&2
    exit 1
fi
misspec_flags=(--run-id "$misspec_run")
misspec_report_run="$misspec_run"
if [[ "$misspec_mode" == pilot ]]; then
    misspec_flags+=(--pilot)
    misspec_report_run+="-pilot"
fi
misspec_session="hz-${misspec_report_run//./_}"
misspec_python="$misspec_repo/.venv/bin/python"
command -v tmux >/dev/null
test -x "$misspec_python"
if tmux has-session -t "=$misspec_session" 2>/dev/null; then
    printf 'Session %s already exists.\n' "$misspec_session" >&2
    exit 1
fi
cd -- "$misspec_repo"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$misspec_python" -m exp.benchmarks.misspecification.run preflight "${misspec_flags[@]}"
printf -v misspec_command '%q ' env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    "$misspec_python" -u -m exp.benchmarks.misspecification.run train --all "${misspec_flags[@]}"
printf -v misspec_log '%q' "$misspec_repo/reports/talent_motivation/hz_misspecification/$misspec_report_run/run.log"
misspec_command+=" >> $misspec_log 2>&1"
tmux new-session -d -s "$misspec_session" -c "$misspec_repo" "$misspec_command"
printf 'Started %s; four workers, one numerical thread each. No automatic test evaluation.\n' "$misspec_session"
