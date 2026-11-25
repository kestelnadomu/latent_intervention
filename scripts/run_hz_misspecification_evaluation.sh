#!/usr/bin/env bash
# Explicit official-test evaluation only: no training or protocol changes.
set -euo pipefail
misspec_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
misspec_run="${1:-hz-misspec-v1}"
misspec_workers="${2:-4}"
if [[ ! "$misspec_run" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || [[ ! "$misspec_workers" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Usage: bash scripts/run_hz_misspecification_evaluation.sh [run-id] [workers]\n' >&2
    exit 1
fi
misspec_session="${misspec_run//./_}-evaluation"
command -v tmux >/dev/null
test -x "$misspec_repo/.venv/bin/python"
test -f "$misspec_repo/reports/talent_motivation/hz_misspecification/$misspec_run/selection.json"
if tmux has-session -t "=$misspec_session" 2>/dev/null; then
    printf 'Session %s already exists.\n' "$misspec_session" >&2
    exit 1
fi
misspec_log="$misspec_repo/reports/talent_motivation/hz_misspecification/$misspec_run/evaluation/run.log"
mkdir -p -- "$(dirname -- "$misspec_log")"
printf -v misspec_command '%q ' env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    "$misspec_repo/.venv/bin/python" -u "$misspec_repo/scripts/hz_misspecification_evaluate.py" \
    --run-id "$misspec_run" --workers "$misspec_workers"
printf -v misspec_quoted_log '%q' "$misspec_log"
misspec_command+=" >> $misspec_quoted_log 2>&1"
tmux new-session -d -s "$misspec_session" -c "$misspec_repo" "$misspec_command"
printf 'Started %s with %s evaluation workers. Log: %s\n' "$misspec_session" "$misspec_workers" "$misspec_log"
