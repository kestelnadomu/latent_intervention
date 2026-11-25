#!/usr/bin/env bash
set -euo pipefail
allocation_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
allocation_role="${1:-}"
allocation_workers="${2:-4}"
if [[ ! "$allocation_role" =~ ^(keep38|return12)$ ]] || [[ ! "$allocation_workers" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Usage: bash scripts/run_hz_misspecification_return12.sh [keep38|return12] [workers]\n' >&2
    exit 1
fi
cd -- "$allocation_repo"
allocation_python="$allocation_repo/.venv/bin/python"
allocation_session="hz-misspec-v1-return12"
if [[ "$allocation_role" == keep38 ]]; then
    # Reuse the old session name so the historical full-50 launcher refuses a competing launch.
    allocation_session="hz-misspec-v1-remote"
fi
allocation_logs="reports/talent_motivation/hz_misspecification/hz-misspec-v1/shards/remote50-20261020/return12-20261022/$allocation_role"
command -v tmux >/dev/null
test -x "$allocation_python"
if tmux has-session -t "=$allocation_session" 2>/dev/null; then
    printf 'Session %s already exists; inspect it before resuming.\n' "$allocation_session" >&2
    exit 1
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$allocation_python" scripts/hz_misspecification_return12.py check --role "$allocation_role" --numerical
mkdir -p -- "$allocation_logs"
printf -v allocation_command '%q ' env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    "$allocation_python" -u scripts/hz_misspecification_return12.py train --role "$allocation_role" --workers "$allocation_workers"
printf -v allocation_log '%q' "$allocation_repo/$allocation_logs/run.log"
allocation_command+=" >> $allocation_log 2>&1"
tmux new-session -d -s "$allocation_session" -c "$allocation_repo" "$allocation_command"
printf 'Started %s: role=%s, workers=%s, one numerical thread each.\n' "$allocation_session" "$allocation_role" "$allocation_workers"
