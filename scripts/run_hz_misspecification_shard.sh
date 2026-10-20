#!/usr/bin/env bash
# Scheduling wrapper only: never edit the frozen YAML or original training sources.
set -euo pipefail
shard_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
shard_name="${1:-remote}"
shard_workers="${2:-4}"
if [[ ! "$shard_name" =~ ^(local|remote)$ ]] || [[ ! "$shard_workers" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Usage: bash scripts/run_hz_misspecification_shard.sh [local|remote] [workers]\n' >&2
    exit 1
fi
cd -- "$shard_repo"
shard_python="$shard_repo/.venv/bin/python"
shard_session="hz-misspec-v1-$shard_name"
shard_logs="reports/talent_motivation/hz_misspecification/hz-misspec-v1/shards/remote50-20261020/$shard_name"
command -v tmux >/dev/null
test -x "$shard_python"
if tmux has-session -t "=$shard_session" 2>/dev/null; then
    printf 'Session %s already exists; inspect it before resuming.\n' "$shard_session" >&2
    exit 1
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$shard_python" scripts/hz_misspecification_shards.py check --shard "$shard_name" --numerical
mkdir -p -- "$shard_logs"
printf -v shard_command '%q ' env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    "$shard_python" -u scripts/hz_misspecification_shards.py train --shard "$shard_name" --workers "$shard_workers"
printf -v shard_log '%q' "$shard_repo/$shard_logs/run.log"
shard_command+=" >> $shard_log 2>&1"
tmux new-session -d -s "$shard_session" -c "$shard_repo" "$shard_command"
printf 'Started %s with %s workers; one numerical thread each. No automatic evaluation.\n' "$shard_session" "$shard_workers"
