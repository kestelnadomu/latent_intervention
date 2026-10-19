#!/usr/bin/env bash
# One detached 768D run; no downstream fitting, pilot encoding or test evaluation.
set -euo pipefail
encoding_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
encoding_run_id="${1:-embeddinggemma-768-v1}"
if [[ ! "$encoding_run_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]]; then
    printf 'Invalid run ID: %s\n' "$encoding_run_id" >&2
    exit 1
fi
encoding_session="encode-talent-motivation-768"
encoding_python="$encoding_repo/.venv-embeddinggemma/bin/python"
encoding_root="$encoding_repo/reports/talent_motivation/encoding/$encoding_run_id"
command -v tmux >/dev/null
test -x "$encoding_python"
test -x "$encoding_repo/.venv/bin/python"
if tmux has-session -t "=$encoding_session" 2>/dev/null; then
    printf 'tmux session %s already exists; inspect it before restarting.\n' "$encoding_session" >&2
    exit 1
fi
cd -- "$encoding_repo"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
"$encoding_repo/.venv/bin/python" -m exp.encoding.talent_motivation validate --run-id "$encoding_run_id"
"$encoding_python" -m exp.encoding.talent_motivation preflight --run-id "$encoding_run_id"
printf -v encoding_command '%q ' env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
    "$encoding_python" -u -m exp.encoding.talent_motivation run --run-id "$encoding_run_id"
printf -v encoding_log '%q' "$encoding_root/run.log"
encoding_command+=" >> $encoding_log 2>&1"
tmux new-session -d -s "$encoding_session" -c "$encoding_repo" "$encoding_command"
printf 'Started tmux session %s.\nLogs/status: %s\n' "$encoding_session" "$encoding_root"
