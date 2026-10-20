#!/usr/bin/env bash
# Explicit detached campaign; preparation and GPU preflight must already have passed.
set -euo pipefail
tr_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$tr_repo"
tr_run_id="${1:?Usage: bash scripts/run_text_realizer.sh RUN_ID GPU [GPU ...]}"
shift
[[ "$tr_run_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || { echo "Invalid run ID" >&2; exit 2; }
[[ "$#" -gt 0 ]] || { echo "Pass explicit physical GPU indices" >&2; exit 2; }
tr_python="$tr_repo/exp/text_realizer/environment/.venv/bin/python"
tr_reports="$tr_repo/reports/talent/embeddinggemma_768/text-realizer/$tr_run_id"
tr_session="text-realizer-$tr_run_id"
command -v tmux >/dev/null
[[ -x "$tr_python" ]]
if tmux has-session -t "$tr_session" 2>/dev/null; then
    echo "tmux session already exists: $tr_session" >&2
    exit 2
fi
"$tr_python" -c 'from exp.text_realizer.settings import read_config; from exp.text_realizer.data import prepared_path; from exp.text_realizer.protocol import read_json, identity; c=read_config(); p=prepared_path(c); r=read_json(p/"preflight.json"); assert r["ready"] and r["identity"] == identity(c,p), "Run current preflight first"'
mkdir -p -- "$tr_reports"
printf -v tr_command '%q ' env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$tr_python" -u -m exp.text_realizer campaign --run-id "$tr_run_id" --gpus "$@"
printf -v tr_log '%q' "$tr_reports/launcher.log"
tmux new-session -d -s "$tr_session" -c "$tr_repo" "$tr_command >> $tr_log 2>&1"
printf 'Started %s\nLog: %s/launcher.log\nStatus: %s/status.json\n' "$tr_session" "$tr_reports" "$tr_reports"
