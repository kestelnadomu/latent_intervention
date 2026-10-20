#!/usr/bin/env bash
# Explicit detached launch of one preflighted counterfactual-exposure pilot arm.
set -euo pipefail
cf_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$cf_repo"
cf_run_id="${1:?Usage: bash scripts/run_text_realizer_cf.sh RUN_ID GPU factual|oracle [--resume]}"
cf_gpu="${2:?Pass one explicit physical GPU index}"
cf_arm="${3:?Pass factual or oracle}"
shift 3
[[ "$cf_run_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || { echo "Invalid run ID" >&2; exit 2; }
[[ "$cf_gpu" =~ ^[0-9]+$ ]] || { echo "Invalid physical GPU index" >&2; exit 2; }
[[ "$cf_arm" == factual || "$cf_arm" == oracle ]] || { echo "Invalid arm" >&2; exit 2; }
[[ "$#" -eq 0 || ( "$#" -eq 1 && "$1" == --resume ) ]] || { echo "Only --resume is allowed" >&2; exit 2; }
cf_python="$cf_repo/exp/text_realizer/environment/.venv/bin/python"
cf_reports="$cf_repo/reports/talent/embeddinggemma_768/text-realizer-cf/$cf_run_id/$cf_arm"
cf_session="text-realizer-cf-$cf_run_id-$cf_arm"
command -v tmux >/dev/null
[[ -x "$cf_python" ]]
if tmux has-session -t "$cf_session" 2>/dev/null; then
    echo "tmux session already exists: $cf_session" >&2
    exit 2
fi
mkdir -p -- "$cf_reports"
printf -v cf_command '%q ' env CUDA_VISIBLE_DEVICES="$cf_gpu" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$cf_python" -u -m exp.text_realizer_cf train --run-id "$cf_run_id" --arm "$cf_arm" --device cuda:0 "$@"
printf -v cf_log '%q' "$cf_reports/launcher.log"
tmux new-session -d -s "$cf_session" -c "$cf_repo" "$cf_command >> $cf_log 2>&1"
printf 'Started %s\nLog: %s/launcher.log\nStatus: %s/status.json\n' "$cf_session" "$cf_reports" "$cf_reports"
