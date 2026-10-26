#!/usr/bin/env bash
# Detached 1B preflight, two GPU workers, then automatic tune-only comparison.
set -euo pipefail
cf_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$cf_repo"
cf_run_id="${1:?Usage: bash scripts/run_text_realizer_cf_1b.sh RUN_ID FACTUAL_GPU ORACLE_GPU [--train-after-preflight] [--resume]}"
cf_factual_gpu="${2:?Pass the factual arm physical GPU index}"
cf_oracle_gpu="${3:?Pass the oracle arm physical GPU index}"
shift 3
[[ "$cf_run_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || { echo "Invalid run ID" >&2; exit 2; }
[[ "$cf_factual_gpu" =~ ^[0-9]+$ && "$cf_oracle_gpu" =~ ^[0-9]+$ && "$cf_factual_gpu" != "$cf_oracle_gpu" ]] || { echo "Select two distinct physical GPU indices" >&2; exit 2; }
for cf_flag in "$@"; do
    [[ "$cf_flag" == --resume || "$cf_flag" == --train-after-preflight ]] || { echo "Unknown option: $cf_flag" >&2; exit 2; }
done
cf_python="$cf_repo/exp/text_realizer/environment/.venv/bin/python"
cf_reports="$cf_repo/reports/talent/embeddinggemma_768/text-realizer-cf-1b/$cf_run_id"
cf_session="text-realizer-cf1b-$cf_run_id"
command -v tmux >/dev/null
[[ -x "$cf_python" ]]
if tmux has-session -t "$cf_session" 2>/dev/null; then
    echo "tmux session already exists: $cf_session" >&2
    exit 2
fi
mkdir -p -- "$cf_reports"
printf -v cf_command '%q ' env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$cf_python" -u -m exp.text_realizer_cf.pilot_1b --run-id "$cf_run_id" --gpus "$cf_factual_gpu" "$cf_oracle_gpu" "$@"
printf -v cf_log '%q' "$cf_reports/launcher.log"
tmux new-session -d -s "$cf_session" -c "$cf_repo" "$cf_command >> $cf_log 2>&1"
printf 'Started %s\nLog: %s/launcher.log\nController: %s/controller/status.json\nMeasured ETA after preflight: %s/eta.json\n' "$cf_session" "$cf_reports" "$cf_reports" "$cf_reports"
