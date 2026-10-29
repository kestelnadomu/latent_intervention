#!/usr/bin/env bash
# Matched 768-D factual-only runs; never modifies either 256-D experiment.
set -euo pipefail
tr_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$tr_repo"
tr_size="${1:?Usage: bash scripts/run_text_realizer_package_768.sh 270m|1b RUN_ID GPU [--resume]}"
tr_run_id="${2:?Pass a unique run ID}"
tr_gpu="${3:?Pass one explicit physical GPU index}"
shift 3
case "$tr_size" in
    270m) tr_config="configs/text_realizer_talent_motivation_768.yaml"; tr_folder="text-realizer" ;;
    1b) tr_config="configs/text_realizer_talent_motivation_768_gemma1b.yaml"; tr_folder="text-realizer-1b" ;;
    *) echo "Backbone must be 270m or 1b" >&2; exit 2 ;;
esac
[[ "$tr_run_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || { echo "Invalid run ID" >&2; exit 2; }
[[ "$tr_gpu" =~ ^[0-9]+$ ]] || { echo "Invalid physical GPU index" >&2; exit 2; }
for tr_flag in "$@"; do
    [[ "$tr_flag" == --resume ]] || { echo "Unknown option: $tr_flag" >&2; exit 2; }
done
tr_python="$tr_repo/exp/text_realizer/environment/.venv/bin/python"
tr_reports="$tr_repo/reports/talent_motivation/embeddinggemma_768/$tr_folder/$tr_run_id"
tr_session="text-realizer-package-768-$tr_size-$tr_run_id"
command -v tmux >/dev/null
[[ -x "$tr_python" ]]
if tmux has-session -t "$tr_session" 2>/dev/null; then
    echo "tmux session already exists: $tr_session" >&2
    exit 2
fi
mkdir -p -- "$tr_reports"
printf -v tr_command '%q ' env CUDA_VISIBLE_DEVICES="$tr_gpu" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$tr_python" -u -m exp.text_realizer_package_768 run --config "$tr_config" --run-id "$tr_run_id" --device cuda:0 --wait-for-gpu "$@"
printf -v tr_log '%q' "$tr_reports/launcher.log"
tmux new-session -d -s "$tr_session" -c "$tr_repo" "$tr_command >> $tr_log 2>&1"
printf 'Started %s\nLog: %s/launcher.log\nStatus: %s/status.json\nMeasured ETA during preflight: %s/eta.json\n' "$tr_session" "$tr_reports" "$tr_reports" "$tr_reports"
