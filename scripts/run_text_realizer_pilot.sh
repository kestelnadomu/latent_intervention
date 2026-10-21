#!/usr/bin/env bash
# Detached one-seed pilot. Training requires an explicit flag after GPU preflight.
set -euo pipefail
tr_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$tr_repo"
tr_run_id="${1:?Usage: bash scripts/run_text_realizer_pilot.sh RUN_ID GPU [--train-after-preflight] [--resume]}"
tr_gpu="${2:?Pass one explicit physical GPU index}"
shift 2
[[ "$tr_run_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || { echo "Invalid run ID" >&2; exit 2; }
[[ "$tr_gpu" =~ ^[0-9]+$ ]] || { echo "Invalid physical GPU index" >&2; exit 2; }
for tr_flag in "$@"; do
    [[ "$tr_flag" == --resume || "$tr_flag" == --train-after-preflight ]] || { echo "Unknown option: $tr_flag" >&2; exit 2; }
done
tr_python="$tr_repo/exp/text_realizer/environment/.venv/bin/python"
tr_reports="$tr_repo/reports/talent/embeddinggemma_768/text-realizer/$tr_run_id"
tr_session="text-realizer-$tr_run_id"
command -v tmux >/dev/null
[[ -x "$tr_python" ]]
if tmux has-session -t "$tr_session" 2>/dev/null; then
    echo "tmux session already exists: $tr_session" >&2
    exit 2
fi
"$tr_python" -c 'from exp.text_realizer.settings import read_config, PILOT_CONFIG; from exp.text_realizer.setup import snapshots; from exp.text_realizer.pilot_protocol import baseline_material; c=read_config(PILOT_CONFIG); baseline_material(c); snapshots(c)'
mkdir -p -- "$tr_reports"
printf -v tr_command '%q ' env CUDA_VISIBLE_DEVICES="$tr_gpu" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$tr_python" -u -m exp.text_realizer pilot --run-id "$tr_run_id" --device cuda:0 "$@"
printf -v tr_log '%q' "$tr_reports/launcher.log"
tmux new-session -d -s "$tr_session" -c "$tr_repo" "$tr_command >> $tr_log 2>&1"
printf 'Started %s\nLog: %s/launcher.log\nStatus: %s/status.json\nMeasured ETA after preflight: %s/eta.json\n' "$tr_session" "$tr_reports" "$tr_reports" "$tr_reports"
