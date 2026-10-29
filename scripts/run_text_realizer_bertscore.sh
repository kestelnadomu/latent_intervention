#!/usr/bin/env bash
# Foreground launcher; use a scheduler or tmux to detach it.
set -euo pipefail
bs_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$bs_repo"
[[ $# == 4 ]] || { echo 'Usage: bash scripts/run_text_realizer_bertscore.sh GPU0 GPU1 GPU2 GPU3' >&2; exit 2; }
bs_gpus=("$@")
for bs_gpu in "${bs_gpus[@]}"; do
    [[ "$bs_gpu" =~ ^[0-9]+$ ]] || { echo 'Use physical GPU indices' >&2; exit 2; }
done
bs_python="$bs_repo/exp/text_realizer_bertscore/environment/.venv/bin/python"
bs_output="$bs_repo/reports/talent_motivation/text-realizer-evaluation/package-v1/bertscore-roberta-large"
mkdir -p -- "$bs_output/logs"
exec 9>"$bs_output/.launch.lock"
flock -n 9 || { echo 'BERTScore launcher already running' >&2; exit 2; }
bs_models=(256-270m 256-1b 768-270m 768-1b)
bs_pids=()
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR="$bs_repo/.cache/matplotlib"
for bs_i in 0 1 2 3; do
    bs_uuid="$(nvidia-smi --id "${bs_gpus[$bs_i]}" --query-gpu=uuid --format=csv,noheader)"
    CUDA_VISIBLE_DEVICES="$bs_uuid" "$bs_python" -u -m exp.text_realizer_bertscore score \
        --decoder "${bs_models[$bs_i]}" --device cuda:0 >> "$bs_output/logs/${bs_models[$bs_i]}.log" 2>&1 &
    bs_pids+=("$!")
done
bs_failed=0
for bs_pid in "${bs_pids[@]}"; do
    if ! wait "$bs_pid"; then bs_failed=1; fi
done
[[ "$bs_failed" == 0 ]] || { echo 'A BERTScore worker failed; see logs' >&2; exit 1; }
"$bs_python" -u -m exp.text_realizer_bertscore merge
