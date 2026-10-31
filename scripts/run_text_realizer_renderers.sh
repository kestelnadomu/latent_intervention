#!/usr/bin/env bash
# Foreground launcher (detach with tmux or a scheduler): decode the renderers' edits (with the
# operator's and with the true target states) and the unedited phi(x) with the 768-D Gemma 3
# 270M decoder, one shard of the 1,000 test units per GPU, then score all texts (text
# similarity, VADER, facts, BERTScore) on the first GPU.
# Prepare the bundle first:
#   exp/text_realizer_package_eval/environment/.venv/bin/python \
#     -m exp.text_realizer_package_eval.renderers prepare --allow-oracle-evaluation
# Rerunning resumes: complete shards are skipped, an interrupted CV is regenerated.
set -euo pipefail
rr_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$rr_repo"
[[ $# == 4 ]] || { echo 'Usage: bash scripts/run_text_realizer_renderers.sh GPU0 GPU1 GPU2 GPU3' >&2; exit 2; }
rr_bundle=models/talent_motivation/text-realizer-evaluation/renderers-v1
rr_output=reports/talent_motivation/text-realizer-evaluation/renderers-v1
rr_python=exp/text_realizer/environment/.venv/bin/python
[[ -f "$rr_bundle/manifest.json" ]] || { echo "Prepare $rr_bundle first" >&2; exit 2; }
mkdir -p -- "$rr_output/logs"
exec 9>"$rr_output/.launch.lock"
flock -n 9 || { echo 'The renderer evaluation is already running' >&2; exit 2; }
rr_uuids=()
# Share the leases used by exp.text_realizer.runtime.gpu_lease.
mkdir -p .cache/text-realizer-gpu-locks
for rr_gpu in "$@"; do
    [[ "$rr_gpu" =~ ^[0-9]+$ ]] || { echo 'Use physical GPU indices' >&2; exit 2; }
    rr_uuid="$(nvidia-smi --id "$rr_gpu" --query-gpu=uuid --format=csv,noheader)"
    exec {rr_gpu_fd}>".cache/text-realizer-gpu-locks/$rr_uuid.lock"
    flock -n "$rr_gpu_fd" || { echo "GPU $rr_gpu is reserved or listed twice" >&2; exit 2; }
    rr_used="$(nvidia-smi --id "$rr_gpu" --query-gpu=memory.used --format=csv,noheader,nounits)"
    (( rr_used <= 256 )) || { echo "GPU $rr_gpu is busy ($rr_used MiB used)" >&2; exit 2; }
    rr_uuids+=("$rr_uuid")
done
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# --frozen locks the project, not the --with overlay; pin its CUDA build explicitly.
rr_score=("${UV:-uv}" run --project exp/text_realizer/environment --frozen --offline
    --index https://download.pytorch.org/whl/cu124 --with torch==2.6.0+cu124
    --with bert-score==0.3.13 --with vaderSentiment==3.3.2)
# Check GPU execution before spending hours generating CVs.
CUDA_VISIBLE_DEVICES="${rr_uuids[0]}" "${rr_score[@]}" python -c \
    'import bert_score, torch, vaderSentiment; print("Scoring GPU ready:", torch.__version__, torch.ones(1, device="cuda:0").item())'
rr_pids=()
for rr_i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="${rr_uuids[$rr_i]}" "$rr_python" -u -m exp.text_realizer_package_eval generate \
        --bundle "$rr_bundle" --output "$rr_output/predictions" --model 768-270m --device cuda:0 \
        --shard-index "$rr_i" --num-shards 4 >> "$rr_output/logs/shard-$rr_i.log" 2>&1 &
    rr_pids+=("$!")
done
rr_failed=0
for rr_pid in "${rr_pids[@]}"; do
    if ! wait "$rr_pid"; then rr_failed=1; fi
done
[[ "$rr_failed" == 0 ]] || { echo 'A generation worker failed; see the shard logs' >&2; exit 1; }
CUDA_VISIBLE_DEVICES="${rr_uuids[0]}" "${rr_score[@]}" \
    python -u -m exp.text_realizer_package_eval.renderers score --device cuda:0 \
    >> "$rr_output/logs/score.log" 2>&1
echo "Done: docs/experiments/results/decoder_package_renderers_v1.json"
