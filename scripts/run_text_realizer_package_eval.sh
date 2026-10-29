#!/usr/bin/env bash
# Foreground worker: use directly, under a scheduler, or inside your own tmux session.
set -euo pipefail
tre_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$tre_repo"
tre_model="${1:?Usage: run_text_realizer_package_eval.sh MODEL cpu|cuda:0 [generate options]}"
tre_device="${2:?Pass an explicit visible device}"
shift 2
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
exec exp/text_realizer/environment/.venv/bin/python -u -m exp.text_realizer_package_eval generate \
    --model "$tre_model" --device "$tre_device" "$@"
