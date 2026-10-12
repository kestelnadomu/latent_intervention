#!/usr/bin/env bash
# Downstream-predictor benchmark over a finished h_Z run (CPU only, no API calls).
set -euo pipefail
ds_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ds_run_id="${1:-downstream-gemma-v1}"
ds_protocol="${2:-$ds_repo/configs/downstream_benchmark.yaml}"
if [[ ! "$ds_run_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]]; then
    printf 'Invalid run ID: %s\n' "$ds_run_id" >&2
    exit 1
fi
ds_python="$ds_repo/.venv/bin/python"
ds_report="$ds_repo/reports/talent/downstream_benchmarks/$ds_run_id"
test -x "$ds_python"
cd -- "$ds_repo"
mkdir -p -- "$ds_report"
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$ds_python" -u -m exp.benchmarks.downstream.run --run-id "$ds_run_id" --protocol "$ds_protocol" \
    2>&1 | tee -a "$ds_report/run.log"
