#!/usr/bin/env bash
# Environment-only by default; gated downloads require explicit --download-models.
set -euo pipefail
tr_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$tr_repo"
tr_uv="${TEXT_REALIZER_UV:-uv}"
if ! command -v "$tr_uv" >/dev/null 2>&1; then
    tr_uv="$tr_repo/.cache/text-realizer-tools/uv"
fi
[[ -x "$tr_uv" ]] || command -v "$tr_uv" >/dev/null
export UV_CACHE_DIR="$tr_repo/.cache/text-realizer-uv"
export UV_PYTHON_INSTALL_DIR="$tr_repo/.cache/text-realizer-python"
"$tr_uv" sync --native-tls --frozen --project exp/text_realizer/environment --python 3.12
export HF_HUB_DISABLE_XET=1
exp/text_realizer/environment/.venv/bin/python -m exp.text_realizer.setup "$@"
