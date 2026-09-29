#!/usr/bin/env bash
# Install only into the isolated Qwen environment; prefetch the pinned public model.
set -euo pipefail
qwen_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$qwen_repo"
command -v uv >/dev/null
if [[ ! -e .venv-qwen3 ]]; then
    uv venv .venv-qwen3 --python .venv/bin/python
fi
uv pip sync --python .venv-qwen3/bin/python requirements/qwen3.lock
HF_HUB_DISABLE_XET=1 .venv-qwen3/bin/python -m src.qwen3_setup
