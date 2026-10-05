#!/usr/bin/env bash
# Model access must already be approved for the user's locally authenticated HF account.
set -euo pipefail
gemma_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$gemma_repo"
command -v uv >/dev/null
if [[ ! -e .venv-embeddinggemma ]]; then
    uv venv .venv-embeddinggemma --python .venv/bin/python
fi
uv pip sync --python .venv-embeddinggemma/bin/python --require-hashes exp/encoding/requirements/embeddinggemma.lock
HF_HUB_DISABLE_XET=1 .venv-embeddinggemma/bin/python -m exp.encoding.embeddinggemma_setup
