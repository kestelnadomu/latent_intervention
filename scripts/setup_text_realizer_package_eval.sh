#!/usr/bin/env bash
# Separate CPU export environment; reuse the pinned decoder environment for generation.
set -euo pipefail
tre_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$tre_repo"
tre_mode="${1:?Usage: setup_text_realizer_package_eval.sh export|decode [--download-models] [--bundle PATH] [--cache-dir PATH]}"
shift
tre_bundle="$tre_repo/models/talent_motivation/text-realizer-evaluation/package-v1"
tre_cache="$tre_repo/.cache/huggingface/hub"
tre_download=0
while (($#)); do
    case "$1" in
        --download-models) tre_download=1; shift ;;
        --bundle) tre_bundle="${2:?Missing bundle path}"; shift 2 ;;
        --cache-dir) tre_cache="${2:?Missing cache path}"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
case "$tre_mode" in
    export) tre_project="exp/text_realizer_package_eval/environment" ;;
    decode) tre_project="exp/text_realizer/environment" ;;
    *) echo "Mode must be export or decode" >&2; exit 2 ;;
esac
if [[ "$tre_mode" == export && "$tre_download" == 1 ]]; then
    echo "Model downloads use decode mode" >&2
    exit 2
fi
tre_uv="${TEXT_REALIZER_UV:-uv}"
if ! command -v "$tre_uv" >/dev/null 2>&1; then
    tre_uv="$tre_repo/.cache/text-realizer-tools/uv"
fi
[[ -x "$tre_uv" ]] || command -v "$tre_uv" >/dev/null
export UV_CACHE_DIR="$tre_repo/.cache/text-realizer-uv"
export UV_PYTHON_INSTALL_DIR="$tre_repo/.cache/text-realizer-python"
"$tre_uv" sync --native-tls --frozen --project "$tre_project" --python 3.12
if [[ "$tre_download" == 1 ]]; then
    "$tre_project/.venv/bin/python" - "$tre_bundle" "$tre_cache" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
from exp.text_realizer.setup import DECODER_FILES
from exp.text_realizer_package_eval.common import load_manifest

manifest = load_manifest(Path(sys.argv[1]))
models = {(row['spec']['name'], row['spec']['revision']) for row in manifest['models'].values()}
for name, revision in sorted(models):
    snapshot_download(repo_id=name, revision=revision, allow_patterns=list(DECODER_FILES), cache_dir=sys.argv[2], max_workers=2)
print('Pinned decoder bases/tokenizers cached. No training or generation was started.')
PY
fi
printf 'Ready: %s/.venv/bin/python\n' "$tre_project"
