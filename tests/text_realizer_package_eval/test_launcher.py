"""The launcher must reject unavailable GPUs and broken scoring before generation."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("gpus", "used", "score_exit", "message"),
    [
        (["0", "0", "2", "3"], "1", "0", "reserved or listed twice"),
        (["0", "1", "2", "3"], "512", "0", "is busy"),
        (["0", "1", "2", "3"], "1", "7", "scoring failed"),
    ],
)
def test_launcher_refuses_before_workers(tmp_path, gpus, used, score_exit, message):
    root = Path(__file__).resolve().parents[2]
    script = tmp_path / "scripts/run_text_realizer_renderers.sh"
    script.parent.mkdir()
    shutil.copyfile(root / "scripts/run_text_realizer_renderers.sh", script)
    manifest = (
        tmp_path
        / "models/talent_motivation/text-realizer-evaluation/renderers-v1/manifest.json"
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    commands = {
        "bin/nvidia-smi": 'case "$*" in *memory.used*) echo "$GPU_USED";; *) echo "GPU-$2";; esac',
        "bin/uv": 'echo "scoring failed" >&2; exit "$SCORE_EXIT"',
        "exp/text_realizer/environment/.venv/bin/python": 'touch "$WORKER_MARKER"',
    }
    for name, command in commands.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"#!/usr/bin/env bash\n{command}\n")
        path.chmod(0o755)
    result = subprocess.run(
        ["bash", str(script), *gpus],
        env={
            **os.environ,
            "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}",
            "UV": str(tmp_path / "bin/uv"),
            "GPU_USED": used,
            "SCORE_EXIT": score_exit,
            "WORKER_MARKER": str(tmp_path / "worker-started"),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert message in result.stderr
    assert not (tmp_path / "worker-started").exists()
