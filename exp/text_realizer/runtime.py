"""Conservative GPU leases; never kill processes or change driver settings."""

import os
import subprocess
from contextlib import contextmanager

from .protocol import lock
from .settings import ROOT


@contextmanager
def gpu_lease(device: str):
    if device == "cpu":
        yield
        return
    index = int(device.split(":", 1)[1]) if ":" in device else 0
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    physical = visible.split(",")[index].strip() if visible is not None else str(index)
    if not physical.isdigit() and not physical.startswith("GPU-"):
        raise ValueError("explicit physical GPU indices/UUIDs required")
    query = subprocess.run(
        [
            "nvidia-smi",
            "--id",
            physical,
            "--query-gpu=uuid,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    uuid, used = query.stdout.strip().split(",")
    with lock(ROOT / ".cache" / "text-realizer-gpu-locks" / f"{uuid.strip()}.lock"):
        if float(used.strip()) > 256:
            raise RuntimeError(
                f"GPU {physical} is already using {used.strip()} MiB; refusing to share it"
            )
        yield
