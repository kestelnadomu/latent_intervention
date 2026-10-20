"""Optional, source-free text realization; importing this package loads no models."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import TextRealizer


def load_text_realizer(
    checkpoint: str | Path, device: str = "cpu", *, cache_dir: str | Path | None = None
) -> TextRealizer:
    """Load a verified inference bundle without reading any training data."""
    from .artifacts import load_text_realizer as load

    return load(checkpoint, device=device, cache_dir=cache_dir)


__all__ = ["load_text_realizer"]
