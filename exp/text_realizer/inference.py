"""Source-free, ID-aligned vector input for standalone decoder inference."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import torch


def read_latents(
    path: str | Path, *, expected_ids: Sequence[int] | None = None
) -> tuple[list[int], torch.Tensor, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {
        "ids",
        "z",
        "encoder",
        "provenance",
    }:
        raise ValueError(
            "inference input must contain exactly ids, z, encoder, provenance; paired artifacts forbidden"
        )
    ids = [int(value) for value in payload["ids"]]
    z = payload["z"]
    if (
        len(ids) != len(set(ids))
        or not isinstance(z, torch.Tensor)
        or not z.is_floating_point()
        or z.shape != (len(ids), 768)
        or not torch.isfinite(z).all()
    ):
        raise ValueError("invalid or misaligned floating-point 768-D input vectors")
    if expected_ids is not None and set(ids) != set(expected_ids):
        raise ValueError("edited vectors must cover exactly the official-test IDs")
    return ids, z.float(), payload
