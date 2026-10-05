"""Matryoshka prefix projection: derive narrower embeddings from one full-width pass.

Slice the normalized full-width vector, then renormalize the prefix. Used by the encoding
queues to publish every supported width and by the oracle targets to verify them.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from src.encoder.protocols import (
    EMBEDDINGGEMMA_DIMENSIONS,
    NOMIC_DIMENSIONS,
    QWEN3_DIMENSIONS,
)
from src.pair_encoding import LatentArtifact

DIMENSIONS = {
    "nomic": NOMIC_DIMENSIONS,
    "qwen3": QWEN3_DIMENSIONS,
    "embeddinggemma": EMBEDDINGGEMMA_DIMENSIONS,
}



def projected_payload(source: LatentArtifact, dim: int) -> dict[str, Any]:
    """Slice a full-width normalized embedding source, then renormalize.

    Do NOT reapply layer norm to the truncated prefix. Full-vector L2 scaling
    cancels during the prefix normalization, up to floating-point rounding.
    """
    variant = source.encoder_info["encoder_variant"]
    dimensions = DIMENSIONS.get(variant, ())
    if not dimensions or source.z.shape[1] != max(dimensions) or dim not in dimensions:
        raise ValueError(
            "projection requires a full-width source and a supported dimension"
        )
    z = F.normalize(source.z[:, :dim], p=2, dim=1).contiguous()
    z_prime = F.normalize(source.z_prime[:, :dim], p=2, dim=1).contiguous()
    positions = {row_id: i for i, row_id in enumerate(source.ids)}
    for i, (row_id, identity) in enumerate(
        zip(source.test_ids, source.is_identity.tolist(), strict=True)
    ):
        if identity:
            z_prime[i] = z[positions[row_id]]
    return {
        "ids": source.ids,
        "z": z,
        "test_ids": source.test_ids,
        "z_prime": z_prime,
        "is_identity": source.is_identity.clone(),
    }



def compare_projection(
    source: LatentArtifact, target: LatentArtifact
) -> dict[str, float]:
    if source.ids != target.ids or source.test_ids != target.test_ids:
        raise ValueError("cross-dimension IDs differ")
    expected = projected_payload(source, target.z.shape[1])
    differences = {}
    for name, tensor in (("z", target.z), ("z_prime", target.z_prime)):
        differences[name] = (tensor - expected[name]).abs().max().item()
        if not torch.allclose(tensor, expected[name], atol=1e-6, rtol=1e-5):
            raise ValueError(
                f"{target.z.shape[1]}D {name} disagrees with {source.z.shape[1]}D source: {differences[name]}"
            )
    return differences

