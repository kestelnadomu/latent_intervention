"""Portable trainable-only checkpoints and data-free reload verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from src.artifact_io import save_torch, sha256_file

from .generation import generate
from .model import TextRealizer, build_model, cached_decoder


def tensor_digest(state: dict) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def metadata_digest(metadata: dict) -> str:
    return hashlib.sha256(
        json.dumps(metadata, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


@torch.inference_mode()
def probe(model) -> dict:
    z = torch.randn(2, model.latent_dim, generator=torch.Generator().manual_seed(1729))
    z = torch.nn.functional.normalize(z, dim=1)
    previous = model.training
    model.eval()
    try:
        prefix = model.prefix(z)
        bos = torch.full((2, 1), model.bos_id, device=model.device, dtype=torch.long)
        embeds = torch.cat(
            (prefix, model.language_model.get_input_embeddings()(bos)), 1
        )
        logits = (
            model.language_model(inputs_embeds=embeds, use_cache=False)
            .logits[:, -1, :64]
            .cpu()
        )
        maximum = min(32, model.generation["max_new_tokens"])
        rows = generate(model, z, max_new_tokens=maximum, batch_size=1)
        return {"z": z, "logits": logits, "tokens": [row["token_ids"] for row in rows]}
    finally:
        model.train(previous)


def save_checkpoint(path: Path, model, *, encoder: dict, provenance: dict) -> None:
    weights = model.trainable_state()
    check = probe(model)
    metadata = {
        "format_version": 1,
        "architecture": "gemma-latent-prefix-v1",
        "spec": model.spec,
        "generation": model.generation,
        "encoder": encoder,
        "provenance": provenance,
        "weights_sha256": tensor_digest(weights),
        "probe_sha256": tensor_digest({"z": check["z"], "logits": check["logits"]}),
        "probe_tokens": check["tokens"],
    }
    save_torch(
        path,
        {
            "metadata": metadata,
            "metadata_sha256": metadata_digest(metadata),
            "weights": weights,
            "probe": check,
        },
    )


def read_checkpoint(path: str | Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if set(payload) != {"metadata", "metadata_sha256", "weights", "probe"}:
        raise ValueError("not an inference checkpoint")
    metadata = payload["metadata"]
    if (
        metadata["format_version"] != 1
        or metadata["architecture"] != "gemma-latent-prefix-v1"
    ):
        raise ValueError("unsupported text-realizer checkpoint")
    if (
        metadata_digest(metadata) != payload["metadata_sha256"]
        or tensor_digest(payload["weights"]) != metadata["weights_sha256"]
        or tensor_digest({key: payload["probe"][key] for key in ("z", "logits")})
        != metadata["probe_sha256"]
        or payload["probe"]["tokens"] != metadata["probe_tokens"]
    ):
        raise ValueError("checkpoint checksum mismatch")
    return payload


def load_text_realizer(
    checkpoint: str | Path,
    device: str | torch.device = "cpu",
    *,
    cache_dir: str | Path | None = None,
) -> TextRealizer:
    payload = read_checkpoint(checkpoint)
    metadata = payload["metadata"]
    expected_files = metadata["provenance"].get("base_files_sha256", {}).get("decoder")
    if expected_files:
        source = Path(cached_decoder(metadata["spec"], cache_dir))
        for name, expected in expected_files.items():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("unsafe base-model file reference")
            if sha256_file(source / name) != expected:
                raise ValueError(f"pinned decoder/tokenizer file changed: {name}")
    model = build_model(
        metadata["spec"], metadata["generation"], device=device, cache_dir=cache_dir
    )
    model.restore_trainable(payload["weights"])
    return model.eval()


def verify_checkpoint(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
    cache_dir: str | Path | None = None,
) -> dict:
    payload = read_checkpoint(checkpoint)
    model = load_text_realizer(checkpoint, device=device, cache_dir=cache_dir)
    observed = probe(model)
    torch.testing.assert_close(
        observed["logits"], payload["probe"]["logits"], rtol=1e-5, atol=2e-5
    )
    if observed["tokens"] != payload["probe"]["tokens"]:
        raise ValueError("fresh generation differs from saved probe")
    return {
        "verified": True,
        "checkpoint_sha256": sha256_file(checkpoint),
        "metadata_sha256": payload["metadata_sha256"],
        "device": str(device),
    }
