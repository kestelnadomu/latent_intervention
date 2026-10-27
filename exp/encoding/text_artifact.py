"""ID-aligned evaluation-only text embeddings, independent of the paired format."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.artifact_io import save_torch, sha256_file, write_json
from src.encoder import encoding_metadata
from src.pair_encoding import _active_encoder_info, _load_csv, _require_new_output

FORMAT = "auxiliary_text_embeddings_v1"


def read_texts(path, expected_ids):
    frame = _load_csv(path, "auxiliary texts", {"id", "text"})
    if frame["id"].tolist() != sorted(expected_ids):
        raise ValueError("auxiliary text IDs do not match the evaluation subset")
    if frame["text"].isna().any() or not frame["text"].str.strip().all():
        raise ValueError("auxiliary texts must be non-empty")
    return frame


def check_vectors(z, rows, encoder_info):
    # LangVAE's posterior means are not unit vectors; the embedding encoders' outputs are.
    unit = encoder_info["encoder_variant"] != "langvae"
    if (
        not isinstance(z, torch.Tensor)
        or z.dtype != torch.float32
        or z.shape != (rows, encoder_info["latent_dimension"])
        or not torch.isfinite(z).all()
        or unit and not torch.allclose(z.norm(dim=1), torch.ones(rows), atol=1e-6, rtol=1e-6)
    ):
        raise ValueError("embeddings must be finite float32 vectors, unit-normalized for embedding encoders")


def load_text_artifact(path, *, texts, expected_ids, kind, encoder_config, provenance):
    """Validate both tensor bytes and their source/encoder identity before reuse."""
    path = Path(path)
    info = json.loads(path.with_suffix(".info.json").read_text())
    encoder_info = _active_encoder_info(encoder_config)
    expected = {
        "format": FORMAT,
        "text_variant": kind,
        "role": "evaluation_only",
        "units": len(expected_ids),
        "source_text_sha256": sha256_file(texts),
        "provenance": provenance,
        **encoder_info,
        **encoding_metadata(encoder_info),
    }
    if any(info.get(key) != value for key, value in expected.items()):
        raise ValueError("auxiliary embedding metadata does not match this run")
    if info.get("artifact_sha256") != sha256_file(path):
        raise ValueError("auxiliary embedding checksum mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["ids"] != sorted(expected_ids) or payload["text_variant"] != kind:
        raise ValueError("auxiliary embedding IDs or text variant mismatch")
    check_vectors(payload["z"], len(expected_ids), encoder_info)
    return payload


def encode_text_artifact(
    path, *, texts, expected_ids, kind, encoder_config, provenance, encoder
):
    """Encode the actual alternate text: never copy same-state factual latents."""
    path = Path(path)
    _require_new_output(path)
    before = sha256_file(texts)
    frame = read_texts(texts, expected_ids)
    encoder_info = _active_encoder_info(encoder_config)
    z = encoder.encode(
        frame["text"].tolist(), deterministic=True,
        batch_size=int(encoder_config["batch_size"]),
    ).detach().cpu()
    check_vectors(z, len(frame), encoder_info)
    if sha256_file(texts) != before:
        raise ValueError("auxiliary input changed during encoding")
    save_torch(path, {"ids": frame["id"].tolist(), "z": z, "text_variant": kind})
    write_json(path.with_suffix(".info.json"), {
        "format": FORMAT, "text_variant": kind, "role": "evaluation_only",
        "units": len(frame), "source_text_sha256": before,
        "artifact_sha256": sha256_file(path), "provenance": provenance,
        **encoder_info, **encoding_metadata(encoder_info),
    })
    return load_text_artifact(
        path, texts=texts, expected_ids=expected_ids, kind=kind,
        encoder_config=encoder_config, provenance=provenance,
    )
