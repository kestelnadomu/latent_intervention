"""Encode factual CVs and held-out counterfactual CVs into one latent artifact."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd
import torch


def _load_csv(path: str | Path, label: str, columns: set[str]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    frame = pd.read_csv(path)
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns {sorted(missing)}")
    ids = pd.to_numeric(frame["id"], errors="coerce")
    if ids.isna().any() or not ids.eq(ids.round()).all():
        raise ValueError(f"{label} IDs must be integers")
    frame["id"] = ids.astype(int)
    if frame["id"].duplicated().any():
        raise ValueError(f"{label} contains duplicate IDs")
    return frame.sort_values("id").reset_index(drop=True)


def _bool(value: Any, row_id: int) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"is_identity for id={row_id} must be true/false")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(Path(path), "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def encode_pairs(config: dict[str, Any], encoder_factory=None) -> dict[str, Any]:
    """Encode all X and test X', copying identity latents exactly."""
    paths = config["paths"]
    pairs = _load_csv(paths["pair_index"], "pair index", {"id", "split", "is_identity"})
    factual = _load_csv(paths["texts"], "factual texts", {"id", "text"})
    counterfactual = _load_csv(
        paths["texts_counterfactual"], "counterfactual texts", {"id", "text"}
    )
    pairs["split"] = pairs["split"].astype(str).str.lower()
    if set(pairs["split"]) != {"train", "test"}:
        raise ValueError("pair index must contain non-empty train and test splits")
    pairs["is_identity"] = [
        _bool(value, row_id)
        for row_id, value in zip(pairs["id"], pairs["is_identity"])
    ]

    all_ids = pairs["id"].tolist()
    test = pairs.loc[pairs["split"] == "test"].copy()
    test_ids = test["id"].tolist()
    if factual["id"].tolist() != all_ids:
        raise ValueError("factual text IDs must equal all pair-index IDs")
    counterfactual_ids = counterfactual["id"].tolist()
    if counterfactual_ids not in (test_ids, all_ids):
        raise ValueError(
            "counterfactual text IDs must equal either test IDs or all "
            "pair-index IDs exactly"
        )
    for label, frame in (("factual", factual), ("counterfactual", counterfactual)):
        if frame["text"].isna().any() or not frame["text"].astype(str).str.strip().all():
            raise ValueError(f"{label} texts must be non-empty")

    factual_by_id = factual.set_index("id")
    counterfactual_by_id = counterfactual.set_index("id")
    identity = test["is_identity"].astype(bool).tolist()
    for row_id, is_identity in zip(test_ids, identity):
        if is_identity and counterfactual_by_id.at[row_id, "text"] != factual_by_id.at[row_id, "text"]:
            raise ValueError(f"identity text differs for id={row_id}")

    nonidentity_ids = [row_id for row_id, flag in zip(test_ids, identity) if not flag]
    texts = factual["text"].astype(str).tolist() + [
        str(counterfactual_by_id.at[row_id, "text"]) for row_id in nonidentity_ids
    ]
    if encoder_factory is None:
        from src.encoder import TextEncoder

        encoder_factory = TextEncoder
    enc = config["encoder"]
    encoder = encoder_factory(
        model_name=enc["model_name"],
        device=enc["device"],
        max_len=int(enc["max_len"]),
        local_checkpoint=enc.get("local_checkpoint"),
    )
    encoded = encoder.encode(
        texts,
        deterministic=True,
        batch_size=int(enc["batch_size"]),
    ).detach().cpu()
    if encoded.ndim != 2 or encoded.shape[0] != len(texts) or not torch.isfinite(encoded).all():
        raise ValueError("encoder returned invalid latent vectors")

    # Detach factual storage from the appended counterfactual encodings before saving.
    z = encoded[: len(all_ids)].clone()
    z_prime = torch.empty((len(test_ids), z.shape[1]), dtype=z.dtype)
    factual_position = {row_id: position for position, row_id in enumerate(all_ids)}
    generated_position = {row_id: len(all_ids) + i for i, row_id in enumerate(nonidentity_ids)}
    for position, (row_id, is_identity) in enumerate(zip(test_ids, identity)):
        source = factual_position[row_id] if is_identity else generated_position[row_id]
        z_prime[position] = encoded[source]

    payload = {
        "ids": all_ids,
        "z": z,
        "test_ids": test_ids,
        "z_prime": z_prime,
        "is_identity": torch.tensor(identity, dtype=torch.bool),
    }
    output = Path(paths["latents"])
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    info_path = output.with_suffix(".info.json")
    info_path.write_text(
        json.dumps(
            {
                "encoder": enc["model_name"],
                "local_checkpoint": (
                    str(enc["local_checkpoint"]) if enc.get("local_checkpoint") else None
                ),
                "langvae_version": _package_version("langvae"),
                "max_length": int(enc["max_len"]),
                "deterministic": True,
                "latent_dimension": int(z.shape[1]),
                "factual_units": len(all_ids),
                "counterfactual_test_units": len(test_ids),
                "identity_test_units": sum(identity),
                "input_sha256": {
                    "pair_index": _sha256(paths["pair_index"]),
                    "factual_text": _sha256(paths["texts"]),
                    "counterfactual_text": _sha256(paths["texts_counterfactual"]),
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"wrote {output}: {len(all_ids)} factual and {len(test_ids)} test pairs")
    return payload
