"""Prepare full factual vectors without opening counterfactual or evaluation targets."""

from __future__ import annotations

from pathlib import Path

import torch

from exp.text_realizer.data import csv_rows
from exp.text_realizer.protocol import immutable_json, lock, read_json
from exp.text_realizer_package.data import make_splits as make_splits, verified_inputs
from src.artifact_io import save_torch, sha256_file, write_json
from src.encoder.protocols import embeddinggemma_protocol

from .settings import digest, resolve


def prepared_path(config: dict) -> Path:
    key = digest(
        {
            "inputs": config["inputs"],
            "split": config["split"],
            "encoder": config["encoder"],
            "preparer_sha256": sha256_file(__file__),
        }
    )
    return resolve(config["paths"]["prepared"]) / key


def canonical_factual(z: torch.Tensor) -> torch.Tensor:
    """Match Encoder('embeddinggemma', 768): use the full cached vector unchanged."""
    if z.ndim != 2 or z.shape[1] != 768 or z.dtype != torch.float32:
        raise ValueError("expected factual FP32 768-D cache")
    if not torch.isfinite(z).all() or not torch.allclose(
        z.norm(dim=1), torch.ones(len(z)), atol=2e-6, rtol=2e-6
    ):
        raise ValueError("canonical factual vectors must be finite and unit normalized")
    return z


def prepare(config: dict) -> Path:
    observed = verified_inputs(config)
    root = prepared_path(config)
    with lock(root / ".prepare.lock"):
        if (root / "manifest.json").exists():
            if read_json(root / "manifest.json")["input_sha256"] != observed:
                raise ValueError("prepared provenance changed")
            load_bundle(root)
            return root
        pairs = csv_rows(resolve(config["inputs"]["pairs"]))
        partition = csv_rows(resolve(config["inputs"]["partition"]))
        # Deserialize the trusted paired container, but access only ids and z.
        # Do not call latent_lab.Encoder._load: it also accesses z_prime.
        payload = torch.load(
            resolve(config["inputs"]["latents"]), map_location="cpu", weights_only=True
        )
        ids = [int(key) for key in payload["ids"]]
        factual = payload["z"]
        if factual.shape != (len(ids), 768):
            raise ValueError("misaligned factual tensor")
        splits = make_splits(ids, pairs, partition, config["split"])
        allowed = set(splits["fit"] + splits["tune"])
        rows = csv_rows(resolve(config["inputs"]["factual"]), allowed_ids=allowed)
        positions = {key: i for i, key in enumerate(ids)}
        parts, groups = {}, {}
        for name in ("fit", "tune"):
            chosen = splits[name]
            texts = [rows[key]["text"] for key in chosen]
            if any(not text.strip() for text in texts):
                raise ValueError("empty development CV")
            z = canonical_factual(factual[[positions[key] for key in chosen]])
            parts[name] = {"ids": chosen, "texts": texts, "z": z}
            groups[name] = {
                str(key): {
                    field: rows[key][field] for field in ("template_id", "persona_id")
                }
                for key in chosen
            }
        texts = parts["fit"]["texts"] + parts["tune"]["texts"]
        if len(set(texts)) != len(texts):
            raise ValueError("duplicate fit/development reconstruction targets")
        common = {"format_version": 1, "split_sha256": digest(splits)}
        save_torch(root / "development.pt", {**common, **parts})
        write_json(root / "development_groups.json", groups)
        immutable_json(
            root / "manifest.json",
            {
                **common,
                "split_ids": splits,
                "input_sha256": observed,
                "encoder": config["encoder"],
                "encoder_protocol": embeddinggemma_protocol(config["encoder"]),
                "latent_derivation": "cached_z.float(); full 768-D factual vectors, unchanged",
                "files": {
                    name: sha256_file(root / name)
                    for name in ("development.pt", "development_groups.json")
                },
                "used_counterfactual_text": False,
                "used_counterfactual_latent": False,
                "used_evaluation_content": False,
            },
        )
    return root


def load_bundle(root: Path) -> dict:
    manifest = read_json(root / "manifest.json")
    if sha256_file(root / "development.pt") != manifest["files"]["development.pt"]:
        raise ValueError("prepared bundle hash mismatch")
    data = torch.load(root / "development.pt", map_location="cpu", weights_only=True)
    if (
        set(data) != {"format_version", "split_sha256", "fit", "tune"}
        or data["format_version"] != 1
    ):
        raise ValueError("only the factual development bundle may enter training")
    splits = manifest["split_ids"]
    flat = [key for part in splits.values() for key in part]
    if len(flat) != len(set(flat)) or data["split_sha256"] != digest(splits):
        raise ValueError("invalid split identity or overlapping partitions")
    for name in ("fit", "tune"):
        part = data[name]
        if set(part) != {"ids", "texts", "z"} or part["ids"] != splits[name]:
            raise ValueError("unexpected factual records")
        if (
            part["z"].shape != (len(part["ids"]), 768)
            or part["z"].dtype != torch.float32
            or not torch.isfinite(part["z"]).all()
            or len(part["texts"]) != len(part["ids"])
            or any(
                not isinstance(text, str) or not text.strip() for text in part["texts"]
            )
            or not torch.allclose(
                part["z"].norm(dim=1),
                torch.ones(len(part["ids"])),
                atol=2e-6,
                rtol=2e-6,
            )
        ):
            raise ValueError("invalid factual 768-D records")
    return data
