"""ID-joined factual preparation; never access oracle tensors or evaluation texts."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from exp.text_realizer.data import csv_rows
from exp.text_realizer.protocol import immutable_json, lock, read_json
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


def verified_inputs(config: dict) -> dict:
    inputs = config["inputs"]
    observed = {}
    for name in ("factual", "pairs", "partition", "latents", "info", "manifest"):
        observed[name] = sha256_file(resolve(inputs[name]))
        if observed[name] != inputs["sha256"][name]:
            raise ValueError(f"input hash mismatch: {name}")
    info = read_json(resolve(inputs["info"]))
    manifest = read_json(resolve(inputs["manifest"]))
    protocol = embeddinggemma_protocol(
        {**config["encoder"], "embeddinggemma_latent_dim": 768}
    )
    if (
        info["artifact_sha256"] != observed["latents"]
        or info["input_sha256"]["factual_text"] != observed["factual"]
        or info["input_sha256"]["pair_index"] != observed["pairs"]
        or info["encoder"] != config["encoder"]["embeddinggemma_model_name"]
        or info["model_revision"] != config["encoder"]["embeddinggemma_model_revision"]
        or info["embeddinggemma_protocol"] != protocol
        or info["latent_dimension"] != 768
        or info["max_length"] != config["encoder"]["embeddinggemma_max_len"]
        or info["deterministic"] is not True
        or manifest["complete"] is not True
        or manifest["protocol"]["input_sha256"]["partition"] != observed["partition"]
        or manifest["protocol"]["input_sha256"]["texts"] != observed["factual"]
        or manifest["protocol"]["input_sha256"]["pair_index"] != observed["pairs"]
    ):
        raise ValueError("incompatible package encoder provenance")
    return observed


def make_splits(ids: list[int], pairs: dict, partition: dict, counts: dict) -> dict:
    if (
        len(ids) != len(set(ids))
        or set(ids) != set(pairs)
        or set(ids) != set(partition)
    ):
        raise ValueError(
            "canonical, pair-index and partition IDs must be unique and identical"
        )
    splits = {"fit": [], "tune": [], "evaluation": []}
    names = {"pool": "fit", "dev": "tune", "eval": "evaluation"}
    # Literal, sorted IDs are independent of physical CSV/tensor row order.
    for key in sorted(ids):
        name = partition[key]["partition"]
        if name not in names:
            raise ValueError("unknown package partition")
        expected = "test" if name == "eval" else "train"
        if pairs[key]["split"] != expected:
            raise ValueError("official pair split disagrees with package partition")
        splits[names[name]].append(key)
    for name, chosen in splits.items():
        if not chosen or len(chosen) != counts[f"{name}_count"]:
            raise ValueError(f"wrong {name} count")
    return splits


def project_factual(z: torch.Tensor) -> torch.Tensor:
    """Exact latent_lab Encoder('embeddinggemma', 256) derivation on factual rows."""
    if z.ndim != 2 or z.shape[1] != 768 or z.dtype != torch.float32:
        raise ValueError("expected factual FP32 768-D cache")
    if not torch.isfinite(z).all() or not torch.allclose(
        z.norm(dim=1), torch.ones(len(z)), atol=2e-6, rtol=2e-6
    ):
        raise ValueError("canonical factual vectors must be finite and unit normalized")
    if (z[:, :256].norm(dim=1) <= 1e-12).any():
        raise ValueError("degenerate 256-D factual prefix")
    return F.normalize(z.float()[:, :256], dim=1)


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
            z = project_factual(factual[[positions[key] for key in chosen]])
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
                "latent_derivation": "F.normalize(cached_z.float()[:, :256], dim=1)",
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
            part["z"].shape != (len(part["ids"]), 256)
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
            raise ValueError("invalid factual 256-D records")
    return data
