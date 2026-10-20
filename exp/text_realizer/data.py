"""Trusted factual extraction. Never use the canonical paired-artifact validator."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from src.artifact_io import save_torch, sha256_file, write_json
from src.encoder.protocols import EMBEDDINGGEMMA_MODEL, embeddinggemma_protocol

from .protocol import immutable_json, lock, read_json
from .settings import digest, resolve


def csv_rows(path: Path, *, allowed_ids: set[int] | None = None) -> dict[int, dict]:
    result = {}
    seen = set()
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = int(row["id"])
            if key in seen:
                raise ValueError(f"duplicate ID in {path.name}: {key}")
            seen.add(key)
            if allowed_ids is None or key in allowed_ids:
                result[key] = row
    if allowed_ids is not None and set(result) != allowed_ids:
        raise ValueError(f"missing IDs in {path.name}")
    return result


def verified_inputs(config: dict) -> dict:
    inputs = config["inputs"]
    observed = {}
    for name in ("factual", "pairs", "latents", "split_plan"):
        path = resolve(inputs[name])
        observed[name] = sha256_file(path)
        if observed[name] != inputs["sha256"][name]:
            raise ValueError(f"input hash mismatch: {name}")
    info = read_json(resolve(inputs["info"]))
    expected = embeddinggemma_protocol(config["encoder"])
    if (
        info["artifact_sha256"] != observed["latents"]
        or info["input_sha256"]["factual_text"] != observed["factual"]
        or info["input_sha256"]["pair_index"] != observed["pairs"]
        or info["encoder"] != EMBEDDINGGEMMA_MODEL
        or info["model_revision"] != config["encoder"]["embeddinggemma_model_revision"]
        or info["embeddinggemma_protocol"] != expected
        or info["latent_dimension"] != 768
        or info["deterministic"] is not True
    ):
        raise ValueError("incompatible canonical encoder provenance")
    observed["info"] = sha256_file(resolve(inputs["info"]))
    return observed


def make_splits(ids: list[int], pairs: dict, settings: dict, plan: dict) -> dict:
    if len(ids) != len(set(ids)) or set(ids) != set(pairs):
        raise ValueError("canonical/pair-index IDs must be unique and identical")
    if any(row["split"] not in {"train", "test"} for row in pairs.values()):
        raise ValueError("unknown official split")
    test = [key for key in ids if pairs[key]["split"] == "test"]
    train = [key for key in ids if pairs[key]["split"] == "train"]
    fit = list(plan["split_ids"]["fit"])
    validation = list(plan["split_ids"]["validation"])
    if set(plan["split_ids"]["official_test"]) != set(test):
        raise ValueError("existing split plan disagrees with official test")
    order = torch.randperm(
        len(validation), generator=torch.Generator().manual_seed(settings["audit_seed"])
    ).tolist()
    middle = len(order) // 2
    splits = {
        "fit": fit,
        "tune": [validation[i] for i in order[:middle]],
        "audit": [validation[i] for i in order[middle:]],
        "official_test": test,
    }
    expected = (settings["fit_count"], middle, middle, settings["test_count"])
    flat = [key for part in splits.values() for key in part]
    if (
        len(validation) != settings["validation_count"]
        or any(
            not part or len(part) != size
            for part, size in zip(splits.values(), expected)
        )
        or len(flat) != len(set(flat))
        or set(flat) != set(ids)
        or set(fit + validation) != set(train)
    ):
        raise ValueError("invalid or overlapping factual splits")
    return splits


def prepared_path(config: dict) -> Path:
    """The key also binds inputs/config; a changed preparation never overwrites another."""
    key = digest(
        {
            "inputs": config["inputs"],
            "split": config["split"],
            "encoder": config["encoder"],
        }
    )
    return resolve(config["paths"]["prepared"]) / key


def prepare(config: dict) -> Path:
    inputs = verified_inputs(config)
    root = prepared_path(config)
    with lock(root / ".prepare.lock"):
        if (root / "manifest.json").exists():
            manifest = read_json(root / "manifest.json")
            if manifest["input_sha256"] != inputs:
                raise ValueError("prepared inputs changed")
            load_bundle(root)
            return root
        pairs = csv_rows(resolve(config["inputs"]["pairs"]))
        # torch must deserialize the container, but no oracle key/value is accessed.
        payload = torch.load(
            resolve(config["inputs"]["latents"]), map_location="cpu", weights_only=True
        )
        ids = [int(key) for key in payload["ids"]]
        factual_z = payload["z"]
        if factual_z.shape != (len(ids), 768) or factual_z.dtype != torch.float32:
            raise ValueError("wrong factual tensor shape/dtype")
        source_plan = resolve(config["inputs"]["split_plan"])
        plan = read_json(source_plan)
        splits = make_splits(ids, pairs, config["split"], plan)
        allowed = set(splits["fit"] + splits["tune"] + splits["audit"])
        rows = csv_rows(resolve(config["inputs"]["factual"]), allowed_ids=allowed)
        positions = {key: i for i, key in enumerate(ids)}
        parts, groups = {}, {}
        for name in ("fit", "tune", "audit"):
            chosen = splits[name]
            z = factual_z[[positions[key] for key in chosen]].clone()
            if not torch.isfinite(z).all() or not torch.allclose(
                z.norm(dim=1), torch.ones(len(z)), atol=2e-6, rtol=2e-6
            ):
                raise ValueError("factual vectors must be finite and unit normalized")
            texts = [rows[key]["text"] for key in chosen]
            if any(not text.strip() for text in texts):
                raise ValueError("empty factual CV")
            parts[name] = {"ids": chosen, "texts": texts, "z": z}
            groups[name] = {
                str(key): {
                    field: rows[key][field] for field in ("template_id", "persona_id")
                }
                for key in chosen
            }
        # Reject exact development duplicates; do not use audit content for model choices.
        development_texts = parts["fit"]["texts"] + parts["tune"]["texts"]
        if len(development_texts) != len(set(development_texts)):
            raise ValueError("duplicate fit/tune CVs would leak reconstruction targets")
        split_hash = digest(splits)
        common = {"format_version": 1, "split_sha256": split_hash}
        save_torch(
            root / "development.pt",
            {**common, "fit": parts["fit"], "tune": parts["tune"]},
        )
        save_torch(root / "audit.pt", {**common, "audit": parts["audit"]})
        write_json(
            root / "development_groups.json", {k: groups[k] for k in ("fit", "tune")}
        )
        write_json(root / "audit_groups.json", {"audit": groups["audit"]})
        files = (
            "development.pt",
            "audit.pt",
            "development_groups.json",
            "audit_groups.json",
        )
        immutable_json(
            root / "manifest.json",
            {
                **common,
                "split_ids": splits,
                "input_sha256": inputs,
                "encoder": config["encoder"],
                "encoder_protocol": embeddinggemma_protocol(config["encoder"]),
                "files": {name: sha256_file(root / name) for name in files},
                "used_counterfactual_text": False,
                "used_counterfactual_latent": False,
                "used_official_test_for_selection": False,
            },
        )
    return root


def load_bundle(
    root: Path, *, audit: bool = False, frozen_root: Path | None = None
) -> dict:
    manifest = read_json(root / "manifest.json")
    if audit:
        _require_audit(root, frozen_root)
    filename = "audit.pt" if audit else "development.pt"
    if sha256_file(root / filename) != manifest["files"][filename]:
        raise ValueError("prepared bundle hash mismatch")
    data = torch.load(root / filename, map_location="cpu", weights_only=True)
    partitions = {"audit"} if audit else {"fit", "tune"}
    if set(data) != partitions | {"format_version", "split_sha256"}:
        raise ValueError(
            "unexpected bundle fields; canonical/oracle artifacts are forbidden"
        )
    if data["split_sha256"] != digest(manifest["split_ids"]):
        raise ValueError("split hash mismatch")
    for name in partitions:
        part = data[name]
        if (
            set(part) != {"ids", "texts", "z"}
            or part["ids"] != manifest["split_ids"][name]
        ):
            raise ValueError("unexpected factual records")
        if set(part["ids"]) & set(manifest["split_ids"]["official_test"]):
            raise ValueError("official test leaked into factual preparation")
        if part["z"].shape != (len(part["ids"]), 768) or len(part["texts"]) != len(
            part["ids"]
        ):
            raise ValueError("misaligned factual records")
    return data


def _require_audit(root: Path, frozen_root: Path | None) -> None:
    if frozen_root is None:
        raise PermissionError("audit requires a frozen campaign")
    from .protocol import require_frozen

    frozen = require_frozen(frozen_root)
    if frozen["prepared_manifest_sha256"] != sha256_file(root / "manifest.json"):
        raise ValueError("frozen campaign refers to different prepared data")


def load_groups(
    root: Path, *, audit: bool = False, frozen_root: Path | None = None
) -> dict:
    if audit:
        _require_audit(root, frozen_root)
    name = "audit_groups.json" if audit else "development_groups.json"
    if sha256_file(root / name) != read_json(root / "manifest.json")["files"][name]:
        raise ValueError("group metadata changed")
    return json.loads((root / name).read_text())
