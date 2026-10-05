"""Privileged training-pair targets, separate from the canonical test-only Z'."""

from __future__ import annotations

import fcntl
import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp

import torch
import torch.nn.functional as F

from src.artifact_io import save_torch, sha256_file, write_json
from src.encoder_protocols import get_encoder_factory
from src.oracle_regression import ORACLE_VARIANT
from src.pair_encoding import (
    _input_hashes,
    _integer_ids,
    _latent_tensor,
    _load_csv,
    _load_pair_index,
    load_latent_artifact,
)
from src.schema import load_intervention


@dataclass(frozen=True)
class OracleTargets:
    ids: list[int]
    z_prime: torch.Tensor
    is_identity: torch.Tensor
    artifact_sha256: str
    metadata: dict


def require_oracle(config):
    if config["latent_intervention"]["variant"] != ORACLE_VARIANT:
        raise ValueError("privileged targets are available only to oracle_regression")


def target_path(config):
    path = Path(config["paths"]["oracle_targets"])
    if path.name != "z_prime.pt" or path.parent.name != "oracle_train":
        raise ValueError(
            "oracle_targets must use a separate oracle_train/z_prime.pt path"
        )
    if path.resolve() == Path(config["paths"]["latents"]).resolve():
        raise ValueError("oracle targets must not replace canonical latents")
    return path


def simulation_hash(config):
    source = config["sim_config"]
    if isinstance(source, dict):
        return hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    return sha256_file(source)


def source_metadata(config, artifact):
    return {
        "format_version": 1,
        "purpose": "oracle_regression_only",
        "split": "train",
        "source_latent_sha256": artifact.artifact_sha256,
        "encoder": artifact.encoder_info,
        "input_sha256": _input_hashes(config["paths"]),
        "sim_config_sha256": simulation_hash(config),
        "intervention": load_intervention(config["sim_config"]),
    }


def training_texts(config, artifact):
    """Exact official-training alignment; reject missing/changed identity texts."""
    require_oracle(config)
    paths = config["paths"]
    pairs = _load_pair_index(paths["pair_index"]).set_index("id")
    factual = _load_csv(paths["texts"], "factual texts", {"id", "text"}).set_index("id")
    counterfactual = _load_csv(
        paths["texts_counterfactual"], "counterfactual texts", {"id", "text"}
    ).set_index("id")
    if set(counterfactual.index) != set(artifact.ids):
        raise ValueError(
            "oracle regression requires counterfactual texts for all training IDs"
        )
    ids = artifact.train_ids
    flags = pairs.loc[ids, "is_identity"].tolist()
    for unit, identity in zip(ids, flags, strict=True):
        text = counterfactual.at[unit, "text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"empty oracle training text for id={unit}")
        if identity and text != factual.at[unit, "text"]:
            raise ValueError(f"identity training text differs for id={unit}")
    return ids, torch.tensor(flags, dtype=torch.bool), factual, counterfactual


def _validate_payload(payload, config, artifact):
    ids, identity, _, _ = training_texts(config, artifact)
    if not isinstance(payload, dict) or set(payload) != {
        "train_ids",
        "z_prime",
        "is_identity",
    }:
        raise ValueError("malformed oracle training-target payload")
    if _integer_ids(payload["train_ids"], "oracle train IDs") != ids:
        raise ValueError(
            "oracle targets must contain exactly official training IDs in order; no test IDs"
        )
    z_prime = _latent_tensor(
        payload["z_prime"], "oracle Z'", len(ids), artifact.z.shape[1]
    )
    if z_prime.dtype != torch.float32:
        raise ValueError("oracle targets must use float32")
    flags = payload["is_identity"]
    if (
        not isinstance(flags, torch.Tensor)
        or flags.dtype != torch.bool
        or not torch.equal(flags, identity)
    ):
        raise ValueError("oracle identity flags disagree with the official split")
    positions = {unit: index for index, unit in enumerate(artifact.ids)}
    for index, (unit, flag) in enumerate(zip(ids, identity.tolist(), strict=True)):
        if flag and not torch.equal(z_prime[index], artifact.z[positions[unit]]):
            raise ValueError(f"oracle identity latent differs for id={unit}")
    if config["encoder"]["variant"] in {"embeddinggemma", "nomic", "qwen3"}:
        if not torch.allclose(
            z_prime.norm(dim=1), torch.ones(len(ids)), atol=1e-6, rtol=1e-6
        ):
            raise ValueError("embedding oracle targets must be unit normalized")
    return ids, z_prime, identity


def load_oracle_targets(config, artifact=None):
    require_oracle(config)
    artifact = load_latent_artifact(config) if artifact is None else artifact
    path = target_path(config)
    if not path.is_file() or not path.with_suffix(".info.json").is_file():
        raise ValueError(
            "oracle training targets are missing; run python -m src.oracle_encoding --run first"
        )
    info = json.loads(path.with_suffix(".info.json").read_text())
    if not isinstance(info, dict):
        raise ValueError("invalid oracle target metadata")
    expected = source_metadata(config, artifact)
    if any(info.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "oracle target provenance does not match the current latent space/data"
        )
    digest = sha256_file(path)
    if info.get("artifact_sha256") != digest:
        raise ValueError("oracle target checksum mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    ids, z_prime, identity = _validate_payload(payload, config, artifact)
    if info.get("training_units") != len(ids) or info.get("identity_units") != int(
        identity.sum()
    ):
        raise ValueError("oracle target counts disagree with payload")
    return OracleTargets(ids, z_prime, identity, digest, info)


def publish_targets(config, artifact, payload, expected_metadata, *, derivation=None):
    """Publish a complete directory atomically; never overwrite existing artifacts."""
    require_oracle(config)
    ids, _, identity = _validate_payload(payload, config, artifact)
    current = load_latent_artifact(config)
    if current.artifact_sha256 != artifact.artifact_sha256:
        raise ValueError("canonical latents changed during oracle encoding")
    if source_metadata(config, artifact) != expected_metadata:
        raise ValueError("oracle encoding inputs changed; refusing publication")
    path = target_path(config)
    directory = path.parent
    directory.parent.mkdir(parents=True, exist_ok=True)
    with (directory.parent / ".oracle_train.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if directory.exists():
            raise FileExistsError(
                f"refusing to overwrite oracle target directory: {directory}"
            )
        staging = Path(mkdtemp(prefix=".oracle_train.", dir=directory.parent))
        staged = staging / path.name
        save_torch(staged, payload)
        write_json(
            staged.with_suffix(".info.json"),
            {
                **expected_metadata,
                "artifact_sha256": sha256_file(staged),
                "training_units": len(ids),
                "identity_units": int(identity.sum()),
                "derivation": derivation,
            },
        )
        os.rename(staging, directory)
    return load_oracle_targets(config, artifact)


def encode_oracle_targets(config, *, encoder_factory=None):
    """Encode only nonidentity training counterfactuals; verify compatibility first."""
    require_oracle(config)
    artifact = load_latent_artifact(config)
    if target_path(config).parent.exists():
        return load_oracle_targets(config, artifact)
    metadata = source_metadata(config, artifact)
    ids, identity, factual, counterfactual = training_texts(config, artifact)
    positions = {unit: index for index, unit in enumerate(artifact.ids)}
    z_prime = artifact.z[[positions[unit] for unit in ids]].clone()
    nonidentity = (~identity).nonzero().flatten()
    if len(nonidentity):
        factory = encoder_factory or get_encoder_factory(config["encoder"])
        encoder = factory(config["encoder"])
        if encoder.latent_dim != artifact.z.shape[1]:
            raise ValueError("oracle encoder width does not match canonical latents")
        encode_kwargs = dict(
            deterministic=True, batch_size=int(config["encoder"]["batch_size"])
        )
        # Training factual probes catch environment/protocol drift without reading
        # held-out Z' or using any held-out target to fit the regression model.
        probe_ids = ids[: min(4, len(ids))]
        probe = (
            encoder.encode(factual.loc[probe_ids, "text"].tolist(), **encode_kwargs)
            .detach()
            .cpu()
        )
        expected = artifact.z[[positions[unit] for unit in probe_ids]]
        if probe.shape != expected.shape or not torch.allclose(
            probe, expected, atol=1e-6, rtol=1e-5
        ):
            raise ValueError(
                "oracle encoder probe differs from the saved factual latent space"
            )
        texts = [
            counterfactual.at[ids[index], "text"] for index in nonidentity.tolist()
        ]
        encoded = encoder.encode(texts, **encode_kwargs).detach().cpu().float()
        _latent_tensor(
            encoded, "oracle encoded targets", len(texts), artifact.z.shape[1]
        )
        z_prime[nonidentity] = encoded
    return publish_targets(
        config,
        artifact,
        {
            "train_ids": ids,
            "z_prime": z_prime,
            "is_identity": identity,
        },
        metadata,
    )


def derive_oracle_targets(config, source_config):
    """Reuse the pinned 768-D EmbeddingGemma pass for the smaller target space."""
    from src.nomic_encoding import compare_projection

    source = load_latent_artifact(source_config)
    artifact = load_latent_artifact(config)
    if any(
        c["encoder"]["variant"] != "embeddinggemma" for c in (config, source_config)
    ):
        raise ValueError(
            "oracle prefix derivation is currently limited to EmbeddingGemma"
        )
    protocols = []
    for item in (source, artifact):
        info = copy.deepcopy(item.encoder_info)
        info.pop("latent_dimension")
        info["embeddinggemma_protocol"].pop("normalization")
        protocols.append(info)
    if protocols[0] != protocols[1] or simulation_hash(config) != simulation_hash(
        source_config
    ):
        raise ValueError(
            "oracle derivation requires identical encoder and SCM protocols"
        )
    if _input_hashes(config["paths"]) != _input_hashes(source_config["paths"]):
        raise ValueError("oracle derivation source/target data differ")
    compare_projection(source, artifact)
    targets = load_oracle_targets(source_config, source)
    ids, identity, _, _ = training_texts(config, artifact)
    if ids != targets.ids or not torch.equal(identity, targets.is_identity):
        raise ValueError("oracle source/target training splits differ")
    z_prime = F.normalize(targets.z_prime[:, : artifact.z.shape[1]], dim=1).contiguous()
    positions = {unit: index for index, unit in enumerate(artifact.ids)}
    for index, flag in enumerate(identity.tolist()):
        if flag:
            z_prime[index] = artifact.z[positions[ids[index]]]
    derivation = {
        "method": "prefix_truncation_then_l2",
        "source_dimension": 768,
        "source_targets_sha256": targets.artifact_sha256,
    }
    if target_path(config).parent.exists():
        existing = load_oracle_targets(config, artifact)
        if existing.metadata.get("derivation") != derivation or not torch.allclose(
            existing.z_prime, z_prime, atol=1e-6, rtol=1e-5
        ):
            raise ValueError(
                "existing oracle targets disagree with their native source"
            )
        return existing
    return publish_targets(
        config,
        artifact,
        {
            "train_ids": ids,
            "z_prime": z_prime,
            "is_identity": identity,
        },
        source_metadata(config, artifact),
        derivation=derivation,
    )
