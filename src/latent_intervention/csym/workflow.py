"""Thin pipeline adapter for one projected C-sym fit and its independent evaluation."""

from __future__ import annotations

import fcntl
import json

import torch

from src.artifact_io import save_torch, sha256_file, write_json
from src.pair_encoding import load_latent_artifact
from .model import ProjectedCSym, VARIANT
from .probes import fit_readers, pack_probe, unpack_probe
from .protocol import (
    DEFAULT_PROTOCOL,
    digest,
    load_data,
    metadata_for,
    paths_for,
    read_settings,
    verify_inputs,
)


def context(config):
    if (
        config["latent_intervention"]["variant"] != VARIANT
        or config["encoder"]["variant"] != "embeddinggemma"
        or config["semantic_decoder"]["variant"] != "independent"
        or config["encoder"]["device"] != "cpu"
    ):
        raise ValueError(
            "supported C-sym workflow uses CPU EmbeddingGemma and independent g"
        )
    settings = read_settings(
        config["latent_intervention"].get(VARIANT, {}).get("protocol", DEFAULT_PROTOCOL)
    )
    torch.set_num_threads(settings["threads"])
    artifact = load_latent_artifact(config)
    if artifact.z.shape[1] not in settings["dimensions"]:
        raise ValueError("embedding dimension outside C-sym protocol")
    return settings, artifact, paths_for(config)


def prepare_readers(paths, data, settings, metadata):
    signature = digest(metadata)
    if paths["readers"].exists():
        payload = torch.load(paths["readers"], map_location="cpu", weights_only=True)
        if (
            payload.get("format") != "csym_readers_v1"
            or payload["signature"] != signature
        ):
            raise ValueError("saved C-sym readers do not match the training protocol")
        if paths["probes"].exists():
            report = json.loads(paths["probes"].read_text())
        else:
            # The atomic reader file is authoritative if report writing was interrupted.
            report = dict(
                payload["report"],
                signature=signature,
                checkpoint_sha256=sha256_file(paths["readers"]),
            )
            report["record_sha256"] = digest(report)
            write_json(paths["probes"], report)
        if (
            report["signature"] != signature
            or report["record_sha256"]
            != digest({k: v for k, v in report.items() if k != "record_sha256"})
            or report["checkpoint_sha256"] != sha256_file(paths["readers"])
        ):
            raise ValueError("saved C-sym readers do not match the training protocol")
        readers = {k: unpack_probe(v) for k, v in payload["readers"].items()}
    else:
        if paths["probes"].exists():
            raise FileExistsError("reader report exists without its checkpoint")
        readers, report = fit_readers(data, settings)
        save_torch(
            paths["readers"],
            dict(
                format="csym_readers_v1",
                signature=signature,
                readers={k: pack_probe(v) for k, v in readers.items()},
                report=report,
            ),
        )
        report.update(
            signature=signature, checkpoint_sha256=sha256_file(paths["readers"])
        )
        report["record_sha256"] = digest(report)
        write_json(paths["probes"], report)
    if not report["readers_qualified"]:
        raise ValueError(
            "guide/primary auditor did not pass the factual qualification gate; no editor fitted"
        )
    return readers


def load_published(paths):
    info = json.loads(paths["info"].read_text())
    if info["checkpoint_sha256"] != sha256_file(paths["model"]):
        raise ValueError("published C-sym checkpoint checksum mismatch")
    model = ProjectedCSym.load(paths["model"])
    if digest(info["metadata"]) != digest(model.checkpoint_metadata):
        raise ValueError("published C-sym metadata mismatch")
    verify_inputs(model.checkpoint_metadata)
    return model


def train_manipulator(config):
    from .training import fit
    from .evaluation import audit

    settings, artifact, paths = context(config)
    paths["model"].parent.mkdir(parents=True, exist_ok=True)
    with (paths["model"].parent / ".train.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if paths["migration"].exists():
            raise FileExistsError(
                "published migrated weights are immutable; set latent_intervention.tag to a fresh run name"
            )
        metadata = metadata_for(config, settings, artifact)
        if paths["protocol"].exists() and json.loads(
            paths["protocol"].read_text()
        ) != json.loads(json.dumps(metadata)):
            raise ValueError(
                "training inputs/code/protocol changed; use a fresh run tag"
            )
        write_json(paths["protocol"], metadata)
        data = load_data(config, metadata["ids"])
        readers = prepare_readers(paths, data, settings, metadata)
        metadata["reader_sha256"] = sha256_file(paths["readers"])
        fit(
            data,
            readers["guide"],
            settings,
            paths["model"],
            paths["training"],
            metadata,
        )
        verify_inputs(metadata)
        model = ProjectedCSym.load(paths["model"], expected_metadata=metadata)
        write_json(
            paths["info"],
            dict(checkpoint_sha256=sha256_file(paths["model"]), metadata=metadata),
        )
        # Factual audit only. Official-test evaluation is a separate explicit command.
        write_json(paths["audit"], audit(config, model, readers, settings))
    print(f"wrote {paths['model']}; official test was not evaluated")


def evaluate_manipulator(config):
    from .evaluation import audit, evaluate

    _, artifact, paths = context(config)
    model = load_published(paths)
    if artifact.z.shape[1] != model.latent_dim:
        raise ValueError("checkpoint and embedding dimensions differ")
    if sha256_file(paths["readers"]) != model.checkpoint_metadata["reader_sha256"]:
        raise ValueError("independent audit readers changed")
    payload = torch.load(paths["readers"], map_location="cpu", weights_only=True)
    readers = {k: unpack_probe(v) for k, v in payload["readers"].items()}
    write_json(
        paths["audit"],
        audit(config, model, readers, model.checkpoint_metadata["settings"]),
    )
    record = evaluate(config, model)
    record["checkpoint_sha256"] = sha256_file(paths["model"])
    record["record_sha256"] = digest(record)
    write_json(paths["evaluation"], record)
    print(f"wrote {paths['evaluation']}")
    return record
