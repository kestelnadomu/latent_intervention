"""Thin adapters for read-only preflight, isolated fitting and explicit evaluation."""

from contextlib import contextmanager
import fcntl
import json

import torch

from src.artifact_io import sha256_file, write_json
from src.pair_encoding import load_latent_artifact
from . import EDITOR_VARIANTS, load_model, make_model
from .capacity import capacity_report
from .protocol import (
    DEFAULT_PROTOCOL,
    decoder_for,
    digest,
    input_hashes,
    load_fitting_data,
    metadata_for,
    paths_for,
    read_settings,
    split_ids,
    support_report,
    validate_ids,
)


def context(config):
    if (
        config["latent_intervention"]["variant"] not in EDITOR_VARIANTS
        or config["encoder"]["variant"] != "embeddinggemma"
        or config["semantic_decoder"]["variant"] != "independent"
        or config["encoder"]["device"] != "cpu"
    ):
        raise ValueError(
            "editor workflow requires CPU EmbeddingGemma and independent g"
        )
    options = config["latent_intervention"].get("editor", {})
    settings = read_settings(
        options.get("protocol", DEFAULT_PROTOCOL), options.get("overrides")
    )
    torch.set_num_threads(settings["threads"])
    artifact = load_latent_artifact(config)
    if artifact.z.shape[1] not in settings["dimensions"]:
        raise ValueError("editor workflow supports only 256D and 768D")
    return settings, artifact, paths_for(config, settings)


@contextmanager
def run_lock(paths):
    paths["lock"].parent.mkdir(parents=True, exist_ok=True)
    with paths["lock"].open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another writer holds this editor run lock") from error
        yield


def construct(variant, dimension, columns, settings):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(settings["seed"])
        return make_model(
            variant,
            latent_dim=dimension,
            columns=columns,
            settings=dict(tier=settings["tier"]),
        )


def preflight(config):
    """Read-only actual-data forward/backward checks, without optimizer steps or test inference."""
    from .objective import total_loss
    from .training import batch_terms, prepare

    settings, artifact, paths = context(config)
    ids = split_ids(artifact, settings)
    if {k: len(v) for k, v in ids.items()} != dict(
        fit=3200, tune=400, audit=400, official_test=1000
    ):
        raise ValueError("unexpected talent split sizes")
    data, decoder = load_fitting_data(config, ids), decoder_for(config, artifact)
    variant, dimension = config["latent_intervention"]["variant"], artifact.z.shape[1]
    capacities, checks = {}, {}
    frozen = {k: v.clone() for k, v in decoder.state_dict().items()}
    for tier in (1, 2):
        model = construct(variant, dimension, data.columns, dict(settings, tier=tier))
        capacities[str(tier)] = capacity_report(model, tier=tier)
        bank, targets, _ = prepare(model, data, decoder, settings)
        before = {k: p.detach().clone() for k, p in model.named_parameters()}
        terms = batch_terms(
            model,
            data.fit,
            torch.arange(2),
            targets["fit"],
            decoder,
            bank,
            create_graph=True,
        )
        loss = total_loss(terms, settings)
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if (
            not torch.isfinite(loss)
            or not gradients
            or not all(torch.isfinite(g).all() for g in gradients)
        ):
            raise FloatingPointError("nonfinite editor preflight forward/backward")
        if any(not torch.equal(before[k], p) for k, p in model.named_parameters()):
            raise ValueError("preflight changed editor weights")
        checks[str(tier)] = dict(
            objective=float(loss.detach()),
            gradient_norm=float(torch.stack([g.norm() for g in gradients]).norm()),
            optimizer_steps=0,
        )
    if any(
        not torch.equal(v, decoder.state_dict()[k]) for k, v in frozen.items()
    ) or any(p.grad is not None for p in decoder.parameters()):
        raise ValueError("preflight changed frozen g")
    if paths["protocol"].exists() and digest(
        json.loads(paths["protocol"].read_text())
    ) != digest(metadata_for(config, settings, artifact, data)):
        raise ValueError(
            "existing editor run has a conflicting signature; choose a new run ID"
        )
    return dict(
        status="ready",
        variant=variant,
        dimension=dimension,
        seed=settings["seed"],
        split_counts={k: len(v) for k, v in ids.items()},
        split_sha256=digest(ids),
        inputs=input_hashes(config, data, ids),
        support=support_report(data, targets["tune"]),
        capacity=capacities,
        forward_backward=checks,
        frozen_g_parameters=sum(p.numel() for p in decoder.parameters()),
        analytic_h_s_parameters=0,
        fitting_launched=False,
        used_official_test=False,
        used_auxiliary_labels=False,
        reserved_audit="unused",
        output_model=str(paths["model"]),
    )


def train_manipulator(config):
    from .training import fit

    settings, artifact, paths = context(config)
    data = load_fitting_data(config, split_ids(artifact, settings))
    metadata = metadata_for(config, settings, artifact, data)
    decoder = decoder_for(config, artifact)
    with run_lock(paths):
        if paths["protocol"].exists():
            if digest(json.loads(paths["protocol"].read_text())) != digest(metadata):
                raise ValueError("editor run signature changed; choose a new run ID")
        elif any(paths[k].exists() for k in ("model", "resume", "training", "info")):
            raise FileExistsError(
                "editor artifacts exist without a protocol; refusing overwrite"
            )
        write_json(paths["protocol"], metadata)
        model = construct(
            metadata["variant"], metadata["dimension"], data.columns, settings
        )
        report = fit(
            model,
            data,
            decoder,
            settings,
            paths["model"],
            paths["training"],
            paths["resume"],
            metadata,
        )
        if (
            input_hashes(
                config, load_fitting_data(config, metadata["ids"]), metadata["ids"]
            )
            != metadata["inputs"]
        ):
            raise ValueError("editor inputs changed during fitting")
        load_model(
            paths["model"],
            expected_variant=metadata["variant"],
            expected_metadata=metadata,
        )
        write_json(
            paths["info"],
            dict(
                checkpoint_sha256=sha256_file(paths["model"]),
                metadata=metadata,
                capacity=report["capacity"],
            ),
        )
    print(
        f"wrote {paths['model']}; reserved-audit and official-test units were not evaluated",
        flush=True,
    )
    return report


def load_selected(config, artifact, paths):
    info = json.loads(paths["info"].read_text())
    if info["checkpoint_sha256"] != sha256_file(paths["model"]):
        raise ValueError("selected editor checkpoint checksum mismatch")
    model = load_model(
        paths["model"],
        expected_variant=config["latent_intervention"]["variant"],
        expected_metadata=info["metadata"],
    )
    metadata = model.checkpoint_metadata
    validate_ids(artifact, metadata["ids"])
    data = load_fitting_data(config, metadata["ids"])
    if (
        model.latent_dim != artifact.z.shape[1]
        or input_hashes(config, data, metadata["ids"]) != metadata["inputs"]
    ):
        raise ValueError("editor evaluation inputs differ from the selected run")
    return model


def evaluate_manipulator(config):
    from .evaluation import evaluate

    _, artifact, paths = context(config)
    with run_lock(paths):
        model = load_selected(config, artifact, paths)
        record = evaluate(config, artifact, model)
        record["checkpoint_sha256"] = sha256_file(paths["model"])
        record["record_sha256"] = digest(record)
        write_json(paths["evaluation"], record)
    print(f"wrote {paths['evaluation']}", flush=True)
    return record
