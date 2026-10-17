"""Thin pipeline/CLI adapter; fitting, independent audit and test evaluation are separate."""

from contextlib import contextmanager
import fcntl
import json

import torch

from src.artifact_io import sha256_file, write_json
from src.pair_encoding import load_latent_artifact
from . import OT_VARIANTS, make_model, load_model
from .capacity import capacity_report
from .protocol import (
    DEFAULT_PROTOCOL,
    decoder_for,
    digest,
    input_hashes,
    load_fitting_data,
    metadata_for,
    model_options,
    paths_for,
    read_settings,
    split_ids,
    support_report,
    validate_ids,
    verify_inputs,
)


def context(config):
    if (
        config["latent_intervention"]["variant"] not in OT_VARIANTS
        or config["encoder"]["variant"] != "embeddinggemma"
        or config["semantic_decoder"]["variant"] != "independent"
        or config["encoder"]["device"] != "cpu"
    ):
        raise ValueError("OT workflow requires CPU EmbeddingGemma and independent g")
    options = config["latent_intervention"].get("ot", {})
    settings = read_settings(
        options.get("protocol", DEFAULT_PROTOCOL), options.get("overrides")
    )
    torch.set_num_threads(settings["threads"])
    artifact = load_latent_artifact(config)
    if artifact.z.shape[1] not in settings["dimensions"]:
        raise ValueError("OT workflow supports only 256D and 768D")
    return settings, artifact, paths_for(config, settings)


@contextmanager
def run_lock(paths):
    paths["lock"].parent.mkdir(parents=True, exist_ok=True)
    with paths["lock"].open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another writer holds this OT run lock") from error
        yield


def construct(variant, dimension, columns, settings):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(settings["seed"])
        return make_model(
            variant,
            latent_dim=dimension,
            columns=columns,
            settings=model_options(variant, settings),
        )


@torch.no_grad()
def preflight(config):
    """Read-only: no fitting, readers, output directories, reports or test predictions."""
    settings, artifact, paths = context(config)
    ids = split_ids(artifact, settings)
    data = load_fitting_data(config, ids)
    decoder = decoder_for(config, artifact)
    support = support_report(data, decoder.log_joint(data.tune.z).exp())
    variant, dimension = config["latent_intervention"]["variant"], artifact.z.shape[1]
    capacities = {}
    for tier in [1] if variant == "ot_gaussian" else [1, 2]:
        model = construct(variant, dimension, data.columns, dict(settings, tier=tier))
        capacities[str(tier)] = capacity_report(model, tier=tier)
    if paths["protocol"].exists():
        expected = metadata_for(config, settings, artifact)
        if digest(json.loads(paths["protocol"].read_text())) != digest(expected):
            raise ValueError(
                "existing OT run has a different signature; choose a new run ID"
            )
    return dict(
        status="ready_for_individual_fit",
        variant=variant,
        dimension=dimension,
        split_counts={k: len(v) for k, v in ids.items()},
        split_sha256=digest(ids),
        inputs=input_hashes(config),
        support=support,
        capacity=capacities,
        frozen_decoder_parameters=sum(p.numel() for p in decoder.parameters()),
        h_s="analytic; no trainable parameters",
        fitting_launched=False,
        used_official_test=False,
        used_oracle_targets=False,
        output_model=str(paths["model"]),
        warnings=[
            "Sparse/unseen states require extrapolation; parameter matching is not compute matching.",
            "Single runs are not a tuned multi-seed benchmark.",
        ],
    )


def train_manipulator(config):
    from .training import fit

    settings, artifact, paths = context(config)
    metadata = metadata_for(config, settings, artifact)
    data = load_fitting_data(config, metadata["ids"])
    decoder = decoder_for(config, artifact)  # never pass g to fitting
    with run_lock(paths):
        if paths["protocol"].exists():
            if digest(json.loads(paths["protocol"].read_text())) != digest(metadata):
                raise ValueError("OT run signature changed; choose a new run ID")
        elif any(
            paths[name].exists() for name in ("model", "resume", "training", "info")
        ):
            raise FileExistsError(
                "OT artifacts exist without their protocol; refusing overwrite"
            )
        write_json(paths["protocol"], metadata)
        model = construct(
            metadata["variant"], metadata["dimension"], data.columns, settings
        )
        record = fit(
            model,
            data,
            settings,
            paths["model"],
            paths["training"],
            paths["resume"],
            metadata,
        )
        verify_inputs(metadata)
        load_model(
            paths["model"],
            expected_variant=metadata["variant"],
            expected_metadata=metadata,
        )
        record["frozen_auxiliary_parameters"] = dict(
            g=sum(p.numel() for p in decoder.parameters()),
            h_s=0,
            used_in_objective=False,
        )
        write_json(paths["training"], record)
        write_json(
            paths["info"],
            dict(
                checkpoint_sha256=sha256_file(paths["model"]),
                metadata=metadata,
                capacity=record["capacity"],
            ),
        )
    print(
        f"wrote {paths['model']}; audit and official-test evaluation were not run",
        flush=True,
    )
    return record


def load_selected(config, artifact, paths):
    info = json.loads(paths["info"].read_text())
    if info["checkpoint_sha256"] != sha256_file(paths["model"]):
        raise ValueError("selected OT checkpoint checksum mismatch")
    model = load_model(
        paths["model"],
        expected_variant=config["latent_intervention"]["variant"],
        expected_metadata=info["metadata"],
    )
    validate_ids(artifact, model.checkpoint_metadata["ids"])
    verify_inputs(model.checkpoint_metadata)
    if (
        model.latent_dim != artifact.z.shape[1]
        or input_hashes(config) != model.checkpoint_metadata["inputs"]
    ):
        raise ValueError("OT evaluation inputs differ from the selected run")
    return model


def audit_manipulator(config):
    from .evaluation import audit

    _, artifact, paths = context(config)
    with run_lock(paths):
        model = load_selected(config, artifact, paths)
        record = audit(config, artifact, model, paths)
        write_json(paths["audit"], record)
    print(f"wrote {paths['audit']}; no official-test evaluation", flush=True)
    return record


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
