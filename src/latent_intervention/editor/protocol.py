"""Factual-only data boundary, fixed split and portable per-seed run layout."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import pandas as pd
import torch
import yaml

from src.artifact_io import sha256_file
from src.pair_encoding import load_latent_artifact
from src.schema import flat_state_index, load_intervention, load_schema
from src.semantic_decoder.model import load_semantic_decoder
from src.symbolic_intervention import load_symbolic_kernel
from . import EDITOR_VARIANTS

DEFAULT_PROTOCOL = Path("configs/hz_editor.yaml")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def read_settings(path=DEFAULT_PROTOCOL, overrides=None):
    settings = yaml.safe_load(Path(path).read_text())
    settings.update(overrides or {})
    for name in (
        "threads",
        "max_epochs",
        "batch_size",
        "patience",
        "scheduler_patience",
        "support_bank_size",
        "evaluation_samples",
        "evaluation_batch_size",
    ):
        if type(settings[name]) is not int or settings[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in (
        "seed",
        "split_seed",
        "audit_split_seed",
        "support_seed",
        "evaluation_seed",
    ):
        if type(settings[name]) is not int or not 0 <= settings[name] < 2**32:
            raise ValueError(f"invalid {name}")
    for name in ("learning_rate", "grad_clip", "min_lr"):
        if not math.isfinite(settings[name]) or settings[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("weight_decay", "min_delta"):
        if not math.isfinite(settings[name]) or settings[name] < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if (
        settings["tier"] not in (1, 2)
        or settings["threads"] != 1
        or settings["validation_fraction"] != 0.2
        or settings["split_seed"] != 42
        or settings["audit_split_seed"] != 20261007
        or settings["support_seed"] != 42
        or settings["support_bank_size"] != 128
        or not 0 < settings["scheduler_factor"] < 1
        or settings["loss"] != dict(movement=0.1, support=0.05, identity=1.0)
        or not settings["dimensions"]
        or not set(settings["dimensions"]) <= {256, 768}
        or len(set(settings["dimensions"])) != len(settings["dimensions"])
        or not settings["seeds"]
        or len(set(settings["seeds"])) != len(settings["seeds"])
        or any(type(s) is not int or not 0 <= s < 2**32 for s in settings["seeds"])
        or settings["evaluation_samples"] != 64
        or settings["evaluation_seed"] != 20260930
        or settings["evaluation_batch_size"] != 32
    ):
        raise ValueError("invalid or non-comparable editor protocol")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", settings["run_id"]):
        raise ValueError("invalid run_id")
    return settings


def split_ids(artifact, settings):
    from src.pipeline import _official_indices, _fit_calibration_split

    train, _ = _official_indices(artifact)
    fit, validation = _fit_calibration_split(
        train, settings["validation_fraction"], settings["split_seed"]
    )
    order = torch.randperm(
        len(validation),
        generator=torch.Generator().manual_seed(settings["audit_split_seed"]),
    )
    middle = len(validation) // 2
    ids = dict(
        fit=[artifact.ids[i] for i in fit.tolist()],
        tune=[artifact.ids[i] for i in validation[order[:middle]].tolist()],
        audit=[artifact.ids[i] for i in validation[order[middle:]].tolist()],
        official_test=list(artifact.test_ids),
    )
    validate_ids(artifact, ids)
    return ids


def validate_ids(artifact, ids):
    if set(ids) != {"fit", "tune", "audit", "official_test"}:
        raise ValueError("invalid editor split IDs")
    flattened = [i for part in ids.values() for i in part]
    if (
        any(not p for p in ids.values())
        or len(flattened) != len(set(flattened))
        or set(ids["fit"] + ids["tune"] + ids["audit"]) != set(artifact.train_ids)
        or ids["official_test"] != list(artifact.test_ids)
    ):
        raise ValueError("editor partitions differ from official splits")


@dataclass(frozen=True)
class FactualPartition:
    z: torch.Tensor
    states: torch.Tensor
    ids: tuple


@dataclass(frozen=True)
class FittingData:
    """Only factual fitting/tuning Z and visible S; no audit/test rows or Z-prime."""

    fit: FactualPartition
    tune: FactualPartition
    columns: tuple
    matrix: torch.Tensor


def load_fitting_data(config, ids):
    artifact = load_latent_artifact(config)
    validate_ids(artifact, ids)
    columns, _ = load_schema(config["sim_config"])
    if [(c.name, c.n_categories) for c in columns] != [
        ("X", 4),
        ("T", 3),
        ("D", 3),
        ("U", 3),
    ] or load_intervention(config["sim_config"]) != {"X": 3}:
        raise ValueError("editors require the four-node talent do(X=3) protocol")
    names = [c.name for c in columns]
    # Auxiliary columns and counterfactual CSVs are never parsed by fitting.
    frame = pd.read_csv(
        config["paths"]["sim_factual"], usecols=["id", *names]
    ).set_index("id")
    if not frame.index.is_unique:
        raise ValueError("duplicate factual IDs")
    lookup = {identifier: i for i, identifier in enumerate(artifact.ids)}
    parts = {}
    for part in ("fit", "tune"):
        raw = frame.loc[ids[part], names].to_numpy()
        if not (raw == raw.astype(int)).all():
            raise ValueError("noninteger factual states")
        states = torch.tensor(raw, dtype=torch.long)
        for i, col in enumerate(columns):
            if ((states[:, i] < 0) | (states[:, i] >= col.n_categories)).any():
                raise ValueError("factual states outside schema")
        z = artifact.z[[lookup[i] for i in ids[part]]].clone()
        if not torch.isfinite(z).all() or not torch.allclose(
            z.norm(dim=-1), torch.ones(len(z)), atol=1e-5
        ):
            raise ValueError("expected finite unit-norm factual EmbeddingGemma inputs")
        parts[part] = FactualPartition(z, states, tuple(ids[part]))
    matrix = (
        load_symbolic_kernel(config["sim_config"])
        .transition_matrix({"X": 3})
        .to_dense()
        .detach()
    )
    return FittingData(parts["fit"], parts["tune"], tuple(columns), matrix)


def decoder_for(config, artifact):
    columns, _ = load_schema(config["sim_config"])
    # Legacy g provenance hashes the entire factual CSV (including auxiliary labels)
    # and paired latent container. Those hashes stay historical metadata, not
    # proxy/oracle-dependent gates for the clean editor. Verify encoder/schema
    # here; pin g's exact bytes and the consumed factual tensors in each editor run.
    model = (
        load_semantic_decoder(
            config["paths"]["decoder_model"],
            expected_variant="independent",
            expected_columns=columns,
            expected_metadata={"encoder": artifact.encoder_info},
        )
        .eval()
        .requires_grad_(False)
    )
    if model.latent_dim != artifact.z.shape[1]:
        raise ValueError("g and factual embeddings have incompatible dimensions")
    return model


def tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(
        str((tuple(value.shape), value.dtype)).encode() + value.numpy().tobytes()
    ).hexdigest()


def input_hashes(config, data, ids):
    # Hash the consumed data, not unused auxiliary columns or oracle payloads.
    result = {
        f"{part}.{field}": tensor_hash(getattr(getattr(data, part), field))
        for part in ("fit", "tune")
        for field in ("z", "states")
    }
    result.update(
        split_ids=digest(ids),
        matrix=tensor_hash(data.matrix),
        decoder=sha256_file(config["paths"]["decoder_model"]),
        schema=digest([(c.name, c.n_categories) for c in data.columns]),
    )
    return result


def metadata_for(config, settings, artifact, data):
    ids = split_ids(artifact, settings)
    sources = sorted(Path(__file__).parent.glob("*.py")) + [
        Path("src/latent_intervention/base/common.py"),
        Path("src/semantic_decoder/model.py"),
        Path("src/schema.py"),
        Path("src/symbolic_intervention.py"),
    ]
    return dict(
        variant=config["latent_intervention"]["variant"],
        dimension=artifact.z.shape[1],
        ids=ids,
        settings=settings,
        schema=[(c.name, c.n_categories) for c in data.columns],
        intervention={"X": 3},
        inputs=input_hashes(config, data, ids),
        source={str(p): sha256_file(p) for p in sources},
        config_sha256=digest(config),
        protocol_sha256=digest(settings),
        torch_version=str(torch.__version__),
        non_oracle=True,
        auxiliary_labels="excluded",
        reserved_audit="unused",
        selection_data="factual tuning objective only",
        newly_fitted=True,
    )


def support_report(data, target):
    size = len(data.matrix)
    counts = {
        name: torch.bincount(
            flat_state_index(getattr(data, name).states, data.columns), minlength=size
        )
        for name in ("fit", "tune")
    }
    missing = counts["fit"] == 0
    mass = target[:, missing].sum(-1)
    return dict(
        states=size,
        counts={k: v.tolist() for k, v in counts.items()},
        missing_fit_states=missing.nonzero().flatten().tolist(),
        sparse_fit_states_below_10=int(
            ((counts["fit"] > 0) & (counts["fit"] < 10)).sum()
        ),
        positive_target_states=int((target > 0).any(0).sum()),
        unsupported_target_mass=dict(
            mean=float(mass.mean()), maximum=float(mass.max())
        ),
        inference_policy="retain all positive targets; absent factual states require extrapolation",
    )


def paths_for(config, settings):
    if config["latent_intervention"]["variant"] not in EDITOR_VARIANTS:
        raise ValueError("not an editor variant")
    suffix = Path("runs") / settings["run_id"] / f"seed-{settings['seed']}"
    model = Path(config["paths"]["manipulator_model"]).parent / suffix
    reports = Path(config["paths"]["eval_report"]).parent / suffix
    paths = dict(
        model=model / "selected/latent_intervention.pt",
        info=model / "selected/latent_intervention.info.json",
        resume=model / "resume.pt",
        lock=model / ".run.lock",
    )
    paths.update(
        {
            name: reports / f"{name}.json"
            for name in ("protocol", "training", "evaluation")
        }
    )
    protected = {
        Path(config["paths"][k]).resolve()
        for k in ("latents", "decoder_model", "sim_factual", "pair_index")
    }
    if len({p.resolve() for p in paths.values()}) != len(paths) or any(
        p.resolve() in protected for p in paths.values()
    ):
        raise ValueError("editor output paths collide with inputs/each other")
    return paths
