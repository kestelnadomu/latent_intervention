"""Factual-only OT data boundary, support accounting and reproducible run layout."""

from __future__ import annotations

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
from src.schema import load_schema, load_intervention, flat_state_index
from src.semantic_decoder.model import load_semantic_decoder
from src.symbolic_intervention import load_symbolic_kernel
from . import OT_VARIANTS

DEFAULT_PROTOCOL = Path("configs/hz_ot.yaml")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def validate_settings(settings):
    for name in (
        "threads",
        "max_epochs",
        "batch_size",
        "patience",
        "scheduler_patience",
        "state_embed_dim",
        "evaluation_samples",
        "evaluation_batch_size",
    ):
        if type(settings[name]) is not int or settings[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("seed", "split_seed", "audit_split_seed", "evaluation_seed"):
        if type(settings[name]) is not int or not 0 <= settings[name] < 2**32:
            raise ValueError(f"invalid {name}")
    for name in ("learning_rate", "grad_clip", "min_lr", "smoothing_std"):
        if not math.isfinite(settings[name]) or settings[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if (
        settings["tier"] not in (1, 2)
        or not 0 < settings["validation_fraction"] < 1
        or not 0 < settings["scheduler_factor"] < 1
        or settings["evaluation_samples"] < 2
        or not settings["dimensions"]
        or len(set(settings["dimensions"])) != len(settings["dimensions"])
        or not set(settings["dimensions"]) <= {256, 768}
        or not math.isfinite(settings["min_delta"])
        or settings["min_delta"] < 0
    ):
        raise ValueError("invalid OT protocol settings")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", settings["run_id"]):
        raise ValueError(
            "run_id must contain only letters, digits, hyphens and underscores"
        )
    solver = settings["sinkhorn"]
    if (
        solver["epsilon"] <= 0
        or solver["tolerance"] <= 0
        or not math.isfinite(solver["epsilon"])
        or not math.isfinite(solver["tolerance"])
        or type(solver["max_iterations"]) is not int
        or solver["max_iterations"] < 1
    ):
        raise ValueError("invalid Sinkhorn settings")
    return settings


def read_settings(path=DEFAULT_PROTOCOL, overrides=None):
    settings = yaml.safe_load(Path(path).read_text())
    settings.update(overrides or {})
    return validate_settings(settings)


def model_options(variant, settings):
    if variant == "ot_gaussian":
        return dict(settings["gaussian"])
    options = dict(tier=settings["tier"], state_embed_dim=settings["state_embed_dim"])
    if variant != "ot_flow":
        raise ValueError("not an active standalone OT variant")
    options.update(settings["flow"])
    return options


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
    result = dict(
        fit=[artifact.ids[i] for i in fit.tolist()],
        tune=[artifact.ids[i] for i in validation[order[:middle]].tolist()],
        audit=[artifact.ids[i] for i in validation[order[middle:]].tolist()],
        official_test=list(artifact.test_ids),
    )
    validate_ids(artifact, result)
    return result


def validate_ids(artifact, ids):
    if set(ids) != {"fit", "tune", "audit", "official_test"}:
        raise ValueError("invalid OT split IDs")
    flattened = [i for part in ids.values() for i in part]
    if (
        any(not part for part in ids.values())
        or len(flattened) != len(set(flattened))
        or set(ids["fit"] + ids["tune"] + ids["audit"]) != set(artifact.train_ids)
        or ids["official_test"] != artifact.test_ids
    ):
        raise ValueError("OT partitions differ from official splits")


@dataclass(frozen=True)
class FactualPartition:
    z: torch.Tensor
    states: torch.Tensor
    ids: tuple


@dataclass(frozen=True)
class FittingData:
    """No decoder, auxiliary labels, audit rows, test rows or oracle targets."""

    fit: FactualPartition
    tune: FactualPartition
    columns: tuple
    matrix: torch.Tensor


def load_factual(config, ids, *, parts=("fit", "tune")):
    if not parts or not set(parts) <= {"fit", "tune", "audit"}:
        raise ValueError("only factual official-training partitions may be loaded")
    artifact = load_latent_artifact(config)
    validate_ids(artifact, ids)
    columns, _ = load_schema(config["sim_config"])
    if [(c.name, c.n_categories) for c in columns] != [
        ("X", 4),
        ("T", 3),
        ("D", 3),
        ("U", 3),
    ] or load_intervention(config["sim_config"]) != {"X": 3}:
        raise ValueError("OT workflow requires the four-node talent do(X=3) protocol")
    names = [c.name for c in columns]
    # Do not even parse P/L/H/A or the counterfactual CSV here.
    frame = pd.read_csv(
        config["paths"]["sim_factual"], usecols=["id", *names]
    ).set_index("id")
    if not frame.index.is_unique:
        raise ValueError("duplicate factual IDs")
    lookup = {identifier: i for i, identifier in enumerate(artifact.ids)}
    result = {}
    for part in parts:
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
            raise ValueError("OT workflow expects finite unit-norm factual embeddings")
        result[part] = FactualPartition(z, states, tuple(ids[part]))
    return result, tuple(columns)


def load_fitting_data(config, ids):
    parts, columns = load_factual(config, ids)
    matrix = (
        load_symbolic_kernel(config["sim_config"])
        .transition_matrix({"X": 3})
        .to_dense()
        .detach()
    )
    return FittingData(parts["fit"], parts["tune"], columns, matrix)


def decoder_for(config, artifact):
    from src.pipeline import _decoder_metadata

    columns, _ = load_schema(config["sim_config"])
    return (
        load_semantic_decoder(
            config["paths"]["decoder_model"],
            expected_variant="independent",
            expected_columns=columns,
            expected_metadata=_decoder_metadata(config, artifact),
        )
        .eval()
        .requires_grad_(False)
    )


def support_report(data, posterior=None):
    size = len(data.matrix)
    counts = {
        name: torch.bincount(
            flat_state_index(getattr(data, name).states, data.columns), minlength=size
        )
        for name in ("fit", "tune")
    }
    src, dst = (data.matrix > 0).nonzero(as_tuple=True)
    changed = src != dst
    fit_ok = (counts["fit"][src] > 0) & (counts["fit"][dst] > 0)
    tune_ok = fit_ok & (counts["tune"][src] > 0) & (counts["tune"][dst] > 0)
    population = counts["fit"].float() / counts["fit"].sum()
    edge_mass = population[src] * data.matrix[src, dst]
    result = dict(
        states=size,
        counts={k: v.tolist() for k, v in counts.items()},
        missing_fit_states=(counts["fit"] == 0).nonzero().flatten().tolist(),
        sparse_fit_states_below_10=int((counts["fit"] < 10).sum()),
        positive_transitions=len(src),
        fitting_nonidentity_edges=int((fit_ok & changed).sum()),
        validation_nonidentity_edges=int((tune_ok & changed).sum()),
        factual_weighted_unsupported_mass=float(edge_mass[~fit_ok].sum()),
        factual_weighted_validation_coverage=float(edge_mass[tune_ok & changed].sum()),
        inference_policy="retain all transitions; unsupported endpoints are extrapolation",
    )
    if posterior is not None:
        mass = (posterior[:, src] * data.matrix[src, dst][None])[:, ~fit_ok].sum(-1)
        result["posterior_unsupported_mass"] = dict(
            mean=float(mass.mean()), maximum=float(mass.max())
        )
    return result


def input_hashes(config):
    paths = [
        config["paths"][name]
        for name in ("latents", "decoder_model", "sim_factual", "pair_index")
    ]
    paths += [
        config["sim_config"],
        str(Path(config["paths"]["latents"]).with_suffix(".info.json")),
    ]
    return {str(p): sha256_file(p) for p in paths}


def verify_inputs(metadata):
    for path, checksum in metadata["inputs"].items():
        if sha256_file(path) != checksum:
            raise ValueError(f"OT input changed: {path}")


def metadata_for(config, settings, artifact):
    columns, _ = load_schema(config["sim_config"])
    return dict(
        variant=config["latent_intervention"]["variant"],
        dimension=artifact.z.shape[1],
        ids=split_ids(artifact, settings),
        settings=settings,
        inputs=input_hashes(config),
        schema=[(c.name, c.n_categories) for c in columns],
        intervention={"X": 3},
        config_sha256=digest(config),
        protocol_sha256=digest(settings),
        source={
            str(p): sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))
        },
        torch_version=str(torch.__version__),
        non_oracle=True,
        preservation="audit_only",
        selection_data="factual tuning partition only",
        flow_teacher=False,
    )


def paths_for(config, settings):
    variant = config["latent_intervention"]["variant"]
    if variant not in OT_VARIANTS:
        raise ValueError("not an OT variant")
    suffix = Path("runs") / settings["run_id"]
    model_root = Path(config["paths"]["manipulator_model"]).parent / suffix
    report_root = Path(config["paths"]["eval_report"]).parent / suffix
    paths = dict(
        model=model_root / "selected" / "latent_intervention.pt",
        info=model_root / "selected" / "latent_intervention.info.json",
        resume=model_root / "resume.pt",
        readers=model_root / "selected" / "audit_readers.pt",
        lock=model_root / ".run.lock",
    )
    paths.update(
        {
            name: report_root / f"{name}.json"
            for name in ("protocol", "training", "audit", "evaluation")
        }
    )
    protected = {
        Path(config["paths"][key]).resolve()
        for key in ("latents", "decoder_model", "sim_factual", "pair_index")
    }
    if len({p.resolve() for p in paths.values()}) != len(paths) or any(
        p.resolve() in protected for p in paths.values()
    ):
        raise ValueError("OT output paths collide with inputs/each other")
    return paths
