"""Small factual-only data protocol; no dependency on prior experiments."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import torch
import yaml

from src.artifact_io import sha256_file
from src.pair_encoding import load_latent_artifact
from src.schema import ColumnSpec, load_intervention, load_schema
from src.semantic_decoder.model import load_semantic_decoder
from src.symbolic_intervention import load_symbolic_kernel
from .model import VARIANT

DEFAULT_PROTOCOL = Path("configs/hz_csym_project.yaml")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def read_settings(path=DEFAULT_PROTOCOL):
    settings = yaml.safe_load(Path(path).read_text())
    for name in (
        "threads",
        "max_epochs",
        "patience",
        "warmup_epochs",
        "warmup_patience",
        "batch_size",
        "support_bank_size",
        "hidden_dim",
        "state_embed_dim",
        "evaluation_samples",
        "evaluation_batch_size",
    ):
        if type(settings[name]) is not int or settings[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("seed", "split_seed", "audit_split_seed", "evaluation_seed"):
        if type(settings[name]) is not int or not 0 <= settings[name] < 2**32:
            raise ValueError(f"invalid {name}")
    if (
        not settings["dimensions"]
        or len(set(settings["dimensions"])) != len(settings["dimensions"])
        or not set(settings["dimensions"]) <= {256, 768}
        or settings["protected_columns"] != ["P", "L", "H", "A"]
        or not 0 < settings["validation_fraction"] < 1
        or settings["evaluation_samples"] < 2
    ):
        raise ValueError("unsupported projected C-sym protocol")
    for value in (
        settings["learning_rate"],
        settings["weight_decay"],
        settings["semantic_tolerance"],
        *settings["loss"].values(),
    ):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("rates and weights must be finite and nonnegative")
    if (
        min(
            settings["learning_rate"],
            settings["loss"]["factual"],
            settings["semantic_tolerance"],
        )
        <= 0
    ):
        raise ValueError(
            "positive learning rate, factual weight and semantic tolerance required"
        )
    probe = settings["probe"]
    for name in ("max_epochs", "patience", "batch_size"):
        if type(probe[name]) is not int or probe[name] < 1:
            raise ValueError(f"invalid probe {name}")
    for name in ("guide_lr", "auditor_lr"):
        if not math.isfinite(probe[name]) or probe[name] <= 0:
            raise ValueError(f"invalid probe {name}")
    return settings


def paths_for(config):
    model = Path(config["paths"]["manipulator_model"])
    report = Path(config["paths"]["eval_report"]).parent
    paths = dict(
        model=model,
        info=model.with_suffix(".info.json"),
        readers=model.parent / "audit_readers.pt",
    )
    paths.update(
        {
            name: report / f"{name}.json"
            for name in (
                "protocol",
                "training",
                "probes",
                "audit",
                "evaluation",
                "migration",
                "original_evaluation",
            )
        }
    )
    protected = {
        Path(p).resolve()
        for key, p in config["paths"].items()
        if key
        not in {
            "manipulator_model",
            "eval_report",
            "flow_manipulator_model",
            "flow_eval_report",
            "oracle_training_report",
        }
        and isinstance(p, str)
    }
    if len({p.resolve() for p in paths.values()}) != len(paths) or any(
        p.resolve() in protected for p in paths.values()
    ):
        raise ValueError(
            "C-sym outputs must be distinct from each other and upstream data"
        )
    return paths


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
    return dict(
        fit=[artifact.ids[i] for i in fit.tolist()],
        tune=[artifact.ids[i] for i in validation[order[:middle]].tolist()],
        audit=[artifact.ids[i] for i in validation[order[middle:]].tolist()],
        official_test=list(artifact.test_ids),
    )


def validate_ids(artifact, ids):
    if set(ids) != {"fit", "tune", "audit", "official_test"}:
        raise ValueError("invalid C-sym partitions")
    flat = [i for part in ids.values() for i in part]
    if (
        any(not part for part in ids.values())
        or len(flat) != len(set(flat))
        or set(ids["fit"] + ids["tune"] + ids["audit"]) != set(artifact.train_ids)
        or ids["official_test"] != artifact.test_ids
    ):
        raise ValueError("C-sym partitions differ from the official split")


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


def input_hashes(config):
    files = [
        config["paths"][k]
        for k in ("latents", "decoder_model", "sim_factual", "pair_index")
    ]
    files += [
        config["sim_config"],
        str(Path(config["paths"]["latents"]).with_suffix(".info.json")),
    ]
    return {str(path): sha256_file(path) for path in files}


def verify_inputs(metadata):
    for path, expected in metadata["inputs"].items():
        if sha256_file(path) != expected:
            raise ValueError(f"C-sym input changed: {path}")


def metadata_for(config, settings, artifact):
    columns, _ = load_schema(config["sim_config"])
    ids = split_ids(artifact, settings)
    validate_ids(artifact, ids)
    return dict(
        variant=VARIANT,
        dimension=artifact.z.shape[1],
        ids=ids,
        settings=settings,
        schema=[(c.name, c.n_categories) for c in columns],
        intervention={"X": 3},
        inputs=input_hashes(config),
        source={
            str(p): sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))
        },
        non_oracle=True,
        flow_teacher=False,
    )


def load_data(config, ids, *, parts=("fit", "tune")):
    """Expose only factual official-training rows; discard the container's test Z'."""
    if not parts or not set(parts) <= {"fit", "tune", "audit"}:
        raise ValueError("only factual official-training partitions are available")
    artifact = load_latent_artifact(config)
    validate_ids(artifact, ids)
    columns, _ = load_schema(config["sim_config"])
    if [(c.name, c.n_categories) for c in columns] != [
        ("X", 4),
        ("T", 3),
        ("D", 3),
        ("U", 3),
    ] or load_intervention(config["sim_config"]) != {"X": 3}:
        raise ValueError(
            "projected C-sym requires the four-node talent do(X=3) protocol"
        )
    if not torch.allclose(
        artifact.z.norm(dim=-1), torch.ones(len(artifact.z)), atol=1e-5
    ):
        raise ValueError("C-sym geometry expects unit-norm input embeddings")
    protected = [ColumnSpec(name, 3) for name in ("P", "L", "H", "A")]
    frame = pd.read_csv(config["paths"]["sim_factual"]).set_index("id")
    if not frame.index.is_unique:
        raise ValueError("duplicate factual IDs")
    lookup = {value: i for i, value in enumerate(artifact.ids)}
    result = {}
    names = [c.name for c in [*columns, *protected]]
    for part in parts:
        raw = frame.loc[ids[part], names].to_numpy()
        if not (raw == raw.astype(int)).all():
            raise ValueError("noninteger factual labels")
        labels = torch.tensor(raw, dtype=torch.long)
        for i, column in enumerate([*columns, *protected]):
            if ((labels[:, i] < 0) | (labels[:, i] >= column.n_categories)).any():
                raise ValueError("factual labels outside schema")
        result[part] = dict(
            z=artifact.z[[lookup[i] for i in ids[part]]].clone(), labels=labels
        )
    result.update(
        decoder=decoder_for(config, artifact),
        kernel=load_symbolic_kernel(config["sim_config"]),
        columns=columns,
        protected=protected,
    )
    return result
