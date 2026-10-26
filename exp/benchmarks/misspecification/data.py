"""Explicit factual boundaries. Auxiliary/CF labels never enter h_Z fitting."""

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd
import torch
import yaml

from exp.benchmarks.reader.protocol import STATE_SPACES, partitions
from src.artifact_io import sha256_file
from src.config import load_config
from src.pair_encoding import load_latent_artifact
from src.schema import ColumnSpec, flat_state_index
from src.semantic_decoder.model import load_semantic_decoder
from . import AUDIT_COLUMNS


@dataclass(frozen=True)
class FactualPartition:
    z: torch.Tensor
    states: torch.Tensor
    ids: tuple


@dataclass(frozen=True)
class FactualData:
    fit: FactualPartition
    tune: FactualPartition
    columns: tuple
    role: str = "h_z"


def columns_for(space):
    return tuple(ColumnSpec(n, 4 if n == "X" else 3) for n in STATE_SPACES[space])


def bundle(cfg):
    encoding = load_config(path=cfg["encoding_config"])
    if (
        encoding["encoder"]["variant"] != "embeddinggemma"
        or encoding["encoder"]["embeddinggemma_latent_dim"] != 768
    ):
        raise ValueError(
            "misspecification requires the existing EmbeddingGemma 768D artifact"
        )
    artifact = load_latent_artifact(encoding)
    manifest = json.loads(
        Path(encoding["paths"]["latents"]).with_name("manifest.json").read_text()
    )
    if not manifest.get("complete"):
        raise ValueError("encoding publication is incomplete")
    for key, path in {
        "sim_factual": encoding["paths"]["sim_factual"],
        "partition": encoding["paths"]["partition"],
        "sim_config": encoding["sim_config"],
    }.items():
        if sha256_file(path) != manifest["protocol"]["input_sha256"][key]:
            raise ValueError(f"encoding input has changed: {key}")
    frame = pd.read_csv(encoding["paths"]["partition"])
    reader = yaml.safe_load(Path(cfg["reader_protocol"]).read_text())
    roles = partitions(
        frame,
        artifact.ids,
        artifact.test_ids,
        reader["expected_counts"],
        reader["label_budget"],
    )
    ids = dict(
        fit=frame.loc[frame.partition.eq("pool"), "id"].tolist(),
        tune=roles["validation"],
        official_test=roles["official_test"],
        reader_fit=roles["evaluation_reader_reserved"],
        reader_L=roles["fit"],
    )
    if (len(ids["fit"]), len(ids["tune"]), len(ids["official_test"])) != (
        2800,
        200,
        1000,
    ):
        raise ValueError("unexpected misspecification split")
    return encoding, artifact, ids


def select_factual(artifact, factual_path, ids, columns, *, role="h_z"):
    if role not in ("h_z", "evaluation_reader"):
        raise ValueError("unknown factual data role")
    names = [c.name for c in columns]
    # usecols is deliberate: the objective cannot access unrequested labels.
    frame = pd.read_csv(factual_path, usecols=["id", *names])
    if frame.id.duplicated().any() or set(frame.id) != set(artifact.ids):
        raise ValueError("factual IDs do not match the encoding")
    frame = frame.set_index("id")
    positions = {v: i for i, v in enumerate(artifact.ids)}
    fitting = ids["fit"] if role == "h_z" else ids["reader_fit"]
    if set(fitting) & set(ids["tune"] + ids["official_test"]):
        raise ValueError("fitting/tuning/test overlap")
    parts = []
    for selected in (fitting, ids["tune"]):
        raw = frame.loc[selected, names].to_numpy(copy=True)
        states = torch.as_tensor(raw)
        if not torch.isfinite(states).all() or not torch.equal(states, states.round()):
            raise ValueError("non-integer factual states")
        states = states.long()
        for i, c in enumerate(columns):
            if (states[:, i] < 0).any() or (states[:, i] >= c.n_categories).any():
                raise ValueError("state outside schema")
        parts.append(
            FactualPartition(
                artifact.z[[positions[v] for v in selected]].clone(),
                states,
                tuple(selected),
            )
        )
    return FactualData(*parts, tuple(columns), role)


def load_factual(cfg, space, loaded=None):
    encoding, artifact, ids = loaded or bundle(cfg)
    return select_factual(
        artifact, encoding["paths"]["sim_factual"], ids, columns_for(space)
    )


def load_reader_data(cfg, loaded=None):
    encoding, artifact, ids = loaded or bundle(cfg)
    columns = tuple(ColumnSpec(n, 4 if n == "X" else 3) for n in AUDIT_COLUMNS)
    return select_factual(
        artifact,
        encoding["paths"]["sim_factual"],
        ids,
        columns,
        role="evaluation_reader",
    )


def guide_path(cfg, space, seed):
    return (
        Path(cfg["models_root"])
        / space
        / "runs"
        / cfg["reader_run_id"]
        / "final"
        / f"seed-{seed}"
        / "semantic_decoder.pt"
    )


def guide(cfg, space, seed, artifact=None):
    path = guide_path(cfg, space, seed)
    expected = (
        None
        if artifact is None
        else {
            "latent_artifact_sha256": artifact.artifact_sha256,
            "encoder": artifact.encoder_info,
        }
    )
    result = load_semantic_decoder(
        path,
        expected_variant="independent",
        expected_columns=columns_for(space),
        expected_metadata=expected,
    )
    identity = result.checkpoint_metadata["reader_fit"]
    if (
        identity["seed"] != seed
        or identity["state_space"] != space
        or identity["phase"] != "final"
    ):
        raise ValueError("wrong guiding reader run/seed/role")
    if identity["run_id"] != cfg["reader_run_id"]:
        raise ValueError("guiding reader run mismatch")
    return result.eval().requires_grad_(False)


def counts(data):
    import math

    return torch.bincount(
        flat_state_index(data.fit.states, data.columns),
        minlength=math.prod(c.n_categories for c in data.columns),
    )


def support_summary(data):
    values = counts(data)
    return dict(
        possible=len(values),
        observed=int((values > 0).sum()),
        unseen=int((values == 0).sum()),
        rare=int(((values > 0) & (values < 5)).sum()),
        counts=values.tolist(),
    )
