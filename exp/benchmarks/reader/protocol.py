"""Allowed factual inputs and provenance; never expose test/CF targets to fitting."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import pandas as pd
import torch
import yaml

from src.artifact_io import sha256_file
from src.config import load_config
from src.pair_encoding import _bool, _load_csv, load_latent_artifact
from src.schema import ColumnSpec
from src.semantic_decoder.model import targets_from_dataframe

CONFIG = "configs/reader_s1_s6.yaml"
STATE_SPACES = {
    "S1": ["X", "T", "D", "U"],
    "S2": ["X", "D", "U"],
    "S3": ["X"],
    "S4": ["X", "T", "U"],
    "S5": ["X", "U"],
    "S6": ["X", "T", "D", "U", "M"],
}


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def read_config(path=CONFIG, run_id=None):
    cfg = yaml.safe_load(Path(path).read_text())
    if run_id is not None:
        cfg["run_id"] = run_id
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", cfg["run_id"]):
        raise ValueError("run-id must be a simple directory name")
    if (
        cfg["state_spaces"] != STATE_SPACES
        or cfg["training"]["variant"] != "independent"
    ):
        raise ValueError("this runner supports independent readers for S1-S6 only")
    if min(cfg["workers"], cfg["threads_per_worker"], cfg["label_budget"]) < 1:
        raise ValueError("workers, threads and label budget must be positive")
    search = cfg["training"]["search"]
    seeds = search["final_seeds"]
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or search["deployment_seed"] not in seeds
    ):
        raise ValueError("final seeds must be unique and include the deployment seed")
    return cfg


def partitions(frame, ids, test_ids, expected, budget):
    """Validate persisted roles, including zero L_rank meaning *not* in L."""
    if frame.id.tolist() != ids or not set(frame.partition) == {"pool", "dev", "eval"}:
        raise ValueError("partition IDs or names do not match the factual artifact")
    ranks = pd.to_numeric(frame.L_rank, errors="coerce")
    if ranks.isna().any() or not ranks.eq(ranks.round()).all() or (ranks < 0).any():
        raise ValueError("invalid L ranks")
    in_p = pd.Series([_bool(v, i) for v, i in zip(frame.in_P, ids, strict=True)])
    labelled = ranks.gt(0)
    pool = frame.partition.eq("pool")
    if ((labelled | in_p) & ~pool).any() or (labelled & in_p).any():
        raise ValueError("L/P labels must be disjoint and restricted to the pool")
    if sorted(ranks[labelled].astype(int).tolist()) != list(
        range(1, expected["L"] + 1)
    ):
        raise ValueError("L ranks must be unique and contiguous")
    counts = {
        **frame.partition.value_counts().to_dict(),
        "L": int(labelled.sum()),
        "P": int(in_p.sum()),
    }
    if counts != expected or not 1 <= budget <= expected["L"]:
        raise ValueError(f"unexpected partition sizes or label budget: {counts}")
    split = {
        "fit": frame.loc[labelled & ranks.le(budget), "id"].tolist(),
        "validation": frame.loc[frame.partition.eq("dev"), "id"].tolist(),
        "evaluation_reader_reserved": frame.loc[in_p, "id"].tolist(),
        "official_test": frame.loc[frame.partition.eq("eval"), "id"].tolist(),
    }
    split["unused_pool"] = frame.loc[
        pool & ~frame.id.isin(split["fit"] + split["evaluation_reader_reserved"]), "id"
    ].tolist()
    if split["official_test"] != test_ids:
        raise ValueError("evaluation partition differs from the official test IDs")
    groups = list(map(set, split.values()))
    if any(a & b for i, a in enumerate(groups) for b in groups[i + 1 :]):
        raise ValueError("reader splits overlap")
    return split


@dataclass
class ReaderData:
    columns: list[ColumnSpec]
    fit_z: torch.Tensor
    fit_targets: dict[str, torch.Tensor]
    validation_z: torch.Tensor
    validation_targets: dict[str, torch.Tensor]
    split_ids: dict
    metadata: dict


def select_factual(artifact, factual_path, columns, split_ids, sim_config):
    """Only selected column labels of L/dev survive this boundary."""
    names = [c.name for c in columns]
    frame = pd.read_csv(factual_path, usecols=["id", *names])
    if frame.id.duplicated().any() or set(frame.id) != set(artifact.ids):
        raise ValueError("factual labels do not align with the latent IDs")
    frame = frame.set_index("id")
    positions = {row_id: i for i, row_id in enumerate(artifact.ids)}
    fit = split_ids["fit"]
    validation = split_ids["validation"]
    metadata = {
        "latent_artifact_sha256": artifact.artifact_sha256,
        "encoder": artifact.encoder_info,
        "training_inputs": {
            "sim_factual_sha256": sha256_file(factual_path),
            "sim_config_sha256": sha256_file(sim_config),
        },
        "split_sha256": digest(split_ids),
        "label_policy": "only L and development labels of the named state space",
    }
    return ReaderData(
        columns,
        artifact.z[[positions[i] for i in fit]].clone(),
        targets_from_dataframe(frame.loc[fit, names], columns),
        artifact.z[[positions[i] for i in validation]].clone(),
        targets_from_dataframe(frame.loc[validation, names], columns),
        split_ids,
        metadata,
    )


def load_data(cfg):
    encoding = load_config(path=cfg["encoding_config"])
    if (
        encoding["encoder"]["variant"] != "embeddinggemma"
        or encoding["encoder"]["embeddinggemma_latent_dim"] != 768
    ):
        raise ValueError("this experiment uses EmbeddingGemma 768D only")
    paths = encoding["paths"]
    # Paired-artifact validation checks integrity, not predictions or test scores.
    # Only factual L/dev tensors are returned; z_prime is never exposed to fitting.
    artifact = load_latent_artifact(encoding)
    manifest_path = Path(paths["latents"]).with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("complete"):
        raise ValueError("encoding publication is incomplete")
    recorded = manifest["protocol"]["input_sha256"]
    for key, path in {
        "sim_factual": paths["sim_factual"],
        "partition": paths["partition"],
        "sim_config": encoding["sim_config"],
    }.items():
        if recorded[key] != sha256_file(path):
            raise ValueError(f"encoding/reader input mismatch: {key}")
    partition = _load_csv(
        paths["partition"], "partition", {"id", "partition", "L_rank", "in_P"}
    )
    split = partitions(
        partition,
        artifact.ids,
        artifact.test_ids,
        cfg["expected_counts"],
        cfg["label_budget"],
    )
    sim = yaml.safe_load(Path(encoding["sim_config"]).read_text())
    cards = {**sim["schema"]["columns"], **sim["schema"]["auxiliary"]}
    cases = {
        name: select_factual(
            artifact,
            paths["sim_factual"],
            [ColumnSpec(c, int(cards[c])) for c in names],
            split,
            encoding["sim_config"],
        )
        for name, names in cfg["state_spaces"].items()
    }
    inputs = {
        str(Path(cfg["encoding_config"])),
        str(Path(encoding["sim_config"])),
        str(manifest_path),
        str(Path(paths["latents"]).with_suffix(".info.json")),
        *(
            paths[key]
            for key in (
                "latents",
                "sim_factual",
                "partition",
                "pair_index",
                "texts",
                "texts_counterfactual",
            )
        ),
    }
    return cases, {p: sha256_file(p) for p in sorted(inputs)}
