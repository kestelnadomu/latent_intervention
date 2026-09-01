"""Structured S/S' state: paths, the run record, and the paired-data contract.

Everything the CV stages need in order to trust that ``data/sim/`` and
``pair_index.csv`` still describe the configuration they are running under.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from exp.sim.helpers import coerce_bool, read_csv, sha256_file
from exp.sim.pairing import PairingError, build_pair_index


def state_columns(config: Mapping[str, Any]) -> list[str]:
    """Names of the structured-state columns, in decode order."""
    return [str(column) for column in config["schema"]["columns"]]


def split_settings(config: Mapping[str, Any]) -> tuple[int, float]:
    """Seed and test fraction of the deterministic train/test split."""
    return int(config["split"]["seed"]), float(config["split"].get("test_fraction", 0.2))


def simulation_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    """Output paths written by the simulate stage."""
    sim_dir = Path(config["paths"]["sim_dir"])
    return {
        "factual": sim_dir / "sim_data_factual.csv",
        "counterfactual": sim_dir / "sim_data_counterfactual.csv",
        "epsilon": sim_dir / "sim_data_epsilon.csv",
        "pair_index": Path(config["paths"]["pair_index"]),
    }


def simulation_record(config: Mapping[str, Any]) -> dict[str, Any]:
    """The provenance record persisted as ``simulation_info.json``."""
    paths = simulation_paths(config)
    split_seed, test_fraction = split_settings(config)
    return {
        "n": int(config["n"]),
        "simulation_seed": config.get("seed"),
        "split_seed": split_seed,
        "test_fraction": test_fraction,
        "intervention": config["intervention"],
        "schema": config["schema"],
        "objects": config["objects"],
        "files": {name: sha256_file(path) for name, path in paths.items()},
    }


def validate_fixed_query(
    factual: pd.DataFrame,
    counterfactual: pd.DataFrame,
    config: Mapping[str, Any],
) -> None:
    """Check that S' realises the one fixed do() query the texts are rendered for."""
    columns = state_columns(config)
    intervention = {str(key): int(value) for key, value in config["intervention"].items()}
    if not set(intervention) <= set(columns):
        raise ValueError("the fixed text-rendering intervention must target schema.columns")
    for column, target in intervention.items():
        if not counterfactual[column].eq(target).all():
            raise ValueError(f"counterfactual {column} does not equal target {target}")
    target_rows = pd.Series(True, index=factual.index)
    for column, target in intervention.items():
        target_rows &= factual[column].eq(target)
    if not factual.loc[target_rows, columns].equals(counterfactual.loc[target_rows, columns]):
        raise ValueError("units already at all intervention targets must be structured identities")


def load_pair_inputs(
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load S, S', and the pair index, re-deriving every invariant they must satisfy."""
    sim_dir = Path(config["paths"]["sim_dir"])
    info_path = sim_dir / "simulation_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"simulation information not found: {info_path}; run simulate")
    if json.loads(info_path.read_text(encoding="utf-8")) != simulation_record(config):
        raise RuntimeError(
            "simulation_info.json is incompatible with the current configuration or files"
        )
    columns = state_columns(config)
    factual = read_csv(sim_dir / "sim_data_factual.csv", "factual simulation", columns)
    counterfactual = read_csv(
        sim_dir / "sim_data_counterfactual.csv", "counterfactual simulation", columns
    )
    pairs = read_csv(config["paths"]["pair_index"], "pair index", ("split", "is_identity"))
    ids = factual["id"].tolist()
    if counterfactual["id"].tolist() != ids or pairs["id"].tolist() != ids:
        raise PairingError("S, S', and pair_index.csv must contain exactly the same IDs")
    if len(ids) != int(config["n"]):
        raise PairingError(f"simulation has {len(ids)} units, expected n={config['n']}")
    validate_fixed_query(factual, counterfactual, config)
    pairs["split"] = pairs["split"].astype(str).str.lower()
    if set(pairs["split"]) != {"train", "test"}:
        raise PairingError("pair index must contain non-empty train and test splits")
    pairs["is_identity"] = [
        coerce_bool(value, f"pair index id={row_id} is_identity")
        for row_id, value in zip(pairs["id"], pairs["is_identity"])
    ]
    split_seed, test_fraction = split_settings(config)
    expected_pairs = build_pair_index(
        factual,
        counterfactual,
        columns,
        seed=split_seed,
        test_fraction=test_fraction,
    )
    try:
        pd.testing.assert_frame_equal(pairs, expected_pairs, check_dtype=False)
    except AssertionError as exc:
        raise PairingError("pair_index.csv does not match the configured seeded split") from exc
    return factual, counterfactual, pairs
