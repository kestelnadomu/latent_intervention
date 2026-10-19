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

from exp.sim.design import (
    PARTITION_COLUMNS,
    build_partition,
    design_paths,
    draw_interventions,
    load_design,
    validate_per_unit_query,
)
from exp.sim.helpers import coerce_bool, read_csv, sha256_file
from exp.sim.pairing import PairingError, build_pair_index
from src.schema import ColumnSpec


def state_columns(config: Mapping[str, Any]) -> list[str]:
    """Names of the structured-state columns, in decode order."""
    return [str(column) for column in config["schema"]["columns"]]


def auxiliary_columns(config: Mapping[str, Any]) -> list[ColumnSpec]:
    """
    Auxiliary columns (``schema.auxiliary``): simulated and verbalized, but outside S.

    They exist only to make hidden S columns inferable from text, so they enter
    neither the decoder heads, the consistency loss, nor the pair identity check.
    Absent in schemas without them.
    """
    auxiliary = config["schema"].get("auxiliary") or {}
    return [ColumnSpec(str(name), int(card)) for name, card in auxiliary.items()]


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
        **design_paths(config),
    }


def simulation_record(config: Mapping[str, Any]) -> dict[str, Any]:
    """The provenance record persisted as ``simulation_info.json``."""
    paths = simulation_paths(config)
    split_seed, test_fraction = split_settings(config)
    record = {
        "n": int(config["n"]),
        "simulation_seed": config.get("seed"),
        "split_seed": split_seed,
        "test_fraction": test_fraction,
        "intervention": config["intervention"],
        "schema": config["schema"],
        "objects": config["objects"],
        "files": {name: sha256_file(path) for name, path in paths.items()},
    }
    if "design" in config:  # absent for fixed-query configs, so their records are unchanged
        record["design"] = config["design"]
    return record


def validate_query(
    factual: pd.DataFrame,
    counterfactual: pd.DataFrame,
    config: Mapping[str, Any],
    interventions: pd.DataFrame | None = None,
) -> None:
    """Check S' against the configured query: the fixed do(), or each unit's drawn one."""
    spec = load_design(config)
    if spec is None:
        validate_fixed_query(factual, counterfactual, config)
        return
    if interventions is None:
        interventions = read_csv(design_paths(config)["interventions"], "interventions", ("do_value",))
    validate_per_unit_query(factual, counterfactual, interventions, spec)


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
    rendered = [*columns, *(column.name for column in auxiliary_columns(config))]
    factual = read_csv(sim_dir / "sim_data_factual.csv", "factual simulation", rendered)
    counterfactual = read_csv(
        sim_dir / "sim_data_counterfactual.csv", "counterfactual simulation", rendered
    )
    pairs = read_csv(config["paths"]["pair_index"], "pair index", ("split", "is_identity"))
    ids = factual["id"].tolist()
    if counterfactual["id"].tolist() != ids or pairs["id"].tolist() != ids:
        raise PairingError("S, S', and pair_index.csv must contain exactly the same IDs")
    if len(ids) != int(config["n"]):
        raise PairingError(f"simulation has {len(ids)} units, expected n={config['n']}")
    validate_query(factual, counterfactual, config)
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
    _validate_design_files(config, factual, pairs)
    return factual, counterfactual, pairs


def _validate_design_files(
    config: Mapping[str, Any], factual: pd.DataFrame, pairs: pd.DataFrame
) -> None:
    """Re-derive the drawn interventions and the partition of a per-unit design."""
    spec = load_design(config)
    if spec is None:
        return
    paths = design_paths(config)
    cardinality = int(config["schema"]["columns"][spec.node])
    expected = draw_interventions(factual, spec.node, cardinality, int(config["seed"]))
    interventions = read_csv(paths["interventions"], "interventions", ("do_value",))
    partition = read_csv(paths["partition"], "partition", PARTITION_COLUMNS)
    expected_partition = build_partition(pairs, spec, split_settings(config)[0])
    try:
        pd.testing.assert_frame_equal(interventions, expected, check_dtype=False)
        pd.testing.assert_frame_equal(partition[PARTITION_COLUMNS], expected_partition, check_dtype=False)
    except AssertionError as exc:
        raise PairingError("interventions.csv or partition.csv does not match the configured design") from exc
