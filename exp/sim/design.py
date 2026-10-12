"""Per-unit counterfactual design: interventions, counterfactual states, partitions.

Opt-in through a ``design:`` block in the sim config (see
``exp/sim_talent_motivation/config.yaml``); configs without it keep the single
fixed ``do()`` query. Generic: node names come from the config only.

With a design, the simulate stage additionally writes

- ``interventions.csv`` (``id, do_value``): one non-identity ``do(node = do_value)``
  per unit, drawn from a per-unit generator;
- ``sim_data_counterfactual_states.csv`` (``id, do_value, target, <nodes>``): every
  unit under every ``do_value`` (identity included) and every target. A target is
  a set of blocked edges; ``{}`` is the total effect;
- ``partition.csv`` (``id, partition, L_rank, in_P, fold, floor``): the training
  pool / dev / evaluation partition and the label roles inside the pool.

``sim_data_counterfactual.csv`` then holds, per unit, the state under its drawn
intervention and the text target: the one counterfactual that is rendered as text.
Design: ``docs/experiments/data_sim.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from exp.sim.pairing import PairingError
from exp.sim.scm import SCM

PARTITION_COLUMNS = ["id", "partition", "L_rank", "in_P", "fold", "floor"]


@dataclass(frozen=True)
class DesignSpec:
    """The ``design:`` block of a sim config, validated."""

    node: str
    targets: dict[str, dict[str, list[str]]]
    text_target: str
    n_dev: int
    n_L: int
    n_P: int
    folds: int
    floor_fraction: float


def load_design(config: Mapping[str, Any]) -> DesignSpec | None:
    """The configured per-unit design, or ``None`` for a fixed-query config."""
    raw = config.get("design")
    if raw is None:
        return None
    if raw.get("interventions") != "per_unit":
        raise ValueError("design.interventions must be 'per_unit'")
    node = str(raw["node"])
    columns = [str(column) for column in config["schema"]["columns"]]
    if node not in columns:
        raise ValueError(f"design.node {node!r} must be a schema column")
    targets = {
        str(name): {str(child): [str(p) for p in parents] for child, parents in (edges or {}).items()}
        for name, edges in raw["targets"].items()
    }
    text_target = str(raw.get("text_target", "total"))
    if text_target not in targets:
        raise ValueError(f"design.text_target {text_target!r} is not a configured target")
    partition = raw["partition"]
    spec = DesignSpec(
        node=node,
        targets=targets,
        text_target=text_target,
        n_dev=int(partition["dev"]),
        n_L=int(partition["L"]),
        n_P=int(partition["P"]),
        folds=int(partition["folds"]),
        floor_fraction=float(partition["floor_fraction"]),
    )
    if min(spec.n_dev, spec.n_L, spec.n_P) < 0 or spec.folds < 2:
        raise ValueError("design.partition sizes must be non-negative and folds >= 2")
    if not 0 <= spec.floor_fraction <= 1:
        raise ValueError("design.partition.floor_fraction must lie in [0, 1]")
    return spec


def design_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    """Extra simulate outputs of a per-unit design (empty without one)."""
    if load_design(config) is None:
        return {}
    sim_dir = Path(config["paths"]["sim_dir"])
    return {
        "interventions": sim_dir / "interventions.csv",
        "counterfactual_states": sim_dir / "sim_data_counterfactual_states.csv",
        "partition": sim_dir / "partition.csv",
    }


def noise_from_epsilon(epsilon: pd.DataFrame) -> dict[str, np.ndarray]:
    """Recover the per-node noise arrays from a simulate ``epsilon`` frame."""
    return {
        column.removeprefix("eps_"): epsilon[column].to_numpy()
        for column in epsilon.columns
        if column.startswith("eps_")
    }


def draw_interventions(
    factual: pd.DataFrame, node: str, cardinality: int, seed: int
) -> pd.DataFrame:
    """One non-identity ``do(node = value)`` per unit, from a per-unit generator."""
    values = []
    for unit_id, current in zip(factual["id"], factual[node]):
        rng = np.random.default_rng([int(seed), 3, int(unit_id)])
        options = [value for value in range(cardinality) if value != int(current)]
        values.append(options[int(rng.integers(len(options)))])
    return pd.DataFrame({"id": factual["id"].to_numpy(), "do_value": values})


def counterfactual_states(
    scm: SCM,
    noise: Mapping[str, np.ndarray],
    factual: pd.DataFrame,
    spec: DesignSpec,
    cardinality: int,
) -> pd.DataFrame:
    """Every unit under every ``do(node = k)`` and every target, sharing the factual noise."""
    reference = factual.drop(columns="id").reset_index(drop=True)
    frames = []
    for value in range(cardinality):
        for target, blocked in spec.targets.items():
            frame = scm.generate(dict(noise), {spec.node: value}, blocked or None, reference)
            frame.insert(0, "id", factual["id"].to_numpy())
            frame.insert(1, "do_value", value)
            frame.insert(2, "target", target)
            frames.append(frame)
    states = pd.concat(frames, ignore_index=True)
    order = {target: rank for rank, target in enumerate(spec.targets)}
    return (
        states.sort_values(["id", "do_value", "target"], key=lambda s: s.map(order) if s.name == "target" else s)
        .reset_index(drop=True)
    )


def text_counterfactual(
    states: pd.DataFrame, interventions: pd.DataFrame, spec: DesignSpec, columns: list[str]
) -> pd.DataFrame:
    """Per unit, the state under its drawn intervention and the text target."""
    chosen = states[states["target"] == spec.text_target].merge(
        interventions, on=["id", "do_value"], how="inner"
    )
    if chosen["id"].tolist() != interventions["id"].tolist():
        raise PairingError("every unit needs exactly one text counterfactual")
    return chosen[["id", *columns]].reset_index(drop=True)


def build_partition(pairs: pd.DataFrame, spec: DesignSpec, seed: int) -> pd.DataFrame:
    """
    Training pool / dev / evaluation partition, and the label roles in the pool.

    The evaluation units are the pair index's test units. Of the rest, ``n_dev``
    form the dev partition and the others the training pool. In the pool, ``L``
    (the reader's labels) gets ranks 1..n_L, so a label budget m is the prefix
    ``L_rank <= m``; ``P`` (the evaluation reader's labels) is disjoint from ``L``.
    Pool units get cross-fitting folds by their position in the shuffled pool, so
    every prefix of ``L`` is spread over all folds. A ``floor_fraction`` of the
    evaluation units get a noise-floor text.
    """
    train = np.sort(pairs.loc[pairs["split"] == "train", "id"].to_numpy())
    test = np.sort(pairs.loc[pairs["split"] == "test", "id"].to_numpy())
    rng = np.random.default_rng([int(seed), 4])
    shuffled = rng.permutation(train)
    dev, pool = shuffled[: spec.n_dev], shuffled[spec.n_dev :]
    if len(dev) < spec.n_dev or len(pool) < spec.n_L + spec.n_P:
        raise PairingError(
            f"{len(train)} training units cannot hold dev={spec.n_dev}, L={spec.n_L}, P={spec.n_P}"
        )
    floor = set(rng.permutation(test)[: round(spec.floor_fraction * len(test))].tolist())

    rows = {}
    for unit_id in dev:
        rows[int(unit_id)] = ("dev", 0, False, -1, False)
    for position, unit_id in enumerate(pool):
        l_rank = position + 1 if position < spec.n_L else 0
        in_p = spec.n_L <= position < spec.n_L + spec.n_P
        rows[int(unit_id)] = ("pool", l_rank, in_p, position % spec.folds, False)
    for unit_id in test:
        rows[int(unit_id)] = ("eval", 0, False, -1, int(unit_id) in floor)
    ids = sorted(rows)
    return pd.DataFrame([[unit_id, *rows[unit_id]] for unit_id in ids], columns=PARTITION_COLUMNS)


def validate_per_unit_query(
    factual: pd.DataFrame,
    counterfactual: pd.DataFrame,
    interventions: pd.DataFrame,
    spec: DesignSpec,
) -> None:
    """Check that S' realises each unit's drawn, non-identity intervention."""
    if interventions["id"].tolist() != factual["id"].tolist():
        raise PairingError("interventions.csv must cover exactly the simulated IDs")
    drawn = interventions["do_value"].to_numpy()
    if (drawn == factual[spec.node].to_numpy()).any():
        raise PairingError("per-unit interventions must change the intervened node")
    if not (counterfactual[spec.node].to_numpy() == drawn).all():
        raise PairingError(f"counterfactual {spec.node} does not equal the drawn intervention")
