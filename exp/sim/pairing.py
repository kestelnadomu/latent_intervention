"""Small deterministic helpers for factual/counterfactual CV pairs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


class PairingError(ValueError):
    """Raised when paired artifacts cannot be aligned safely by unit ID."""


def checked_ids(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Return a copy with unique integer IDs, sorted independently of row order."""
    if "id" not in frame:
        raise PairingError(f"{label} has no 'id' column")
    result = frame.copy()
    ids = pd.to_numeric(result["id"], errors="coerce")
    if ids.isna().any() or not np.equal(ids, np.floor(ids)).all():
        raise PairingError(f"{label} IDs must be integers")
    result["id"] = ids.astype("int64")
    if result["id"].duplicated().any():
        duplicates = result.loc[result["id"].duplicated(), "id"].tolist()
        raise PairingError(f"{label} has duplicate IDs: {duplicates}")
    return result.sort_values("id").reset_index(drop=True)


def build_pair_index(
    factual: pd.DataFrame,
    counterfactual: pd.DataFrame,
    state_columns: Sequence[str],
    *,
    seed: int,
    test_fraction: float = 0.2,
) -> pd.DataFrame:
    """Join S/S' by ID and assign a seeded train/test split."""
    factual = checked_ids(factual, "factual simulation")
    counterfactual = checked_ids(counterfactual, "counterfactual simulation")
    if factual["id"].tolist() != counterfactual["id"].tolist():
        raise PairingError("factual and counterfactual simulations have different IDs")
    if len(factual) < 2:
        raise PairingError("at least two units are needed for non-empty train/test splits")
    if not 0 < test_fraction < 1:
        raise PairingError("test_fraction must lie strictly between 0 and 1")

    columns = list(state_columns)
    missing = [
        column
        for column in columns
        if column not in factual or column not in counterfactual
    ]
    if not columns or missing:
        raise PairingError(f"invalid structured-state columns; missing={missing}")
    if factual[columns].isna().any().any() or counterfactual[columns].isna().any().any():
        raise PairingError("structured states must not contain missing values")

    ids = factual["id"].to_numpy()
    n_test = min(max(round(test_fraction * len(ids)), 1), len(ids) - 1)
    test_ids = set(np.random.default_rng(seed).choice(ids, n_test, replace=False).tolist())
    identity = factual[columns].eq(counterfactual[columns]).all(axis=1)
    return pd.DataFrame(
        {
            "id": ids,
            "split": ["test" if unit_id in test_ids else "train" for unit_id in ids],
            "is_identity": identity.astype(bool),
        }
    )


def build_render_plan(
    unit_ids: Iterable[int],
    template_ids: Iterable[int],
    persona_ids: Iterable[int],
    binned_columns: Sequence[str],
    *,
    seed: int,
) -> pd.DataFrame:
    """Choose one shared template, persona, and bin quantile per unit."""
    ids = sorted({int(value) for value in unit_ids})
    templates = sorted({int(value) for value in template_ids})
    personas = sorted({int(value) for value in persona_ids})
    if not ids or not templates or not personas:
        raise PairingError("unit, template, and persona ID sets must be non-empty")

    rows: list[dict[str, Any]] = []
    for unit_id in ids:
        rng = np.random.default_rng([int(seed), 2, unit_id])
        row: dict[str, Any] = {
            "id": unit_id,
            "template_id": templates[int(rng.integers(len(templates)))],
            "persona_id": personas[int(rng.integers(len(personas)))],
        }
        row.update({f"{column}_quantile": float(rng.random()) for column in binned_columns})
        rows.append(row)
    return pd.DataFrame(rows)


def materialize_binned_values(
    state: Mapping[str, Any],
    bins: Mapping[str, Mapping[int, tuple[int, int]]],
    plan: Mapping[str, Any],
) -> dict[str, int]:
    """Map shared quantiles into the factual or counterfactual state's bins."""
    concrete: dict[str, int] = {}
    for column, levels in bins.items():
        category = int(state[column])
        if category not in levels:
            raise PairingError(f"no bin configured for {column}={category}")
        lower, upper = levels[category]
        quantile = float(plan[f"{column}_quantile"])
        if not 0 <= quantile < 1:
            raise PairingError(f"{column} quantile must lie in [0, 1)")
        concrete[column] = min(upper, lower + int(quantile * (upper - lower + 1)))
    return concrete
