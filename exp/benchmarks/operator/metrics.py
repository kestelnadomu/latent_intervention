"""State-level metrics of an assumed vs. a true counterfactual kernel (generic, exact).

Every metric compares two transition matrices over the same state space for one
intervention ``do(node = k)``, weighted by the factual state distribution restricted to
units whose ``node`` differs from ``k`` (the identity rows are trivially equal).
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from exp.sim.symbolic import SymbolicIntervention


def dense(kernel: SymbolicIntervention, delta: dict[str, int]) -> np.ndarray:
    """Row-normalized dense transition matrix of ``kernel`` under ``delta``."""
    matrix = kernel.transition_matrix(delta).to_dense().numpy().astype(float)
    return matrix / matrix.sum(axis=1, keepdims=True)


def all_states(kernel: SymbolicIntervention) -> pd.DataFrame:
    """Every state of the kernel's space, one row per flat state index."""
    names = [c.name for c in kernel.columns]
    rows = list(product(*(range(c.n_categories) for c in kernel.columns)))
    frame = pd.DataFrame(rows, columns=names)
    order = np.array([kernel.state_index(dict(zip(names, r))) for r in rows])
    return frame.iloc[np.argsort(order)].reset_index(drop=True)


def indices(frame: pd.DataFrame, kernel: SymbolicIntervention) -> np.ndarray:
    """Flat state index of each row of ``frame`` in the kernel's space (vectorized)."""
    index = np.zeros(len(frame), dtype=int)
    for column in kernel.columns:
        index = index * column.n_categories + frame[column.name].to_numpy(int)
    return index


def intervention_weights(states: pd.DataFrame, probs: np.ndarray, node: str, value: int) -> np.ndarray:
    """Factual P(s) restricted to states whose ``node`` is not already ``value``, normalized."""
    weights = probs * (states[node].to_numpy() != value)
    return weights / weights.sum()


def compare(assumed: np.ndarray, true: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    """Expected TV distance and the true mass outside the assumed kernel's support."""
    tv_rows = 0.5 * np.abs(assumed - true).sum(axis=1)
    missing_rows = (true * (assumed <= 1e-12)).sum(axis=1)
    return {
        "tv": float(weights @ tv_rows),
        "missing_mass": float(weights @ missing_rows),
        "tv_rows": tv_rows,
    }


def mean_shift(
    assumed: np.ndarray,
    true: np.ndarray,
    weights: np.ndarray,
    states: pd.DataFrame,
    direction: np.ndarray,
) -> dict[str, dict[str, float]]:
    """
    Error of E[V' | s] per state column: ``directional`` is signed along the intervention
    (``direction`` = sign of the intervened change per factual state), so positive means the
    assumed operator moves V further in that direction than the truth; ``absolute`` is the
    mean absolute error. A plain signed mean would cancel between increases and decreases.
    """
    out = {}
    for column in states.columns:
        values = states[column].to_numpy(float)
        error = assumed @ values - true @ values
        out[column] = {
            "directional": float(weights @ (direction * error)),
            "absolute": float(weights @ np.abs(error)),
        }
    return out


def downstream(
    assumed: np.ndarray,
    true: np.ndarray,
    weights: np.ndarray,
    outcome_probs: np.ndarray,
    direction: np.ndarray,
) -> dict[str, float]:
    """Bayes-optimal outcome prediction for s': expected TV, and the error of E[Y'] along the intervention."""
    predicted_a, predicted_t = assumed @ outcome_probs, true @ outcome_probs
    levels = np.arange(outcome_probs.shape[1], dtype=float)
    error = predicted_a @ levels - predicted_t @ levels
    return {
        "y_tv": float(weights @ (0.5 * np.abs(predicted_a - predicted_t).sum(axis=1))),
        "y_shift": float(weights @ (direction * error)),
        "y_abs": float(weights @ np.abs(error)),
    }


def log_score(matrix: np.ndarray, source: np.ndarray, target: np.ndarray, floor: float = 1e-12) -> float:
    """Mean negative log probability of realized transitions ``source -> target``."""
    return float(-np.log(np.maximum(matrix[source, target], floor)).mean())
