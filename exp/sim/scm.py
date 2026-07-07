"""
Python port of the SCM simulation in exp/sim/R/utils/{sim_scm.R, sim_data.R}.

Generates factual data, and counterfactual data under a do() intervention by
reusing the stored exogenous noise (Pearl's abduction step is exact because the
noise is known). The R implementation remains in the repo as the reference;
keep the structural equations here in sync with it.

Structural equations (all children clip-rounded to their category range):
    R = eps_R                                   (root, 0..3)
    G = eps_G                                   (root, 0..1)
    A = eps_A                                   (root, 0..2)
    E = clip(round(0.4(R + A + G) + eps_E), 0, 3)
    S = clip(round(0.45 E + 0.25 A + eps_S), 0, 2)
    W = clip(round(0.5 A + 0.3 E + eps_W), 0, 2)
    V = clip(round(0.2 E + 0.3 S + eps_V), 0, 1)
    C = clip(round(0.15(E + W) + eps_C), 0, 1)
    Q = clip(round(0.3(E + V + C + W) + eps_Q), 0, 2)
"""

from typing import Callable

import numpy as np
import pandas as pd

from src.semantic_decoder import OUTCOME_COLUMN, SCM_COLUMNS

# All SCM nodes in topological order, with cardinalities from the shared schema.
NODES: dict[str, int] = {col.name: col.n_categories for col in [*SCM_COLUMNS, OUTCOME_COLUMN]}
ROOT_NODES = ("R", "G", "A")

NoiseSpec = dict[str, Callable[[np.random.Generator, int], np.ndarray]]
StructuralFn = Callable[[dict[str, np.ndarray], dict[str, np.ndarray]], np.ndarray]

NOISE_SPEC: NoiseSpec = {
    "R": lambda rng, n: rng.integers(0, 4, size=n),
    "G": lambda rng, n: rng.integers(0, 2, size=n),
    "A": lambda rng, n: rng.choice([0, 1, 2], size=n, p=[0.25, 0.50, 0.25]),
    "E": lambda rng, n: rng.normal(0.35, 0.50, size=n),
    "S": lambda rng, n: rng.normal(0.25, 0.35, size=n),
    "W": lambda rng, n: rng.normal(0.00, 0.50, size=n),
    "V": lambda rng, n: rng.normal(-0.35, 0.20, size=n),
    "C": lambda rng, n: rng.normal(0.00, 0.30, size=n),
    "Q": lambda rng, n: rng.normal(0.00, 0.30, size=n),
}


def clip_round(x: np.ndarray, lower: int, upper: int) -> np.ndarray:
    """Round half-to-even (matching R's round()) and clip to [lower, upper]."""
    return np.clip(np.rint(x), lower, upper).astype(np.int64)


STRUCTURAL_FNS: dict[str, StructuralFn] = {
    "E": lambda d, e: clip_round(0.4 * (d["R"] + d["A"] + d["G"]) + e["E"], 0, 3),
    "S": lambda d, e: clip_round(0.45 * d["E"] + 0.25 * d["A"] + e["S"], 0, 2),
    "W": lambda d, e: clip_round(0.5 * d["A"] + 0.3 * d["E"] + e["W"], 0, 2),
    "V": lambda d, e: clip_round(0.2 * d["E"] + 0.3 * d["S"] + e["V"], 0, 1),
    "C": lambda d, e: clip_round(0.15 * (d["E"] + d["W"]) + e["C"], 0, 1),
    "Q": lambda d, e: clip_round(0.3 * (d["E"] + d["V"] + d["C"] + d["W"]) + e["Q"], 0, 2),
}


def _generate(noise: dict[str, np.ndarray], intervention: dict[str, int], n: int) -> dict[str, np.ndarray]:
    """One topological pass over the SCM; intervened nodes are held fixed."""
    data: dict[str, np.ndarray] = {}
    for node in NODES:
        if node in intervention:
            data[node] = np.full(n, intervention[node], dtype=np.int64)
        elif node in ROOT_NODES:
            data[node] = noise[node].astype(np.int64)
        else:
            data[node] = STRUCTURAL_FNS[node](data, noise)
    return data


def validate_intervention(intervention: dict[str, int]) -> dict[str, int]:
    """Check intervention nodes and value ranges against the SCM schema."""
    unknown = set(intervention) - set(NODES)
    if unknown:
        raise ValueError(f"Unknown intervention nodes: {sorted(unknown)}. Available: {list(NODES)}")
    for node, value in intervention.items():
        if not 0 <= int(value) < NODES[node]:
            raise ValueError(f"do({node}={value}) outside 0..{NODES[node] - 1}")
    return {node: int(value) for node, value in intervention.items()}


def simulate(
    n: int,
    intervention: dict[str, int],
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Simulate paired factual/counterfactual data from the SCM.

    Returns (factual, counterfactual, epsilon) data frames, each with an `id`
    column. The counterfactual applies do(intervention) and reuses the factual
    exogenous noise, so rows are true counterfactual pairs.
    """
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    intervention = validate_intervention(intervention)
    if not intervention:
        raise ValueError("Intervention must target at least one node, e.g. {'G': 1}.")

    rng = np.random.default_rng(seed)
    noise = {node: fn(rng, n) for node, fn in NOISE_SPEC.items()}

    factual = _generate(noise, intervention={}, n=n)
    counterfactual = _generate(noise, intervention=intervention, n=n)

    ids = pd.RangeIndex(1, n + 1, name=None)
    factual_df = pd.DataFrame({"id": ids, **factual})
    counterfactual_df = pd.DataFrame({"id": ids, **counterfactual})
    epsilon_df = pd.DataFrame({"id": ids, **{f"eps_{node}": eps for node, eps in noise.items()}})
    return factual_df, counterfactual_df, epsilon_df


if __name__ == "__main__":
    # Smoke test: seeding, invariances, and counterfactual semantics.
    n = 2000
    factual, counterfactual, epsilon = simulate(n, intervention={"G": 1}, seed=1)
    factual2, _, _ = simulate(n, intervention={"G": 1}, seed=1)
    assert factual.equals(factual2), "seeding must be reproducible"

    # do(G=1): non-descendants (R, A) never change; G is 1 everywhere.
    assert (counterfactual["G"] == 1).all()
    assert factual["R"].equals(counterfactual["R"])
    assert factual["A"].equals(counterfactual["A"])

    # Rows whose factual G already equals 1 are their own counterfactuals.
    already = factual["G"] == 1
    assert factual[already].equals(counterfactual[already])
    changed = (factual != counterfactual).any(axis=1)
    assert (changed == ~already).all(), "exactly the G=0 rows must change"

    # Values stay within the schema's category ranges.
    for node, card in NODES.items():
        assert factual[node].between(0, card - 1).all()
        assert counterfactual[node].between(0, card - 1).all()

    print(f"factual marginals (n={n}):")
    print(pd.DataFrame({node: factual[node].value_counts(normalize=True).sort_index() for node in NODES}).round(3))
    print(f"\nshare of rows changed by do(G=1): {changed.mean():.3f}")
