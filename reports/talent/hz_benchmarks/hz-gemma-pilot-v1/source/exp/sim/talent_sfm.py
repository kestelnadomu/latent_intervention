"""
The talent SFM experiment: SCM and closed-form symbolic kernel.

The canonical Standard Fairness Model of Plecko & Bareinboim (X protected
attribute, Z confounder, W mediators, Y outcome), instantiated for CV screening
and extended by a measurement model for the latent confounder. This is the
single place the numbers live; the schema (names and cardinalities) is data in
``exp/sim/config.yaml`` and read via ``src.schema.load_schema``.

Nodes:
- roots X (country of origin, 4 regions ordinal by simulated access gap) and
  T (talent, the SFM's Z; renamed so it does not clash with the latent space);
- mediators W = {D (degree), U (university rank tier)}, both caused by X and T;
- auxiliary proxies P, L, H, A of talent (caused by T only): simulated and
  verbalized so T is inferable from text, but outside the structured state S;
- Y (qualification), the downstream outcome, excluded from S.

S = {X, T, D, U}. Every parent of D and U is in S, so the symbolic kernel over S
is exact without the proxies. T is a non-descendant of X, so do(X) leaves T and
its proxies unchanged.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from exp.sim.scm import SCM, LinearMechanism, NoiseSampler, linear_mechanism
from exp.sim.symbolic import SymbolicIntervention
from src.schema import ColumnSpec, load_schema

ROOT_NODES = ("X", "T")

# eps_i ~ Normal(mu_i, sigma_i) per non-root node, added inside the linear mechanism.
NOISE_PARAMS: dict[str, tuple[float, float]] = {
    "D": (-0.10, 0.50),
    "U": (-0.15, 0.50),
}
_AUXILIARY_NOISE: dict[str, tuple[float, float]] = {
    "P": (0.05, 0.40),
    "L": (0.10, 0.45),
    "H": (0.10, 0.45),
    "A": (0.15, 0.50),
}
_OUTCOME_NOISE: dict[str, tuple[float, float]] = {"Y": (0.00, 0.30)}

# Linear mechanism weights m_i(pa_i) = sum_j coef_j * pa_j (no bias term).
# MECHANISM_COEFFS covers exactly the structured state S (the symbolic kernel);
# the proxies and the outcome are simulated but kept out of it.
MECHANISM_COEFFS: dict[str, dict[str, float]] = {
    "D": {"X": 0.30, "T": 0.60},
    "U": {"X": 0.35, "T": 0.50},
}
_AUXILIARY_COEFFS: dict[str, dict[str, float]] = {
    "P": {"T": 0.90},
    "L": {"T": 0.80},
    "H": {"T": 0.80},
    "A": {"T": 0.70},
}
_OUTCOME_COEFFS: dict[str, dict[str, float]] = {
    "Y": {"X": 0.15, "T": 0.40, "D": 0.20, "U": 0.20},
}


def _normal(mu: float, sigma: float) -> NoiseSampler:
    return lambda rng, n: rng.normal(mu, sigma, size=n)


_NOISE: dict[str, NoiseSampler] = {
    "X": lambda rng, n: rng.integers(0, 4, size=n),
    "T": lambda rng, n: rng.choice([0, 1, 2], size=n, p=[0.25, 0.50, 0.25]),
    **{name: _normal(*params) for name, params in {
        **NOISE_PARAMS, **_AUXILIARY_NOISE, **_OUTCOME_NOISE,
    }.items()},
}


def _as_config(sim_config: dict[str, Any] | str | Path | None) -> dict[str, Any]:
    if isinstance(sim_config, dict):
        return sim_config
    from exp.sim.helpers import CONFIG_PATH, load_sim_config

    return load_sim_config(sim_config if sim_config is not None else CONFIG_PATH)


def _nodes(sim_config: dict[str, Any] | str | Path | None = None) -> list[ColumnSpec]:
    """S columns, auxiliary proxies and the outcome, in topological order."""
    from exp.sim.paired_data import auxiliary_columns

    config = _as_config(sim_config)
    columns, outcome = load_schema(config)
    return [*columns, *auxiliary_columns(config), outcome]


def build_scm(sim_config: dict[str, Any] | str | Path | None = None) -> SCM:
    """Assemble the talent SFM using the supplied schema config."""
    nodes = _nodes(sim_config)
    card = {c.name: c.n_categories for c in nodes}
    mechanisms: dict[str, LinearMechanism] = {
        name: linear_mechanism(coeffs, 0, card[name] - 1)
        for name, coeffs in {**MECHANISM_COEFFS, **_AUXILIARY_COEFFS, **_OUTCOME_COEFFS}.items()
    }
    return SCM(nodes=nodes, roots=ROOT_NODES, noise=dict(_NOISE), mechanisms=mechanisms)


def build_symbolic_kernel(
    sim_config: dict[str, Any] | str | Path | None = None,
) -> SymbolicIntervention:
    """Assemble closed-form h_S over S = schema.columns (proxies and Y excluded)."""
    columns, _ = load_schema(_as_config(sim_config))
    return SymbolicIntervention.from_scm(columns, MECHANISM_COEFFS, NOISE_PARAMS)


_erf = np.vectorize(math.erf, otypes=[float])


def _category_likelihood(m: np.ndarray, mu: float, sigma: float, k: int) -> np.ndarray:
    """P(clip_round(m + eps) = c), shape (len(m), k), for eps ~ Normal(mu, sigma)."""
    edges = np.concatenate([[-np.inf], np.arange(k - 1) + 0.5, [np.inf]])
    z = (edges[None, :] - m[:, None] - mu) / (sigma * math.sqrt(2.0))
    return np.diff(0.5 * (1.0 + _erf(z)), axis=1)


def talent_posterior_accuracy(
    factual: pd.DataFrame,
    given: list[str],
    cardinalities: dict[str, int],
    t_prior: tuple[float, ...] = (0.25, 0.50, 0.25),
) -> float:
    """
    Bayes-optimal accuracy of recovering T from the observed columns ``given``.

    Uses the exact posterior P(T | given) under the known mechanisms: the ceiling
    any reader of the text, and hence the decoder's T head, can reach.
    """
    coeffs = {**MECHANISM_COEFFS, **_AUXILIARY_COEFFS}
    noise = {**NOISE_PARAMS, **_AUXILIARY_NOISE}
    rows = np.arange(len(factual))
    log_post = np.tile(np.log(np.asarray(t_prior)), (len(factual), 1))
    for t in range(len(t_prior)):
        for name in given:
            if name not in coeffs:
                continue  # X is independent of T: no likelihood term
            m = np.zeros(len(factual))
            for parent, coef in coeffs[name].items():
                m += coef * (t if parent == "T" else factual[parent].to_numpy(float))
            lik = _category_likelihood(m, *noise[name], cardinalities[name])
            log_post[:, t] += np.log(lik[rows, factual[name].to_numpy()] + 1e-300)
    return float((log_post.argmax(axis=1) == factual["T"].to_numpy()).mean())


if __name__ == "__main__":
    from exp.sim.paired_data import auxiliary_columns
    from src.schema import load_intervention

    config = _as_config(None)
    columns, outcome = load_schema(config)
    proxies = [c.name for c in auxiliary_columns(config)]
    card = {c.name: c.n_categories for c in _nodes(config)}
    delta = load_intervention(config)

    scm = build_scm(config)
    factual, counterfactual, _ = scm.simulate(20_000, intervention=delta, seed=1)

    print("marginals:")
    for name in ["X", "T", "D", "U", *proxies, outcome.name]:
        print(f"  {name}: {np.round(factual[name].value_counts(normalize=True).sort_index().to_numpy(), 3)}")

    corr = factual[["T", *proxies]].corr()["T"].drop("T")
    print("corr(proxy, T):", corr.round(3).to_dict())
    items = factual[proxies].to_numpy(float)
    k = len(proxies)
    alpha = k / (k - 1) * (1 - items.var(axis=0, ddof=1).sum() / items.sum(axis=1).var(ddof=1))
    print(f"Cronbach's alpha of proxies: {alpha:.3f}")
    print(f"corr(D, U): {factual['D'].corr(factual['U']):.3f}")

    acc_proxies = talent_posterior_accuracy(factual, proxies, card)
    acc_all = talent_posterior_accuracy(factual, ["X", "D", "U", *proxies], card)
    print(f"Bayes-optimal T accuracy | proxies: {acc_proxies:.3f}; | X, D, U, proxies: {acc_all:.3f}"
          f" (prior-only: {factual['T'].value_counts(normalize=True).max():.3f})")

    invariant = ["T", *proxies]
    assert factual[invariant].equals(counterfactual[invariant]), "T and proxies must be invariant under do(X)"
    changed = (factual[["D", "U"]] != counterfactual[["D", "U"]]).any(axis=1)
    print(f"do({delta}): X changed for {(factual['X'] != counterfactual['X']).mean():.1%}, "
          f"D or U changed for {changed.mean():.1%}; T and proxies invariant")

    h_s = build_symbolic_kernel(config)
    m = h_s.transition_matrix(delta).to_dense()
    assert np.allclose(m.sum(1).numpy(), 1.0, atol=1e-4)
    print(f"h_S transition matrix over S={[c.name for c in columns]}: {tuple(m.shape)}, rows sum to 1")
