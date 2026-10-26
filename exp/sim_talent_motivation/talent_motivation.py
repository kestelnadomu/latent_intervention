"""
The talent-motivation experiment: the talent SFM extended by motivation M.

A sibling of ``exp/sim_talent/talent_sfm.py`` (design: ``docs/experiments/data_sim.md``).
Everything of the talent SFM is reused unchanged; the extension adds

- M (motivation, 3 tiers): a root with the same marginal as T, independent of
  X, T, D and U, verbalized through evidence guidance, outside the structured
  state S (``schema.auxiliary``);
- M as a parent of Y with the weight of talent, beta_YM = beta_YT = 0.40. The
  noise mean of Y is lowered by beta_YM * E[M] = 0.40, so the marginal of Y
  stays as in the talent SFM instead of piling onto its top level.

S = {X, T, D, U} as before, so the symbolic kernel is the talent SFM's. M is a
non-descendant of X: do(X) leaves it unchanged. The builders default to
``exp/sim_talent_motivation/config.yaml``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from exp.sim_talent import talent_sfm as base
from exp.sim.scm import SCM, LinearMechanism, NoiseSampler, linear_mechanism
from exp.sim.symbolic import SymbolicIntervention
from exp.sim.schema import ColumnSpec, load_schema

CONFIG_PATH = Path(__file__).parent / "config.yaml"

T_PRIOR: tuple[float, ...] = (0.25, 0.50, 0.25)
M_PRIOR: tuple[float, ...] = (0.25, 0.50, 0.25)
ROOT_NODES = (*base.ROOT_NODES, "M")

# The only numbers that differ from the talent SFM.
_OUTCOME_COEFFS: dict[str, dict[str, float]] = {
    "Y": {**base._OUTCOME_COEFFS["Y"], "M": 0.40},
}
_OUTCOME_NOISE: dict[str, tuple[float, float]] = {"Y": (-0.40, 0.30)}

_NOISE: dict[str, NoiseSampler] = {
    **base._NOISE,
    "M": lambda rng, n: rng.choice([0, 1, 2], size=n, p=list(M_PRIOR)),
    "Y": lambda rng, n: rng.normal(*_OUTCOME_NOISE["Y"], size=n),
}


def _as_config(sim_config: dict[str, Any] | str | Path | None) -> dict[str, Any]:
    if isinstance(sim_config, dict):
        return sim_config
    from exp.sim.helpers import load_sim_config

    return load_sim_config(sim_config if sim_config is not None else CONFIG_PATH)


def _nodes(sim_config: dict[str, Any] | str | Path | None = None) -> list[ColumnSpec]:
    """S columns, auxiliary columns (proxies, M) and the outcome, in topological order."""
    from exp.sim.paired_data import auxiliary_columns

    config = _as_config(sim_config)
    columns, outcome = load_schema(config)
    return [*columns, *auxiliary_columns(config), outcome]


def build_scm(sim_config: dict[str, Any] | str | Path | None = None) -> SCM:
    """Assemble the talent SFM with motivation using the supplied schema config."""
    nodes = _nodes(sim_config)
    names = {c.name for c in nodes}
    if "M" not in names:
        raise ValueError("schema.auxiliary must contain M for the talent-motivation experiment")
    card = {c.name: c.n_categories for c in nodes}
    mechanisms: dict[str, LinearMechanism] = {
        name: linear_mechanism(coeffs, 0, card[name] - 1)
        for name, coeffs in {
            **base.MECHANISM_COEFFS, **base._AUXILIARY_COEFFS, **_OUTCOME_COEFFS,
        }.items()
    }
    return SCM(nodes=nodes, roots=ROOT_NODES, noise=dict(_NOISE), mechanisms=mechanisms)


def build_symbolic_kernel(
    sim_config: dict[str, Any] | str | Path | None = None,
) -> SymbolicIntervention:
    """Closed-form h_S over S = schema.columns: identical to the talent SFM's."""
    columns, _ = load_schema(_as_config(sim_config))
    return SymbolicIntervention.from_scm(columns, base.MECHANISM_COEFFS, base.NOISE_PARAMS)


talent_posterior_accuracy = base.talent_posterior_accuracy  # M carries no information on T


def outcome_posterior(
    frame: pd.DataFrame,
    given: list[str],
    cardinalities: dict[str, int],
    t_prior: tuple[float, ...] = T_PRIOR,
    m_prior: tuple[float, ...] = M_PRIOR,
) -> np.ndarray:
    """
    Exact P(Y | given), shape (len(frame), card(Y)), under the known mechanisms.

    ``given`` must contain Y's observed parents X, D, U. T and M are conditioned
    on when given, otherwise marginalized: T over P(T | given) (so the proxies
    carry their talent information), M over its prior (nothing else informs it).
    """
    parents = set(_OUTCOME_COEFFS["Y"])
    required = parents - {"T", "M"}
    if not required <= set(given):
        raise ValueError(f"outcome posterior needs {sorted(required)} in given")
    n = len(frame)
    mu, sigma = _OUTCOME_NOISE["Y"]

    if "T" in given:
        t_weights = np.eye(len(t_prior))[frame["T"].to_numpy()]
    else:
        log_post = base._talent_log_posterior(frame, given, cardinalities, t_prior)
        t_weights = np.exp(log_post - log_post.max(axis=1, keepdims=True))
        t_weights /= t_weights.sum(axis=1, keepdims=True)
    if "M" in given:
        m_weights = np.eye(len(m_prior))[frame["M"].to_numpy()]
    else:
        m_weights = np.tile(np.asarray(m_prior, dtype=float), (n, 1))

    observed = {p: frame[p].to_numpy(float) for p in required}
    probs = np.zeros((n, cardinalities["Y"]))
    for t in range(len(t_prior)):
        for m in range(len(m_prior)):
            mean = np.zeros(n)
            for parent, coef in _OUTCOME_COEFFS["Y"].items():
                value = t if parent == "T" else m if parent == "M" else observed[parent]
                mean += coef * value
            weight = (t_weights[:, t] * m_weights[:, m])[:, None]
            probs += weight * base._category_likelihood(mean, mu, sigma, cardinalities["Y"])
    return probs


# Information sets for the Bayes-optimal outcome reference (benchmark ceilings).
OUTCOME_INFORMATION_SETS: dict[str, list[str]] = {
    "all parents (T, M known)": ["X", "T", "D", "U", "M"],
    "state only (T known, M unknown)": ["X", "T", "D", "U"],
    "text content (X, D, U, M, proxies)": ["X", "D", "U", "M", "P", "L", "H", "A"],
    "text content without M": ["X", "D", "U", "P", "L", "H", "A"],
    "no talent or motivation information (X, D, U)": ["X", "D", "U"],
}


def outcome_bayes_predictions(
    frame: pd.DataFrame,
    sim_config: dict[str, Any] | str | Path | None = None,
) -> dict[str, np.ndarray]:
    """P(Y | information set) per row of ``frame`` for each OUTCOME_INFORMATION_SETS entry."""
    cardinalities = {c.name: c.n_categories for c in _nodes(sim_config)}
    return {
        name: outcome_posterior(frame, given, cardinalities)
        for name, given in OUTCOME_INFORMATION_SETS.items()
    }


if __name__ == "__main__":
    from exp.sim.paired_data import auxiliary_columns
    from exp.sim.schema import load_intervention

    config = _as_config(None)
    columns, outcome = load_schema(config)
    auxiliary = [c.name for c in auxiliary_columns(config)]
    proxies = [name for name in auxiliary if name != "M"]
    card = {c.name: c.n_categories for c in _nodes(config)}
    delta = load_intervention(config)

    scm = build_scm(config)
    factual, counterfactual, _ = scm.simulate(50_000, intervention=delta, seed=1)

    print("marginals:")
    for name in ["X", "T", "D", "U", *auxiliary, outcome.name]:
        print(f"  {name}: {np.round(factual[name].value_counts(normalize=True).sort_index().to_numpy(), 3)}")
    print("corr(M, ·):", factual[["M", "X", "T", "D", "U"]].corr()["M"].drop("M").round(3).to_dict())

    acc_all = talent_posterior_accuracy(factual, ["X", "D", "U", *proxies], card)
    print(f"Bayes-optimal T accuracy | X, D, U, proxies: {acc_all:.3f}")
    y = factual[outcome.name].to_numpy()
    for name, probs in outcome_bayes_predictions(factual, config).items():
        assert np.allclose(probs.sum(axis=1), 1.0)
        print(f"Bayes-optimal Y accuracy | {name}: {(probs.argmax(1) == y).mean():.3f}")

    invariant = ["T", "M", *proxies]
    assert factual[invariant].equals(counterfactual[invariant]), "T, M and proxies must be invariant under do(X)"
    print(f"do({delta}): T, M and proxies invariant")

    h_s = build_symbolic_kernel(config)
    m = h_s.transition_matrix(delta).to_dense()
    assert np.allclose(m.sum(1).numpy(), 1.0, atol=1e-4)
    print(f"h_S transition matrix over S={[c.name for c in columns]}: {tuple(m.shape)}, rows sum to 1")
