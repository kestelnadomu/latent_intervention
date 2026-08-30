"""
The concrete CV-screening experiment: SCM and closed-form symbolic kernel.

This is the single place the CV-screening numbers live. The structural equations
are Table 9 of the LIBERTy paper (arXiv 2601.10700); text generation (elsewhere)
follows its Appendix D.3. The structured-state schema (column names and
cardinalities) is data in ``exp/sim/config.yaml`` and read via
``src.schema.load_schema``; ``exp/sim/config.yaml`` points the framework here
through its ``objects:`` block.

Nodes: roots R (race), G (gender), A (age); then E (education), S (socio-economic
status), W (work experience), V (volunteering), C (certificates); Q is the
downstream outcome Y (simulated, excluded from the structured state S).
"""

from __future__ import annotations

import numpy as np

from exp.sim.scm import SCM, LinearMechanism, NoiseSampler, linear_mechanism
from exp.sim.symbolic import SymbolicIntervention
from src.schema import ColumnSpec, load_schema

ROOT_NODES = ("R", "G", "A")

# Exogenous noise eps_i. Roots are drawn as their category directly; non-roots are
# Normal(mu_i, sigma_i) added inside the linear mechanism.
_NOISE: dict[str, NoiseSampler] = {
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

# Linear mechanism weights m_i(pa_i) = sum_j coef_j * pa_j (no bias term).
# Q is the outcome and is kept out of MECHANISM_COEFFS (the symbolic kernel over S).
MECHANISM_COEFFS: dict[str, dict[str, float]] = {
    "E": {"R": 0.4, "A": 0.4, "G": 0.4},
    "S": {"E": 0.45, "A": 0.25},
    "W": {"A": 0.5, "E": 0.3},
    "V": {"E": 0.2, "S": 0.3},
    "C": {"E": 0.15, "W": 0.15},
}
_OUTCOME_COEFFS = {"Q": {"E": 0.3, "V": 0.3, "C": 0.3, "W": 0.3}}

# eps_i ~ Normal(mu_i, sigma_i) for the non-root structured-state nodes.
NOISE_PARAMS: dict[str, tuple[float, float]] = {
    "E": (0.35, 0.50),
    "S": (0.25, 0.35),
    "W": (0.00, 0.50),
    "V": (-0.35, 0.20),
    "C": (0.00, 0.30),
}


def _nodes() -> list[ColumnSpec]:
    """Structured-state columns plus the outcome, from the sim config schema."""
    columns, outcome = load_schema()
    return [*columns, outcome]


def build_scm() -> SCM:
    """Assemble the CV-screening SCM (all nodes, including the outcome Q)."""
    nodes = _nodes()
    card = {c.name: c.n_categories for c in nodes}
    mechanisms: dict[str, LinearMechanism] = {
        name: linear_mechanism(coeffs, 0, card[name] - 1)
        for name, coeffs in {**MECHANISM_COEFFS, **_OUTCOME_COEFFS}.items()
    }
    return SCM(nodes=nodes, roots=ROOT_NODES, noise=dict(_NOISE), mechanisms=mechanisms)


def build_symbolic_kernel() -> SymbolicIntervention:
    """Assemble the closed-form h_S over the structured state S (Q excluded)."""
    columns, _ = load_schema()
    return SymbolicIntervention.from_scm(columns, MECHANISM_COEFFS, NOISE_PARAMS)


if __name__ == "__main__":
    scm = build_scm()
    factual, counterfactual, _ = scm.simulate(2000, intervention={"G": 1}, seed=1)
    changed = (factual.drop(columns="id") != counterfactual.drop(columns="id")).any(axis=1)
    print(f"do(G=1): {changed.mean():.1%} of rows changed")
    print(factual.head())

    h_s = build_symbolic_kernel()
    m = h_s.transition_matrix({"G": 1}).to_dense()
    assert np.allclose(m.sum(1).numpy(), 1.0, atol=1e-4)
    print(f"h_S transition matrix: {tuple(m.shape)}, rows sum to 1")
