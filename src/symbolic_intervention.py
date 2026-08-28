"""
Symbolic counterfactual kernel h_S(s' | s, delta).

See docs/architecture/symbolic_intervention.md. h_S is available in closed form: for the
CV-screening SCM (exp/sim/scm.py) each non-root node is `clip_round(m_i(pa_i) + eps_i)`
with `eps_i ~ Normal(mu_i, sigma_i)`, so the three-step counterfactual recipe is exact:

1. **Abduction** - given the full factual state s every parent is observed, so the noise
   posterior factorises into independent truncated normals; `clip_round` inversion gives
   each interval [l_i, u_i).
2. **Action** - set the do() nodes.
3. **Prediction** - push the factorised posterior forward in topological order by exact
   enumeration over the descendant closure of the intervention.

Enumerating all |S| = 3456 factual states precomputes a sparse 3456 x 3456 transition
matrix per delta; h_S is then a lookup, and (h_S . g)(. | z) = M_delta^T g(. | z).

The noise parameters (mu_i, sigma_i), and optionally the mechanism coefficients, can be
taken from the known SCM (`SymbolicIntervention.from_scm`) or estimated from a factual
sample (`SymbolicIntervention.fit`) when the simulator is not available (Studies 2/3).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd
import torch

from src.semantic_decoder import SCM_COLUMNS, ColumnSpec

# --- SCM structure (mirrors exp/sim/scm.py; Q is excluded - not part of S) -------------

ROOT_NODES: tuple[str, ...] = ("R", "G", "A")

# Linear mechanism m_i(pa_i) = sum_j coef[j] * pa_j (no bias term in this SCM).
MECHANISM_COEFFS: dict[str, dict[str, float]] = {
    "E": {"R": 0.4, "A": 0.4, "G": 0.4},
    "S": {"E": 0.45, "A": 0.25},
    "W": {"A": 0.5, "E": 0.3},
    "V": {"E": 0.2, "S": 0.3},
    "C": {"E": 0.15, "W": 0.15},
}

# eps_i ~ Normal(mu_i, sigma_i), from exp/sim/R/utils/sim_scm.R / exp/sim/scm.py NOISE_SPEC.
NOISE_PARAMS: dict[str, tuple[float, float]] = {
    "E": (0.35, 0.50),
    "S": (0.25, 0.35),
    "W": (0.00, 0.50),
    "V": (-0.35, 0.20),
    "C": (0.00, 0.30),
}

_SQRT2 = math.sqrt(2.0)


def _children() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {n: [] for n in [*ROOT_NODES, *MECHANISM_COEFFS]}
    for child, parents in MECHANISM_COEFFS.items():
        for p in parents:
            out[p].append(child)
    return out


def _descendants(nodes: set[str]) -> set[str]:
    """Transitive descendant closure of `nodes` (excluding the nodes themselves)."""
    children = _children()
    seen: set[str] = set()
    stack = [c for n in nodes for c in children[n]]
    while stack:
        n = stack.pop()
        if n not in seen:
            seen.add(n)
            stack.extend(children[n])
    return seen


def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    if x == math.inf:
        return 1.0
    if x == -math.inf:
        return 0.0
    return 0.5 * math.erfc(-(x - mu) / (sigma * _SQRT2))


def _abduction_interval(s_i: int, m_fac: float, k: int) -> tuple[float, float]:
    """Invert `clip_round(m_fac + eps) = s_i` for the noise interval [l, u)."""
    if s_i <= 0:
        return (-math.inf, 0.5 - m_fac)
    if s_i >= k - 1:
        return (k - 1.5 - m_fac, math.inf)
    return (s_i - m_fac - 0.5, s_i - m_fac + 0.5)


def _category_probs(m_prime: float, lo: float, hi: float, mu: float, sigma: float, k: int) -> np.ndarray:
    """P(clip_round(m_prime + eps) = c) for eps ~ TruncNormal(mu, sigma, [lo, hi))."""
    z = _normal_cdf(hi, mu, sigma) - _normal_cdf(lo, mu, sigma)
    probs = np.zeros(k)
    if z <= 0.0:  # degenerate interval (should not happen for a valid factual state)
        probs[min(max(int(round(m_prime)), 0), k - 1)] = 1.0
        return probs
    for c in range(k):
        if c == 0:
            pre_lo, pre_hi = -math.inf, 0.5 - m_prime
        elif c == k - 1:
            pre_lo, pre_hi = k - 1.5 - m_prime, math.inf
        else:
            pre_lo, pre_hi = c - 0.5 - m_prime, c + 0.5 - m_prime
        a, b = max(lo, pre_lo), min(hi, pre_hi)
        if b > a:
            probs[c] = _normal_cdf(b, mu, sigma) - _normal_cdf(a, mu, sigma)
    return probs / probs.sum()


@dataclass
class SymbolicIntervention:
    """
    Closed-form symbolic counterfactual kernel h_S(s' | s, delta) for the CV-screening SCM.

    Attributes:
        columns: the structured state schema (single source of truth: SCM_COLUMNS).
        mechanism_coeffs: linear mechanism weights m_i(pa_i) = sum_j coef[j] pa_j.
        noise_params: (mu_i, sigma_i) of eps_i ~ Normal for each non-root node.
    """

    columns: list[ColumnSpec] = field(default_factory=lambda: list(SCM_COLUMNS))
    mechanism_coeffs: dict[str, dict[str, float]] = field(
        default_factory=lambda: {k: dict(v) for k, v in MECHANISM_COEFFS.items()}
    )
    noise_params: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(NOISE_PARAMS)
    )

    def __post_init__(self) -> None:
        self._names = [c.name for c in self.columns]
        self._card = {c.name: c.n_categories for c in self.columns}
        self._n_states = int(np.prod([c.n_categories for c in self.columns]))

    # --- constructors -----------------------------------------------------------------

    @classmethod
    def from_scm(cls, columns: list[ColumnSpec] = SCM_COLUMNS) -> "SymbolicIntervention":
        """Use the known simulator mechanism and noise parameters."""
        return cls(columns=list(columns))

    @classmethod
    def fit(
        cls,
        factual: pd.DataFrame,
        columns: list[ColumnSpec] = SCM_COLUMNS,
        fit_mechanisms: bool = False,
    ) -> "SymbolicIntervention":
        """
        Estimate the noise parameters (and optionally the mechanism coefficients) from a
        factual sample, for when the simulator is unavailable.

        `sigma_i` is corrected for the uniform rounding jitter (Var -= 1/12); this is a
        method-of-moments estimate on the interior (non-censored) rows. The principled
        alternative is MLE of the rounded-censored-normal likelihood; empirical transition
        counts on (factual, counterfactual) pairs remain the validation check.
        """
        coeffs = {k: dict(v) for k, v in MECHANISM_COEFFS.items()}
        noise: dict[str, tuple[float, float]] = {}
        for name, parents in coeffs.items():
            k = next(c.n_categories for c in columns if c.name == name)
            pa = list(parents)
            if fit_mechanisms:
                x = np.column_stack([factual[p].to_numpy(float) for p in pa])
                beta, *_ = np.linalg.lstsq(x, factual[name].to_numpy(float), rcond=None)
                parents = {p: float(b) for p, b in zip(pa, beta)}
                coeffs[name] = parents
            m = sum(coef * factual[p].to_numpy(float) for p, coef in parents.items())
            resid = factual[name].to_numpy(float) - m
            interior = (factual[name].to_numpy() > 0) & (factual[name].to_numpy() < k - 1)
            r = resid[interior] if interior.any() else resid
            mu = float(r.mean())
            sigma = float(np.sqrt(max(r.var() - 1.0 / 12.0, 1e-4)))
            noise[name] = (mu, sigma)
        return cls(columns=list(columns), mechanism_coeffs=coeffs, noise_params=noise)

    # --- core -------------------------------------------------------------------------

    def state_index(self, state: dict[str, int]) -> int:
        """Flat column-major index of a state (matches SemanticDecoder.log_joint order)."""
        idx = 0
        for name in self._names:
            idx = idx * self._card[name] + int(state[name])
        return idx

    def _mechanism(self, node: str, assign: dict[str, int]) -> float:
        return sum(coef * assign[p] for p, coef in self.mechanism_coeffs[node].items())

    def counterfactual(self, state: dict[str, int], delta: dict[str, int]) -> dict[int, float]:
        """h_S(. | state, delta) as {state_index: probability} over the sparse support."""
        movable = _descendants(set(delta))
        fixed: dict[str, int] = {}
        for name in self._names:
            if name in delta:
                fixed[name] = int(delta[name])
            elif name not in movable:
                fixed[name] = int(state[name])

        intervals = {
            name: _abduction_interval(state[name], self._mechanism(name, state), self._card[name])
            for name in movable
        }

        beam: list[tuple[dict[str, int], float]] = [(dict(fixed), 1.0)]
        for name in self._names:
            if name not in movable:
                continue
            mu, sigma = self.noise_params[name]
            k = self._card[name]
            nxt: list[tuple[dict[str, int], float]] = []
            for assign, p in beam:
                probs = _category_probs(
                    self._mechanism(name, assign), *intervals[name], mu, sigma, k
                )
                for c, pc in enumerate(probs):
                    if pc > 1e-9:
                        nxt.append(({**assign, name: c}, p * pc))
            beam = nxt

        out: dict[int, float] = {}
        for assign, p in beam:
            out[self.state_index(assign)] = out.get(self.state_index(assign), 0.0) + p
        return out

    def transition_matrix(self, delta: dict[str, int]) -> torch.Tensor:
        """
        Sparse (|S|, |S|) COO tensor M_delta with M[s, s'] = h_S(s' | s, delta).

        Rows sum to 1. `(h_S . g)(. | z) = M_delta.T @ g(. | z)` (see `compose`).
        """
        self._validate_delta(delta)
        rows, cols, vals = [], [], []
        ranges = [range(self._card[n]) for n in self._names]
        for values in product(*ranges):
            state = dict(zip(self._names, values))
            src = self.state_index(state)
            for dst, p in self.counterfactual(state, delta).items():
                rows.append(src)
                cols.append(dst)
                vals.append(p)
        n = self._n_states
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Sparse invariant checks")
            return torch.sparse_coo_tensor(
                torch.tensor([rows, cols]),
                torch.tensor(vals, dtype=torch.float32),
                (n, n),
            ).coalesce()

    def compose(self, g_probs: torch.Tensor, delta: dict[str, int]) -> torch.Tensor:
        """
        Pushforward (h_S . g): map a batch of dense g(. | z) distributions, shape
        (batch, |S|), through M_delta -> (batch, |S|).
        """
        m = self.transition_matrix(delta)
        return torch.sparse.mm(m.t(), g_probs.t()).t()

    def _validate_delta(self, delta: dict[str, int]) -> None:
        if not delta:
            raise ValueError("delta must target at least one node, e.g. {'G': 1}")
        for name, value in delta.items():
            if name not in self._card:
                raise ValueError(f"unknown node {name!r}; known: {self._names}")
            if not 0 <= int(value) < self._card[name]:
                raise ValueError(f"do({name}={value}) outside 0..{self._card[name] - 1}")


if __name__ == "__main__":
    # Smoke test: closed-form h_S vs. empirical transition counts from exp/sim/scm.py.
    from exp.sim.scm import simulate

    h_s = SymbolicIntervention.from_scm()
    delta = {"G": 1}
    m = h_s.transition_matrix(delta)
    dense = m.to_dense()
    per_row = (dense > 0).sum(1)
    reachable = int((dense.sum(0) > 0).sum())
    print(f"support: {m._nnz()} nonzeros over {reachable} reachable states; "
          f"per-row nnz mean {per_row.float().mean():.2f}, max {int(per_row.max())}")
    print(f"row sums in [{dense.sum(1).min():.4f}, {dense.sum(1).max():.4f}]  (want 1)")
    assert torch.allclose(dense.sum(1), torch.ones(dense.shape[0]), atol=1e-4)

    factual, counterfactual, _ = simulate(50_000, intervention=delta, seed=0)
    idx = lambda df, i: h_s.state_index(df.iloc[i].to_dict())  # noqa: E731
    n = dense.shape[0]
    fac_count = torch.zeros(n)
    joint = torch.zeros(n, n)
    for i in range(len(factual)):
        s, sp = idx(factual, i), idx(counterfactual, i)
        fac_count[s] += 1
        joint[s, sp] += 1

    # Row-wise TVD on factual states with enough samples to estimate a row.
    well_sampled = fac_count >= 100
    emp_rows = joint[well_sampled] / fac_count[well_sampled][:, None]
    tvd_rows = 0.5 * (dense[well_sampled] - emp_rows).abs().sum(1)
    w = fac_count[well_sampled] / fac_count[well_sampled].sum()
    print(f"count-weighted mean row TVD ({int(well_sampled.sum())} states): {(w * tvd_rows).sum():.4f}")

    # Marginal pushforward vs. empirical counterfactual marginal (no per-row noise).
    pred_marginal = (fac_count / fac_count.sum()) @ dense
    emp_marginal = joint.sum(0) / joint.sum()
    print(f"marginal TVD: {0.5 * (pred_marginal - emp_marginal).abs().sum():.4f}")
    assert 0.5 * (pred_marginal - emp_marginal).abs().sum() < 0.03

    print("fit() from factual sample:", SymbolicIntervention.fit(factual).noise_params)
