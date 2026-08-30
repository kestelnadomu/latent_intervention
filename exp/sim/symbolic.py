"""
Closed-form symbolic counterfactual kernel h_S(s' | s, delta) for a linear-Gaussian
rounded-and-clipped SCM.

Generic over the SCM: the graph, mechanism weights and noise parameters are all
passed in (use :meth:`SymbolicIntervention.from_scm` with a known SCM, or
:meth:`SymbolicIntervention.fit` to estimate the noise from a factual sample).
The concrete CV-screening kernel is assembled in ``exp/sim/cv_screening.py``.

For each non-root node ``s_i = clip_round(m_i(pa_i) + eps_i)`` with
``eps_i ~ Normal(mu_i, sigma_i)``, the three-step counterfactual recipe is exact:

1. **Abduction** - given the full factual state every parent is observed, so the
   noise posterior factorises into independent truncated normals; ``clip_round``
   inversion gives each interval ``[l_i, u_i)``.
2. **Action** - set the do() nodes.
3. **Prediction** - push the factorised posterior forward in topological order by
   exact enumeration over the descendant closure of the intervention.

Enumerating all ``|S| = prod(cardinalities)`` factual states precomputes a sparse
``|S| x |S|`` transition matrix per delta; h_S is then a lookup, and
``(h_S . g)(. | z) = M_delta^T g(. | z)``.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
import torch

from src.schema import ColumnSpec

_SQRT2 = math.sqrt(2.0)


def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    if x == math.inf:
        return 1.0
    if x == -math.inf:
        return 0.0
    return 0.5 * math.erfc(-(x - mu) / (sigma * _SQRT2))


def _abduction_interval(s_i: int, m_fac: float, k: int) -> tuple[float, float]:
    """Invert ``clip_round(m_fac + eps) = s_i`` for the noise interval ``[l, u)``."""
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
    Closed-form symbolic counterfactual kernel h_S(s' | s, delta) for a linear-Gaussian
    rounded-and-clipped SCM. Construct via :meth:`from_scm` or :meth:`fit`.

    Attributes:
        columns: the structured-state schema, in the same order as the framework's
            decoder (column-major flat state index).
        mechanism_coeffs: linear mechanism weights ``m_i(pa_i) = sum_j coef_j pa_j``;
            the keys are the non-root nodes and each value's keys are that node's
            parents, so this doubles as the graph structure.
        noise_params: ``(mu_i, sigma_i)`` of ``eps_i ~ Normal`` per non-root node.
    """

    columns: list[ColumnSpec]
    mechanism_coeffs: dict[str, dict[str, float]]
    noise_params: dict[str, tuple[float, float]]

    def __post_init__(self) -> None:
        self._names = [c.name for c in self.columns]
        self._card = {c.name: c.n_categories for c in self.columns}
        self._n_states = int(np.prod([c.n_categories for c in self.columns]))

    # --- constructors -----------------------------------------------------------------

    @classmethod
    def from_scm(
        cls,
        columns: list[ColumnSpec],
        mechanism_coeffs: dict[str, dict[str, float]],
        noise_params: dict[str, tuple[float, float]],
    ) -> "SymbolicIntervention":
        """Build the kernel from a fully known linear-Gaussian SCM."""
        return cls(
            columns=list(columns),
            mechanism_coeffs={k: dict(v) for k, v in mechanism_coeffs.items()},
            noise_params=dict(noise_params),
        )

    @classmethod
    def fit(
        cls,
        factual: pd.DataFrame,
        mechanism_coeffs: dict[str, dict[str, float]],
        columns: list[ColumnSpec],
        fit_mechanisms: bool = False,
    ) -> "SymbolicIntervention":
        """
        Estimate the noise parameters from a factual sample, for when the simulator is
        unavailable. ``mechanism_coeffs`` supplies the graph (nodes and their parents);
        its coefficient values are used as given, or re-estimated by least squares when
        ``fit_mechanisms=True``.

        ``sigma_i`` is corrected for the uniform rounding jitter (Var -= 1/12); this is a
        method-of-moments estimate on the interior (non-censored) rows.
        """
        coeffs = {k: dict(v) for k, v in mechanism_coeffs.items()}
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
        """Flat column-major index of a state (matches the decoder's log_joint order)."""
        idx = 0
        for name in self._names:
            idx = idx * self._card[name] + int(state[name])
        return idx

    def _mechanism(self, node: str, assign: dict[str, int]) -> float:
        return sum(coef * assign[p] for p, coef in self.mechanism_coeffs[node].items())

    def _children(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {name: [] for name in self._names}
        for child, parents in self.mechanism_coeffs.items():
            for p in parents:
                out[p].append(child)
        return out

    def _descendants(self, nodes: set[str]) -> set[str]:
        """Transitive descendant closure of ``nodes`` (excluding the nodes themselves)."""
        children = self._children()
        seen: set[str] = set()
        stack = [c for n in nodes for c in children[n]]
        while stack:
            n = stack.pop()
            if n not in seen:
                seen.add(n)
                stack.extend(children[n])
        return seen

    def counterfactual(self, state: dict[str, int], delta: dict[str, int]) -> dict[int, float]:
        """h_S(. | state, delta) as ``{state_index: probability}`` over the sparse support."""
        movable = self._descendants(set(delta))
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
        Sparse ``(|S|, |S|)`` COO tensor M_delta with ``M[s, s'] = h_S(s' | s, delta)``.

        Rows sum to 1. ``(h_S . g)(. | z) = M_delta.T @ g(. | z)`` (see :meth:`compose`).
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
        Pushforward ``(h_S . g)``: map a batch of dense ``g(. | z)`` distributions,
        shape ``(batch, |S|)``, through M_delta -> ``(batch, |S|)``.
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
    # Smoke test: closed-form h_S vs. empirical transition counts from the concrete
    # CV-screening SCM/kernel (the one place a real experiment is referenced here).
    from exp.sim.cv_screening import build_scm, build_symbolic_kernel

    h_s = build_symbolic_kernel()
    scm = build_scm()
    delta = {"G": 1}
    m = h_s.transition_matrix(delta)
    dense = m.to_dense()
    assert torch.allclose(dense.sum(1), torch.ones(dense.shape[0]), atol=1e-4)
    print(f"support: {m._nnz()} nonzeros; row sums in "
          f"[{dense.sum(1).min():.4f}, {dense.sum(1).max():.4f}]")

    factual, counterfactual, _ = scm.simulate(50_000, intervention=delta, seed=0)
    idx = lambda df, i: h_s.state_index(df.iloc[i].to_dict())  # noqa: E731
    n = dense.shape[0]
    fac_count = torch.zeros(n)
    joint = torch.zeros(n, n)
    for i in range(len(factual)):
        s, sp = idx(factual, i), idx(counterfactual, i)
        fac_count[s] += 1
        joint[s, sp] += 1

    pred_marginal = (fac_count / fac_count.sum()) @ dense
    emp_marginal = joint.sum(0) / joint.sum()
    tvd = 0.5 * (pred_marginal - emp_marginal).abs().sum()
    print(f"marginal TVD: {tvd:.4f}")
    assert tvd < 0.03
    print("fit() from factual sample:", SymbolicIntervention.fit(
        factual, h_s.mechanism_coeffs, h_s.columns).noise_params)
