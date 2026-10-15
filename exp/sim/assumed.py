"""Assumed (possibly wrong) symbolic operators, derived from a known kernel.

Generic over the SCM: every function takes and returns a
:class:`~exp.sim.symbolic.SymbolicIntervention` and names nodes only through its
arguments. Used by the operator benchmark (``exp/benchmarks/operator/``) to build the
misspecified kernels of ``docs/experiments/operator_benchmark.md``:

- :func:`restrict` — the kernel over a subset of the state columns; parents outside
  the subset are deleted from the mechanisms ("unaware"), optionally followed by
  :func:`refit` of the noise ("refit");
- :func:`drop_parents` — delete edges on the full state (the operator ignores a parent);
- :func:`block` — cut edges in the prediction step only (path-specific operator);
- :func:`scale` — multiply one parameter group (a parent's coefficients, noise mu or sigma);
- :func:`add_mechanism` — give a root a mechanism (a false edge into it);
- :func:`refit` — re-estimate noise (and optionally coefficients) from a factual frame.

Plus the exact factual state distribution :func:`state_distribution` and
:func:`project`, which marginalizes a transition matrix onto a column subset.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from itertools import product

import numpy as np
import pandas as pd

from exp.sim.symbolic import SymbolicIntervention, _category_probs


def _kernel(
    base: SymbolicIntervention,
    *,
    columns=None,
    mechanism_coeffs=None,
    noise_params=None,
    blocked=None,
) -> SymbolicIntervention:
    return SymbolicIntervention(
        columns=list(columns if columns is not None else base.columns),
        mechanism_coeffs=copy.deepcopy(mechanism_coeffs if mechanism_coeffs is not None else base.mechanism_coeffs),
        noise_params=dict(noise_params if noise_params is not None else base.noise_params),
        blocked=copy.deepcopy(blocked if blocked is not None else base.blocked),
    )


def restrict(base: SymbolicIntervention, keep: Sequence[str]) -> SymbolicIntervention:
    """The kernel over the columns ``keep``; parents outside ``keep`` are deleted ("unaware")."""
    keep = list(keep)
    names = [c.name for c in base.columns]
    if not set(keep) <= set(names):
        raise ValueError(f"{sorted(set(keep) - set(names))} are not state columns")
    columns = [c for c in base.columns if c.name in keep]
    mechanisms = {
        child: {p: coef for p, coef in parents.items() if p in keep}
        for child, parents in base.mechanism_coeffs.items()
        if child in keep
    }
    noise = {child: base.noise_params[child] for child in mechanisms}
    blocked = {
        child: {p for p in parents if p in keep}
        for child, parents in base.blocked.items()
        if child in keep
    }
    return _kernel(base, columns=columns, mechanism_coeffs=mechanisms, noise_params=noise, blocked=blocked)


def drop_parents(base: SymbolicIntervention, edges: Mapping[str, Sequence[str]]) -> SymbolicIntervention:
    """Delete edges ``{child: [parent, ...]}``: the operator ignores those parents entirely."""
    mechanisms = copy.deepcopy(base.mechanism_coeffs)
    for child, parents in edges.items():
        for parent in parents:
            if parent not in mechanisms.get(child, {}):
                raise ValueError(f"{parent} -> {child} is not an edge of the kernel")
            del mechanisms[child][parent]
    return _kernel(base, mechanism_coeffs=mechanisms)


def block(base: SymbolicIntervention, edges: Mapping[str, Sequence[str]]) -> SymbolicIntervention:
    """Cut edges ``{child: [parent, ...]}`` in the prediction step (path-specific operator)."""
    blocked = copy.deepcopy(base.blocked)
    for child, parents in edges.items():
        for parent in parents:
            if parent not in base.mechanism_coeffs.get(child, {}):
                raise ValueError(f"{parent} -> {child} is not an edge of the kernel")
            blocked.setdefault(child, set()).add(parent)
    return _kernel(base, blocked=blocked)


def scale(
    base: SymbolicIntervention,
    factor: float,
    *,
    parent: str | None = None,
    noise: str | None = None,
) -> SymbolicIntervention:
    """Multiply one parameter group: every coefficient of ``parent``, or the noise ``mu``/``sigma``."""
    if (parent is None) == (noise is None):
        raise ValueError("scale exactly one group: a parent's coefficients or noise 'mu'/'sigma'")
    mechanisms = copy.deepcopy(base.mechanism_coeffs)
    noise_params = dict(base.noise_params)
    if parent is not None:
        for parents in mechanisms.values():
            if parent in parents:
                parents[parent] *= factor
    elif noise == "mu":
        noise_params = {k: (mu * factor, sigma) for k, (mu, sigma) in noise_params.items()}
    elif noise == "sigma":
        noise_params = {k: (mu, sigma * factor) for k, (mu, sigma) in noise_params.items()}
    else:
        raise ValueError("noise must be 'mu' or 'sigma'")
    return _kernel(base, mechanism_coeffs=mechanisms, noise_params=noise_params)


def add_mechanism(
    base: SymbolicIntervention,
    child: str,
    coeffs: Mapping[str, float],
    noise: tuple[float, float],
) -> SymbolicIntervention:
    """Give the root ``child`` a linear mechanism (a false edge into it) with the given noise."""
    if child in base.mechanism_coeffs:
        raise ValueError(f"{child} already has a mechanism")
    if child not in {c.name for c in base.columns}:
        raise ValueError(f"{child} is not a state column")
    mechanisms = copy.deepcopy(base.mechanism_coeffs)
    mechanisms[child] = dict(coeffs)
    noise_params = {**base.noise_params, child: tuple(noise)}
    order = [c.name for c in base.columns]
    if any(order.index(p) > order.index(child) for p in coeffs):
        raise ValueError("the kernel's column order must stay topological")
    return _kernel(base, mechanism_coeffs=mechanisms, noise_params=noise_params)


ESTIMATORS = ("naive", "intercept", "mle")


def refit(
    base: SymbolicIntervention,
    factual: pd.DataFrame,
    *,
    fit_mechanisms: bool = False,
    estimator: str = "naive",
    only: Sequence[str] | None = None,
) -> SymbolicIntervention:
    """
    Re-estimate the noise (and with ``fit_mechanisms`` the coefficients) from ``factual``.

    - ``naive``: ``SymbolicIntervention.fit`` as is — least squares without an intercept, and
      sigma from the interior level's residual variance minus 1/12. For 3-level variables the
      interior residuals lie in a window of width 1, so sigma is badly underestimated.
    - ``intercept``: slopes by least squares with an intercept (absorbed into the noise mean),
      noise as in ``naive``.
    - ``mle``: maximum likelihood of the ordered-probit model the SCM actually is
      (``clip_round(m + eps)`` has cut points at k - 0.5), for slopes, mu and sigma jointly;
      with ``fit_mechanisms=False`` only mu and sigma are estimated.

    ``only`` restricts the refit to those mechanisms; all others keep ``base``'s parameters.
    """
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {ESTIMATORS}")
    card = {c.name: c.n_categories for c in base.columns}
    nodes = list(only) if only is not None else list(base.mechanism_coeffs)
    mechanisms = copy.deepcopy(base.mechanism_coeffs)
    noise = dict(base.noise_params)
    if estimator == "mle":
        for node in nodes:
            parents = list(base.mechanism_coeffs[node])
            beta, mu, sigma = ordered_probit(
                factual[node].to_numpy(int),
                np.column_stack([factual[p].to_numpy(float) for p in parents]) if parents else np.zeros((len(factual), 0)),
                card[node],
                fixed_beta=None if fit_mechanisms else np.array([base.mechanism_coeffs[node][p] for p in parents]),
            )
            mechanisms[node] = {p: float(b) for p, b in zip(parents, beta)}
            noise[node] = (mu, sigma)
        return _kernel(base, mechanism_coeffs=mechanisms, noise_params=noise)
    coeffs = base.mechanism_coeffs
    if fit_mechanisms and estimator == "intercept":
        coeffs = {}
        for child, parents in base.mechanism_coeffs.items():
            names = list(parents)
            design = np.column_stack(
                [np.ones(len(factual))] + [factual[p].to_numpy(float) for p in names]
            )
            beta, *_ = np.linalg.lstsq(design, factual[child].to_numpy(float), rcond=None)
            coeffs[child] = {p: float(b) for p, b in zip(names, beta[1:])}
        fit_mechanisms = False
    fitted = SymbolicIntervention.fit(factual, coeffs, base.columns, fit_mechanisms=fit_mechanisms)
    for node in nodes:
        mechanisms[node] = dict(fitted.mechanism_coeffs[node])
        noise[node] = fitted.noise_params[node]
    return _kernel(base, mechanism_coeffs=mechanisms, noise_params=noise)


def ordered_probit(
    levels: np.ndarray,
    parents: np.ndarray,
    k: int,
    fixed_beta: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float]:
    """
    Maximum likelihood for ``level = clip_round(parents @ beta + eps)``, ``eps ~ N(mu, sigma)``.

    Cut points are known (``c +- 0.5``, open at both ends), so beta, mu and sigma are all
    identified. With ``fixed_beta`` only mu and sigma are estimated. Starts from least
    squares with an intercept; optimized with L-BFGS (torch, float64).
    """
    import torch

    y = torch.tensor(np.asarray(levels, dtype=float))
    x = torch.tensor(np.asarray(parents, dtype=float))
    bottom, top = y <= 0, y >= k - 1  # open intervals: their outer CDF term is 0 or 1
    lo, hi = y - 0.5, y + 0.5  # finite everywhere, so no 0 * inf in the gradient
    design = np.column_stack([np.ones(len(levels)), parents])
    start, *_ = np.linalg.lstsq(design, levels.astype(float), rcond=None)
    beta0 = start[1:] if fixed_beta is None else np.asarray(fixed_beta, dtype=float)
    mu0 = float(np.mean(levels - parents @ beta0)) if parents.shape[1] else float(levels.mean())
    sigma0 = max(float(np.std(levels - parents @ beta0 - mu0)), 0.1)
    free_beta = torch.tensor(beta0, dtype=torch.float64, requires_grad=fixed_beta is None)
    mu = torch.tensor(mu0, dtype=torch.float64, requires_grad=True)
    log_sigma = torch.tensor(math.log(sigma0), dtype=torch.float64, requires_grad=True)
    variables = [mu, log_sigma] + ([free_beta] if fixed_beta is None else [])
    optimizer = torch.optim.LBFGS(variables, max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        mean = x @ free_beta + mu if x.shape[1] else mu.expand_as(y)
        sigma = log_sigma.exp()
        upper = torch.where(top, torch.ones_like(y), torch.special.ndtr((hi - mean) / sigma))
        lower = torch.where(bottom, torch.zeros_like(y), torch.special.ndtr((lo - mean) / sigma))
        probs = upper - lower
        loss = -torch.log(probs.clamp_min(1e-300)).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    return free_beta.detach().numpy().copy(), float(mu.detach()), float(log_sigma.detach().exp())


def matched_root_noise(prior: Sequence[float], mean_parent: float, coef: float) -> tuple[float, float]:
    """
    Noise for a root turned into ``clip_round(coef * parent + eps)``: ``eps`` is centred so the
    mean level is kept and scaled so the probability of the middle level matches ``prior``
    (3-level roots). This keeps the assumed model's marginal close to the data's.
    """
    if len(prior) != 3:
        raise ValueError("matched_root_noise supports 3-level roots")
    sigma = 0.5 / _std_normal_quantile(0.5 + prior[1] / 2)
    mean_level = sum(level * p for level, p in enumerate(prior))
    return mean_level - coef * mean_parent, sigma


def _std_normal_quantile(q: float) -> float:
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if 0.5 * math.erfc(-mid / math.sqrt(2)) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def state_distribution(
    kernel: SymbolicIntervention, root_priors: Mapping[str, Sequence[float]]
) -> np.ndarray:
    """Exact factual P(s) over the kernel's state index: root priors times mechanism likelihoods."""
    names = [c.name for c in kernel.columns]
    card = {c.name: c.n_categories for c in kernel.columns}
    roots = [n for n in names if n not in kernel.mechanism_coeffs]
    if set(roots) - set(root_priors):
        raise ValueError(f"missing root priors for {sorted(set(roots) - set(root_priors))}")
    probs = np.zeros(int(np.prod([card[n] for n in names])))
    for values in product(*(range(card[n]) for n in names)):
        state = dict(zip(names, values))
        p = 1.0
        for name in names:
            if name in kernel.mechanism_coeffs:
                mu, sigma = kernel.noise_params[name]
                m = sum(c * state[pa] for pa, c in kernel.mechanism_coeffs[name].items())
                p *= _category_probs(m, -math.inf, math.inf, mu, sigma, card[name])[state[name]]
            else:
                p *= root_priors[name][state[name]]
        probs[kernel.state_index(state)] = p
    return probs / probs.sum()


def projection(full: SymbolicIntervention, sub: SymbolicIntervention) -> np.ndarray:
    """Index map: full state index -> index of its projection onto ``sub``'s columns."""
    names = [c.name for c in full.columns]
    card = {c.name: c.n_categories for c in full.columns}
    out = np.zeros(int(np.prod([card[n] for n in names])), dtype=int)
    keep = [c.name for c in sub.columns]
    for values in product(*(range(card[n]) for n in names)):
        state = dict(zip(names, values))
        out[full.state_index(state)] = sub.state_index({n: state[n] for n in keep})
    return out


def project(matrix: np.ndarray, weights: np.ndarray, index: np.ndarray, n_sub: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Marginalize a full transition matrix onto a column subset.

    Returns ``(P(s'_k | s_k), P(s_k))``: rows are averaged over the full states in each
    sub-state cell, weighted by ``weights`` (the factual P(s)); columns are summed per cell.
    """
    to_cell = np.zeros((len(index), n_sub))
    to_cell[np.arange(len(index)), index] = 1.0
    cell_mass = to_cell.T @ weights
    joint = (weights[:, None] * matrix) @ to_cell  # P(s, s'_k)
    rows = to_cell.T @ joint  # P(s_k, s'_k)
    with np.errstate(invalid="ignore", divide="ignore"):
        conditional = np.where(cell_mass[:, None] > 0, rows / cell_mass[:, None], 0.0)
    return conditional, cell_mass
