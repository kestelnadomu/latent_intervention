"""A discrete linear SCM, one source of operators omega_S.

Each non-root node is ``clip(round(sum_j w_j * parent_j + eps), 0, k - 1)`` with
``eps ~ Normal(mu, sd)``; roots are taken as given. ``do()`` turns an intervention into
an operator: the unit-level SCM counterfactual (abduction, action, prediction).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

# A do() value: one level for every unit, or a function of the states returning one
# level per unit (a random or unit-dependent intervention, drawn at each call).
DoValue = int | Callable[[pd.DataFrame], np.ndarray]


class LinearSCM:
    """``LinearSCM(levels, parents, noise)``.

    Args:
        levels: number of levels per node, e.g. ``{"X": 4, "T": 3, "D": 3, "U": 3}``.
        parents: ``{node: {parent: weight}}`` for every non-root node, in topological
            order. Nodes in ``levels`` without an entry are roots.
        noise: ``{node: (mu, sd)}`` of each non-root node's Gaussian noise.
        seed: seeds the noise draws of ``counterfactual``.
    """

    def __init__(
        self,
        levels: dict[str, int],
        parents: dict[str, dict[str, float]],
        noise: dict[str, tuple[float, float]],
        seed: int = 0,
    ) -> None:
        if set(noise) != set(parents):
            raise ValueError("noise needs exactly one (mu, sd) per non-root node")
        seen = {node for node in levels if node not in parents}
        for node, weights in parents.items():
            if node not in levels:
                raise ValueError(f"{node} has parents but no levels")
            unknown = set(weights) - seen
            if unknown:
                raise ValueError(f"parents {sorted(unknown)} of {node} are unknown or come later")
            seen.add(node)
        self.levels = dict(levels)
        self.parents = {node: dict(weights) for node, weights in parents.items()}
        self.noise = {node: (float(mu), float(sd)) for node, (mu, sd) in noise.items()}
        self.rng = np.random.default_rng(seed)

    @property
    def roots(self) -> list[str]:
        return [node for node in self.levels if node not in self.parents]

    def __repr__(self) -> str:
        lines = [f"LinearSCM, roots {self.roots}"]
        for node, weights in self.parents.items():
            mean = " + ".join(f"{w:.2f}·{p}" for p, w in weights.items()) or "0"
            mu, sd = self.noise[node]
            lines.append(
                f"  {node} = clip(round({mean} + ε), 0, {self.levels[node] - 1}),  ε ~ N({mu:.2f}, {sd:.2f})"
            )
        return "\n".join(lines)

    def counterfactual(
        self,
        states: pd.DataFrame,
        do: dict[str, DoValue],
        blocked: dict[str, list[str]] | None = None,
    ) -> pd.DataFrame:
        """One draw of each unit's counterfactual under ``do``.

        1. Abduction: draw each node's noise from the range consistent with its observed value.
        2. Action: set the ``do`` values.
        3. Prediction: recompute the non-root nodes in order.

        A ``do`` value is a level, or a function of ``states`` that returns one level
        per unit, e.g. ``lambda s: rng.choice(4, len(s), p=p_X)`` for do(X ~ P(X)).
        ``blocked={"D": ["X"]}`` lets D read the *observed* X (a path-specific effect).
        Columns not in the SCM are passed through unchanged.
        """
        unknown = set(do) - set(self.levels)
        if unknown:
            raise ValueError(f"unknown do() nodes {sorted(unknown)}")
        blocked = blocked or {}
        new = states.copy()
        for node, value in do.items():
            value = np.broadcast_to(value(states) if callable(value) else value, len(states))
            if not np.isin(value, np.arange(self.levels[node])).all():
                raise ValueError(f"do({node}) outside the levels 0..{self.levels[node] - 1}")
            new[node] = value
        for node, weights in self.parents.items():
            if node in do:
                continue
            mu, sd = self.noise[node]
            k = self.levels[node]
            value = states[node].to_numpy()
            factual_mean = sum(w * states[p].to_numpy() for p, w in weights.items())
            lo = np.where(value == 0, -np.inf, value - 0.5) - factual_mean
            hi = np.where(value == k - 1, np.inf, value + 0.5) - factual_mean
            eps = truncnorm.rvs((lo - mu) / sd, (hi - mu) / sd, loc=mu, scale=sd, random_state=self.rng)
            mean = sum(
                w * (states if p in blocked.get(node, []) else new)[p].to_numpy()
                for p, w in weights.items()
            )
            new[node] = np.clip(np.rint(mean + eps), 0, k - 1).astype(int)
        return new

    def do(
        self, blocked: dict[str, list[str]] | None = None, **values: DoValue
    ) -> Callable[[pd.DataFrame], pd.DataFrame]:
        """The operator of do(values), a function for ``Pipeline``.

        ``scm.do(X=3)``; ``scm.do(X=3, blocked={"D": ["X"]})`` for the path-specific one;
        ``scm.do(X=lambda s: rng.choice(4, len(s), p=p_X))`` for a random intervention,
        drawn anew for every call.
        """
        if not values:
            raise ValueError("name at least one node, e.g. scm.do(X=3)")
        return lambda states: self.counterfactual(states, values, blocked)
