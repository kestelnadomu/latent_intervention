"""
Generic structural causal model engine for the data-generation half.

Holds no knowledge of any specific experiment: an :class:`SCM` is built from an
ordered list of nodes, a set of root nodes, a per-node exogenous-noise sampler
and a per-node structural function. The concrete CV-screening SCM is assembled
in ``exp/sim/cv_screening.py`` and pointed to from ``exp/sim/config.yaml``.

``SCM.simulate`` generates factual data and, under a ``do()`` intervention,
counterfactual data by reusing the stored exogenous noise (Pearl's abduction
step is exact because the noise is known), so the two frames are exact
counterfactual pairs row by row. The original R implementation
(``exp/sim/R/utils/{sim_scm.R, sim_data.R}``) remains the reference.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from src.schema import ColumnSpec

NoiseSampler = Callable[[np.random.Generator, int], np.ndarray]
StructuralFn = Callable[[dict[str, np.ndarray], dict[str, np.ndarray]], np.ndarray]


def clip_round(x: np.ndarray, lower: int, upper: int) -> np.ndarray:
    """Round half-to-even (matching R's ``round()``) and clip to ``[lower, upper]``."""
    return np.clip(np.rint(x), lower, upper).astype(np.int64)


@dataclass(frozen=True)
class LinearMechanism:
    """
    ``clip_round(sum_j coef_j * pa_j + eps_i, lower, upper)`` structural function.

    ``coeffs`` maps each parent node name to its linear weight (no bias term). The
    node's own exogenous noise term is added by :class:`SCM` at evaluation time.
    """

    coeffs: dict[str, float]
    lower: int
    upper: int


def linear_mechanism(coeffs: dict[str, float], lower: int, upper: int) -> LinearMechanism:
    """Convenience constructor for :class:`LinearMechanism`."""
    return LinearMechanism(dict(coeffs), lower, upper)


@dataclass
class SCM:
    """
    A structural causal model over discrete nodes.

    Attributes:
        nodes: every node (structured-state columns and the downstream outcome),
            in topological order, with its cardinality.
        roots: names of the exogenous root nodes (set directly from noise).
        noise: per-node exogenous-noise sampler ``(rng, n) -> (n,) array``.
        mechanisms: per-non-root structural function ``(data, noise) -> (n,) array``
            of integer category values; may be a plain callable or one produced by
            :func:`linear_mechanism` (whose noise term is added automatically).
    """

    nodes: list[ColumnSpec]
    roots: Sequence[str]
    noise: dict[str, NoiseSampler]
    mechanisms: dict[str, StructuralFn | LinearMechanism]
    _card: dict[str, int] = field(init=False)
    _order: list[str] = field(init=False)

    def __post_init__(self) -> None:
        self._card = {c.name: c.n_categories for c in self.nodes}
        self._order = [c.name for c in self.nodes]
        roots = set(self.roots)
        non_roots = [n for n in self._order if n not in roots]
        missing = [n for n in non_roots if n not in self.mechanisms]
        if missing:
            raise ValueError(f"No mechanism for non-root node(s): {missing}")
        if not set(self._order) <= set(self.noise):
            raise ValueError(f"Missing noise samplers for {set(self._order) - set(self.noise)}")

    def _apply(self, name: str, data: dict[str, np.ndarray], noise: dict[str, np.ndarray]) -> np.ndarray:
        fn = self.mechanisms[name]
        if isinstance(fn, LinearMechanism):  # add this node's own noise term
            acc = sum(coef * data[p] for p, coef in fn.coeffs.items())
            return clip_round(acc + noise[name], fn.lower, fn.upper)
        return fn(data, noise)

    def _generate(self, noise: dict[str, np.ndarray], intervention: dict[str, int], n: int) -> dict[str, np.ndarray]:
        """One topological pass; intervened nodes are held fixed."""
        data: dict[str, np.ndarray] = {}
        for node in self._order:
            if node in intervention:
                data[node] = np.full(n, intervention[node], dtype=np.int64)
            elif node in set(self.roots):
                data[node] = noise[node].astype(np.int64)
            else:
                data[node] = self._apply(node, data, noise)
        return data

    def validate_intervention(self, intervention: dict[str, int]) -> dict[str, int]:
        """Check intervention node names and value ranges against the schema."""
        unknown = set(intervention) - set(self._order)
        if unknown:
            raise ValueError(f"Unknown intervention nodes: {sorted(unknown)}. Available: {self._order}")
        for node, value in intervention.items():
            if not 0 <= int(value) < self._card[node]:
                raise ValueError(f"do({node}={value}) outside 0..{self._card[node] - 1}")
        return {node: int(value) for node, value in intervention.items()}

    def simulate(
        self,
        n: int,
        intervention: dict[str, int],
        seed: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Simulate paired factual/counterfactual data.

        Returns ``(factual, counterfactual, epsilon)`` frames, each with an ``id``
        column. The counterfactual applies ``do(intervention)`` and reuses the
        factual exogenous noise, so rows are exact counterfactual pairs.
        """
        if n < 1:
            raise ValueError(f"n must be positive, got {n}")
        intervention = self.validate_intervention(intervention)
        if not intervention:
            raise ValueError("Intervention must target at least one node, e.g. {'G': 1}.")

        rng = np.random.default_rng(seed)
        noise = {node: self.noise[node](rng, n) for node in self._order}

        factual = self._generate(noise, intervention={}, n=n)
        counterfactual = self._generate(noise, intervention=intervention, n=n)

        ids = pd.RangeIndex(1, n + 1, name=None)
        factual_df = pd.DataFrame({"id": ids, **factual})
        counterfactual_df = pd.DataFrame({"id": ids, **counterfactual})
        epsilon_df = pd.DataFrame({"id": ids, **{f"eps_{node}": eps for node, eps in noise.items()}})
        return factual_df, counterfactual_df, epsilon_df


if __name__ == "__main__":
    # Smoke test on a random SCM: seeding, do() semantics, category ranges.
    rng = np.random.default_rng(0)
    nodes = [ColumnSpec(name, int(rng.integers(2, 5))) for name in ("a", "b", "c", "d")]
    roots = ("a", "b")

    def root_noise(card: int) -> NoiseSampler:
        return lambda r, n: r.integers(0, card, n)

    noise: dict[str, NoiseSampler] = {
        c.name: root_noise(c.n_categories) if c.name in roots else (lambda r, n: r.normal(0.0, 0.5, n))
        for c in nodes
    }
    mechanisms: dict[str, StructuralFn | LinearMechanism] = {
        "c": linear_mechanism({"a": 0.5, "b": 0.3}, 0, nodes[2].n_categories - 1),
        "d": linear_mechanism({"a": 0.2, "c": 0.4}, 0, nodes[3].n_categories - 1),
    }
    scm = SCM(nodes, roots, noise, mechanisms)

    intervention = {"a": 1}
    factual, counterfactual, epsilon = scm.simulate(2000, intervention=intervention, seed=1)
    factual2, _, _ = scm.simulate(2000, intervention=intervention, seed=1)
    assert factual.equals(factual2), "seeding must be reproducible"
    assert (counterfactual["a"] == 1).all()
    assert factual["b"].equals(counterfactual["b"]), "non-descendant of a must not change"
    for c in nodes:
        assert factual[c.name].between(0, c.n_categories - 1).all()
        assert counterfactual[c.name].between(0, c.n_categories - 1).all()
    print("random-SCM smoke test passed")
    print(factual.head())