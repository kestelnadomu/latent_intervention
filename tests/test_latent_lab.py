"""latent_lab on synthetic data: renderers keep unchanged states, the pipeline runs."""

import numpy as np
import pandas as pd
import pytest
import torch

from latent_lab import RENDERERS, IdentityRenderer, LinearSCM, Pipeline, Reader, Renderer, StateEffectRenderer


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    states = pd.DataFrame({"X": rng.integers(0, 3, 400), "D": rng.integers(0, 2, 400)})
    z = torch.randn(400, 6) + torch.tensor(states.to_numpy(), dtype=torch.float32).repeat(1, 3)
    return z, states


@pytest.mark.parametrize("name", RENDERERS)
def test_renderers_keep_unchanged_states(toy, name):
    z, states = toy
    options = {"epochs": 2} if name == "flow" else {}
    renderer = RENDERERS[name](["X", "D"], **options).fit(z, states)
    assert isinstance(renderer, Renderer)
    z_prime = renderer(z, states, states)
    assert z_prime.shape == z.shape
    if name != "constant":
        assert torch.allclose(z_prime, z, atol=1e-5)


def test_unfitted_renderer_raises(toy):
    z, states = toy
    assert torch.equal(IdentityRenderer(["X"])(z, states, states), z)
    with pytest.raises(ValueError):
        StateEffectRenderer(["X", "D"])(z, states, states)


def test_pipeline_with_a_plain_function(toy):
    z, states = toy
    reader = Reader(["X", "D"]).fit(z, states, epochs=5)
    renderer = StateEffectRenderer(["X", "D"]).fit(z, states)
    result = Pipeline(reader, lambda s: s.assign(X=2), renderer)(z)
    assert result.s_prime["X"].eq(2).all() and result.z_prime.shape == z.shape
    true_states = Pipeline(None, lambda s: s, renderer)(z, states)
    assert torch.allclose(true_states.z_prime, z, atol=1e-5)


LEVELS = {"X": 4, "T": 3, "D": 3, "U": 3}
PARENTS = {"D": {"X": 0.30, "T": 0.60}, "U": {"X": 0.35, "T": 0.50}}
NOISE = {"D": (-0.10, 0.50), "U": (-0.15, 0.50)}


def _all_states(reps: int) -> pd.DataFrame:
    from itertools import product

    grid = pd.DataFrame(list(product(*[range(k) for k in LEVELS.values()])), columns=list(LEVELS))
    return grid.loc[grid.index.repeat(reps)].reset_index(drop=True)


def _flat(df: pd.DataFrame) -> np.ndarray:
    return (((df["X"] * 3 + df["T"]) * 3 + df["D"]) * 3 + df["U"]).to_numpy()


@pytest.mark.parametrize("blocked", [{}, {"D": ["X"]}])
def test_linear_scm_matches_exact_kernel(blocked):
    from exp.sim.symbolic import SymbolicIntervention
    from src.schema import ColumnSpec

    kernel = SymbolicIntervention.from_scm([ColumnSpec(c, k) for c, k in LEVELS.items()], PARENTS, NOISE)
    kernel.blocked = {child: set(parents) for child, parents in blocked.items()}
    exact = kernel.transition_matrix({"X": 3}).to_dense().numpy()
    reps = 2000
    states = _all_states(reps)
    out = LinearSCM(LEVELS, PARENTS, NOISE).counterfactual(states, {"X": 3}, blocked)
    empirical = np.zeros_like(exact)
    np.add.at(empirical, (_flat(states), _flat(out)), 1 / reps)
    assert np.abs(empirical - exact).max() < 0.05


def test_linear_scm_blocking_and_validation():
    scm = LinearSCM(LEVELS, PARENTS, NOISE)
    states = _all_states(5)
    out = scm.do(X=0, blocked={"D": ["X"], "U": ["X"]})(states)
    assert out["X"].eq(0).all() and out[["T", "D", "U"]].equals(states[["T", "D", "U"]])
    assert "D = clip(round(0.30·X + 0.60·T + ε), 0, 2)" in repr(scm)
    per_unit = scm.do(X=lambda s: (s["X"] + 1) % 4, blocked={"D": ["X"], "U": ["X"]})(states)
    assert per_unit["X"].equals((states["X"] + 1) % 4) and per_unit["D"].equals(states["D"])
    with pytest.raises(ValueError):
        scm.do(X=4)(states)
    with pytest.raises(ValueError):
        LinearSCM(LEVELS, {"U": {"D": 1.0}, "D": {"X": 1.0}}, {"U": (0, 1), "D": (0, 1)})  # not topological
    with pytest.raises(ValueError):
        LinearSCM(LEVELS, PARENTS, {"D": (0, 1)})
