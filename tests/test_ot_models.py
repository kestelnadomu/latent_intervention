"""Mathematical/inference contracts for the isolated OT methods (synthetic data only)."""

import copy
import math

import pytest
import torch

from src.schema import ColumnSpec, flat_state_index, unflatten_state_index
from src.semantic_decoder.model import SemanticDecoder
from src.latent_intervention.ot import OT_VARIANTS, make_model, load_model
from src.latent_intervention.ot.capacity import capacity_report
from src.latent_intervention.ot.common import coupled_weights
from src.latent_intervention.ot.transport import (
    integrate,
    sinkhorn,
    sinkhorn_divergence,
)
from src.latent_intervention.dispatch import (
    make_latent_intervention,
    load_latent_intervention,
    sample_counterfactual,
)


@pytest.fixture(autouse=True)
def deterministic():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(87)
        yield
    torch.set_num_threads(old)


COLS = [ColumnSpec("X", 4), ColumnSpec("T", 2)]


class Kernel:
    def transition_matrix(self, action):
        states = unflatten_state_index(torch.arange(8), COLS)
        if action:
            states[:, 0] = 3
        dst = flat_state_index(states, COLS)
        return torch.sparse_coo_tensor(
            torch.stack([torch.arange(8), dst]),
            torch.ones(8),
            (8, 8),
            check_invariants=True,
        ).coalesce()


def model_for(variant, dim=4):
    options = {} if variant == "ot_gaussian" else dict(hidden_dim=8, state_embed_dim=2)
    if variant == "ot_flow":
        options.update(ode_steps=2, ode_max_steps=8)
    return make_model(variant, latent_dim=dim, columns=COLS, settings=options)


def fitted(variant):
    model = model_for(variant)
    states = unflatten_state_index(torch.arange(8), COLS).repeat(3, 1)
    z = torch.randn(len(states), 4)
    if variant == "ot_gaussian":
        model.fit(z, states)
    else:
        model.fit_normalization(z, states)
    return model.eval(), z, states


def test_sinkhorn_marginals_divergence_and_failures():
    x, y = torch.randn(9, 3), torch.randn(11, 3)
    plan, record = sinkhorn(x, y)
    torch.testing.assert_close(
        plan.sum(1), torch.full((9,), 1 / 9, dtype=torch.float64), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        plan.sum(0), torch.full((11,), 1 / 11, dtype=torch.float64), atol=1e-6, rtol=0
    )
    assert record["marginal_error"] <= 1e-6
    assert sinkhorn_divergence(x, x) == pytest.approx(0, abs=1e-8)
    assert sinkhorn_divergence(x, y) > 0
    permutation = torch.randperm(len(x))
    permuted, _ = sinkhorn(x[permutation], y)
    torch.testing.assert_close(permuted, plan[permutation])
    with pytest.raises(FloatingPointError, match="converge"):
        sinkhorn(x, y, max_iterations=1, tolerance=1e-14)
    with pytest.raises(ValueError):
        sinkhorn(x * float("nan"), y)


def test_sinkhorn_sparse_pair_can_exceed_original_iteration_budget():
    rng = torch.Generator().manual_seed(19)
    x = torch.randn(5, 3, generator=rng)
    y = torch.randn(5, 3, generator=rng)
    with pytest.raises(FloatingPointError, match="converge"):
        sinkhorn(x, y, max_iterations=1000)
    plan, record = sinkhorn(x, y)
    assert record["iterations"] > 1000
    assert record["marginal_error"] <= 1e-6
    torch.testing.assert_close(
        plan.sum(1), torch.full((5,), 0.2, dtype=torch.float64), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        plan.sum(0), torch.full((5,), 0.2, dtype=torch.float64), atol=1e-6, rtol=0
    )


def test_gaussian_known_map_sparse_states_and_full_basis():
    model, z, states = fitted("ot_gaussian")
    q, _ = torch.linalg.qr(torch.randn(4, 4))
    model.mean.zero_()
    model.scale.fill_(1)
    model.basis.copy_(q)
    model.state_mean.zero_()
    model.state_mean[6] = torch.tensor([1.0, -1.0, 2.0, 0.0])
    model.state_variance.fill_(1)
    model.state_variance[6] = torch.tensor([0.25, 4.0, 1.0, 9.0])
    source, target = (
        model.states[0].expand(len(z), -1),
        model.states[6].expand(len(z), -1),
    )
    expected = (model.state_mean[6] + (z @ q) * model.state_variance[6].sqrt()) @ q.T
    torch.testing.assert_close(model.transport(z, source, target), expected)
    sparse = model_for("ot_gaussian").fit(z[:4], states[:4])
    assert (sparse.fit_counts == 0).sum() == 4
    assert torch.isfinite(sparse.state_mean).all() and (sparse.state_variance > 0).all()
    moved = sparse.transport(z[:2], sparse.states[[0, 1]], sparse.states[[6, 7]])
    assert torch.isfinite(moved).all()
    assert sparse.basis.shape == (4, 4)  # no dimensionality reduction


def test_ode_inverse_composition_and_failure():
    x, states = torch.randn(8, 4), torch.zeros(8, 2, dtype=torch.long)

    def velocity(point, time, state):
        return 0.2 * point + state[:, :1] * 0.3

    y, diag = integrate(velocity, x, states, steps=4, max_steps=16, tolerance=1e-5)
    torch.testing.assert_close(y, x * math.exp(0.2), atol=1e-5, rtol=1e-5)
    original, _ = integrate(
        velocity, y, states, reverse=True, steps=4, max_steps=16, tolerance=1e-5
    )
    torch.testing.assert_close(original, x, atol=1e-5, rtol=1e-5)
    assert diag["nfe"] == 4 * (4 + 8)
    with pytest.raises(FloatingPointError, match="tolerance"):
        integrate(
            lambda p, t, s: 20 * p, x, states, steps=1, max_steps=2, tolerance=1e-12
        )
    model, z, source = fitted("ot_flow")
    model.velocity = velocity
    model.ode_steps, model.ode_max_steps = 4, 16
    middle, end = source.roll(1, 0), source.roll(2, 0)
    first = model.transport_normalized(z, source, middle)
    composed = model.transport_normalized(first, middle, end)
    direct = model.transport_normalized(z, source, end)
    torch.testing.assert_close(composed, direct, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("variant", OT_VARIANTS)
def test_coupling_identity_roundtrip_sampling_and_frozen_g(variant, tmp_path):
    model, z, states = fitted(variant)
    z = z[:3]
    decoder = (
        SemanticDecoder(4, COLS, hidden_dim=8, n_hidden=1).eval().requires_grad_(False)
    )
    frozen = copy.deepcopy(decoder.state_dict())
    kernel = Kernel()
    posterior = torch.zeros(3, 8)
    posterior[:, :2] = 0.5
    matrix = kernel.transition_matrix({"X": 3}).to_dense()
    locations, weights = model.components(
        z, posterior.requires_grad_(), matrix.requires_grad_()
    )
    src, dst, joint = coupled_weights(posterior, matrix)
    torch.testing.assert_close(
        weights.exp(), posterior.detach()[:, src] * matrix.detach()[src, dst]
    )
    assert torch.isneginf(weights).sum() == 18 and not weights.requires_grad
    assert torch.isfinite(locations).all()
    assert torch.equal(model.transport(z, states[:3], states[:3]), z)
    assert torch.equal(
        model.sample_symbolic(z, {}, n_samples=5), z[None].expand(5, -1, -1)
    )
    path = tmp_path / "selected.pt"
    model.save(
        path,
        metadata={"test": True, "schema": [(c.name, c.n_categories) for c in COLS]},
    )
    restored = load_latent_intervention(
        path,
        expected_variant=variant,
        expected_columns=COLS,
        expected_metadata={
            "test": True,
            "schema": [[c.name, c.n_categories] for c in COLS],
        },
    )
    outputs = [
        sample_counterfactual(
            m,
            z,
            {"X": 3},
            n_samples=7,
            decoder=decoder,
            h_s=kernel,
            generator=torch.Generator().manual_seed(9),
        )
        for m in (model, restored)
    ]
    torch.testing.assert_close(*outputs, atol=0, rtol=0)
    assert outputs[0].shape == (7, 3, 4)
    with torch.inference_mode():
        samples = sample_counterfactual(model, z, {"X": 3}, decoder=decoder, h_s=kernel)
        assert torch.isfinite(samples).all()
    for key, value in decoder.state_dict().items():
        assert torch.equal(value, frozen[key])
    assert all(p.grad is None for p in decoder.parameters())
    with pytest.raises(ValueError, match="supports"):
        model.sample_symbolic(z, {"X": 1}, decoder=decoder, h_s=kernel)
    with pytest.raises(ValueError, match="supports"):
        model.sample_symbolic(z, {"X": 3.0}, decoder=decoder, h_s=kernel)
    with pytest.raises(ValueError, match="observed"):
        sample_counterfactual(model, z, {"X": 3}, source_states=states[:3])
    with pytest.raises(ValueError, match="provenance"):
        load_model(path, expected_metadata={"test": False})
    with pytest.raises(ValueError, match="variant"):
        load_model(path, expected_variant="state_flow")
    payload = torch.load(path, weights_only=True)
    payload["state_dict"]["scale"][0] = float("nan")
    torch.save(payload, path)
    with pytest.raises(ValueError, match="nonfinite"):
        load_model(path)


@pytest.mark.parametrize("variant", OT_VARIANTS)
def test_sample_reuse_matches_all_original_joint_draws(variant):
    model, z, _ = fitted(variant)
    z = z[:2]
    decoder = SemanticDecoder(4, COLS, hidden_dim=8, n_hidden=1).eval()
    kernel = Kernel()
    src, dst, weights = coupled_weights(
        decoder.log_joint(z).exp(), kernel.transition_matrix({"X": 3}).to_dense()
    )
    picks = torch.multinomial(
        weights, 64, replacement=True, generator=torch.Generator().manual_seed(10)
    ).t()
    expected = model.transport(
        z[None].expand(64, -1, -1).reshape(-1, 4),
        model.states[src[picks]].flatten(0, 1),
        model.states[dst[picks]].flatten(0, 1),
    )
    actual = model.sample_symbolic(
        z,
        {"X": 3},
        decoder=decoder,
        h_s=kernel,
        n_samples=64,
        generator=torch.Generator().manual_seed(10),
    )
    torch.testing.assert_close(actual, expected.reshape(64, 2, 4), atol=1e-6, rtol=1e-6)
    assert model.last_diagnostics["sampled_pairs"] == 128
    assert model.last_diagnostics["unique_pairs"] <= 16
    model.sample_symbolic(z, {})
    assert model.last_diagnostics == {}


def test_invalid_coupled_weights_fail_explicitly():
    for posterior, matrix in (
        (torch.empty(2, 0), torch.empty(0, 0)),
        (torch.zeros(1, 2), torch.eye(2)),
        (torch.ones(1, 2) / 2, -torch.eye(2)),
    ):
        with pytest.raises(ValueError, match="probabilities"):
            coupled_weights(posterior, matrix)


@pytest.mark.parametrize("dimension", (256, 768))
@pytest.mark.parametrize("tier", (1, 2))
@pytest.mark.parametrize("variant", ("ot_flow",))
def test_instantiated_capacity(dimension, tier, variant):
    columns = [
        ColumnSpec("X", 4),
        ColumnSpec("T", 3),
        ColumnSpec("D", 3),
        ColumnSpec("U", 3),
    ]
    model = make_latent_intervention(
        variant,
        latent_dim=dimension,
        columns=columns,
        ot_settings={"tier": tier},
        seed=42,
    )
    result = capacity_report(model, tier=tier)
    assert result["relative_difference"] < 0.007
    assert result["training_only"] == 0
