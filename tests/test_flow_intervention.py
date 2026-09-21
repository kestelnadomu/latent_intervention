from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from src.flow_intervention import (
    DirectSemanticFlowIntervention,
    DistilledFlowIntervention,
    StateConditionalFlow,
    counterfactual,
    load_latent_intervention,
    make_latent_intervention,
    multivariate_energy_distance,
    sample_counterfactual,
    sample_symbolic_states,
    train_direct_semantic_flow,
    train_distilled_flow,
    train_latent_intervention_model,
    train_state_conditional_flow,
)
from src.latent_intervention import LatentIntervention
from src.schema import ColumnSpec, flat_state_index, unflatten_state_index
from src.semantic_decoder import SemanticDecoder


@pytest.fixture
def columns() -> list[ColumnSpec]:
    return [ColumnSpec("a", 2), ColumnSpec("b", 3)]


@pytest.fixture
def latents() -> torch.Tensor:
    generator = torch.Generator().manual_seed(19)
    return torch.randn(12, 6, generator=generator)


def _small_kwargs() -> dict[str, int]:
    return {
        "n_blocks": 4,
        "hidden_dim": 16,
        "condition_dim": 8,
        "state_embed_dim": 4,
        "permutation_seed": 7,
    }


def _states(n: int) -> torch.Tensor:
    row = torch.tensor([[0, 0], [1, 1], [0, 2]], dtype=torch.long)
    return row.repeat((n + len(row) - 1) // len(row), 1)[:n]


def _objective(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.tensor([[1, 0]], dtype=torch.long).expand(n, -1).clone()
    mask = torch.tensor([[True, False]]).expand(n, -1).clone()
    return values, mask


def _make_nontrivial(model: nn.Module) -> None:
    """Move coupling heads away from their intentional identity initialization."""
    generator = torch.Generator().manual_seed(23)
    with torch.no_grad():
        for block in model.flow.blocks:  # type: ignore[attr-defined]
            final = block.conditioner[-1]
            final.weight.copy_(
                0.03
                * torch.randn(
                    final.weight.shape,
                    generator=generator,
                    dtype=final.weight.dtype,
                )
            )
            final.bias.copy_(
                0.03
                * torch.randn(
                    final.bias.shape,
                    generator=generator,
                    dtype=final.bias.dtype,
                )
            )


class _DeterministicKernel:
    def __init__(self, columns: list[ColumnSpec]) -> None:
        self.columns = list(columns)

    def state_index(self, state: dict[str, int]) -> int:
        values = torch.tensor([[state[column.name] for column in self.columns]])
        return int(flat_state_index(values, self.columns).item())

    def transition_matrix(self, delta: dict[str, int]) -> torch.Tensor:
        n_states = 1
        for column in self.columns:
            n_states *= column.n_categories
        source = unflatten_state_index(torch.arange(n_states), self.columns)
        target = source.clone()
        for index, column in enumerate(self.columns):
            if column.name in delta:
                target[:, index] = delta[column.name]
        matrix = torch.zeros(n_states, n_states)
        matrix[torch.arange(n_states), flat_state_index(target, self.columns)] = 1.0
        return matrix

    def compose(self, g_probs: torch.Tensor, delta: dict[str, int]) -> torch.Tensor:
        return g_probs @ self.transition_matrix(delta).to(g_probs)


class _RecordingDecoder(nn.Module):
    def __init__(self, latent_dim: int, columns: list[ColumnSpec]) -> None:
        super().__init__()
        self.columns = list(columns)
        n_states = 1
        for column in columns:
            n_states *= column.n_categories
        self.linear = nn.Linear(latent_dim, n_states)
        self.calls: list[torch.Tensor] = []

    def log_joint(self, z: torch.Tensor) -> torch.Tensor:
        result = F.log_softmax(self.linear(z), dim=-1)
        self.calls.append(result.detach().clone())
        return result


def test_realnvp_round_trip_and_log_determinants_cancel(
    columns: list[ColumnSpec],
) -> None:
    model = StateConditionalFlow(6, columns, **_small_kwargs())
    _make_nontrivial(model)
    base = torch.randn(5, 6, generator=torch.Generator().manual_seed(3))
    condition = model._condition(_states(5), len(base))

    transformed, forward_log_det = model.flow.forward_transform(base, condition)
    restored, inverse_log_det = model.flow.inverse_transform(transformed, condition)

    assert torch.allclose(restored, base, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        forward_log_det + inverse_log_det,
        torch.zeros_like(forward_log_det),
        atol=1e-6,
        rtol=1e-6,
    )


def test_alternating_couplings_transform_every_coordinate(
    columns: list[ColumnSpec],
) -> None:
    model = StateConditionalFlow(6, columns, **_small_kwargs())
    labels = torch.arange(model.latent_dim)
    transformed: set[int] = set()
    for block in model.flow.blocks:
        labels = labels[block.permutation.cpu()]
        transformed.update(int(value) for value in labels[block.transform_index].tolist())
    assert transformed == set(range(6))


@pytest.mark.parametrize("n_blocks", [2, 8])
def test_seeded_permutations_cover_all_production_latent_coordinates(
    columns: list[ColumnSpec], n_blocks: int
) -> None:
    model = StateConditionalFlow(
        128,
        columns,
        n_blocks=n_blocks,
        hidden_dim=8,
        condition_dim=4,
        state_embed_dim=2,
        permutation_seed=42,
    )
    labels = torch.arange(model.latent_dim)
    transformed: set[int] = set()
    for block in model.flow.blocks:
        labels = labels[block.permutation.cpu()]
        transformed.update(int(value) for value in labels[block.transform_index].tolist())

    assert transformed == set(range(128))


def test_flow_inputs_are_colocated_and_cast_to_model_dtype(
    columns: list[ColumnSpec], latents: torch.Tensor
) -> None:
    state = StateConditionalFlow(6, columns, **_small_kwargs())
    state.set_latent_statistics(latents)
    source = _states(len(latents))
    abducted = state.abduct(latents.double(), source)

    direct = DistilledFlowIntervention(6, columns, **_small_kwargs())
    direct.set_latent_statistics(latents)
    values, mask = _objective(len(latents))
    samples = direct.sample(latents.double(), values, mask, n_samples=2)

    assert abducted.dtype == state.latent_mean.dtype
    assert samples.dtype == direct.latent_mean.dtype
    assert torch.isfinite(abducted).all()
    assert torch.isfinite(samples).all()


def test_state_transport_is_identity_and_preserves_abducted_noise(
    columns: list[ColumnSpec], latents: torch.Tensor
) -> None:
    model = StateConditionalFlow(6, columns, **_small_kwargs())
    model.set_latent_statistics(latents)
    _make_nontrivial(model)
    source = _states(len(latents))
    target = torch.stack((1 - source[:, 0], (source[:, 1] + 1) % 3), dim=-1)

    factual_noise = model.abduct(latents, source)
    identity = model.transport(latents, source, source)
    transported = model.transport(latents, source, target)
    recovered_noise = model.abduct(transported, target)

    assert torch.allclose(identity, latents, atol=1e-5, rtol=1e-5)
    assert torch.allclose(recovered_noise, factual_noise, atol=1e-5, rtol=1e-5)
    assert torch.isfinite(model.log_prob(latents, source)).all()


@pytest.mark.parametrize(
    "model_type", [DistilledFlowIntervention, DirectSemanticFlowIntervention]
)
def test_direct_flow_sampling_is_seeded_and_empty_intervention_is_exact_identity(
    model_type,
    columns: list[ColumnSpec],
    latents: torch.Tensor,
) -> None:
    model = model_type(6, columns, **_small_kwargs())
    model.set_latent_statistics(latents)
    _make_nontrivial(model)
    values, mask = _objective(len(latents))

    first = model.sample(
        latents,
        values,
        mask,
        n_samples=3,
        generator=torch.Generator().manual_seed(31),
    )
    second = model.sample(
        latents,
        values,
        mask,
        n_samples=3,
        generator=torch.Generator().manual_seed(31),
    )
    identity = model.sample(
        latents,
        torch.zeros_like(values),
        torch.zeros_like(mask),
        n_samples=3,
        generator=torch.Generator().manual_seed(99),
    )

    assert first.shape == (3, len(latents), 6)
    assert torch.equal(first, second)
    assert torch.equal(identity, latents.unsqueeze(0).expand_as(identity))
    assert torch.isfinite(first).all()


def test_flow_log_prob_is_finite_and_differentiable(
    columns: list[ColumnSpec], latents: torch.Tensor
) -> None:
    model = DistilledFlowIntervention(6, columns, **_small_kwargs())
    model.set_latent_statistics(latents)
    _make_nontrivial(model)
    values, mask = _objective(len(latents))
    sample = model.sample(
        latents,
        values,
        mask,
        noise=torch.zeros(1, len(latents), 6),
    )[0].detach()

    loss = -model.log_prob(sample, latents, values, mask).mean()
    loss.backward()

    assert torch.isfinite(loss)
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_multivariate_energy_distance_has_expected_invariants() -> None:
    samples = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.5, 0.5], [0.25, 0.75]],
        ]
    )
    shifted = samples + torch.tensor([2.0, -1.0])

    same = multivariate_energy_distance(samples, samples)
    forward = multivariate_energy_distance(samples, shifted)
    reverse = multivariate_energy_distance(shifted, samples)

    assert same == pytest.approx(0.0, abs=1e-7)
    assert forward > 0
    assert torch.allclose(forward, reverse)


def test_model_validation_rejects_malformed_inputs(
    columns: list[ColumnSpec], latents: torch.Tensor
) -> None:
    with pytest.raises(ValueError, match="latent_dim"):
        StateConditionalFlow(0, columns)
    with pytest.raises(ValueError, match="duplicate"):
        StateConditionalFlow(2, [ColumnSpec("a", 2), ColumnSpec("a", 3)])
    with pytest.raises(ValueError, match="n_blocks must be at least 2"):
        StateConditionalFlow(6, columns, n_blocks=1)

    state_model = StateConditionalFlow(6, columns, **_small_kwargs())
    with pytest.raises(ValueError, match="shape"):
        state_model.abduct(latents[:, :5], _states(len(latents)))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        state_model.abduct(latents, torch.tensor([[2, 0]]).expand(len(latents), -1))

    direct_model = DirectSemanticFlowIntervention(6, columns, **_small_kwargs())
    values, mask = _objective(len(latents))
    with pytest.raises(ValueError, match="point mass"):
        direct_model.log_prob(
            latents,
            latents,
            torch.zeros_like(values),
            torch.zeros_like(mask),
        )
    bad = latents.clone()
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        direct_model.sample(bad, values, mask)


def test_symbolic_sampling_respects_intervention_and_preserves_other_columns(
    columns: list[ColumnSpec],
) -> None:
    kernel = _DeterministicKernel(columns)
    source = _states(7)

    target = sample_symbolic_states(
        kernel,
        source,
        columns,
        {"a": 1},
        n_samples=4,
        generator=torch.Generator().manual_seed(8),
    )

    assert target.shape == (4, len(source), len(columns))
    assert torch.equal(target[..., 0], torch.ones_like(target[..., 0]))
    assert torch.equal(target[..., 1], source[None, :, 1].expand(4, -1))


@pytest.mark.parametrize(
    ("variant", "expected_type"),
    [
        ("state_flow", StateConditionalFlow),
        ("distilled_flow", DistilledFlowIntervention),
        ("direct_semantic_flow", DirectSemanticFlowIntervention),
    ],
)
def test_factory_selects_flow_variant_and_seed_controls_initialization(
    variant: str,
    expected_type: type[nn.Module],
    columns: list[ColumnSpec],
) -> None:
    first = make_latent_intervention(
        variant, latent_dim=6, columns=columns, seed=41, **_small_kwargs()
    )
    second = make_latent_intervention(
        variant, latent_dim=6, columns=columns, seed=41, **_small_kwargs()
    )

    assert isinstance(first, expected_type)
    assert type(first) is type(second)
    assert first.state_dict().keys() == second.state_dict().keys()
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])


@pytest.mark.parametrize(
    "variant", ["baseline", "pre_additive", "noise_token", "dist", "particles"]
)
def test_central_dispatch_preserves_legacy_training_and_inference(
    variant: str, columns: list[ColumnSpec], latents: torch.Tensor
) -> None:
    model = make_latent_intervention(
        variant,
        latent_dim=6,
        columns=columns,
        seed=17,
        d_model=8,
        nhead=1,
        dim_feedforward=12,
        dropout=0.0,
        noise_std=0.1,
        noise_dim=2,
        embed_dim=2,
        top_k=4,
        n_particles=3,
    )
    decoder = SemanticDecoder(6, columns, hidden_dim=8, n_hidden=1, dropout=0.0)

    class SparseKernel(_DeterministicKernel):
        def transition_matrix(self, delta: dict[str, int]) -> torch.Tensor:
            return super().transition_matrix(delta).to_sparse().coalesce()

    states = _states(len(latents))
    target_states = states.clone()
    target_states[:, 0] = 1
    s_prime = {
        column.name: target_states[:, index]
        for index, column in enumerate(columns)
    }
    history = train_latent_intervention_model(
        model,
        latents=latents,
        intervention={"a": 1},
        decoder=decoder,
        h_s=SparseKernel(columns),
        s_prime=s_prime,
        epochs=1,
        pretrain_epochs=1,
        joint_epochs=1,
        batch_size=len(latents),
        lr=1e-3,
        n_samples=2,
        seed=5,
        verbose=False,
    )
    samples = sample_counterfactual(
        model,
        latents[:3],
        {"a": 1},
        n_samples=2,
        generator=torch.Generator().manual_seed(9),
    )

    assert history
    assert all(
        torch.isfinite(torch.tensor(value))
        for epoch in history
        for value in epoch.values()
    )
    assert samples.shape == (2, 3, 6)
    assert torch.isfinite(samples).all()


def test_save_load_round_trip_preserves_outputs_metadata_and_device(
    tmp_path: Path,
    columns: list[ColumnSpec],
    latents: torch.Tensor,
) -> None:
    model = make_latent_intervention(
        "direct_semantic_flow",
        latent_dim=6,
        columns=columns,
        seed=4,
        **_small_kwargs(),
    )
    assert isinstance(model, DirectSemanticFlowIntervention)
    model.set_latent_statistics(latents)
    _make_nontrivial(model)
    checkpoint = tmp_path / "direct.pt"
    metadata = {"latent_artifact_sha256": "latent-hash", "dependency": {"seed": 4}}
    model.save(
        checkpoint,
        metadata=metadata,
        supported_interventions=[{"a": 1}, {}],
    )

    loaded = load_latent_intervention(
        checkpoint,
        device="cpu",
        expected_variant="direct_semantic_flow",
        expected_columns=columns,
        expected_metadata={"latent_artifact_sha256": "latent-hash"},
    )
    assert isinstance(loaded, DirectSemanticFlowIntervention)
    assert not loaded.training
    assert next(loaded.parameters()).device.type == "cpu"
    assert loaded.supported_interventions == [{"a": 1}, {}]
    values, mask = _objective(len(latents))
    noise = torch.randn(2, len(latents), 6, generator=torch.Generator().manual_seed(5))
    assert torch.equal(
        model.sample(latents, values, mask, 2, noise=noise),
        loaded.sample(latents, values, mask, 2, noise=noise),
    )

    with pytest.raises(ValueError, match="re-encode/retrain"):
        load_latent_intervention(
            checkpoint,
            expected_metadata={"latent_artifact_sha256": "different"},
        )


def test_uniform_inference_warns_for_unseen_intervention(
    columns: list[ColumnSpec], latents: torch.Tensor
) -> None:
    model = DistilledFlowIntervention(6, columns, **_small_kwargs())
    model.set_latent_statistics(latents)
    model.supported_interventions = [{"a": 1}, {}]

    with pytest.warns(RuntimeWarning, match="unsupported extrapolation"):
        result = counterfactual(
            model,
            latents[:3],
            {"b": 2},
            generator=torch.Generator().manual_seed(1),
        )

    assert result.shape == (3, 6)
    assert torch.isfinite(result).all()


def test_state_flow_uniform_inference_supports_kernel_and_explicit_target(
    columns: list[ColumnSpec], latents: torch.Tensor
) -> None:
    model = StateConditionalFlow(6, columns, **_small_kwargs())
    model.set_latent_statistics(latents)
    _make_nontrivial(model)
    source = _states(3)
    explicit_target = source.clone()
    explicit_target[:, 0] = 1
    kernel = _DeterministicKernel(columns)

    via_kernel = sample_counterfactual(
        model,
        latents[:3],
        {"a": 1},
        n_samples=2,
        source_states=source,
        h_s=kernel,
        generator=torch.Generator().manual_seed(2),
    )
    via_target = sample_counterfactual(
        model,
        latents[:3],
        {"a": 1},
        n_samples=2,
        source_states=source,
        target_states=explicit_target,
    )

    assert via_kernel.shape == (2, 3, 6)
    assert torch.allclose(via_kernel, via_target)


def test_state_training_is_deterministic_and_returns_finite_weighted_history(
    columns: list[ColumnSpec], latents: torch.Tensor
) -> None:
    first = make_latent_intervention(
        "state_flow", latent_dim=6, columns=columns, seed=13, **_small_kwargs()
    )
    second = make_latent_intervention(
        "state_flow", latent_dim=6, columns=columns, seed=13, **_small_kwargs()
    )
    assert isinstance(first, StateConditionalFlow)
    assert isinstance(second, StateConditionalFlow)
    kwargs = {
        "epochs": 2,
        "batch_size": 5,
        "lr": 1e-3,
        "seed": 29,
        "verbose": False,
    }

    first_history = train_state_conditional_flow(first, latents, _states(12), **kwargs)
    second_history = train_state_conditional_flow(
        second, latents, _states(12), **kwargs
    )

    assert first_history == pytest.approx(second_history)
    assert all(
        torch.isfinite(torch.tensor(value))
        for epoch in first_history
        for value in epoch.values()
    )
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])
    assert not first.training and not second.training


class _BatchSizeLikelihoodFlow(StateConditionalFlow):
    def log_prob(self, z: torch.Tensor, states) -> torch.Tensor:
        del states
        anchor = sum(parameter.sum() * 0.0 for parameter in self.parameters())
        return anchor.expand(len(z)) - z.new_tensor(float(len(z)))


def test_state_training_history_is_sample_weighted(
    columns: list[ColumnSpec], latents: torch.Tensor
) -> None:
    model = _BatchSizeLikelihoodFlow(6, columns, **_small_kwargs())

    history = train_state_conditional_flow(
        model,
        latents,
        _states(len(latents)),
        epochs=1,
        batch_size=5,
        seed=1,
        verbose=False,
    )

    # Batch losses are 5, 5, and 2; weighting by their sample counts gives 4.5.
    assert history[0]["nll"] == pytest.approx((5 * 5 + 5 * 5 + 2 * 2) / 12)


def test_distillation_freezes_teacher_and_student_has_standalone_inference(
    columns: list[ColumnSpec], latents: torch.Tensor
) -> None:
    teacher = StateConditionalFlow(6, columns, **_small_kwargs())
    teacher.set_latent_statistics(latents)
    _make_nontrivial(teacher)
    student = DistilledFlowIntervention(6, columns, **_small_kwargs())
    kernel = _DeterministicKernel(columns)

    history = train_distilled_flow(
        student,
        teacher,
        kernel,
        latents,
        _states(len(latents)),
        {"a": 1},
        epochs=1,
        batch_size=len(latents),
        n_samples=2,
        identity_fraction=0.0,
        seed=7,
        verbose=False,
    )
    result = counterfactual(
        student,
        latents[:2],
        {"a": 1},
        generator=torch.Generator().manual_seed(3),
    )

    assert len(history) == 1 and torch.isfinite(torch.tensor(history[0]["energy"]))
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert not student.training
    assert result.shape == (2, 6)


def test_direct_training_averages_probabilities_before_semantic_kl(
    columns: list[ColumnSpec], latents: torch.Tensor
) -> None:
    model = DirectSemanticFlowIntervention(6, columns, **_small_kwargs())
    decoder = _RecordingDecoder(6, columns)
    kernel = _DeterministicKernel(columns)

    history = train_direct_semantic_flow(
        model,
        decoder,
        kernel,
        latents,
        {"a": 1},
        epochs=1,
        batch_size=len(latents),
        n_samples=2,
        identity_fraction=0.0,
        seed=11,
        verbose=False,
    )

    assert len(decoder.calls) == 3  # factual target, followed by two flow samples
    transition = kernel.transition_matrix({"a": 1})
    # The trainer shuffles once with the dedicated seed+1 generator before this
    # single full-batch update; align its precomputed target to the observed calls.
    order = torch.randperm(len(latents), generator=torch.Generator().manual_seed(12))
    target = (decoder.calls[0].exp() @ transition)[order]
    composed = torch.logsumexp(torch.stack(decoder.calls[1:]), dim=0) - torch.log(
        torch.tensor(2.0)
    )
    expected_kl = F.kl_div(composed, target, reduction="batchmean")

    assert history[0]["semantic_kl"] == pytest.approx(expected_kl.item(), rel=1e-6)
    assert all(torch.isfinite(torch.tensor(value)) for value in history[0].values())
    assert all(not parameter.requires_grad for parameter in decoder.parameters())
    assert not model.training


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_load_places_flow_on_cuda(tmp_path: Path, columns: list[ColumnSpec]) -> None:
    model = StateConditionalFlow(6, columns, **_small_kwargs())
    checkpoint = tmp_path / "state.pt"
    model.save(checkpoint)

    loaded = load_latent_intervention(checkpoint, device="cuda")

    assert next(loaded.parameters()).device.type == "cuda"
    assert not loaded.training


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_load_places_legacy_intervention_on_cuda(
    tmp_path: Path, columns: list[ColumnSpec]
) -> None:
    model = LatentIntervention(
        6, columns, d_model=8, nhead=1, dim_feedforward=12, dropout=0.0
    )
    checkpoint = tmp_path / "legacy.pt"
    model.save(checkpoint)

    loaded = LatentIntervention.load(checkpoint, device="cuda")

    assert next(loaded.parameters()).device.type == "cuda"
    assert not loaded.training
