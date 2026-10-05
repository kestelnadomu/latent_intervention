"""``distilled_flow``: the deployable direct flow fitted to a frozen ``state_flow`` teacher.

Trained on samples from the teacher with an energy-distance objective.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch

from src.symbolic_intervention import SymbolicKernel

from src.latent_intervention.flow.common import (
    StateLike,
    _finish_epoch,
    _intervention_support,
    _make_generator,
    _objective_tensors,
    _positive_int,
    _state_tensor,
    _unit_interval,
    _validate_training_inputs,
    _weighted_log_update,
    multivariate_energy_distance,
    sample_symbolic_states,
)
from src.latent_intervention.flow.direct import (
    _DirectFlowIntervention,
)
from src.latent_intervention.flow.state_conditional import (
    StateConditionalFlow,
)

__all__ = [
    "DistilledFlowIntervention",
    "train_distilled_flow",
]


class DistilledFlowIntervention(_DirectFlowIntervention):
    """Standalone direct flow trained from a state-flow teacher."""

    variant = "distilled_flow"


def train_distilled_flow(
    model: DistilledFlowIntervention,
    teacher: StateConditionalFlow,
    h_s: SymbolicKernel,
    latents: torch.Tensor,
    states: StateLike,
    intervention: Mapping[str, int],
    *,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    n_samples: int = 8,
    identity_fraction: float = 0.2,
    grad_clip: float = 5.0,
    seed: int | None = None,
    device: str | torch.device | None = None,
    verbose: bool = True,
    epoch_callback: Callable | None = None,
) -> list[dict[str, float]]:
    """Distil a state-flow teacher with scale-normalized energy distance."""
    if not isinstance(model, DistilledFlowIntervention):
        raise TypeError("model must be a DistilledFlowIntervention")
    if not isinstance(teacher, StateConditionalFlow):
        raise TypeError("teacher must be a StateConditionalFlow")
    if teacher.latent_dim != model.latent_dim or teacher.columns != model.columns:
        raise ValueError("teacher and student flow schemas must match")
    _validate_training_inputs(
        model,
        latents,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
    )
    _positive_int("n_samples", n_samples)
    identity_fraction = _unit_interval("identity_fraction", identity_fraction)
    source = _state_tensor(states, model.columns, batch_size=latents.shape[0])
    model.set_latent_statistics(latents)
    run_device = torch.device(device or "cpu")
    model.to(run_device).train()
    teacher.to(run_device).eval().requires_grad_(False)
    z = latents.to(device=run_device, dtype=model.latent_mean.dtype)
    source = source.to(run_device)
    values, mask = _objective_tensors(
        intervention, model.columns, 1, run_device
    )
    shuffle = _make_generator(run_device, seed, offset=1)
    symbolic_rng = _make_generator(run_device, seed, offset=2)
    flow_rng = _make_generator(run_device, seed, offset=3)
    identity_rng = _make_generator(run_device, seed, offset=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        totals: dict[str, float] = {}
        order = torch.randperm(z.shape[0], generator=shuffle, device=run_device)
        for start in range(0, z.shape[0], batch_size):
            index = order[start : start + batch_size]
            batch_z = z[index]
            batch_source = source[index]
            batch_values = values.expand(len(index), -1).clone()
            batch_mask = mask.expand(len(index), -1).clone()
            identity = (
                torch.rand(len(index), generator=identity_rng, device=run_device)
                < identity_fraction
            )
            batch_mask[identity] = False

            with torch.no_grad():
                target_states = sample_symbolic_states(
                    h_s,
                    batch_source,
                    model.columns,
                    intervention,
                    n_samples,
                    generator=symbolic_rng,
                )
                target_states[:, identity] = batch_source[identity]
                u = teacher.abduct(batch_z, batch_source)
                repeated_u = u.unsqueeze(0).expand(n_samples, -1, -1).reshape(
                    n_samples * len(index), model.latent_dim
                )
                teacher_samples = teacher.render(
                    repeated_u,
                    target_states.reshape(n_samples * len(index), -1),
                ).reshape(n_samples, len(index), model.latent_dim)

            student_samples, _ = model._sample_with_log_det(
                batch_z,
                batch_values,
                batch_mask,
                n_samples,
                generator=flow_rng,
                enforce_identity=False,
            )
            energy = multivariate_energy_distance(
                student_samples, teacher_samples, model.latent_scale
            )
            optimizer.zero_grad(set_to_none=True)
            energy.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip, error_if_nonfinite=True
            )
            optimizer.step()
            spread = (
                (student_samples / model.latent_scale)
                .std(dim=0, unbiased=False)
                .norm(dim=-1)
                .mean()
            )
            _weighted_log_update(
                totals,
                {
                    "total": energy.item(),
                    "energy": energy.item(),
                    "spread": spread.item(),
                    "identity_fraction": identity.float().mean().item(),
                    "grad_norm": float(grad_norm),
                },
                len(index),
            )
        history.append(_finish_epoch(totals, len(z), epoch, epochs, verbose))
        if epoch_callback is not None and epoch_callback(
            model, optimizer, "train", history[-1]
        ):
            break

    model.supported_interventions = _intervention_support(intervention)
    model.eval()
    return history
