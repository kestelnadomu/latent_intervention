"""``direct_semantic_flow``: the deployable direct flow fitted against g and h_S.

Trained directly against the frozen semantic decoder and the symbolic transition kernel.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

import torch
import torch.nn.functional as F

from src.semantic_decoder.model import SemanticDecoderModel
from src.symbolic_intervention import SymbolicKernel

from src.latent_intervention.flow.common import (
    _finish_epoch,
    _intervention_support,
    _make_generator,
    _objective_tensors,
    _positive_int,
    _unit_interval,
    _validate_training_inputs,
    _weighted_log_update,
)
from src.latent_intervention.flow.direct import (
    _DirectFlowIntervention,
)

__all__ = [
    "DirectSemanticFlowIntervention",
    "train_direct_semantic_flow",
]


class DirectSemanticFlowIntervention(_DirectFlowIntervention):
    """Standalone direct flow trained only through frozen ``g`` and ``h_S``."""

    variant = "direct_semantic_flow"


@torch.no_grad()
def _semantic_target(
    decoder: SemanticDecoderModel,
    h_s: SymbolicKernel,
    latents: torch.Tensor,
    intervention: Mapping[str, int],
    *,
    chunk_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = h_s.transition_matrix(dict(intervention)).to(latents.device)
    transposed = matrix.t().coalesce() if matrix.is_sparse else matrix.t()
    factual_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    for start in range(0, latents.shape[0], chunk_size):
        factual = decoder.log_joint(latents[start : start + chunk_size]).exp()
        target = (
            torch.sparse.mm(transposed, factual.t()).t()
            if transposed.is_sparse
            else factual @ matrix
        )
        factual_parts.append(factual)
        target_parts.append(target)
    return torch.cat(target_parts), torch.cat(factual_parts)


def train_direct_semantic_flow(
    model: DirectSemanticFlowIntervention,
    decoder: SemanticDecoderModel,
    h_s: SymbolicKernel,
    latents: torch.Tensor,
    intervention: Mapping[str, int],
    *,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    n_samples: int = 8,
    identity_fraction: float = 0.2,
    entropy_weight: float = 0.1,
    proximity_weight: float = 0.1,
    support_weight: float = 0.05,
    identity_weight: float = 1.0,
    support_bank_size: int = 512,
    grad_clip: float = 5.0,
    seed: int | None = None,
    device: str | torch.device | None = None,
    verbose: bool = True,
    epoch_callback: Callable | None = None,
) -> list[dict[str, float]]:
    """Fit the direct flow by semantic consistency, confidence, and support."""
    if not isinstance(model, DirectSemanticFlowIntervention):
        raise TypeError("model must be a DirectSemanticFlowIntervention")
    if list(decoder.columns) != model.columns or list(h_s.columns) != model.columns:
        raise ValueError("decoder, symbolic kernel, and flow schemas must match")
    _validate_training_inputs(
        model,
        latents,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
    )
    if n_samples < 2 or n_samples % 2:
        raise ValueError("n_samples must be an even integer >= 2 for antithetic sampling")
    identity_fraction = _unit_interval("identity_fraction", identity_fraction)
    for name, weight in (
        ("entropy_weight", entropy_weight),
        ("proximity_weight", proximity_weight),
        ("support_weight", support_weight),
        ("identity_weight", identity_weight),
    ):
        if not isinstance(weight, (int, float)) or not math.isfinite(float(weight)) or weight < 0:
            raise ValueError(f"{name} must be a finite non-negative number")
    _positive_int("support_bank_size", support_bank_size)

    model.set_latent_statistics(latents)
    run_device = torch.device(device or "cpu")
    model.to(run_device).train()
    decoder.to(run_device).eval().requires_grad_(False)
    z = latents.to(device=run_device, dtype=model.latent_mean.dtype)
    with torch.no_grad():
        target, factual = _semantic_target(decoder, h_s, z, intervention)
    values, mask = _objective_tensors(intervention, model.columns, 1, run_device)
    shuffle = _make_generator(run_device, seed, offset=1)
    flow_rng = _make_generator(run_device, seed, offset=2)
    identity_rng = _make_generator(run_device, seed, offset=3)
    bank_rng = _make_generator(run_device, seed, offset=4)
    bank_count = min(support_bank_size, len(z))
    bank_index = torch.randperm(len(z), generator=bank_rng, device=run_device)[:bank_count]
    support_bank = ((z[bank_index] - model.latent_mean) / model.latent_scale).detach()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        totals: dict[str, float] = {}
        order = torch.randperm(z.shape[0], generator=shuffle, device=run_device)
        for start in range(0, z.shape[0], batch_size):
            index = order[start : start + batch_size]
            batch_z = z[index]
            batch_values = values.expand(len(index), -1).clone()
            batch_mask = mask.expand(len(index), -1).clone()
            identity = (
                torch.rand(len(index), generator=identity_rng, device=run_device)
                < identity_fraction
            )
            batch_mask[identity] = False
            batch_target = target[index].clone()
            batch_target[identity] = factual[index][identity]

            half_noise = model.base_std * torch.randn(
                n_samples // 2,
                len(index),
                model.latent_dim,
                device=run_device,
                dtype=z.dtype,
                generator=flow_rng,
            )
            noise = torch.cat((half_noise, -half_noise), dim=0)
            samples, log_det = model._sample_with_log_det(
                batch_z,
                batch_values,
                batch_mask,
                n_samples,
                noise=noise,
                enforce_identity=False,
            )
            log_joints = torch.stack([decoder.log_joint(sample) for sample in samples])
            composed = torch.logsumexp(log_joints, dim=0) - math.log(n_samples)
            semantic_kl = F.kl_div(composed, batch_target, reduction="batchmean")
            entropy = -(log_joints.exp() * log_joints).sum(dim=-1).mean()
            scaled_shift = (samples - batch_z.unsqueeze(0)) / model.latent_scale
            proximity = F.smooth_l1_loss(scaled_shift, torch.zeros_like(scaled_shift))
            standardized_samples = (
                (samples - model.latent_mean) / model.latent_scale
            ).reshape(n_samples * len(index), model.latent_dim)
            support = (
                torch.cdist(standardized_samples, support_bank).min(dim=-1).values
                / math.sqrt(model.latent_dim)
            ).mean()
            identity_loss = (
                F.smooth_l1_loss(
                    scaled_shift[:, identity], torch.zeros_like(scaled_shift[:, identity])
                )
                if identity.any()
                else samples.sum() * 0.0
            )
            total = (
                semantic_kl
                + entropy_weight * entropy
                + proximity_weight * proximity
                + support_weight * support
                + identity_weight * identity_loss
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip, error_if_nonfinite=True
            )
            optimizer.step()
            spread = scaled_shift.std(dim=0, unbiased=False).norm(dim=-1).mean()
            _weighted_log_update(
                totals,
                {
                    "total": total.item(),
                    "semantic_kl": semantic_kl.item(),
                    "entropy": entropy.item(),
                    "proximity": proximity.item(),
                    "support": support.item(),
                    "identity": identity_loss.item(),
                    "spread": spread.item(),
                    "log_det": log_det.mean().item(),
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
