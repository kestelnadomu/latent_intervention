"""Plan A (`pre_additive`, engression): z' = z + Delta_theta(z + eps, delta).

eps ~ N(0, sigma^2 I); Monte-Carlo composition.
"""

from collections.abc import Callable

import torch

from src.schema import ColumnSpec
from src.semantic_decoder.model import SemanticDecoderModel
from src.symbolic_intervention import SymbolicKernel

from src.latent_intervention.base.common import (
    _DeltaNet,
    _ManipulatorBase,
    _forward_kl,
    _mc_mixture,
    _penalties,
    _train,
    consistency_target,
)

__all__ = ["LatentInterventionPreAdditive", "train_latent_intervention_preadditive"]


class LatentInterventionPreAdditive(_ManipulatorBase):
    """z' = z + Delta_theta(z + eps, delta), eps ~ N(0, noise_std^2 I)."""

    def __init__(
        self,
        latent_dim: int,
        columns: list[ColumnSpec],
        noise_std: float = 1.0,
        d_model: int = 128,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            latent_dim,
            columns,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            noise_std=noise_std,
        )
        self.noise_std = noise_std
        self.delta = _DeltaNet(latent_dim, self.columns, d_model, nhead, dim_feedforward, dropout)

    def forward(
        self,
        z: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """One sample z' ~ h_Z(. | z, delta)."""
        eps = self.noise_std * torch.randn(z.shape, device=z.device, generator=generator)
        return z + self.delta(z + eps, values, mask)

    def sample(
        self,
        z: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        n_samples: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """(n_samples, batch, latent_dim) draws from h_Z."""
        return torch.stack(
            [self.forward(z, values, mask, generator) for _ in range(n_samples)]
        )

    def composed_log_joint(
        self,
        z: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        decoder: SemanticDecoderModel,
        n_samples: int,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Monte-Carlo estimate of log (g . h_Z)(. | z, delta), shape (batch, |S|),
        plus the sampled latent shifts (n_samples, batch, latent_dim) for penalties.
        """
        zs = self.sample(z, values, mask, n_samples, generator)
        composed, _ = _mc_mixture(zs, decoder)
        return composed, zs - z


def train_latent_intervention_preadditive(
    model: LatentInterventionPreAdditive,
    decoder: SemanticDecoderModel,
    latents: torch.Tensor,
    intervention: dict[str, int],
    s_prime: dict[str, torch.Tensor] | None = None,  # unused; uniform call site
    *,
    h_s: SymbolicKernel | None = None,
    n_samples: int = 8,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-4,
    proximity_weight: float = 1.0,
    sparsity_weight: float = 1.0,
    seed: int | None = None,
    device: str | torch.device | None = None,
    verbose: bool = True,
    epoch_callback: Callable | None = None,
    grad_clip: float | None = None,
) -> list[dict[str, float]]:
    """Train Plan A on the forward-KL consistency objective (decoder + h_S frozen)."""
    if h_s is None:
        raise ValueError("pre-additive training needs the symbolic kernel `h_s`")

    def setup(device: torch.device, _gen: torch.Generator, z: torch.Tensor) -> dict:
        return {"target": consistency_target(decoder, h_s, z, intervention).to(device)}

    def batch_loss(_phase, ctx, generator, z, v, m, idx):
        composed, shifts = model.composed_log_joint(z, v, m, decoder, n_samples, generator)
        kl = _forward_kl(composed, ctx["target"][idx])
        sparsity, proximity, penalty = _penalties(shifts, sparsity_weight, proximity_weight)
        total = kl + penalty
        return total, {
            "total": total.item(),
            "kl": kl.item(),
            "sparsity": sparsity.item(),
            "proximity": proximity.item(),
        }

    return _train(
        model,
        decoder,
        latents,
        intervention,
        phases=[("train", epochs)],
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        device=device,
        verbose=verbose,
        setup=setup,
        batch_loss=batch_loss,
        epoch_callback=epoch_callback,
        grad_clip=grad_clip,
    )
