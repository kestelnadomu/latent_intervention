"""Plan B (`noise_token`, outsourced noise): z' = z + Delta_theta(z, eps, delta).

eps ~ N(0, I_r) enters as its own token, so z reaches Delta intact. Monte-Carlo
composition + per-sample entropy term.
"""

from collections.abc import Callable

import torch

from src.schema import ColumnSpec
from src.semantic_decoder.model import SemanticDecoder
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

__all__ = ["LatentInterventionNoiseToken", "train_latent_intervention_noise_token"]


class LatentInterventionNoiseToken(_ManipulatorBase):
    """z' = z + Delta_theta(z, eps, delta), eps ~ N(0, I_k) fed as its own token."""

    def __init__(
        self,
        latent_dim: int,
        columns: list[ColumnSpec],
        noise_dim: int = 4,
        d_model: int = 128,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        if noise_dim < 1:
            raise ValueError(f"noise_dim must be >= 1, got {noise_dim}")
        super().__init__(
            latent_dim,
            columns,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            noise_dim=noise_dim,
        )
        self.noise_dim = noise_dim
        self.delta = _DeltaNet(
            latent_dim, self.columns, d_model, nhead, dim_feedforward, dropout, noise_dim
        )

    def forward(
        self,
        z: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """One sample z' ~ h_Z(. | z, delta)."""
        eps = torch.randn(z.shape[0], self.noise_dim, device=z.device, generator=generator)
        return z + self.delta(z, values, mask, eps)

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
        decoder: SemanticDecoder,
        n_samples: int,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Monte-Carlo log (g . h_Z)(. | z, delta), shape (batch, |S|), the sampled latent
        shifts (n_samples, batch, latent_dim), and the mean per-sample decoder entropy
        E_eps[H(g(. | z'_eps))] (scalar).
        """
        zs = self.sample(z, values, mask, n_samples, generator)
        composed, log_joints = _mc_mixture(zs, decoder)
        entropy = -(log_joints.exp() * log_joints).sum(dim=-1).mean()
        return composed, zs - z, entropy


def train_latent_intervention_noise_token(
    model: LatentInterventionNoiseToken,
    decoder: SemanticDecoder,
    latents: torch.Tensor,
    intervention: dict[str, int],
    s_prime: dict[str, torch.Tensor] | None = None,  # unused; uniform call site
    *,
    h_s: SymbolicKernel | None = None,
    n_samples: int = 8,
    entropy_weight: float = 0.1,
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
    """
    Train Plan B on forward KL + a per-sample entropy penalty (decoder + h_S frozen).

    The KL alone cannot tell "ignore eps, park z' where g is ambiguous" (Plan 0's
    solution) from "use eps to scatter z' over points where g is confident": both give
    the same g . h_Z. `entropy_weight` * E_eps[H(g(. | z'_eps))] prefers the latter, so
    the mixture has to come from eps. The logged `spread` (mean norm of the per-dimension
    std of z' across eps) is the diagnostic for eps being ignored (spread ~ 0).
    """
    if h_s is None:
        raise ValueError("noise-token training needs the symbolic kernel `h_s`")

    def setup(device: torch.device, _gen: torch.Generator, z: torch.Tensor) -> dict:
        return {"target": consistency_target(decoder, h_s, z, intervention).to(device)}

    def batch_loss(_phase, ctx, generator, z, v, m, idx):
        composed, shifts, entropy = model.composed_log_joint(
            z, v, m, decoder, n_samples, generator
        )
        kl = _forward_kl(composed, ctx["target"][idx])
        sparsity, proximity, penalty = _penalties(shifts, sparsity_weight, proximity_weight)
        total = kl + entropy_weight * entropy + penalty
        return total, {
            "total": total.item(),
            "kl": kl.item(),
            "entropy": entropy.item(),
            "spread": shifts.detach().std(dim=0).norm(dim=-1).mean().item(),
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
