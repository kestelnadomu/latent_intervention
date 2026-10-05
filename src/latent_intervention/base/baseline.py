"""Plan 0 (`baseline`): deterministic z' = z + Delta_theta(z, delta).

Trained by per-column CE against the simulated S'; learns the per-column marginals of
the counterfactual and cannot represent its multimodality.
"""

from collections.abc import Callable

import torch

from src.schema import ColumnSpec
from src.semantic_decoder.model import SemanticDecoderModel
from src.symbolic_intervention import SymbolicKernel

from src.latent_intervention.base.common import (
    _DeltaNet,
    _ManipulatorBase,
    _penalties,
    _train,
)

__all__ = ["LatentIntervention", "train_latent_intervention"]


class LatentIntervention(_ManipulatorBase):
    """Single-layer transformer mapping (latent, do() spec) to an intervened latent."""

    def __init__(
        self,
        latent_dim: int,
        columns: list[ColumnSpec],
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
        )
        self.delta = _DeltaNet(latent_dim, self.columns, d_model, nhead, dim_feedforward, dropout)

    def forward(self, z: torch.Tensor, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply the intervention.

        Args:
            z: (batch, latent_dim) latent representations.
            values: (batch, n_columns) long tensor of do() category indices
                (entries where mask is False are ignored).
            mask: (batch, n_columns) bool tensor marking intervened columns.

        Returns:
            (batch, latent_dim) intervened latent representations.
        """
        return z + self.delta(z, values, mask)


def train_latent_intervention(
    model: LatentIntervention,
    decoder: SemanticDecoderModel,
    latents: torch.Tensor,
    intervention: dict[str, int],
    s_prime: dict[str, torch.Tensor] | None = None,
    *,
    h_s: SymbolicKernel | None = None,  # unused; accepted for a uniform call site
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
    Train the baseline manipulator on counterfactual pairs; the decoder stays frozen.

    `latents` are the factual latent representations (n, latent_dim); `s_prime` maps each
    S column to its (n,) counterfactual values (see targets_from_dataframe on the
    counterfactual sim CSV); `intervention` is the do() spec those counterfactuals were
    generated under, e.g. {"G": 1}. The consistency term is the decoder's mean
    cross-entropy over *all* S columns: descendants of the intervened node must move to
    their S' values while non-descendants keep their factual ones.
    """
    if s_prime is None:
        raise ValueError("baseline training needs the counterfactual targets `s_prime`")

    def setup(device: torch.device, _gen: torch.Generator, _z: torch.Tensor) -> dict:
        return {"s": {k: v.to(device) for k, v in s_prime.items()}}

    def batch_loss(_phase, ctx, _gen, z, v, m, idx):
        z_prime = model(z, v, m)
        consistency = decoder.nll(z_prime, {k: val[idx] for k, val in ctx["s"].items()})
        sparsity, proximity, penalty = _penalties(z_prime - z, sparsity_weight, proximity_weight)
        total = consistency + penalty
        return total, {
            "total": total.item(),
            "consistency": consistency.item(),
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
