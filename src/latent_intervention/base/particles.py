"""Plan D (`particles`): deterministic particle set.

h_Z = sum_j w_j(z, delta) * dirac_{z + Delta_j(z, delta)}, j = 1..d, from d learned query
tokens. Exact composition + weighted per-particle entropy term.
"""

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from src.schema import ColumnSpec
from src.semantic_decoder.model import SemanticDecoder
from src.symbolic_intervention import SymbolicKernel

from src.latent_intervention.base.common import (
    _DoSpecTokens,
    _ManipulatorBase,
    _forward_kl,
    _penalties,
    _train,
    consistency_target,
)

__all__ = ["LatentInterventionParticles", "train_latent_intervention_particles"]


class LatentInterventionParticles(_ManipulatorBase):
    """h_Z(. | z, delta) = sum_j w_j(z, delta) dirac_{z + Delta_j(z, delta)}, j = 1..d.

    d learned query tokens attend to [latent token, do() tokens]; each query's output is
    one particle's shift (zero-init -> every particle starts at z) and, unless
    `uniform_weights`, one mixture logit (zero-init -> uniform weights at start). The
    particles carry no S labels; the random query embeddings break their symmetry.
    """

    def __init__(
        self,
        latent_dim: int,
        columns: list[ColumnSpec],
        n_particles: int = 16,
        uniform_weights: bool = False,
        d_model: int = 128,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        if n_particles < 1:
            raise ValueError(f"n_particles must be >= 1, got {n_particles}")
        super().__init__(
            latent_dim,
            columns,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            n_particles=n_particles,
            uniform_weights=uniform_weights,
        )
        self.n_particles = n_particles
        self.uniform_weights = uniform_weights
        self.vec_proj = nn.Linear(latent_dim, d_model)
        self.cond = _DoSpecTokens(self.columns, d_model)
        self.queries = nn.Parameter(0.02 * torch.randn(n_particles, d_model))
        self.layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.out = nn.Linear(d_model, latent_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)  # every particle at z at init
        self.weight_head = None if uniform_weights else nn.Linear(d_model, 1)
        if self.weight_head is not None:
            nn.init.zeros_(self.weight_head.weight)
            nn.init.zeros_(self.weight_head.bias)  # uniform weights at init

    def particles(
        self, z: torch.Tensor, values: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(n_particles, batch, latent_dim) shifts and (batch, n_particles) log-weights."""
        batch = z.shape[0]
        tokens = torch.cat(
            [
                self.vec_proj(z).unsqueeze(1),
                self.cond(values, mask),
                self.queries.unsqueeze(0).expand(batch, -1, -1),
            ],
            dim=1,
        )
        h = self.layer(tokens)[:, -self.n_particles :]  # (batch, d, d_model)
        shifts = self.out(h).transpose(0, 1)
        if self.weight_head is None:
            log_w = z.new_full((batch, self.n_particles), -math.log(self.n_particles))
        else:
            log_w = F.log_softmax(self.weight_head(h).squeeze(-1), dim=-1)
        return shifts, log_w

    def forward(self, z: torch.Tensor, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Inference: the highest-weight particle (deterministic)."""
        shifts, log_w = self.particles(z, values, mask)
        best = log_w.argmax(dim=-1)
        return z + shifts[best, torch.arange(z.shape[0], device=z.device)]

    def sample(
        self,
        z: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        n_samples: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """(n_samples, batch, latent_dim) draws: pick particles by weight."""
        shifts, log_w = self.particles(z, values, mask)
        j = torch.multinomial(log_w.exp(), n_samples, replacement=True, generator=generator)
        rows = torch.arange(z.shape[0], device=z.device)
        return z + shifts[j.t(), rows]

    def composed_log_joint(
        self,
        z: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        decoder: SemanticDecoder,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        """
        Exact log (g . h_Z)(. | z, delta), shape (batch, |S|); the (n_particles, batch,
        latent_dim) shifts; the weight-averaged per-particle decoder entropy (scalar); and
        collapse diagnostics (effective and distinct-state particle counts).
        """
        shifts, log_w = self.particles(z, values, mask)
        log_joints = torch.stack([decoder.log_joint(z + s) for s in shifts])  # (d, batch, |S|)
        composed = torch.logsumexp(log_w.t()[..., None] + log_joints, dim=0)
        w = log_w.exp()
        entropy = (w.t() * -(log_joints.exp() * log_joints).sum(dim=-1)).sum(dim=0).mean()
        with torch.no_grad():
            states = log_joints.argmax(dim=-1).sort(dim=0).values  # (d, batch)
            distinct = 1 + (states[1:] != states[:-1]).sum(dim=0)
            diag = {
                "eff_particles": (1.0 / w.pow(2).sum(dim=-1)).mean().item(),
                "distinct_states": distinct.float().mean().item(),
            }
        return composed, shifts, entropy, diag


def train_latent_intervention_particles(
    model: LatentInterventionParticles,
    decoder: SemanticDecoder,
    latents: torch.Tensor,
    intervention: dict[str, int],
    s_prime: dict[str, torch.Tensor] | None = None,  # unused; uniform call site
    *,
    h_s: SymbolicKernel | None = None,
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
    Train Plan D on the exact forward KL + weighted per-particle entropy (decoder + h_S frozen).

    Penalties apply to every particle, unweighted, so low-weight particles stay near z.
    Logged `eff_particles` (1 / sum_j w_j^2) and `distinct_states` (distinct argmax g over
    particles) flag collapse and dead particles.
    """
    if h_s is None:
        raise ValueError("particle training needs the symbolic kernel `h_s`")

    def setup(device: torch.device, _gen: torch.Generator, z: torch.Tensor) -> dict:
        return {"target": consistency_target(decoder, h_s, z, intervention).to(device)}

    def batch_loss(_phase, ctx, _gen, z, v, m, idx):
        composed, shifts, entropy, diag = model.composed_log_joint(z, v, m, decoder)
        kl = _forward_kl(composed, ctx["target"][idx])
        sparsity, proximity, penalty = _penalties(shifts, sparsity_weight, proximity_weight)
        total = kl + entropy_weight * entropy + penalty
        return total, {
            "total": total.item(),
            "kl": kl.item(),
            "entropy": entropy.item(),
            **diag,
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
