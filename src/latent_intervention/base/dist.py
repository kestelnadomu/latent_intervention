"""Plan C (`dist`): discrete mixture over counterfactual states.

h_Z = sum_{s' in top-k} w_theta(s' | z, delta) * dirac_{z + Delta_phi(z, s')}. w_theta is an
autoregressive head distilled from h_S . g; Delta_phi is a do()-agnostic realiser.
Exact composition; trained pretrain-then-joint.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from src.schema import ColumnSpec, unflatten_state_index
from src.semantic_decoder.model import SemanticDecoderModel
from src.symbolic_intervention import SymbolicKernel

from src.latent_intervention.base.common import (
    _DeltaNet,
    _DoSpecTokens,
    _ManipulatorBase,
    _forward_kl,
    _penalties,
    _train,
    consistency_target,
)

__all__ = ["LatentInterventionDist", "train_latent_intervention_dist"]


def _autoreg_log_joint(
    context: torch.Tensor,
    columns: list[ColumnSpec],
    prefix_embed: nn.ModuleDict,
    heads: nn.ModuleDict,
) -> torch.Tensor:
    """Dense (batch, |S|) log-probabilities of an autoregressive product over `columns`.

    Each head sees `context` plus embeddings of the already-decoded prefix; states are
    enumerated in mixed-radix schema order, with the last column varying fastest.
    """
    batch = context.shape[0]
    acc = context.new_zeros(batch, 1)
    prefix = context.new_zeros(batch, 1, 0)
    for col in columns:
        n_states = acc.shape[1]
        ctx = context[:, None, :].expand(batch, n_states, -1).reshape(batch * n_states, -1)
        logits = heads[col.name](torch.cat([ctx, prefix.reshape(batch * n_states, -1)], dim=-1))
        lp = F.log_softmax(logits, dim=-1).reshape(batch, n_states, col.n_categories)
        acc = (acc[..., None] + lp).reshape(batch, n_states * col.n_categories)
        emb = prefix_embed[col.name](torch.arange(col.n_categories, device=context.device))
        prefix = torch.cat(
            [
                prefix[:, :, None, :].expand(batch, n_states, col.n_categories, -1),
                emb[None, None].expand(batch, n_states, col.n_categories, -1),
            ],
            dim=-1,
        ).reshape(batch, n_states * col.n_categories, -1)
    return acc


class _AutoregWeights(nn.Module):
    """w_theta(s' | z, delta): autoregressive over S, conditioned on z and the do() spec."""

    def __init__(
        self,
        latent_dim: int,
        columns: list[ColumnSpec],
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.columns = list(columns)
        self.z_proj = nn.Linear(latent_dim, d_model)
        self.cond = _DoSpecTokens(self.columns, d_model)
        self.layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.prefix_embed = nn.ModuleDict(
            {c.name: nn.Embedding(c.n_categories, embed_dim) for c in self.columns}
        )
        self.heads = nn.ModuleDict(
            {
                c.name: nn.Linear(d_model + i * embed_dim, c.n_categories)
                for i, c in enumerate(self.columns)
            }
        )

    def context(self, z: torch.Tensor, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tokens = torch.cat([self.z_proj(z).unsqueeze(1), self.cond(values, mask)], dim=1)
        return self.layer(tokens)[:, 0]

    def log_joint(self, z: torch.Tensor, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Dense (batch, |S|) log-weights."""
        return _autoreg_log_joint(
            self.context(z, values, mask), self.columns, self.prefix_embed, self.heads
        )

    def top_k(
        self, z: torch.Tensor, values: torch.Tensor, mask: torch.Tensor, k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(batch, k) retained state indices and their (renormalised) log-weights."""
        log_joint = self.log_joint(z, values, mask)
        logw, idx = log_joint.topk(min(k, log_joint.shape[-1]), dim=-1)
        return idx, logw - torch.logsumexp(logw, dim=-1, keepdim=True)


class LatentInterventionDist(_ManipulatorBase):
    """h_Z(. | z, delta) = sum_s' w_theta(s' | z, delta) dirac_{z + Delta_phi(z, s')}."""

    def __init__(
        self,
        latent_dim: int,
        columns: list[ColumnSpec],
        d_model: int = 128,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        embed_dim: int = 16,
        top_k: int = 16,
    ) -> None:
        super().__init__(
            latent_dim,
            columns,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            embed_dim=embed_dim,
            top_k=top_k,
        )
        self.top_k = top_k
        self.w_theta = _AutoregWeights(
            latent_dim, self.columns, d_model, nhead, dim_feedforward, dropout, embed_dim
        )
        # Realiser Delta_phi conditions on a full state s' (every column masked in).
        self.realiser = _DeltaNet(
            latent_dim, self.columns, d_model, nhead, dim_feedforward, dropout
        )

    def realise(self, z: torch.Tensor, s_prime: torch.Tensor) -> torch.Tensor:
        """z' = z + Delta_phi(z, s'); `s_prime` is (batch, n_cols) category indices."""
        mask = torch.ones_like(s_prime, dtype=torch.bool)
        return z + self.realiser(z, s_prime, mask)

    def forward(self, z: torch.Tensor, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Inference: realise the single most likely counterfactual state (argmax w_theta)."""
        idx, _ = self.w_theta.top_k(z, values, mask, 1)
        return self.realise(z, unflatten_state_index(idx[:, 0], self.columns))

    def sample(
        self,
        z: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        n_samples: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """(n_samples, batch, latent_dim) draws: s' ~ renormalised top-k w_theta, then realise."""
        idx, logw = self.w_theta.top_k(z, values, mask, self.top_k)  # (batch, k)
        pick = torch.multinomial(logw.exp(), n_samples, replacement=True, generator=generator)
        states = idx.gather(1, pick)  # (batch, n_samples)
        return torch.stack(
            [self.realise(z, unflatten_state_index(states[:, m], self.columns)) for m in range(n_samples)]
        )

    def composed_log_joint(
        self,
        z: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        decoder: SemanticDecoderModel,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Exact log (g . h_Z)(. | z, delta) over the retained top-k s', shape (batch, |S|),
        plus the per-component (k, batch, latent_dim) latent shifts for penalties.
        """
        idx, logw = self.w_theta.top_k(z, values, mask, self.top_k)  # (batch, k)
        components, shifts = [], []
        for j in range(idx.shape[1]):
            z_j = self.realise(z, unflatten_state_index(idx[:, j], self.columns))
            components.append(logw[:, j : j + 1] + decoder.log_joint(z_j))
            shifts.append(z_j - z)
        composed = torch.logsumexp(torch.stack(components), dim=0)
        return composed, torch.stack(shifts)


def train_latent_intervention_dist(
    model: LatentInterventionDist,
    decoder: SemanticDecoderModel,
    latents: torch.Tensor,
    intervention: dict[str, int],
    s_prime: dict[str, torch.Tensor] | None = None,
    *,
    h_s: SymbolicKernel | None = None,
    pretrain_epochs: int = 30,
    joint_epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-4,
    proximity_weight: float = 1.0,
    sparsity_weight: float = 1.0,
    realiser_l1: float = 1.0,
    realiser_l2: float = 1.0,
    seed: int | None = None,
    device: str | torch.device | None = None,
    verbose: bool = True,
    epoch_callback: Callable | None = None,
    grad_clip: float | None = None,
) -> list[dict[str, float]]:
    """
    Plan C, pretrain-then-joint (decoder + h_S frozen):

    - **Pretrain** splits the bilevel problem into two supervised ones:
      w_theta distils the dense target (h_S . g)(. | z, delta); the realiser Delta_phi
      minimises -log g(s' | z + Delta_phi(z, s')) + l1||Delta||_1 + l2||Delta||_2^2 on the
      true counterfactual states `s_prime`.
    - **Joint** fine-tunes both on the exact forward-KL consistency objective.

    Returns per-epoch mean loss components; the "phase" key is 0 for pretrain, 1 for joint.
    """
    if h_s is None or s_prime is None:
        raise ValueError("dist training needs both `h_s` and `s_prime`")

    def setup(device: torch.device, _gen: torch.Generator, z: torch.Tensor) -> dict:
        s_cols = {k: v.to(device) for k, v in s_prime.items()}
        return {
            "s_cols": s_cols,
            "s_idx": torch.stack([s_cols[c.name] for c in model.columns], dim=1),  # (n, n_cols)
            "target": consistency_target(decoder, h_s, z, intervention).to(device),
        }

    def batch_loss(phase, ctx, _gen, z, v, m, idx):
        if phase == "pretrain":
            w_kl = _forward_kl(model.w_theta.log_joint(z, v, m), ctx["target"][idx])
            z_r = model.realise(z, ctx["s_idx"][idx])
            realise_ce = decoder.nll(z_r, {k: val[idx] for k, val in ctx["s_cols"].items()})
            _, _, realise_reg = _penalties(z_r - z, realiser_l1, realiser_l2)
            total = w_kl + realise_ce + realise_reg
            return total, {
                "phase": 0.0,
                "total": total.item(),
                "w_kl": w_kl.item(),
                "realise_ce": realise_ce.item(),
            }
        composed, shifts = model.composed_log_joint(z, v, m, decoder)
        kl = _forward_kl(composed, ctx["target"][idx])
        sparsity, proximity, penalty = _penalties(shifts, sparsity_weight, proximity_weight)
        total = kl + penalty
        return total, {
            "phase": 1.0,
            "total": total.item(),
            "kl": kl.item(),
            "sparsity": sparsity.item(),
        }

    return _train(
        model,
        decoder,
        latents,
        intervention,
        phases=[("pretrain", pretrain_epochs), ("joint", joint_epochs)],
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
