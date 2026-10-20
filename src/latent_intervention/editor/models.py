"""Two target-conditioned maps with the same semantic objective and inference contract."""

import math

import torch
from torch import nn
import torch.nn.functional as F

from src.latent_intervention.base.common import _DeltaNet
from .common import TargetEditor


class SemanticEditor(TargetEditor):
    variant = "semantic_editor"

    def __init__(
        self, latent_dim, columns, *, d_model=128, nhead=4, dim_feedforward=256
    ):
        super().__init__(latent_dim, columns)
        self.d_model, self.nhead, self.dim_feedforward = d_model, nhead, dim_feedforward
        self.delta = _DeltaNet(
            latent_dim, self.columns, d_model, nhead, dim_feedforward, 0.0
        )

    def transport_normalized(self, x, targets, *, create_graph=False):
        return x + self.delta(x, targets, torch.ones_like(targets, dtype=torch.bool))

    def constructor(self):
        return dict(
            super().constructor(),
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
        )


class TargetPotential(nn.Module):
    """Convex in x for fixed target, including a learnable quadratic (not fixed I)."""

    def __init__(
        self, latent_dim, columns, hidden_dim, state_embed_dim=16, quadratic_floor=1e-3
    ):
        super().__init__()
        if hidden_dim < 1 or state_embed_dim < 1 or not 0 < quadratic_floor < 1:
            raise ValueError("invalid convex-potential settings")
        self.embeddings = nn.ModuleList(
            [nn.Embedding(c.n_categories, state_embed_dim) for c in columns]
        )
        context = len(columns) * state_embed_dim
        self.input1 = nn.Linear(latent_dim + context, hidden_dim)
        self.input2 = nn.Linear(latent_dim + context, hidden_dim)
        initial = math.log(math.expm1(1 / hidden_dim))
        self.positive_hidden = nn.Parameter(
            torch.full((hidden_dim, hidden_dim), initial)
        )
        self.positive_output = nn.Parameter(torch.full((hidden_dim,), initial))
        self.linear = nn.Parameter(torch.zeros(latent_dim))
        self.quadratic = nn.Parameter(
            torch.full((latent_dim,), math.log(math.expm1(1 - quadratic_floor)))
        )
        self.quadratic_floor = quadratic_floor

    def forward(self, x, targets):
        context = torch.cat(
            [layer(targets[:, i]) for i, layer in enumerate(self.embeddings)], -1
        )
        inputs = torch.cat([x, context], -1)
        hidden = F.softplus(self.input1(inputs))
        hidden = F.softplus(
            F.linear(hidden, F.softplus(self.positive_hidden)) + self.input2(inputs)
        )
        return (
            (hidden * F.softplus(self.positive_output)).sum(-1)
            + x @ self.linear
            + 0.5
            * (x.square() * (F.softplus(self.quadratic) + self.quadratic_floor)).sum(-1)
        )

    def gradient(self, x, targets, *, create_graph=False):
        with torch.inference_mode(False), torch.enable_grad():
            point = x.detach().clone().requires_grad_(True)
            (result,) = torch.autograd.grad(
                self(point, targets.clone()).sum(), point, create_graph=create_graph
            )
        return result if create_graph else result.detach()


class OTStructuredEditor(TargetEditor):
    variant = "ot_editor"

    def __init__(
        self,
        latent_dim,
        columns,
        *,
        hidden_dim,
        state_embed_dim=16,
        quadratic_floor=1e-3,
    ):
        super().__init__(latent_dim, columns)
        self.hidden_dim, self.state_embed_dim = hidden_dim, state_embed_dim
        self.quadratic_floor = quadratic_floor
        self.potential = TargetPotential(
            latent_dim, self.columns, hidden_dim, state_embed_dim, quadratic_floor
        )

    def transport_normalized(self, x, targets, *, create_graph=False):
        return self.potential.gradient(x, targets, create_graph=create_graph)

    def constructor(self):
        return dict(
            super().constructor(),
            hidden_dim=self.hidden_dim,
            state_embed_dim=self.state_embed_dim,
            quadratic_floor=self.quadratic_floor,
        )
