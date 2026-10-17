"""Conditional OT flow matching and deterministic shared-reference counterfactuals."""

import math
import torch
from torch import nn

from .common import OTModel, StateEmbedding
from .transport import integrate


class ReferenceOTFlow(OTModel):
    variant = "ot_flow"

    def __init__(
        self,
        latent_dim,
        columns,
        *,
        hidden_dim,
        state_embed_dim=16,
        ode_steps=64,
        ode_max_steps=256,
        ode_tolerance=1e-4,
    ):
        super().__init__(latent_dim, columns)
        if (
            any(
                type(v) is not int or v < 1
                for v in (hidden_dim, state_embed_dim, ode_steps, ode_max_steps)
            )
            or ode_max_steps < 2 * ode_steps
            or not math.isfinite(ode_tolerance)
            or ode_tolerance <= 0
        ):
            raise ValueError("invalid OT flow architecture/solver")
        self.hidden_dim, self.state_embed_dim = hidden_dim, state_embed_dim
        self.ode_steps, self.ode_max_steps, self.ode_tolerance = (
            ode_steps,
            ode_max_steps,
            ode_tolerance,
        )
        self.embedding = StateEmbedding(columns, state_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + len(columns) * state_embed_dim + 16, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def velocity(self, x, time, states):
        time = (
            torch.as_tensor(time, dtype=x.dtype, device=x.device)
            .expand(len(x))
            .reshape(-1, 1)
        )
        angle = (
            time * torch.arange(1, 9, dtype=x.dtype, device=x.device)[None] * math.pi
        )
        return self.net(
            torch.cat([x, self.embedding(states), angle.sin(), angle.cos()], -1)
        )

    def integrate(self, x, states, *, reverse=False):
        return integrate(
            self.velocity,
            x,
            states,
            reverse=reverse,
            steps=self.ode_steps,
            max_steps=self.ode_max_steps,
            tolerance=self.ode_tolerance,
        )

    def transport_normalized(self, x, source, target):
        reference, backward = self.integrate(x, source, reverse=True)
        result, forward = self.integrate(reference, target)
        self.last_diagnostics = dict(
            nfe=backward["nfe"] + forward["nfe"], backward=backward, forward=forward
        )
        return result

    def constructor(self):
        return dict(
            super().constructor(),
            hidden_dim=self.hidden_dim,
            state_embed_dim=self.state_embed_dim,
            ode_steps=self.ode_steps,
            ode_max_steps=self.ode_max_steps,
            ode_tolerance=self.ode_tolerance,
        )
