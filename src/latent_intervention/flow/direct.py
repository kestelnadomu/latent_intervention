"""Deployable conditional flow ``q(Z' | Z, delta)`` shared by the two direct variants.

``distilled_flow`` and ``direct_semantic_flow`` use the same architecture and need only ``Z``
and ``delta`` after training; they differ only in their training objective.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from src.schema import ColumnSpec

from src.latent_intervention.flow.common import (
    _ConditionalRealNVP,
    _FlowModelBase,
    _model_device,
    _positive_float,
    _positive_int,
    _validate_latents,
    _validate_objective,
)

__all__ = [
]


class _DirectContext(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        columns: Sequence[ColumnSpec],
        state_embed_dim: int,
        condition_dim: int,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(column.n_categories + 1, state_embed_dim) for column in columns]
        )
        self.null_values = [column.n_categories for column in columns]
        self.z_project = nn.Linear(latent_dim, condition_dim)
        self.do_project = nn.Linear(len(columns) * state_embed_dim, condition_dim)
        self.combine = nn.Sequential(
            nn.Linear(2 * condition_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )

    def forward(
        self, standardized_z: torch.Tensor, values: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        embedded = []
        for index, embedding in enumerate(self.embeddings):
            category = torch.where(
                mask[:, index],
                values[:, index],
                torch.full_like(values[:, index], self.null_values[index]),
            )
            embedded.append(embedding(category))
        do_context = self.do_project(torch.cat(embedded, dim=-1))
        return self.combine(torch.cat((self.z_project(standardized_z), do_context), dim=-1))


class _DirectFlowIntervention(_FlowModelBase):
    """Conditional flow over the standardized residual ``(Z' - Z) / scale``."""

    def __init__(
        self,
        latent_dim: int,
        columns: Sequence[ColumnSpec],
        *,
        n_blocks: int = 8,
        hidden_dim: int = 128,
        condition_dim: int = 64,
        state_embed_dim: int = 16,
        permutation_seed: int = 0,
        scale_floor: float = 1e-6,
        base_std: float = 0.05,
    ) -> None:
        self.base_std = _positive_float("base_std", base_std)
        super().__init__(
            latent_dim,
            columns,
            n_blocks=n_blocks,
            hidden_dim=hidden_dim,
            condition_dim=condition_dim,
            state_embed_dim=state_embed_dim,
            permutation_seed=permutation_seed,
            scale_floor=scale_floor,
            base_std=self.base_std,
        )
        self.context = _DirectContext(
            latent_dim, self.columns, state_embed_dim, condition_dim
        )
        self.flow = _ConditionalRealNVP(
            latent_dim, condition_dim, hidden_dim, n_blocks, permutation_seed
        )

    def _condition(
        self, z: torch.Tensor, values: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = _model_device(self)
        z = z.to(device=device, dtype=self.latent_mean.dtype)
        values, mask = _validate_objective(
            values, mask, self.columns, z.shape[0], device
        )
        standardized_z = (z - self.latent_mean) / self.latent_scale
        return self.context(standardized_z, values, mask), values, mask

    def _sample_with_log_det(
        self,
        z: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        n_samples: int,
        *,
        generator: torch.Generator | None = None,
        noise: torch.Tensor | None = None,
        enforce_identity: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_latents(z, self.latent_dim)
        _positive_int("n_samples", n_samples)
        device = _model_device(self)
        z = z.to(device=device, dtype=self.latent_mean.dtype)
        condition, values, mask = self._condition(z, values, mask)
        if noise is None:
            noise = self.base_std * torch.randn(
                n_samples,
                z.shape[0],
                self.latent_dim,
                device=device,
                dtype=z.dtype,
                generator=generator,
            )
        else:
            if noise.ndim == 2:
                noise = noise.unsqueeze(0)
            expected = (n_samples, z.shape[0], self.latent_dim)
            if tuple(noise.shape) != expected:
                raise ValueError(f"noise must have shape {expected}, got {tuple(noise.shape)}")
            if not noise.is_floating_point() or not torch.isfinite(noise).all():
                raise ValueError("noise must be finite and floating point")
            noise = noise.to(device=device, dtype=z.dtype)

        repeated_condition = condition.unsqueeze(0).expand(n_samples, -1, -1).reshape(
            n_samples * z.shape[0], -1
        )
        residual, log_det = self.flow.forward_transform(
            noise.reshape(n_samples * z.shape[0], self.latent_dim), repeated_condition
        )
        residual = residual.reshape(n_samples, z.shape[0], self.latent_dim)
        samples = z.unsqueeze(0) + self.latent_scale * residual
        if enforce_identity:
            no_op = ~mask.any(dim=-1)
            samples = torch.where(no_op[None, :, None], z[None], samples)
        return samples, log_det.reshape(n_samples, z.shape[0])

    def sample(
        self,
        z: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        n_samples: int = 1,
        generator: torch.Generator | None = None,
        *,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Draw counterfactual latents, hard-routing empty interventions to ``Z``."""
        return self._sample_with_log_det(
            z,
            values,
            mask,
            n_samples,
            generator=generator,
            noise=noise,
            enforce_identity=True,
        )[0]

    def forward(
        self,
        z: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Return one draw with shape ``(batch, latent_dim)``."""
        return self.sample(z, values, mask, 1, generator)[0]

    def log_prob(
        self,
        z_prime: torch.Tensor,
        z: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``log q(Z' | Z, delta)`` for non-empty interventions."""
        _validate_latents(z, self.latent_dim)
        _validate_latents(z_prime, self.latent_dim)
        if z_prime.shape[0] != z.shape[0]:
            raise ValueError("z and z_prime must have equal batch sizes")
        device = _model_device(self)
        z = z.to(device=device, dtype=self.latent_mean.dtype)
        z_prime = z_prime.to(device=device, dtype=self.latent_mean.dtype)
        condition, _, mask = self._condition(z, values, mask)
        if (~mask.any(dim=-1)).any():
            raise ValueError("the exact no-op route is a point mass and has no flow density")
        residual = (z_prime - z) / self.latent_scale
        base, inverse_log_det = self.flow.inverse_transform(residual, condition)
        scaled = base / self.base_std
        base_log_prob = -0.5 * (
            scaled.pow(2) + math.log(2.0 * math.pi) + 2.0 * math.log(self.base_std)
        ).sum(dim=-1)
        return base_log_prob + inverse_log_det - self.latent_scale.log().sum()
