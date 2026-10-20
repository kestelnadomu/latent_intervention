"""Shared inference contract; no training data, oracle targets or guide dependency."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn

from src.artifact_io import save_torch
from src.schema import flat_state_index, unflatten_state_index
from src.latent_intervention.flow.common import _validate_columns, _state_tensor
from . import CHECKPOINT_FORMAT


def metadata_matches(actual, expected):
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and all(
            k in actual and metadata_matches(actual[k], v) for k, v in expected.items()
        )
    if isinstance(expected, (list, tuple)):
        return (
            isinstance(actual, (list, tuple))
            and len(actual) == len(expected)
            and all(
                metadata_matches(a, b) for a, b in zip(actual, expected, strict=True)
            )
        )
    return actual == expected


def coupled_weights(posterior, matrix):
    """Return every positive kernel edge and its *joint* probability (including zeros)."""
    posterior, matrix = posterior.detach(), matrix.detach()
    size = matrix.shape[0] if matrix.ndim == 2 else -1
    if (
        size < 1
        or matrix.shape != (size, size)
        or posterior.ndim != 2
        or posterior.shape[1] != size
        or not torch.isfinite(matrix).all()
        or not torch.isfinite(posterior).all()
        or (matrix < 0).any()
        or (posterior < 0).any()
        or not torch.allclose(matrix.sum(-1), torch.ones_like(matrix[:, 0]), atol=1e-5)
        or not torch.allclose(
            posterior.sum(-1), torch.ones_like(posterior[:, 0]), atol=1e-5
        )
    ):
        raise ValueError("invalid coupled state probabilities")
    source, target = (matrix > 0).nonzero(as_tuple=True)
    weights = posterior[:, source] * matrix[source, target][None]
    return source, target, weights


class OTModel(nn.Module):
    """Deterministic transport per state pair; uncertainty comes from coupled states."""

    supported_interventions = [{"X": 3}, {}]

    def __init__(self, latent_dim, columns):
        super().__init__()
        self.columns = _validate_columns(columns)
        if type(latent_dim) is not int or latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        self.latent_dim = latent_dim
        self.register_buffer("mean", torch.zeros(latent_dim))
        self.register_buffer("scale", torch.ones(latent_dim))
        states = unflatten_state_index(
            torch.arange(math.prod(c.n_categories for c in self.columns)), self.columns
        )
        self.register_buffer("states", states)
        self.register_buffer("fit_counts", torch.zeros(len(states), dtype=torch.long))
        self.register_buffer("fitted", torch.tensor(False))
        self.checkpoint_metadata = {}
        self.last_diagnostics = {}

    @torch.no_grad()
    def fit_normalization(self, z, states):
        self.validate_z(z)
        states = self.validate_states(states, len(z))
        if not len(z):
            raise ValueError("empty fitting data")
        self.mean.copy_(z.mean(0))
        self.scale.copy_(z.std(0, unbiased=False).clamp_min(1e-4))
        self.fit_counts.copy_(
            torch.bincount(
                flat_state_index(states, self.columns), minlength=len(self.states)
            )
        )
        self.fitted.fill_(True)

    def validate_z(self, z):
        if (
            not isinstance(z, torch.Tensor)
            or z.ndim != 2
            or z.shape[1] != self.latent_dim
            or z.dtype != self.mean.dtype
            or z.device != self.mean.device
            or not torch.isfinite(z).all()
        ):
            raise ValueError(
                "invalid OT latents: shape, dtype, device or nonfinite values"
            )

    def validate_states(self, states, batch):
        return _state_tensor(
            states, self.columns, batch_size=batch, device=self.mean.device
        )

    def validate_buffers(self):
        if any(not torch.isfinite(v).all() for v in self.state_dict().values()):
            raise ValueError("nonfinite OT checkpoint")
        expected = unflatten_state_index(
            torch.arange(len(self.states), device=self.states.device), self.columns
        )
        if (
            (self.scale <= 0).any()
            or (self.fit_counts < 0).any()
            or not self.fitted
            or not torch.equal(self.states, expected)
        ):
            raise ValueError("invalid OT normalization, support or fitted state")

    def transport(self, z, source, target):
        self.validate_z(z)
        self.last_diagnostics = {}
        if not self.fitted:
            raise ValueError("OT model has not been fitted")
        source, target = (
            self.validate_states(source, len(z)),
            self.validate_states(target, len(z)),
        )
        changed = (source != target).any(-1)
        result = z.clone()
        if changed.any():
            x = (z[changed] - self.mean) / self.scale
            result[changed] = self.mean + self.scale * self.transport_normalized(
                x, source[changed], target[changed]
            )
        self.last_diagnostics["transported_pairs"] = int(changed.sum())
        self.last_diagnostics["velocity_point_evaluations"] = self.last_diagnostics.get(
            "nfe", 0
        ) * int(changed.sum())
        if not torch.isfinite(result).all():
            raise FloatingPointError("nonfinite OT transport")
        return result

    @torch.no_grad()
    def components(self, z, posterior, matrix):
        self.validate_z(z)
        self.last_diagnostics = {}
        if len(posterior) != len(z) or matrix.shape != (
            len(self.states),
            len(self.states),
        ):
            raise ValueError("coupled probabilities differ from OT schema/batch")
        src, dst, weights = coupled_weights(posterior.to(z), matrix.to(z))
        locations = torch.stack(
            [
                self.transport(
                    z,
                    self.states[s].expand(len(z), -1),
                    self.states[t].expand(len(z), -1),
                )
                for s, t in zip(src, dst)
            ]
        )
        return locations, weights.log()

    @torch.no_grad()
    def sample_symbolic(
        self, z, intervention, *, decoder=None, h_s=None, n_samples=1, generator=None
    ):
        self.validate_z(z)
        self.last_diagnostics = {}
        if type(n_samples) is not int or n_samples < 1:
            raise ValueError("n_samples must be positive")
        if (
            not isinstance(intervention, Mapping)
            or intervention not in self.supported_interventions
            or any(type(v) is not int for v in intervention.values())
        ):
            raise ValueError("OT supports only do(X=3) and the empty identity action")
        if not intervention or not len(z):
            return z[None].expand(n_samples, -1, -1).clone()
        if decoder is None or h_s is None or list(decoder.columns) != self.columns:
            raise ValueError(
                "OT latent-only inference requires matching frozen g and h_S"
            )
        if decoder.training:
            raise ValueError("OT inference requires g in eval mode")
        matrix = h_s.transition_matrix(intervention).to_dense().to(z)
        src, dst, weights = coupled_weights(decoder.log_joint(z).exp(), matrix)
        picks = torch.multinomial(
            weights, n_samples, replacement=True, generator=generator
        ).t()
        # State-pair transport is deterministic. Reuse repeated draws for a unit,
        # especially for expensive ODE solves; preserve every draw and its order.
        keys = torch.arange(len(z), device=z.device)[None] * len(src) + picks
        unique, inverse = torch.unique(keys.flatten(), sorted=True, return_inverse=True)
        unit, edge = unique // len(src), unique % len(src)
        result = self.transport(z[unit], self.states[src[edge]], self.states[dst[edge]])
        self.last_diagnostics.update(
            sampled_pairs=keys.numel(), unique_pairs=len(unique)
        )
        return result[inverse].reshape(n_samples, len(z), self.latent_dim)

    def constructor(self):
        return dict(
            latent_dim=self.latent_dim,
            columns=[(c.name, c.n_categories) for c in self.columns],
        )

    def save(self, path, *, metadata=None):
        self.validate_buffers()
        save_torch(
            path,
            dict(
                format=CHECKPOINT_FORMAT,
                variant=self.variant,
                config=self.constructor(),
                state_dict=self.state_dict(),
                supported_interventions=self.supported_interventions,
                metadata=self.checkpoint_metadata if metadata is None else metadata,
            ),
        )


class StateEmbedding(nn.Module):
    def __init__(self, columns, width=16):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Embedding(c.n_categories, width) for c in columns]
        )

    def forward(self, states):
        return torch.cat(
            [layer(states[:, i]) for i, layer in enumerate(self.layers)], -1
        )
