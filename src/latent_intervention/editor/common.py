"""Target-marginal inference and self-contained persistence; no auxiliary readers."""

from collections.abc import Mapping
import math

import torch
from torch import nn

from src.artifact_io import save_torch
from src.schema import unflatten_state_index
from src.latent_intervention.flow.common import _validate_columns, _state_tensor
from . import CHECKPOINT_FORMAT


def metadata_matches(actual, expected):
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and all(
            key in actual and metadata_matches(actual[key], value)
            for key, value in expected.items()
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


def probabilities(value, rows, states):
    if (
        value.shape != (rows, states)
        or not torch.isfinite(value).all()
        or (value < 0).any()
        or not torch.allclose(
            value.sum(-1),
            torch.ones(rows, device=value.device, dtype=value.dtype),
            atol=1e-5,
        )
    ):
        raise ValueError("expected normalized, finite, nonnegative state probabilities")
    return value.detach()


def target_marginal(decoder, kernel, z, action, columns):
    if list(decoder.columns) != list(columns) or list(kernel.columns) != list(columns):
        raise ValueError("editor, g and h_S schemas must match")
    if decoder.training:
        raise ValueError("semantic editor requires frozen g in eval mode")
    with torch.no_grad():
        matrix = kernel.transition_matrix(action).to(z)
        if matrix.is_sparse:
            matrix = matrix.to_dense()
        states = math.prod(c.n_categories for c in columns)
        matrix = probabilities(matrix, states, states)
        posterior = probabilities(decoder.log_joint(z).exp(), len(z), states)
        return probabilities(posterior @ matrix, len(z), states)


class TargetEditor(nn.Module):
    supported_interventions = [{"X": 3}, {}]

    def __init__(self, latent_dim, columns):
        super().__init__()
        if type(latent_dim) is not int or latent_dim < 1:
            raise ValueError("invalid latent dimension")
        self.latent_dim, self.columns = latent_dim, _validate_columns(columns)
        self.register_buffer("mean", torch.zeros(latent_dim))
        self.register_buffer("scale", torch.ones(latent_dim))
        self.register_buffer("fitted", torch.tensor(False))
        self.checkpoint_metadata = {}

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
                "invalid editor embeddings: shape, dtype, device or values"
            )

    @torch.no_grad()
    def fit_normalization(self, z):
        self.validate_z(z)
        if not len(z):
            raise ValueError("empty fitting embeddings")
        self.mean.copy_(z.mean(0))
        self.scale.copy_(z.std(0, unbiased=False).clamp_min(1e-4))
        self.fitted.fill_(True)

    def validate_buffers(self):
        if (
            not self.fitted
            or (self.scale <= 0).any()
            or any(not torch.isfinite(v).all() for v in self.state_dict().values())
        ):
            raise ValueError("invalid editor weights/normalization")

    def transport(self, z, targets, *, create_graph=False):
        self.validate_z(z)
        if not self.fitted:
            raise ValueError("editor normalization has not been fitted")
        targets = _state_tensor(
            targets, self.columns, batch_size=len(z), device=z.device
        )
        x = (z - self.mean) / self.scale
        result = self.mean + self.scale * self.transport_normalized(
            x, targets, create_graph=create_graph
        )
        if not torch.isfinite(result).all():
            raise FloatingPointError("nonfinite edited embeddings")
        return result

    def components(self, z, target, *, create_graph=False):
        self.validate_z(z)
        size = math.prod(c.n_categories for c in self.columns)
        target = probabilities(target.to(z), len(z), size)
        selected = (target > 0).any(0).nonzero().flatten()
        states = unflatten_state_index(selected, self.columns).to(z.device)
        count = len(states)
        locations = self.transport(
            z[None].expand(count, -1, -1).reshape(count * len(z), -1),
            states[:, None].expand(-1, len(z), -1).reshape(count * len(z), -1),
            create_graph=create_graph,
        ).reshape(count, len(z), self.latent_dim)
        return locations, target[:, selected].log()

    @torch.no_grad()
    def sample_symbolic(
        self, z, intervention, *, decoder=None, h_s=None, n_samples=1, generator=None
    ):
        self.validate_z(z)
        if type(n_samples) is not int or n_samples < 1:
            raise ValueError("n_samples must be positive")
        if (
            not isinstance(intervention, Mapping)
            or intervention not in self.supported_interventions
            or any(type(v) is not int for v in intervention.values())
        ):
            raise ValueError(
                "editors support only do(X=3) and the empty identity action"
            )
        if not intervention or not len(z):
            return z[None].expand(n_samples, -1, -1).clone()
        if decoder is None or h_s is None:
            raise ValueError("editor inference requires frozen g and h_S")
        target = target_marginal(decoder, h_s, z, intervention, self.columns)
        locations, log_weights = self.components(z, target)
        picks = torch.multinomial(
            log_weights.exp(), n_samples, replacement=True, generator=generator
        )
        return locations[picks.t(), torch.arange(len(z), device=z.device)]

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
