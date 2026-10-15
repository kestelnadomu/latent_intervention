"""Projected factual state-effect editor with fixed, coupled symbolic weights."""

from __future__ import annotations

import json
import math

import torch
from torch import nn

from src.artifact_io import save_torch
from src.schema import ColumnSpec, unflatten_state_index
from .probes import AttributeProbe, pack_probe, unpack_probe

VARIANT = "c_sym_project"
CHECKPOINT_FORMAT = "csym_project_v1"


class LinearPreservationGuard(nn.Module):
    """Remove residual components that alter centered linear-probe logits.

    Work in the probe's standardized coordinates. An SVD keeps the independent
    score constraints (K-1 per categorical head). The frozen orthogonal projection
    is differentiable w.r.t. the editor and preserves probe probabilities exactly
    up to floating-point error. It does NOT enforce unit norm or semantic validity.
    """

    def __init__(self, guide: AttributeProbe):
        super().__init__()
        if guide.hidden:
            raise ValueError("exact preservation projection requires a linear guide")
        if (
            not torch.isfinite(guide.scale).all()
            or (guide.scale <= 0).any()
            or not torch.isfinite(guide.temperature).all()
            or (guide.temperature <= 0).any()
        ):
            raise ValueError(
                "probe scales and temperatures must be finite and positive"
            )
        weights = (
            torch.cat(
                [
                    (head.weight - head.weight.mean(0, keepdim=True))
                    / guide.temperature[i]
                    for i, head in enumerate(guide.heads)
                ]
            )
            .detach()
            .double()
        )
        _, singular, vh = torch.linalg.svd(weights, full_matrices=False)
        tolerance = singular.max() * max(weights.shape) * torch.finfo(torch.float32).eps
        basis = vh[singular > tolerance].t().float()
        self.register_buffer("basis", basis)
        self.register_buffer("scale", guide.scale.detach().clone())

    @property
    def rank(self):
        return self.basis.shape[1]

    def forward(self, z, samples):
        residual = (samples - z.unsqueeze(0)) / self.scale
        residual = residual - (residual @ self.basis) @ self.basis.t()
        return z.unsqueeze(0) + self.scale * residual


class FactualStateEffect(nn.Module):
    def __init__(self, latent_dim, columns, hidden_dim=128, state_embed_dim=16):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(c.n_categories, state_embed_dim) for c in columns]
        )
        width = len(columns) * state_embed_dim
        self.net = nn.Sequential(
            nn.Linear(width, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def features(self, states):
        return torch.cat([e(states[..., i]) for i, e in enumerate(self.embeddings)], -1)

    def forward(self, states):
        return self.net(self.features(states))


class ProjectedCSym(nn.Module):
    """z' = z + Project[m(s') - m(s)], with no unguarded public prediction path.

    Fixed mixture probabilities are g(s|z) h_S(s'|s,a). The guard preserves
    its own predictions, not necessarily true attributes or causal identity.
    """

    variant = VARIANT

    def __init__(
        self,
        latent_dim,
        columns,
        *,
        guide,
        hidden_dim=128,
        state_embed_dim=16,
        intervention=None,
    ):
        super().__init__()
        self.latent_dim, self.columns = latent_dim, list(columns)
        if (
            type(latent_dim) is not int
            or latent_dim < 1
            or not self.columns
            or hidden_dim < 1
            or state_embed_dim < 1
            or len({c.name for c in self.columns}) != len(self.columns)
            or any(c.n_categories < 2 for c in self.columns)
        ):
            raise ValueError("invalid C-sym dimensions/schema")
        if (
            not isinstance(guide, AttributeProbe)
            or guide.hidden
            or guide.latent_dim != latent_dim
        ):
            raise ValueError(
                "C-sym requires a matching frozen linear preservation guide"
            )
        self.hidden_dim, self.state_embed_dim = hidden_dim, state_embed_dim
        self.intervention = dict({"X": 3} if intervention is None else intervention)
        if self.intervention != {"X": 3}:
            raise ValueError("only the trained do(X=3) protocol is supported")
        if not any(c.name == "X" and c.n_categories > 3 for c in self.columns):
            raise ValueError("do(X=3) requires a compatible X column")
        # Preserve the original initialization order: only the state-effect net is new.
        self.effect = FactualStateEffect(
            latent_dim, self.columns, hidden_dim, state_embed_dim
        )
        self.register_buffer("mean", torch.zeros(latent_dim))
        self.register_buffer("scale", torch.ones(latent_dim))
        self.register_buffer(
            "states",
            unflatten_state_index(
                torch.arange(math.prod(c.n_categories for c in self.columns)),
                self.columns,
            ),
        )
        self.guide = guide.eval().requires_grad_(False)
        self.guard = LinearPreservationGuard(guide)
        self.checkpoint_metadata = {}

    def train(self, mode=True):
        super().train(mode)
        self.guide.eval()
        return self

    def _validate_z(self, z):
        if (
            z.ndim != 2
            or not len(z)
            or z.shape[1] != self.latent_dim
            or z.dtype != self.mean.dtype
            or z.device != self.mean.device
            or not torch.isfinite(z).all()
        ):
            raise ValueError(
                "expected finite, nonempty latents matching model dimension/device/dtype"
            )

    @torch.no_grad()
    def set_latent_statistics(self, z):
        self._validate_z(z)
        if len(z) < 2:
            raise ValueError("at least two factual fit embeddings are required")
        self.mean.copy_(z.mean(0))
        self.scale.copy_(z.std(0, unbiased=False).clamp_min(1e-4))

    def factual_loss(self, z, states):
        return (self.effect(states) - (z - self.mean) / self.scale).square().mean()

    def realize(self, z, source, destination):
        self._validate_z(z)
        change = self.effect(destination) - self.effect(source)
        return self.guard(z, (z + self.scale * change)[None])[0]

    def components(self, z, source_probs, transition):
        """Return projected locations (edges,batch,dim), fixed log weights (batch,edges)."""
        self._validate_z(z)
        count = len(self.states)
        source_probs, transition = (
            source_probs.detach().to(z),
            transition.detach().to(z),
        )
        if source_probs.shape != (len(z), count) or transition.shape != (count, count):
            raise ValueError("coupled symbolic shapes do not match schema")
        for value in (source_probs, transition):
            if (
                not torch.isfinite(value).all()
                or (value < 0).any()
                or not torch.allclose(
                    value.sum(-1), torch.ones_like(value[:, 0]), atol=1e-5
                )
            ):
                raise ValueError(
                    "coupled probabilities must be finite, nonnegative and normalized"
                )
        src, dst = (transition > 0).nonzero().unbind(1)
        probabilities = source_probs[:, src] * transition[src, dst][None]
        log_weights = probabilities.clamp_min(torch.finfo(z.dtype).tiny).log()
        log_weights = log_weights.masked_fill(probabilities == 0, -torch.inf)
        effects = self.effect(self.states)
        shift = (effects[dst] - effects[src])[:, None].expand(-1, len(z), -1)
        return self.guard(z, z[None] + self.scale * shift), log_weights

    @torch.no_grad()
    def sample_symbolic(
        self, z, intervention, *, decoder, h_s, n_samples=64, generator=None
    ):
        self._validate_z(z)
        if type(n_samples) is not int or n_samples < 1:
            raise ValueError("n_samples must be a positive integer")
        if intervention and dict(intervention) != self.intervention:
            raise ValueError("intervention outside training support")
        if not intervention:
            return z[None].expand(n_samples, -1, -1).clone()
        if decoder is None or h_s is None:
            raise ValueError("C-sym inference requires frozen decoder and h_s")
        if list(decoder.columns) != self.columns or list(h_s.columns) != self.columns:
            raise ValueError("editor, g and h_S schemas must match")
        locations, weights = self.components(
            z,
            decoder.log_joint(z).exp(),
            h_s.transition_matrix(intervention).to_dense(),
        )
        picks = torch.multinomial(
            weights.exp(), n_samples, replacement=True, generator=generator
        )
        return locations[picks.t(), torch.arange(len(z), device=z.device)]

    def save(self, path, *, metadata=None):
        metadata = self.checkpoint_metadata if metadata is None else metadata
        save_torch(
            path,
            dict(
                format=CHECKPOINT_FORMAT,
                variant=self.variant,
                config=dict(
                    latent_dim=self.latent_dim,
                    columns=[(c.name, c.n_categories) for c in self.columns],
                    hidden_dim=self.hidden_dim,
                    state_embed_dim=self.state_embed_dim,
                    intervention=self.intervention,
                ),
                guide=pack_probe(self.guide),
                state_dict=self.state_dict(),
                metadata=metadata,
            ),
        )

    @classmethod
    def load(cls, path, *, device=None, expected_columns=None, expected_metadata=None):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            payload.get("format") != CHECKPOINT_FORMAT
            or payload.get("variant") != VARIANT
        ):
            raise ValueError("not a projected C-sym checkpoint")
        config = dict(payload["config"])
        config["columns"] = [ColumnSpec(*c) for c in config["columns"]]
        if expected_columns is not None and config["columns"] != list(expected_columns):
            raise ValueError("C-sym checkpoint schema mismatch")
        metadata = payload["metadata"]
        if not isinstance(metadata, dict) or (
            expected_metadata is not None
            and any(
                json.dumps(metadata.get(k), sort_keys=True)
                != json.dumps(v, sort_keys=True)
                for k, v in expected_metadata.items()
            )
        ):
            raise ValueError("C-sym checkpoint provenance mismatch")
        model = cls(**config, guide=unpack_probe(payload["guide"]))
        model.load_state_dict(payload["state_dict"], strict=True)
        if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
            raise ValueError("nonfinite C-sym checkpoint")
        if (model.scale <= 0).any() or (model.guard.scale <= 0).any():
            raise ValueError("invalid C-sym normalization")
        model.checkpoint_metadata = metadata
        return model.to(device or "cpu").eval()
