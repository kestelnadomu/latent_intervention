"""Shrinkage Gaussian OT in a common, full-dimensional factual PCA basis."""

import itertools
import math
import torch
import torch.nn.functional as F

from src.schema import flat_state_index
from .common import OTModel


def state_features(states, columns):
    main = [
        F.one_hot(states[:, i], c.n_categories)[:, 1:].double()
        for i, c in enumerate(columns)
    ]
    pairs = [
        (a[:, :, None] * b[:, None, :]).flatten(1)
        for a, b in itertools.combinations(main, 2)
    ]
    return torch.cat(
        [torch.ones(len(states), 1, device=states.device), *main, *pairs], -1
    )


class GaussianOT(OTModel):
    variant = "ot_gaussian"

    def __init__(
        self, latent_dim, columns, *, ridge=1.0, shrinkage=32.0, variance_floor=1e-4
    ):
        super().__init__(latent_dim, columns)
        if any(
            not math.isfinite(v) or v <= 0 for v in (ridge, shrinkage, variance_floor)
        ):
            raise ValueError("Gaussian regularization must be finite and positive")
        self.ridge, self.shrinkage, self.variance_floor = (
            ridge,
            shrinkage,
            variance_floor,
        )
        self.register_buffer("basis", torch.eye(latent_dim))
        self.register_buffer("state_mean", torch.zeros(len(self.states), latent_dim))
        self.register_buffer("state_variance", torch.ones(len(self.states), latent_dim))

    @torch.no_grad()
    def fit(self, z, states):
        self.fit_normalization(z, states)
        states = self.validate_states(states, len(z))
        x = ((z - self.mean) / self.scale).double()
        _, basis = torch.linalg.eigh(x.T @ x / len(x))
        basis = basis.flip(1)
        # Resolve eigenvector signs; save the actual basis, never recompute on load.
        signs = basis[
            basis.abs().argmax(0), torch.arange(self.latent_dim, device=z.device)
        ].sign()
        basis = basis * signs
        x = x @ basis
        design, all_design = (
            state_features(states, self.columns),
            state_features(self.states, self.columns),
        )
        penalty = (
            torch.eye(design.shape[1], dtype=x.dtype, device=x.device) * self.ridge
        )
        penalty[0, 0] = (
            0  # unpenalized intercept; categorical contrasts are regularized
        )
        coefficients = torch.linalg.solve(design.T @ design + penalty, design.T @ x)
        prior_mean = all_design @ coefficients
        pooled_variance = (
            (x - design @ coefficients).square().mean(0).clamp_min(self.variance_floor)
        )
        index = flat_state_index(states, self.columns)
        count = self.fit_counts.double()[:, None]
        total, squares = torch.zeros_like(prior_mean), torch.zeros_like(prior_mean)
        total.index_add_(0, index, x)
        squares.index_add_(0, index, x.square())
        empirical_mean = total / count.clamp_min(1)
        empirical_variance = (
            squares / count.clamp_min(1) - empirical_mean.square()
        ).clamp_min(0)
        weight = count / (count + self.shrinkage)
        mean = weight * empirical_mean + (1 - weight) * prior_mean
        # Mixture second moments, including uncertainty between empirical/prior means.
        variance = weight * (empirical_variance + (empirical_mean - mean).square())
        variance += (1 - weight) * (pooled_variance + (prior_mean - mean).square())
        self.basis.copy_(basis)
        self.state_mean.copy_(mean)
        self.state_variance.copy_(variance.clamp_min(self.variance_floor))
        self.validate_buffers()
        return self

    def transport_normalized(self, x, source, target):
        s, t = (
            flat_state_index(source, self.columns),
            flat_state_index(target, self.columns),
        )
        coordinates = x @ self.basis
        moved = self.state_mean[t] + (
            self.state_variance[t] / self.state_variance[s]
        ).sqrt() * (coordinates - self.state_mean[s])
        return moved @ self.basis.T

    @torch.no_grad()
    def nll(self, z, states):
        index = flat_state_index(states, self.columns)
        x = ((z - self.mean) / self.scale) @ self.basis
        variance = self.state_variance[index]
        return float(
            (
                0.5
                * (
                    math.log(2 * math.pi)
                    + variance.log()
                    + (x - self.state_mean[index]).square() / variance
                )
            ).mean()
            + self.scale.log().mean()
        )

    def validate_buffers(self):
        super().validate_buffers()
        if (self.state_variance <= 0).any() or not torch.allclose(
            self.basis.T @ self.basis,
            torch.eye(
                self.latent_dim, device=self.basis.device, dtype=self.basis.dtype
            ),
            atol=2e-5,
            rtol=1e-5,
        ):
            raise ValueError("invalid Gaussian covariance/basis")

    def constructor(self):
        return dict(
            super().constructor(),
            ridge=self.ridge,
            shrinkage=self.shrinkage,
            variance_floor=self.variance_floor,
        )
