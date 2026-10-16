"""rho: write a target state s' into a latent z while keeping what s does not describe.

Every renderer is fitted on factual ``(z, s)`` only and called as
``renderer(z, s, s_prime) -> z'``. The subclasses are the options compared in
docs/experiments/renderer_benchmark.md:

=======================  =========================================================  ==========
class                    z'                                                         uses
=======================  =========================================================  ==========
``IdentityRenderer``     z (lower reference)                                        --
``ConstantRenderer``     z + E[z | c = s'_c] - E[z]: one direction per target level  s'_c
``GroupOTRenderer``      Gaussian OT map p(z | c = s_c) -> p(z | c = s'_c) (N2)     s_c, s'_c
``StateEffectRenderer``  z + m(s') - m(s), m additive in the levels (C-sym); with   s, s'
                         ``guard=<linear Reader>`` the edit is projected so the
                         guard's probabilities stay unchanged (projected C-sym)
``FlowRenderer``         u = F^-1(z; s), z' = F(u; s'), F a conditional RealNVP     s, s'
                         (F1)
=======================  =========================================================  ==========

``c`` is the ``on`` column (default ``"X"``). All return ``z`` unchanged where
``s' = s``, except ``ConstantRenderer``, which ignores the factual state.

A new renderer subclasses ``Renderer`` and implements ``_fit`` and ``_render``.
"""

from __future__ import annotations

import copy

import pandas as pd
import torch
from torch import nn


class Renderer:
    """Base class: ``SomeRenderer(columns).fit(z, states)``, then ``renderer(z, s, s_prime)``.

    ``columns`` are the state columns the renderer conditions on. Levels are 0..k-1,
    with k taken from the training states unless ``levels`` gives it.
    """

    def __init__(self, columns: list[str], levels: dict[str, int] | None = None) -> None:
        self.columns = list(columns)
        self._given_levels = dict(levels or {})
        self._fitted = False

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.columns})"

    def fit(self, z: torch.Tensor, states: pd.DataFrame, seed: int = 0) -> "Renderer":
        """Fit on factual latents and their states."""
        torch.manual_seed(seed)
        self.levels = [self._given_levels.get(c, int(states[c].max()) + 1) for c in self.columns]
        self._fit(z, states)
        self._fitted = True
        return self

    @torch.no_grad()
    def __call__(self, z: torch.Tensor, s: pd.DataFrame, s_prime: pd.DataFrame) -> torch.Tensor:
        """z' for latents ``z`` with factual states ``s`` and target states ``s_prime``."""
        if not self._fitted:
            raise ValueError("call fit() first")
        return self._render(z, s, s_prime)

    def _fit(self, z: torch.Tensor, states: pd.DataFrame) -> None:
        pass

    def _render(self, z: torch.Tensor, s: pd.DataFrame, s_prime: pd.DataFrame) -> torch.Tensor:
        raise NotImplementedError

    def one_hot(self, states: pd.DataFrame) -> torch.Tensor:
        """The concatenated one-hot encoding of ``states``' columns."""
        return _one_hot(states, self.columns, self.levels)


class IdentityRenderer(Renderer):
    """z' = z: the lower reference. Needs no fitting."""

    def __init__(self, columns: list[str], levels: dict[str, int] | None = None) -> None:
        super().__init__(columns, levels)
        self._fitted = True

    def _render(self, z, s, s_prime):
        return z.clone()


class ConstantRenderer(Renderer):
    """z' = z + E[z | on = s'_on] - E[z]: one fixed direction per target level."""

    def __init__(self, columns: list[str], on: str = "X", levels: dict[str, int] | None = None) -> None:
        super().__init__(columns, levels)
        self.on = on

    def _fit(self, z, states):
        group = torch.tensor(states[self.on].to_numpy())
        self.group_mean = {int(g): z[group == g].mean(0) for g in group.unique()}
        self.mean = z.mean(0)

    def _render(self, z, s, s_prime):
        target = s_prime[self.on].to_numpy()
        return z + torch.stack([self.group_mean[int(t)] for t in target]) - self.mean


class GroupOTRenderer(Renderer):
    """Gaussian optimal-transport map from p(z | on = s_on) to p(z | on = s'_on) (N2).

    ``shrink`` pulls each group covariance towards a scaled identity.
    """

    def __init__(
        self, columns: list[str], on: str = "X", shrink: float = 0.1, levels: dict[str, int] | None = None
    ) -> None:
        super().__init__(columns, levels)
        self.on, self.shrink = on, shrink

    def _fit(self, z, states):
        group = torch.tensor(states[self.on].to_numpy())
        self.group_mean = {int(g): z[group == g].mean(0) for g in group.unique()}
        self.group_cov = {int(g): _shrunk_cov(z[group == g], self.shrink) for g in group.unique()}
        self._maps: dict[tuple[int, int], torch.Tensor] = {}

    def _render(self, z, s, s_prime):
        source, target = s[self.on].to_numpy(), s_prime[self.on].to_numpy()
        out = z.clone()
        for a, b in {(int(x), int(y)) for x, y in zip(source, target) if x != y}:
            rows = torch.as_tensor((source == a) & (target == b))
            if (a, b) not in self._maps:
                self._maps[(a, b)] = _gaussian_ot(self.group_cov[a], self.group_cov[b])
            out[rows] = self.group_mean[b] + (z[rows] - self.group_mean[a]) @ self._maps[(a, b)]
        return out


class StateEffectRenderer(Renderer):
    """z' = z + m(s') - m(s), m a ridge regression of z on the one-hot state (C-sym).

    With ``guard=<linear Reader>``, the edit is projected onto the directions that leave
    the guard's probabilities unchanged (projected C-sym).
    """

    def __init__(
        self,
        columns: list[str],
        ridge: float = 1e-3,
        guard=None,
        levels: dict[str, int] | None = None,
    ) -> None:
        super().__init__(columns, levels)
        self.ridge, self.guard = ridge, guard

    def _fit(self, z, states):
        x = torch.cat([self.one_hot(states), torch.ones(len(states), 1)], 1)
        penalty = self.ridge * torch.eye(x.shape[1])
        self.effect = torch.linalg.solve(x.T @ x + penalty, x.T @ z)[:-1]
        self.basis = _probe_basis(self.guard) if self.guard is not None else None

    def _render(self, z, s, s_prime):
        edit = (self.one_hot(s_prime) - self.one_hot(s)) @ self.effect
        if self.basis is not None:
            edit = edit - (edit @ self.basis) @ self.basis.T
        return z + edit


class FlowRenderer(Renderer):
    """State-conditioned flow (F1): abduct u = F^-1(z; s), render z' = F(u; s').

    F is a conditional RealNVP fitted by maximum likelihood of p(z | s); the epoch
    with the best likelihood on a ``validation`` share of the rows is kept.
    """

    def __init__(
        self,
        columns: list[str],
        blocks: int = 8,
        hidden: int = 256,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 256,
        validation: float = 0.1,
        levels: dict[str, int] | None = None,
    ) -> None:
        super().__init__(columns, levels)
        self.settings = dict(
            blocks=blocks, hidden=hidden, epochs=epochs, lr=lr, batch_size=batch_size, validation=validation
        )

    def _fit(self, z, states):
        self.flow = _fit_flow(z, self.one_hot(states), **self.settings)

    def _render(self, z, s, s_prime):
        a, b = self.one_hot(s), self.one_hot(s_prime)
        changed = (a != b).any(1)
        out = z.clone()
        if changed.any():
            out[changed] = self.flow.render(z[changed], a[changed], b[changed])
        return out


RENDERERS: dict[str, type[Renderer]] = {
    "identity": IdentityRenderer,
    "constant": ConstantRenderer,
    "group_ot": GroupOTRenderer,
    "state_effect": StateEffectRenderer,
    "flow": FlowRenderer,
}


# --- helpers ---------------------------------------------------------------------------


def _one_hot(states: pd.DataFrame, columns: list[str], levels: list[int]) -> torch.Tensor:
    return torch.cat(
        [
            nn.functional.one_hot(torch.tensor(states[c].to_numpy(), dtype=torch.long), k).float()
            for c, k in zip(columns, levels)
        ],
        1,
    )


def _shrunk_cov(z: torch.Tensor, shrink: float) -> torch.Tensor:
    cov = torch.cov(z.T.double())
    return (1 - shrink) * cov + shrink * cov.trace() / len(cov) * torch.eye(len(cov), dtype=cov.dtype)


def _sqrtm(m: torch.Tensor, power: float = 0.5) -> torch.Tensor:
    values, vectors = torch.linalg.eigh(m)
    return vectors @ torch.diag(values.clamp(min=1e-12) ** power) @ vectors.T


def _gaussian_ot(cov_a: torch.Tensor, cov_b: torch.Tensor) -> torch.Tensor:
    """Linear map A of the OT map N(0, cov_a) -> N(0, cov_b), applied as x @ A (A symmetric)."""
    root, inv_root = _sqrtm(cov_a), _sqrtm(cov_a, -0.5)
    return (inv_root @ _sqrtm(root @ cov_b @ root) @ inv_root).float()


def _probe_basis(reader) -> torch.Tensor:
    """Orthonormal basis of the directions that change a linear probe's probabilities."""
    if reader.hidden != 0:
        raise ValueError("the guard must be a linear Reader (hidden=0)")
    weights = torch.cat(
        [head.weight - head.weight.mean(0, keepdim=True) for head in reader._fitted().heads.values()]
    ).detach().double()
    _, singular, vh = torch.linalg.svd(weights, full_matrices=False)
    keep = singular > singular.max() * max(weights.shape) * torch.finfo(torch.float32).eps
    return vh[keep].T.float()


class _Coupling(nn.Module):
    def __init__(self, mask: torch.Tensor, cond_dim: int, hidden: int) -> None:
        super().__init__()
        self.register_buffer("mask", mask)
        dim = len(mask)
        self.net = nn.Sequential(
            nn.Linear(dim + cond_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _params(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale, shift = self.net(torch.cat([x * self.mask, c], 1)).chunk(2, 1)
        free = 1 - self.mask
        return torch.tanh(scale) * free, shift * free

    def forward(self, u: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        scale, shift = self._params(u, c)
        return u * scale.exp() + shift

    def inverse(self, z: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale, shift = self._params(z, c)
        return (z - shift) * (-scale).exp(), scale.sum(1)


class _StateFlow(nn.Module):
    """Conditional RealNVP z = F(u; s) on standardized latents, u ~ N(0, I)."""

    def __init__(self, z: torch.Tensor, cond_dim: int, blocks: int, hidden: int) -> None:
        super().__init__()
        self.register_buffer("mean", z.mean(0))
        self.register_buffer("std", z.std(0).clamp(min=1e-6))
        g = torch.Generator().manual_seed(0)
        masks = [(torch.randperm(z.shape[1], generator=g) < z.shape[1] // 2).float() for _ in range(blocks // 2)]
        masks = [m for mask in masks for m in (mask, 1 - mask)]
        self.layers = nn.ModuleList(_Coupling(m, cond_dim, hidden) for m in masks)

    def abduct(self, z: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, log_det = (z - self.mean) / self.std, torch.zeros(len(z))
        for layer in reversed(self.layers):
            x, step = layer.inverse(x, c)
            log_det = log_det + step
        return x, log_det

    def log_prob(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        u, log_det = self.abduct(z, c)
        return -0.5 * (u**2).sum(1) - log_det

    def render(self, z: torch.Tensor, c: torch.Tensor, c_prime: torch.Tensor) -> torch.Tensor:
        x, _ = self.abduct(z, c)
        for layer in self.layers:
            x = layer(x, c_prime)
        return x * self.std + self.mean


def _fit_flow(
    z: torch.Tensor,
    c: torch.Tensor,
    blocks: int = 8,
    hidden: int = 256,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 256,
    validation: float = 0.1,
) -> _StateFlow:
    """Maximum likelihood of p(z | s); keeps the epoch with the best held-out likelihood."""
    flow = _StateFlow(z, c.shape[1], blocks, hidden)
    order = torch.randperm(len(z))
    val, train = order[: int(validation * len(z))], order[int(validation * len(z)) :]
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr, weight_decay=1e-5)
    best, best_state = float("inf"), None
    for _ in range(epochs):
        flow.train()
        for batch in train[torch.randperm(len(train))].split(batch_size):
            loss = -flow.log_prob(z[batch], c[batch]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        flow.eval()
        with torch.no_grad():
            val_loss = -flow.log_prob(z[val], c[val]).mean().item() if len(val) else loss.item()
        if val_loss < best:
            best, best_state = val_loss, copy.deepcopy(flow.state_dict())
    flow.load_state_dict(best_state)
    return flow.eval()
