"""rho: write a target state s' into a latent z while keeping what s does not describe.

Every renderer is fitted on observed ``(z, s)`` only, never on edited pairs, and called
as ``renderer(z, s, s_prime) -> z'``:

=======================  =========================================================  ==========
class                    z'                                                         uses
=======================  =========================================================  ==========
``IdentityRenderer``     z (lower reference)                                        --
``ConstantRenderer``     z + E[z | c = s'_c] - E[z]: one direction per target level  s'_c
``GroupOTRenderer``      Gaussian OT map p(z | c = s_c) -> p(z | c = s'_c)          s_c, s'_c
``StateEffectRenderer``  z + m(s') - m(s), m additive in the levels; with          s, s'
                         ``guard=<linear Reader>`` the edit is projected so the
                         guard's probabilities stay unchanged
``FlowRenderer``         u = F^-1(z; s), z' = F(u; s'), F a conditional RealNVP     s, s'
``EditorRenderer``       z + Delta(z, s'), trained so that a reader fitted on       s'
                         (z, s) reads s', with random targets s' ~ Uniform(S)
                         (a minimal edit; no operator in training)
``OTEditorRenderer``     grad_x Phi(x; s'), Phi input-convex; trained like the      s'
                         editor (OT-structured minimal edit)
``OTFlowRenderer``       u = F_s^-1(z), z' = F_s'(u), F_s a continuous flow fitted  s, s'
                         by OT flow matching within each state
=======================  =========================================================  ==========

``c`` is the ``on`` column (default ``"X"``). All return ``z`` unchanged where
``s' = s``, except ``ConstantRenderer``, which ignores the source state.

A new renderer subclasses ``Renderer`` and implements ``_fit`` and ``_render``.
"""

from __future__ import annotations

import copy
import math
from time import perf_counter

import pandas as pd
import torch
from torch import nn

from latent_lab.reader import fit_log


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
        """Fit on observed latents and their states."""
        started = perf_counter()
        torch.manual_seed(seed)
        self.levels = [self._given_levels.get(c, int(states[c].max()) + 1) for c in self.columns]
        log = self._fit(z, states)  # trained renderers return a fit_log; closed-form ones None
        self._fit_log = {**(log or {}), "seconds": perf_counter() - started}
        self._fitted = True
        return self

    @property
    def fit_log(self) -> dict:
        """Training time and convergence of ``fit`` (see :func:`latent_lab.reader.fit_log`)."""
        return getattr(self, "_fit_log", {})

    @torch.no_grad()
    def __call__(self, z: torch.Tensor, s: pd.DataFrame, s_prime: pd.DataFrame) -> torch.Tensor:
        """z' for latents ``z`` with source states ``s`` and target states ``s_prime``."""
        if not self._fitted:
            raise ValueError("call fit() first")
        return self._render(z, s, s_prime)

    def _fit(self, z: torch.Tensor, states: pd.DataFrame) -> dict | None:
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
    """Gaussian optimal-transport map from p(z | on = s_on) to p(z | on = s'_on).

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
    """z' = z + m(s') - m(s), m a ridge regression of z on the one-hot state.

    With ``guard=<linear Reader>``, the edit is projected onto the directions that leave
    the guard's probabilities unchanged.
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
    """State-conditioned flow: infer the residual u = F^-1(z; s), render z' = F(u; s').

    F is a conditional RealNVP fitted by maximum likelihood of p(z | s). The state enters
    every coupling through one learned code (an embedding per column, then a small MLP).
    A ``validation`` share of the rows selects the epoch with the lowest per-coordinate
    NLL; the learning rate halves after 10 stale epochs and training stops after
    ``patience`` of them. Defaults: the settings selected for EmbeddingGemma at d = 256.
    """

    def __init__(
        self,
        columns: list[str],
        blocks: int = 8,
        hidden: int = 136,
        condition: int = 64,
        embed: int = 16,
        epochs: int = 500,
        lr: float = 1e-3,
        batch_size: int = 64,
        weight_decay: float = 1e-5,
        grad_clip: float = 5.0,
        validation: float = 0.2,
        patience: int = 30,
        levels: dict[str, int] | None = None,
    ) -> None:
        super().__init__(columns, levels)
        self.validation = validation
        self.settings = dict(
            blocks=blocks, hidden=hidden, condition=condition, embed=embed, epochs=epochs, lr=lr,
            batch_size=batch_size, weight_decay=weight_decay, grad_clip=grad_clip, patience=patience,
        )

    def _fit(self, z, states):
        index = torch.tensor(states[self.columns].to_numpy(), dtype=torch.long)
        seed = int(torch.initial_seed())
        order = torch.randperm(len(z), generator=torch.Generator().manual_seed(seed))
        n_val = int(self.validation * len(z))
        val, fit = order[:n_val], order[n_val:]
        self.flow, log = _fit_state_flow(z[fit], index[fit], z[val], index[val], self.levels, seed, **self.settings)
        return log

    def _render(self, z, s, s_prime):
        a = torch.tensor(s[self.columns].to_numpy(), dtype=torch.long)
        b = torch.tensor(s_prime[self.columns].to_numpy(), dtype=torch.long)
        changed = (a != b).any(1)
        out = z.clone()
        if changed.any():
            out[changed] = self.flow.transport(z[changed], a[changed], b[changed])
        return out


class EditorRenderer(Renderer):
    """Minimal edit: z' = z + Delta(z, s') on standardized latents.

    A ``Reader`` (the critic) is fitted on the training ``(z, s)`` and frozen. Delta is
    then trained on random targets s' drawn uniformly from S, so the critic reads s' in
    z' (cross-entropy) with a ``proximity`` penalty on the squared edit per coordinate.
    No operator appears in training, so any s' can be rendered. A ``validation`` share of
    the rows, with fixed random targets, selects the best epoch and stops training after
    ``patience`` epochs without improvement.
    """

    def __init__(
        self,
        columns: list[str],
        proximity: float = 0.1,
        hidden: int = 512,
        critic_hidden: int = 1,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 256,
        validation: float = 0.2,
        patience: int = 30,
        levels: dict[str, int] | None = None,
    ) -> None:
        super().__init__(columns, levels)
        self.proximity, self.hidden, self.critic_hidden = proximity, hidden, critic_hidden
        self.epochs, self.lr, self.batch_size = epochs, lr, batch_size
        self.validation, self.patience = validation, patience

    def _fit(self, z, states):
        from latent_lab.reader import Reader

        seed = int(torch.initial_seed())
        val, fit = _split(len(z), self.validation, seed)
        levels = dict(zip(self.columns, self.levels))
        self.critic = Reader(self.columns, hidden=self.critic_hidden, levels=levels).fit(z[fit], states.iloc[fit.numpy()])
        critic = self.critic.model.requires_grad_(False)
        self.mean, self.std = z[fit].mean(0), z[fit].std(0).clamp(min=1e-6)
        x = (z - self.mean) / self.std
        x_fit, x_val = x[fit], x[val]
        draw = lambda n, g=None: torch.stack([torch.randint(k, (n,), generator=g) for k in self.levels], 1)  # noqa: E731
        val_target = draw(len(x_val), torch.Generator().manual_seed(seed + 1))  # fixed, so epochs are comparable

        def objective(xb, target, train):
            moved = self._move(xb, target, train=train)
            logits = critic(moved * self.std + self.mean)
            loss = sum(nn.functional.cross_entropy(logits[c], target[:, i]) for i, c in enumerate(self.columns))
            return loss + self.proximity * (moved - xb).pow(2).mean()

        self.net = self._build(z.shape[1])
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        stop, history = _EarlyStop(self.patience), []
        for epoch in range(1, self.epochs + 1):
            total = 0.0
            for batch in torch.randperm(len(x_fit)).split(self.batch_size):
                loss = objective(x_fit[batch], draw(len(batch)), train=True)  # s' ~ Uniform(S)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * len(batch)
            history.append({"epoch": epoch, "train": total / len(x_fit), "lr": self.lr})
            if len(x_val):
                with torch.no_grad():
                    history[-1]["validation"] = value = objective(x_val, val_target, train=False).item()
                if stop.step(self.net, value, epoch):
                    break
        stop.restore(self.net)
        self.net.eval()
        return fit_log(history, self.epochs, stop.best_epoch or len(history), 0.0)

    def _build(self, dim: int) -> nn.Module:
        net = nn.Sequential(nn.Linear(dim + sum(self.levels), self.hidden), nn.SiLU(), nn.Linear(self.hidden, dim))
        nn.init.zeros_(net[-1].weight)
        nn.init.zeros_(net[-1].bias)
        return net

    def _move(self, x: torch.Tensor, target: torch.Tensor, train: bool = False) -> torch.Tensor:
        """Standardized x' for standardized x and integer targets (n, columns)."""
        one_hot = torch.cat([nn.functional.one_hot(target[:, i], k).float() for i, k in enumerate(self.levels)], 1)
        return x + self.net(torch.cat([x, one_hot], 1))

    def _render(self, z, s, s_prime):
        changed = (self.one_hot(s) != self.one_hot(s_prime)).any(1)
        out = z.clone()
        if changed.any():
            target = torch.tensor(s_prime[self.columns].to_numpy(), dtype=torch.long)[changed]
            out[changed] = self._move((z[changed] - self.mean) / self.std, target) * self.std + self.mean
        return out


class OTEditorRenderer(EditorRenderer):
    """OT-structured minimal edit: x' = grad_x Phi(x; s'), Phi convex in x for each target.

    Same random-target training as ``EditorRenderer``; only the map differs.
    """

    def _build(self, dim):
        return _ConvexPotential(dim, self.levels, self.hidden)

    def _move(self, x, target, train=False):
        return self.net.gradient(x, target, create_graph=train)


class OTFlowRenderer(Renderer):
    """Shared-reference OT flow: u = F_s^-1(z), z' = F_s'(u), F_s a state-conditioned
    continuous flow fitted by conditional minibatch OT flow matching.

    Each step draws one state (by its frequency), a batch of its standardized latents
    (plus ``smoothing_std`` noise) and as many N(0, I) reference points, couples them by
    an entropic OT plan, and regresses the velocity on the straight paths between the
    coupled pairs. Reference and latents are never coupled across states. A
    ``validation`` share of the rows, with one fixed coupled batch per state (weighted by
    its frequency), selects the best epoch and stops training after ``patience`` epochs
    without improvement.
    """

    def __init__(
        self,
        columns: list[str],
        hidden: int = 512,
        epochs: int = 100,
        lr: float = 3e-4,
        batch_size: int = 64,
        smoothing_std: float = 0.01,
        validation: float = 0.2,
        patience: int = 30,
        levels: dict[str, int] | None = None,
    ) -> None:
        super().__init__(columns, levels)
        self.hidden, self.epochs, self.lr = hidden, epochs, lr
        self.batch_size, self.smoothing_std = batch_size, smoothing_std
        self.validation, self.patience = validation, patience

    def _fit(self, z, states):
        seed = int(torch.initial_seed())
        val, fit = _split(len(z), self.validation, seed)
        index = torch.tensor(states[self.columns].to_numpy(), dtype=torch.long)
        self.mean, self.std = z[fit].mean(0), z[fit].std(0, unbiased=False).clamp(min=1e-4)
        self.velocity = _VelocityField(z.shape[1], self.levels, self.hidden)
        x = (z - self.mean) / self.std
        x_fit = x[fit]
        state_ids, rows = index[fit].unique(dim=0, return_inverse=True)
        groups = [(rows == g).nonzero().flatten() for g in range(len(state_ids))]
        rng, n = torch.Generator().manual_seed(seed), self.batch_size
        batch = lambda points, state, g: _coupled_batch(points, state, n, self.smoothing_std, g)  # noqa: E731
        val_rng, bank = torch.Generator().manual_seed(seed + 1), []
        val_ids, val_rows = index[val].unique(dim=0, return_inverse=True)
        for g, state in enumerate(val_ids):  # one fixed coupled batch per state, weighted by its frequency
            members = val[val_rows == g]
            bank.append((len(members) / len(val), batch(x[members], state, val_rng)))
        optimizer = torch.optim.Adam(self.velocity.parameters(), lr=self.lr)
        # shortcut: no learning-rate schedule or gradient clipping (src's benchmark had both).
        stop, history, steps = _EarlyStop(self.patience), [], -(-len(x_fit) // n)
        for epoch in range(1, self.epochs + 1):
            total = 0.0
            for _ in range(steps):
                group = int(rows[torch.randint(len(x_fit), (1,), generator=rng)])
                path, t, cond, velocity = batch(x_fit[groups[group]], state_ids[group], rng)
                loss = (self.velocity(path, t, cond) - velocity).square().mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()
            history.append({"epoch": epoch, "train": total / steps, "lr": self.lr})
            if bank:
                with torch.no_grad():
                    value = sum(w * (self.velocity(p, t, c) - v).square().mean().item() for w, (p, t, c, v) in bank)
                history[-1]["validation"] = value
                if stop.step(self.velocity, value, epoch):
                    break
        stop.restore(self.velocity)
        self.velocity.eval()
        return fit_log(history, self.epochs, stop.best_epoch or len(history), 0.0)

    def _render(self, z, s, s_prime):
        a = torch.tensor(s[self.columns].to_numpy(), dtype=torch.long)
        b = torch.tensor(s_prime[self.columns].to_numpy(), dtype=torch.long)
        changed = (a != b).any(1)
        out = z.clone()
        if changed.any():
            x = (z[changed] - self.mean) / self.std
            u = _integrate(self.velocity, x, a[changed], reverse=True)
            out[changed] = _integrate(self.velocity, u, b[changed]) * self.std + self.mean
        return out


RENDERERS: dict[str, type[Renderer]] = {
    "identity": IdentityRenderer,
    "constant": ConstantRenderer,
    "group_ot": GroupOTRenderer,
    "state_effect": StateEffectRenderer,
    "flow": FlowRenderer,
    "editor": EditorRenderer,
    "ot_editor": OTEditorRenderer,
    "ot_flow": OTFlowRenderer,
}


# --- helpers ---------------------------------------------------------------------------


def _split(n: int, validation: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Seeded (validation rows, fitting rows)."""
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    return order[: int(validation * n)], order[int(validation * n) :]


class _EarlyStop:
    """Keeps the weights of the epoch with the lowest validation loss; ``step`` returns
    True after ``patience`` epochs without an improvement of at least ``min_delta``."""

    def __init__(self, patience: int, min_delta: float = 1e-7) -> None:
        self.patience, self.min_delta = patience, min_delta
        self.best, self.reference, self.stale, self.best_epoch, self.state = math.inf, math.inf, 0, 0, None

    def step(self, module: nn.Module, value: float, epoch: int) -> bool:
        if value < self.best:
            self.best, self.best_epoch, self.state = value, epoch, copy.deepcopy(module.state_dict())
        if value < self.reference - self.min_delta:
            self.reference, self.stale = value, 0
        else:
            self.stale += 1
        return self.stale >= self.patience

    def restore(self, module: nn.Module) -> None:
        if self.state is not None:
            module.load_state_dict(self.state)


def _coupled_batch(points: torch.Tensor, state: torch.Tensor, n: int, smoothing: float, rng: torch.Generator):
    """One flow-matching batch within one state: ``n`` of its latents (with replacement,
    plus smoothing noise) coupled to ``n`` N(0, I) points by an entropic OT plan; returns
    the points on the straight paths, their times, the state and the target velocities."""
    target = points[torch.randint(len(points), (n,), generator=rng)]
    target = target + smoothing * torch.randn(target.shape, generator=rng)
    reference = torch.randn(target.shape, generator=rng)
    picks = torch.multinomial(_sinkhorn(reference, target).flatten(), n, replacement=True, generator=rng)
    u, end = reference[picks // n], target[picks % n]
    t = torch.rand(n, generator=rng)
    return (1 - t[:, None]) * u + t[:, None] * end, t, state.expand(n, -1), end - u


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


def _permutations(dim: int, blocks: int, seed: int) -> list[torch.Tensor]:
    """One coordinate permutation per coupling, applied before its even/odd split.

    The second is built so that it transforms every coordinate the first leaves
    untouched; the rest are random.
    """
    if dim < 2 or blocks < 2:
        raise ValueError("the state flow needs at least 2 latent dimensions and 2 blocks")
    g = torch.Generator().manual_seed(seed)
    coords = torch.arange(dim)
    first = torch.randperm(dim, generator=g)
    covered = torch.zeros(dim, dtype=torch.bool)
    covered[first[coords % 2 != 0]] = True  # block 0 transforms the odd positions
    untouched, done = coords[~covered[first]], coords[covered[first]]
    untouched = untouched[torch.randperm(len(untouched), generator=g)]
    done = done[torch.randperm(len(done), generator=g)]
    second = torch.empty(dim, dtype=torch.long)
    second[coords % 2 == 0], second[coords % 2 == 1] = untouched, done  # block 1 transforms the even ones
    return [first, second] + [torch.randperm(dim, generator=g) for _ in range(blocks - 2)]


class _AffineCoupling(nn.Module):
    """Permute, then transform one half of the coordinates (by parity) affinely given
    the other half and the state code; log-scales are bounded to [-2, 2]."""

    def __init__(self, dim: int, condition: int, hidden: int, parity: int, permutation: torch.Tensor) -> None:
        super().__init__()
        coords = torch.arange(dim)
        self.register_buffer("keep", coords[coords % 2 == parity])
        self.register_buffer("change", coords[coords % 2 != parity])
        self.register_buffer("permutation", permutation)
        self.register_buffer("inverse_permutation", torch.argsort(permutation))
        self.net = nn.Sequential(
            nn.Linear(len(self.keep) + condition, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * len(self.change)),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _params(self, permuted: torch.Tensor, code: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw_scale, shift = self.net(torch.cat([permuted[:, self.keep], code], 1)).chunk(2, 1)
        return 2 * torch.tanh(raw_scale / 2), shift

    def forward(self, x: torch.Tensor, code: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        permuted = x[:, self.permutation]
        log_scale, shift = self._params(permuted, code)
        out = permuted.clone()
        out[:, self.change] = permuted[:, self.change] * log_scale.exp() + shift
        return out, log_scale.sum(1)

    def inverse(self, y: torch.Tensor, code: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_scale, shift = self._params(y, code)
        out = y.clone()
        out[:, self.change] = (y[:, self.change] - shift) * (-log_scale).exp()
        return out[:, self.inverse_permutation], -log_scale.sum(1)


class _StateFlow(nn.Module):
    """Conditional RealNVP z = F(u; s) on standardized latents, u ~ N(0, I)."""

    def __init__(self, dim: int, levels: list[int], blocks: int, hidden: int, condition: int, embed: int, seed: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("scale", torch.ones(dim))
        self.embeddings = nn.ModuleList(nn.Embedding(k, embed) for k in levels)
        self.project = nn.Sequential(nn.Linear(len(levels) * embed, condition), nn.SiLU(), nn.Linear(condition, condition))
        self.blocks = nn.ModuleList(
            _AffineCoupling(dim, condition, hidden, i % 2, p) for i, p in enumerate(_permutations(dim, blocks, seed))
        )

    def code(self, states: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([e(states[:, i]) for i, e in enumerate(self.embeddings)], 1))

    def infer(self, z: torch.Tensor, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """u = F^-1(z; s) and log |det dF^-1/dz| in standardized units."""
        x, code, log_det = (z - self.mean) / self.scale, self.code(states), torch.zeros(len(z))
        for block in reversed(self.blocks):
            x, step = block.inverse(x, code)
            log_det = log_det + step
        return x, log_det

    def log_prob(self, z: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        u, log_det = self.infer(z, states)
        return -0.5 * (u.square() + math.log(2 * math.pi)).sum(1) + log_det - self.scale.log().sum()

    def transport(self, z: torch.Tensor, s: torch.Tensor, s_prime: torch.Tensor) -> torch.Tensor:
        x, code = self.infer(z, s)[0], self.code(s_prime)
        for block in self.blocks:
            x = block(x, code)[0]
        return x * self.scale + self.mean


def _fit_state_flow(
    z_fit, s_fit, z_val, s_val, levels, seed, blocks, hidden, condition, embed,
    epochs, lr, batch_size, weight_decay, grad_clip, patience,
) -> _StateFlow:
    """Conditional maximum likelihood with validation selection, as in the h_Z benchmark."""
    with torch.random.fork_rng(devices=[]):  # the seed alone fixes the initial weights
        torch.manual_seed(seed)
        flow = _StateFlow(z_fit.shape[1], levels, blocks, hidden, condition, embed, seed)
    flow.mean.copy_(z_fit.mean(0))
    flow.scale.copy_(z_fit.std(0, unbiased=False).clamp(min=1e-6))
    shuffle = torch.Generator().manual_seed(seed + 1)
    optimizer = torch.optim.AdamW(flow.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, threshold=1e-7, threshold_mode="abs", min_lr=1e-6
    )
    best, best_state, reference, stale, best_epoch, history = math.inf, None, math.inf, 0, 0, []
    flow.train()
    for epoch in range(1, epochs + 1):
        lr_now, total = optimizer.param_groups[0]["lr"], 0.0
        for batch in torch.randperm(len(z_fit), generator=shuffle).split(batch_size):
            loss = -flow.log_prob(z_fit[batch], s_fit[batch]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), grad_clip, error_if_nonfinite=True)
            optimizer.step()
            total += loss.item() * len(batch)
        history.append({"epoch": epoch, "train": total / len(z_fit) / z_fit.shape[1], "lr": lr_now})
        if not len(z_val):
            continue
        with torch.no_grad():  # mean NLL per coordinate, in batches of 32 like the benchmark
            value = sum(
                float(-flow.log_prob(zb, sb).mean() / zb.shape[1]) * len(zb)
                for zb, sb in zip(z_val.split(32), s_val.split(32))
            ) / len(z_val)
        history[-1]["validation"] = value
        if value < best:
            best, best_state, best_epoch = value, copy.deepcopy(flow.state_dict()), epoch
        if value < reference - 1e-7:
            reference, stale = value, 0
        else:
            stale += 1
        scheduler.step(value)
        if stale >= patience:
            break
    if best_state is not None:
        flow.load_state_dict(best_state)
    return flow.eval(), fit_log(history, epochs, best_epoch or len(history), 0.0)


class _ConvexPotential(nn.Module):
    """Phi(x; s'), convex in x for every fixed target s' (an input-convex network).

    Softplus layers with non-negative hidden and output weights plus affine skips of
    the input, a linear term and a learnable diagonal quadratic (floor ``floor``), so
    grad_x Phi can contract. At initialization grad_x Phi is close to the identity.
    """

    def __init__(self, dim: int, levels: list[int], hidden: int, embed: int = 16, floor: float = 1e-3) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(k, embed) for k in levels)
        self.input1 = nn.Linear(dim + len(levels) * embed, hidden)
        self.input2 = nn.Linear(dim + len(levels) * embed, hidden)
        start = math.log(math.expm1(1 / hidden))  # softplus^-1(1 / hidden)
        self.positive_hidden = nn.Parameter(torch.full((hidden, hidden), start))
        self.positive_output = nn.Parameter(torch.full((hidden,), start))
        self.linear = nn.Parameter(torch.zeros(dim))
        self.quadratic = nn.Parameter(torch.full((dim,), math.log(math.expm1(1 - floor))))
        self.floor = floor

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([x, *(e(target[:, i]) for i, e in enumerate(self.embeddings))], 1)
        softplus = nn.functional.softplus
        h = softplus(self.input1(inputs))
        h = softplus(nn.functional.linear(h, softplus(self.positive_hidden)) + self.input2(inputs))
        quadratic = 0.5 * (x.square() * (softplus(self.quadratic) + self.floor)).sum(1)
        return (h * softplus(self.positive_output)).sum(1) + x @ self.linear + quadratic

    def gradient(self, x: torch.Tensor, target: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
        with torch.enable_grad():  # renderers are called under no_grad
            x = x.detach().requires_grad_(True)
            (grad,) = torch.autograd.grad(self(x, target).sum(), x, create_graph=create_graph)
        return grad if create_graph else grad.detach()


class _VelocityField(nn.Module):
    """v(x, t; s): an MLP on x, one embedding per state column and 16 sinusoidal time
    features. The last layer starts at zero, so the untrained flow is the identity."""

    def __init__(self, dim: int, levels: list[int], hidden: int, embed: int = 16) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(k, embed) for k in levels)
        self.net = nn.Sequential(
            nn.Linear(dim + len(levels) * embed + 16, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, t, states: torch.Tensor) -> torch.Tensor:
        angle = torch.as_tensor(t, dtype=x.dtype).expand(len(x)).reshape(-1, 1) * torch.arange(1, 9) * math.pi
        embedded = [e(states[:, i]) for i, e in enumerate(self.embeddings)]
        return self.net(torch.cat([x, *embedded, angle.sin(), angle.cos()], 1))


def _rk4(velocity, x: torch.Tensor, states: torch.Tensor, start: float, end: float, steps: int) -> torch.Tensor:
    h = (end - start) / steps
    for i in range(steps):
        t = start + i * h
        k1 = velocity(x, t, states)
        k2 = velocity(x + h * k1 / 2, t + h / 2, states)
        k3 = velocity(x + h * k2 / 2, t + h / 2, states)
        k4 = velocity(x + h * k3, t + h, states)
        x = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x


@torch.no_grad()
def _integrate(velocity, x, states, reverse: bool = False, steps: int = 64, max_steps: int = 256, tol: float = 1e-4):
    """Solve the flow ODE from t=0 to 1 (or back) by RK4, doubling the steps until two
    solutions agree to ``tol`` (max per-unit RMS, standardized units)."""
    start, end = (1.0, 0.0) if reverse else (0.0, 1.0)
    previous = _rk4(velocity, x, states, start, end, steps)
    error = math.inf
    while 2 * steps <= max_steps:
        steps *= 2
        current = _rk4(velocity, x, states, start, end, steps)
        error = float((current - previous).square().mean(1).sqrt().max())
        if error <= tol and torch.isfinite(current).all():
            return current
        previous = current
    raise FloatingPointError(f"OT flow ODE did not reach tolerance {tol}: error {error:.3g} at {steps} steps")


@torch.no_grad()
def _sinkhorn(x: torch.Tensor, y: torch.Tensor, epsilon: float = 0.05, tol: float = 1e-6, max_iterations: int = 100_000):
    """Entropic OT plan between two uniformly weighted point sets, in log domain and
    float64, with cost |x - y|^2 / dim. Raises if the marginals do not converge."""
    kernel = -torch.cdist(x.double(), y.double()).square() / x.shape[1] / epsilon
    log_a, log_b = -math.log(len(x)), -math.log(len(y))
    u, v = torch.zeros(len(x), dtype=torch.double), torch.zeros(len(y), dtype=torch.double)
    for i in range(1, max_iterations + 1):
        u = log_a - torch.logsumexp(kernel + v[None], dim=1)
        v = log_b - torch.logsumexp(kernel + u[:, None], dim=0)
        if i % 10 == 0:
            plan = (kernel + u[:, None] + v[None]).exp()
            if (plan.sum(1) - 1 / len(x)).abs().max() <= tol:  # columns are exact after the v step
                return plan
    raise FloatingPointError(f"Sinkhorn did not converge in {max_iterations} iterations")
