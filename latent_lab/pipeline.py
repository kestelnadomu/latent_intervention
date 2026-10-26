"""The modular latent operation (MLO) read -> operate -> render, and its metrics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from latent_lab.reader import Reader, state_metrics
from latent_lab.renderer import Renderer


@dataclass
class Result:
    z_prime: torch.Tensor
    s: pd.DataFrame
    s_prime: pd.DataFrame


@dataclass
class Pipeline:
    """The MLO omega_Z(z) = rho(z; s, s') with s ~ psi(. | z) and s' = operator(s).

    ``operator`` is any function from a frame of states to a frame of target states
    (one row each, same columns), deterministic or random. It sees every column read
    (or passed); the renderer then conditions on its own ``columns`` of s and s' only,
    so a column outside them is left to the renderer's residual. ``reader=None`` uses
    the true source states passed to ``__call__`` instead, so only the renderer can err.
    """

    reader: Reader | None
    operator: Callable[[pd.DataFrame], pd.DataFrame]
    renderer: Renderer

    def __call__(
        self,
        z: torch.Tensor,
        states: pd.DataFrame | None = None,
        sample: bool = False,
        seed: int = 0,
    ) -> Result:
        """s is psi's mode, or a draw from psi if ``sample=True``."""
        if self.reader is None:
            if states is None:
                raise ValueError("without a reader, pass the true source states")
            s = states.reset_index(drop=True)
        else:
            s = self.reader.sample(z, seed) if sample else self.reader.predict(z)
        s_prime = self.operator(s.copy()).reset_index(drop=True)
        return Result(self.renderer(z, s, s_prime), s, s_prime)


def evaluate(
    z: torch.Tensor,
    z_prime: torch.Tensor,
    z_true: torch.Tensor,
    reader: Reader | None = None,
    true_states: pd.DataFrame | None = None,
    joint: list[str] | None = None,
    predict: Callable[[torch.Tensor], torch.Tensor] | None = None,
    outcome: pd.Series | None = None,
    groups: pd.Series | None = None,
    scale: torch.Tensor | None = None,
) -> pd.Series:
    """Compare edited latents z' with target latents phi(x'), the encodings of real
    texts that have the target states.

    Latent space, in coordinates divided by ``scale`` (default: the std of ``z``) and
    by sqrt(d):

    - ``gap_closed``: 1 - |z' - phi(x')| / |z - phi(x')|, averaged over units that
      move (1 = perfect, 0 = no better than leaving z alone, < 0 = worse).
    - ``distance``: mean |z' - phi(x')| (the per-unit energy score of a point edit).
    - ``energy``: energy distance between the sets {z'} and {phi(x')}.
    - ``shift``: mean |z' - z|.
    - ``cosine``: mean cosine similarity of z' and phi(x').

    With an (ideally independent) ``reader`` and the true target states:
    ``acc_<c>`` and ``joint_acc`` (over ``joint``, default all of the reader's columns)
    of its reading of z', next to ``real_*``, the same on phi(x') itself.

    With a downstream ``predict`` (latents -> class) and the true target
    ``outcome``: ``down_acc`` of predict(z') against it, ``down_agree`` with predict(z),
    and ``parity_gap``, the range over ``groups`` (e.g. the target X) of the
    mean predicted class; ``real_*`` the same on phi(x').
    """
    scale = z.std(0).clamp(min=1e-6) if scale is None else scale
    root = z.shape[1] ** 0.5
    before = (z - z_true).norm(dim=1)
    after = (z_prime - z_true).norm(dim=1)
    moving = before > 1e-6
    out = {
        "gap_closed": float((1 - after[moving] / before[moving]).mean()),
        "distance": float(((z_prime - z_true) / scale).norm(dim=1).mean() / root),
        "energy": _energy_distance(z_prime / scale, z_true / scale) / root,
        "shift": float(((z_prime - z) / scale).norm(dim=1).mean() / root),
        "cosine": float(torch.nn.functional.cosine_similarity(z_prime, z_true).mean()),
    }
    if reader is not None and true_states is not None:
        joint = reader.names if joint is None else joint
        for prefix, latent in (("", z_prime), ("real_", z_true)):
            metrics = state_metrics(reader.proba(latent), true_states, joint)
            out.update({f"{prefix}{k}": v for k, v in metrics.items() if k.startswith(("acc_", "joint_acc"))})
    if predict is not None and outcome is not None:
        y = torch.tensor(outcome.to_numpy())
        source = predict(z)
        for prefix, latent in (("", z_prime), ("real_", z_true)):
            predicted = predict(latent)
            out[f"{prefix}down_acc"] = float((predicted == y).float().mean())
            out[f"{prefix}down_agree"] = float((predicted == source).float().mean())
            if groups is not None:
                means = pd.Series(predicted.float().numpy()).groupby(groups.to_numpy()).mean()
                out[f"{prefix}parity_gap"] = float(means.max() - means.min())
    return pd.Series(out)


def factual_score(
    renderer: Renderer, z: torch.Tensor, states: pd.DataFrame, reader: Reader, seed: int = 0, min_units: int = 10
) -> pd.Series:
    """Score a renderer on held-out observed units, without target latents.

    Every unit is edited to the observed state of another, randomly drawn unit, so the
    targets follow the observed joint distribution.

    - ``fidelity``: share of units for which ``reader`` (not a renderer's own critic)
      reads every target column in z'.
    - ``realism``: for every column and level, the energy distance between the edited
      latents with that target level and the observed latents with that level (both
      with at least ``min_units`` units), averaged; standardized and divided by sqrt(d).
      The identity edit scores > 0 here. An edit that keeps the target's statistics but
      discards the unit's residual is not caught: that needs target latents.
    - ``shift``: mean |z' - z| in the same units.
    """
    s = states[renderer.columns].reset_index(drop=True)
    s_prime = s.iloc[np.random.default_rng(seed).permutation(len(s))].reset_index(drop=True)
    z_prime = renderer(z, s, s_prime)
    scale, root = z.std(0).clamp(min=1e-6), z.shape[1] ** 0.5
    read = reader.predict(z_prime)
    fidelity = np.mean(np.all([read[c].to_numpy() == s_prime[c].to_numpy() for c in renderer.columns], axis=0))
    distances = []
    for c in renderer.columns:
        for level in np.unique(s_prime[c]):
            edited = torch.tensor((s_prime[c] == level).to_numpy())
            observed = torch.tensor((s[c] == level).to_numpy())
            if edited.sum() >= min_units and observed.sum() >= min_units:
                distances.append(_energy_distance(z_prime[edited] / scale, z[observed] / scale) / root)
    return pd.Series({
        "fidelity": float(fidelity),
        "realism": float(np.mean(distances)),
        "shift": float(((z_prime - z) / scale).norm(dim=1).mean() / root),
    })


def _energy_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Biased empirical energy distance 2 E|a - b| - E|a - a'| - E|b - b'| (>= 0)."""
    mean = lambda x, y: torch.cdist(x, y).mean()  # noqa: E731
    return float((2 * mean(a, b) - mean(a, a) - mean(b, b)).clamp(min=0))
