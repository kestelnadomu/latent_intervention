"""Read -> operate -> render, and the few metrics needed to judge the result."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

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
    """omega_Z(z) = rho(z; s, s') with s ~ psi(. | z) and s' = operator(s).

    ``operator`` is any function from a frame of states to a frame of target states
    (one row each, same columns), deterministic or random. ``reader=None`` uses the
    true factual states passed to ``__call__`` instead (the privileged setting of the
    renderer benchmark).
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
                raise ValueError("without a reader, pass the true factual states")
            s = states[self.renderer.columns].reset_index(drop=True)
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
    """Compare edited latents with the paired counterfactual latents phi(x').

    Latent space, in coordinates divided by ``scale`` (default: the std of ``z``) and
    by sqrt(d):

    - ``gap_closed``: 1 - |z' - phi(x')| / |z - phi(x')|, averaged over units that
      move (1 = perfect, 0 = no better than leaving z alone, < 0 = worse).
    - ``distance``: mean |z' - phi(x')| (the per-unit energy score of a point edit).
    - ``energy``: energy distance between the sets {z'} and {phi(x')}.
    - ``shift``: mean |z' - z|.
    - ``cosine``: mean cosine similarity of z' and phi(x').

    With an (ideally independent) ``reader`` and the true counterfactual states:
    ``acc_<c>`` and ``joint_acc`` (over ``joint``, default all of the reader's columns)
    of its reading of z', next to ``real_*``, the same on phi(x') itself.

    With a downstream ``predict`` (latents -> class) and the true counterfactual
    ``outcome``: ``down_acc`` of predict(z') against it, ``down_agree`` with predict(z),
    and ``parity_gap``, the range over ``groups`` (e.g. the counterfactual X) of the
    mean predicted class; ``real_*`` the same on phi(x').
    """
    from src.latent_intervention.flow.common import multivariate_energy_distance

    scale = z.std(0).clamp(min=1e-6) if scale is None else scale
    root = z.shape[1] ** 0.5
    before = (z - z_true).norm(dim=1)
    after = (z_prime - z_true).norm(dim=1)
    moving = before > 1e-6
    out = {
        "gap_closed": float((1 - after[moving] / before[moving]).mean()),
        "distance": float(((z_prime - z_true) / scale).norm(dim=1).mean() / root),
        "energy": float(multivariate_energy_distance(z_prime[:, None], z_true[:, None], scale)),
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
        factual = predict(z)
        for prefix, latent in (("", z_prime), ("real_", z_true)):
            predicted = predict(latent)
            out[f"{prefix}down_acc"] = float((predicted == y).float().mean())
            out[f"{prefix}down_agree"] = float((predicted == factual).float().mean())
            if groups is not None:
                means = pd.Series(predicted.float().numpy()).groupby(groups.to_numpy()).mean()
                out[f"{prefix}parity_gap"] = float(means.max() - means.min())
    return pd.Series(out)
