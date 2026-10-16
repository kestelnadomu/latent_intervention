"""Read -> operate -> render, and the few metrics needed to judge the result."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
import torch

from latent_lab.reader import Reader
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
) -> pd.Series:
    """Compare edited latents with the paired counterfactual latents phi(x').

    - ``gap_closed``: 1 - |z' - phi(x')| / |z - phi(x')|, averaged over units that
      move (1 = perfect, 0 = no better than leaving z alone, < 0 = worse).
    - ``cosine``: mean cosine similarity of z' and phi(x').
    - ``acc_<c>``: an (ideally independent) reader's accuracy on z' against the true
      counterfactual states, next to ``real_acc_<c>``, its accuracy on phi(x') itself.
    """
    before = (z - z_true).norm(dim=1)
    after = (z_prime - z_true).norm(dim=1)
    moving = before > 1e-6
    out = {
        "gap_closed": float((1 - after[moving] / before[moving]).mean()),
        "cosine": float(torch.nn.functional.cosine_similarity(z_prime, z_true).mean()),
    }
    if reader is not None and true_states is not None:
        for column, acc in reader.accuracy(z_prime, true_states).items():
            out[f"acc_{column}"] = acc
        for column, acc in reader.accuracy(z_true, true_states).items():
            out[f"real_acc_{column}"] = acc
    return pd.Series(out)
