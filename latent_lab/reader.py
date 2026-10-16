"""psi: a probe that reads a structured state from a latent.

A thin wrapper around ``src.semantic_decoder.SemanticDecoder`` (one categorical head
per column): ``hidden=0`` is a linear probe, ``hidden=k`` an MLP probe with ``k``
hidden layers. Any simulated column can be read, also ones outside S (e.g. a
proxy, motivation ``M`` or the outcome ``Y``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.schema import ColumnSpec
from src.semantic_decoder.model import SemanticDecoder
from src.semantic_decoder.training import train_semantic_decoder


class Reader:
    """``Reader(["X", "D"]).fit(z, states)``: one head per column of ``states``.

    Levels are 0..k-1, with k taken from the training states unless ``levels`` gives it.
    """

    def __init__(
        self,
        columns: list[str],
        hidden: int = 0,
        width: int = 256,
        dropout: float = 0.1,
        levels: dict[str, int] | None = None,
    ) -> None:
        self.names = list(columns)
        self.hidden, self.width, self.dropout = hidden, width, dropout
        self.levels = dict(levels or {})
        self.model: SemanticDecoder | None = None

    def __repr__(self) -> str:
        kind = "linear" if self.hidden == 0 else f"mlp{self.hidden}"
        return f"Reader({self.names}, {kind})"

    def fit(
        self,
        z: torch.Tensor,
        states: pd.DataFrame,
        epochs: int = 300,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        validation: float = 0.2,
        seed: int = 0,
    ) -> "Reader":
        """Fit on (z, states); ``validation`` of the rows select the best epoch."""
        columns = [ColumnSpec(c, self.levels.get(c, int(states[c].max()) + 1)) for c in self.names]
        self.model = SemanticDecoder(
            z.shape[1], columns, hidden_dim=self.width, n_hidden=self.hidden, dropout=self.dropout
        )
        targets = {c: torch.tensor(states[c].to_numpy()) for c in self.names}
        order = torch.randperm(len(z), generator=torch.Generator().manual_seed(seed))
        n_val = int(validation * len(z))
        val, train = order[:n_val], order[n_val:]
        train_semantic_decoder(
            self.model,
            z[train],
            {c: t[train] for c, t in targets.items()},
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            seed=seed,
            verbose=False,
            validation_latents=z[val] if n_val else None,
            validation_targets={c: t[val] for c, t in targets.items()} if n_val else None,
        )
        self.model.eval()
        return self

    @torch.no_grad()
    def proba(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """psi(s_c | z) per column, each of shape (n, levels)."""
        return {c: logits.softmax(-1) for c, logits in self._fitted()(z).items()}

    @torch.no_grad()
    def joint(self, z: torch.Tensor) -> torch.Tensor:
        """psi(s | z) over the flat joint state space, shape (n, |S|)."""
        return self._fitted().log_joint(z).exp()

    def predict(self, z: torch.Tensor) -> pd.DataFrame:
        """The most likely level per column."""
        return pd.DataFrame({c: p.argmax(-1).numpy() for c, p in self.proba(z).items()})

    def sample(self, z: torch.Tensor, seed: int = 0) -> pd.DataFrame:
        """One draw per column from psi(. | z)."""
        g = torch.Generator().manual_seed(seed)
        return pd.DataFrame(
            {c: torch.multinomial(p, 1, generator=g).squeeze(1).numpy() for c, p in self.proba(z).items()}
        )

    def accuracy(self, z: torch.Tensor, states: pd.DataFrame) -> pd.Series:
        """Per-column accuracy of the argmax reading against ``states``."""
        predicted = self.predict(z)
        return pd.Series(
            {c: float(np.mean(predicted[c].to_numpy() == states[c].to_numpy())) for c in self.names}
        )

    def _fitted(self) -> SemanticDecoder:
        if self.model is None:
            raise ValueError("call fit() first")
        return self.model
