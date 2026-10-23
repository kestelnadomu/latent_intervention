"""psi: a probe that reads a structured state from a latent.

A thin wrapper around ``src.semantic_decoder.SemanticDecoder`` (one categorical head
per column): ``hidden=0`` is a linear probe, ``hidden=k`` an MLP probe with ``k``
hidden layers. Any simulated column can be read, also ones outside S (e.g. a
proxy, motivation ``M`` or the outcome ``Y``).
"""

from __future__ import annotations

import copy

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

    def metrics(self, z: torch.Tensor, states: pd.DataFrame, joint: list[str] | None = None) -> pd.Series:
        """``state_metrics`` of this reader on (z, states)."""
        return state_metrics(self.proba(z), states, joint)

    def _fitted(self) -> SemanticDecoder:
        if self.model is None:
            raise ValueError("call fit() first")
        return self.model


def cross_fit(
    reader: Reader,
    z: torch.Tensor,
    states: pd.DataFrame,
    folds,
    labelled=None,
    seed: int = 0,
) -> tuple[pd.DataFrame, list[Reader]]:
    """Out-of-fold reading: per fold k, a copy of ``reader`` fitted on the ``labelled``
    rows outside k reads all rows of k. Returns the readings (row order of ``z``) and
    one fitted reader per fold."""
    folds = np.asarray(folds)
    labelled = np.ones(len(z), bool) if labelled is None else np.asarray(labelled, bool)
    readings = pd.DataFrame(0, index=range(len(z)), columns=reader.names)
    readers = []
    for k in np.unique(folds):
        train, held_out = labelled & (folds != k), folds == k
        fold_reader = copy.deepcopy(reader).fit(z[torch.as_tensor(train)], states[train], seed=seed)
        readings.loc[held_out] = fold_reader.predict(z[torch.as_tensor(held_out)]).to_numpy()
        readers.append(fold_reader)
    return readings, readers


def mean_proba(readers: list[Reader], z: torch.Tensor) -> dict[str, torch.Tensor]:
    """The readers' averaged psi(s_c | z) per column."""
    probas = [r.proba(z) for r in readers]
    return {c: sum(p[c] for p in probas) / len(probas) for c in readers[0].names}


def mean_predict(readers: list[Reader], z: torch.Tensor) -> pd.DataFrame:
    """The most likely level per column under the readers' averaged probabilities."""
    return pd.DataFrame({c: p.argmax(-1).numpy() for c, p in mean_proba(readers, z).items()})


def state_metrics(
    proba: dict[str, torch.Tensor], states: pd.DataFrame, joint: list[str] | None = None, bins: int = 10
) -> pd.Series:
    """Per column: ``acc_c``, ``nll_c`` and ``ece_c`` (equal-width confidence bins);
    over the ``joint`` columns (default all): ``joint_acc`` (every column right) and
    ``joint_nll`` (sum of the column NLLs, i.e. independent heads)."""
    out, right, nll = {}, [], []
    for c, p in proba.items():
        truth = torch.tensor(states[c].to_numpy(), dtype=torch.long)
        confidence, predicted = p.max(-1)
        correct = (predicted == truth).float()
        column_nll = -p.gather(1, truth[:, None]).squeeze(1).clamp(min=1e-12).log()
        which = (confidence * bins).long().clamp(max=bins - 1)
        ece = sum(
            float((which == b).float().mean() * (confidence[which == b].mean() - correct[which == b].mean()).abs())
            for b in range(bins) if (which == b).any()
        )
        out.update({f"acc_{c}": float(correct.mean()), f"nll_{c}": float(column_nll.mean()), f"ece_{c}": ece})
        if joint is None or c in joint:
            right.append(correct)
            nll.append(column_nll)
    out["joint_acc"] = float(torch.stack(right).prod(0).mean())
    out["joint_nll"] = float(torch.stack(nll).sum(0).mean())
    return pd.Series(out)
