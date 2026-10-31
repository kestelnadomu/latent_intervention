"""psi: a probe that reads a structured state from a latent.

An MLP with one categorical head per column, so the columns are read conditionally
independently given z: ``hidden=0`` is a linear probe, ``hidden=k`` an MLP probe with
``k`` hidden layers. Any simulated column can be read, also ones outside S (e.g. a
proxy, motivation ``M`` or the outcome ``Y``).
"""

from __future__ import annotations

import copy
import math
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch import nn


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
        activation: str = "gelu",
    ) -> None:
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(_ACTIVATIONS)}, got {activation!r}")
        self.names = list(columns)
        self.hidden, self.width, self.dropout, self.activation = hidden, width, dropout, activation
        self.levels = dict(levels or {})
        self.model: _Probe | None = None

    def __repr__(self) -> str:
        kind = "linear" if self.hidden == 0 else f"mlp{self.hidden}-{self.activation}"
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
        batch_size: int = 64,
        patience: int = 30,
    ) -> "Reader":
        """Minimize the mean per-column cross-entropy with Adam.

        A ``validation`` share of the rows selects the epoch with the lowest joint NLL;
        the learning rate halves after 10 epochs without improvement and training stops
        after ``patience``. The global RNG is left untouched.
        """
        started = perf_counter()
        levels = {c: self.levels.get(c, int(states[c].max()) + 1) for c in self.names}
        targets = torch.tensor(states[self.names].to_numpy(), dtype=torch.long)
        order = torch.randperm(len(z), generator=torch.Generator().manual_seed(seed))
        n_val = int(validation * len(z))
        val, train = order[:n_val], order[n_val:]
        z_train, y_train = z[train], targets[train]
        best, best_state, reference, stale, best_epoch, history = math.inf, None, math.inf, 0, 0, []
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.model = _Probe(z.shape[1], levels, self.width, self.hidden, self.dropout, self.activation)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.5, patience=10, threshold=1e-4, threshold_mode="abs", min_lr=1e-6
            )
            self.model.train()
            for epoch in range(1, epochs + 1):
                lr_now, total = optimizer.param_groups[0]["lr"], 0.0
                for batch in torch.randperm(len(z_train)).split(batch_size):
                    loss = self._nll(z_train[batch], y_train[batch]) / len(self.names)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total += loss.item() * len(self.names) * len(batch)
                history.append({"epoch": epoch, "train": total / len(z_train), "lr": lr_now})
                if not n_val:
                    continue
                self.model.eval()
                with torch.no_grad():
                    val_loss = float(self._nll(z[val], targets[val]))
                self.model.train()
                history[-1]["validation"] = val_loss
                if val_loss < best:
                    best, best_state, best_epoch = val_loss, copy.deepcopy(self.model.state_dict()), epoch
                if val_loss < reference - 1e-4:  # patience counts improvements of at least 1e-4
                    reference, stale = val_loss, 0
                else:
                    stale += 1
                scheduler.step(val_loss)
                if stale >= patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        self._fit_log = fit_log(history, epochs, best_epoch or len(history), perf_counter() - started)
        return self

    @property
    def fit_log(self) -> dict:
        """Training time and convergence of the last ``fit`` (see :func:`fit_log`)."""
        return getattr(self, "_fit_log", {})

    def _nll(self, z: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Mean joint NLL: the sum of the per-column cross-entropies."""
        logits = self.model(z)
        return sum(nn.functional.cross_entropy(logits[c], targets[:, i]) for i, c in enumerate(self.names))

    @torch.no_grad()
    def proba(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """psi(s_c | z) per column, each of shape (n, levels)."""
        return {c: logits.softmax(-1) for c, logits in self._fitted()(z).items()}

    @torch.no_grad()
    def joint(self, z: torch.Tensor) -> torch.Tensor:
        """psi(s | z) over the flat joint state space, shape (n, |S|)."""
        log_p = [logits.log_softmax(-1) for logits in self._fitted()(z).values()]
        joint = log_p[0]
        for column in log_p[1:]:  # the last column varies fastest
            joint = (joint[:, :, None] + column[:, None, :]).reshape(len(z), -1)
        return joint.exp()

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

    def _fitted(self) -> _Probe:
        if self.model is None:
            raise ValueError("call fit() first")
        return self.model


_ACTIVATIONS = {"gelu": nn.GELU, "relu": nn.ReLU}


def fit_log(history: list[dict], max_epochs: int, best_epoch: int, seconds: float) -> dict:
    """``seconds``, ``max_epochs``, ``epochs_run``, ``best_epoch`` (the epoch whose weights
    were kept), ``stopped_early`` and ``history``: one dict per epoch with the training
    loss, the validation loss where there is one, and the learning rate."""
    return {
        "seconds": seconds, "max_epochs": max_epochs, "epochs_run": len(history), "best_epoch": best_epoch,
        "stopped_early": len(history) < max_epochs, "history": history,
    }


class _Probe(nn.Module):
    """``hidden`` x (Linear, activation, Dropout), then one linear head per column."""

    def __init__(
        self, dim: int, levels: dict[str, int], width: int, hidden: int, dropout: float, activation: str = "gelu"
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(hidden):
            layers += [nn.Linear(dim, width), _ACTIVATIONS[activation](), nn.Dropout(dropout)]
            dim = width
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleDict({c: nn.Linear(dim, k) for c, k in levels.items()})

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.trunk(z)
        return {c: head(hidden) for c, head in self.heads.items()}


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


class OutOfFold:
    """The fold readers of :func:`cross_fit` as one reader of the cross-fitted rows:
    row i is read by the reader of fold ``folds[i]``, which never saw it. Reads only
    latents aligned with ``folds`` (e.g. those rows' edited or target latents);
    ``rows(index)`` restricts it to a subset of them."""

    def __init__(self, readers: list[Reader], folds) -> None:
        self.readers, self.folds = list(readers), np.asarray(folds)
        self.labels = np.unique(self.folds)  # readers[i] belongs to fold labels[i], as in cross_fit
        if len(self.labels) != len(self.readers):
            raise ValueError(f"{len(self.readers)} readers for {len(self.labels)} folds")
        self.names = self.readers[0].names

    def __repr__(self) -> str:
        return f"OutOfFold({self.readers[0]!r}, {len(self.readers)} folds)"

    def rows(self, index) -> "OutOfFold":
        """The same readers for the rows ``index`` (positions or a boolean mask) only."""
        subset = copy.copy(self)
        subset.folds = self.folds[np.asarray(index)]
        return subset

    @torch.no_grad()
    def proba(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """psi(s_c | z) per column, row i read by the reader of fold ``folds[i]``."""
        if len(z) != len(self.folds):
            raise ValueError(f"OutOfFold reads its {len(self.folds)} rows, got {len(z)} latents")
        out = None
        for k, reader in zip(self.labels, self.readers):
            rows = torch.as_tensor(self.folds == k)
            if not rows.any():
                continue
            proba = reader.proba(z[rows])
            out = out or {c: torch.zeros(len(z), p.shape[1]) for c, p in proba.items()}
            for c, p in proba.items():
                out[c][rows] = p
        return out

    def predict(self, z: torch.Tensor) -> pd.DataFrame:
        """The most likely level per column."""
        return pd.DataFrame({c: p.argmax(-1).numpy() for c, p in self.proba(z).items()})

    def metrics(self, z: torch.Tensor, states: pd.DataFrame, joint: list[str] | None = None) -> pd.Series:
        """``state_metrics`` of the out-of-fold reading of (z, states)."""
        return state_metrics(self.proba(z), states, joint)


def mean_proba(readers: list[Reader], z: torch.Tensor) -> dict[str, torch.Tensor]:
    """The readers' averaged psi(s_c | z) per column."""
    probas = [r.proba(z) for r in readers]
    return {c: sum(p[c] for p in probas) / len(probas) for c in readers[0].names}


def mean_predict(readers: list[Reader], z: torch.Tensor) -> pd.DataFrame:
    """The most likely level per column under the readers' averaged probabilities."""
    return pd.DataFrame({c: p.argmax(-1).numpy() for c, p in mean_proba(readers, z).items()})


def mean_sample(readers: list[Reader], z: torch.Tensor, seed: int = 0) -> pd.DataFrame:
    """One draw per column from the readers' averaged psi(s_c | z)."""
    g = torch.Generator().manual_seed(seed)
    return pd.DataFrame(
        {c: torch.multinomial(p, 1, generator=g).squeeze(1).numpy() for c, p in mean_proba(readers, z).items()}
    )


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
