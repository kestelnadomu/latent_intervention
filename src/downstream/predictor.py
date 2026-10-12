"""A plain MLP outcome classifier p: Z -> Y, trained on factual latents only.

Generic: the outcome cardinality and any group labels are supplied by the caller.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.artifact_io import save_torch

OUTCOME_PREDICTOR_FORMAT = "outcome_predictor_v1"


class OutcomePredictor(nn.Module):
    """MLP returning class logits for a categorical outcome."""

    def __init__(
        self,
        latent_dim: int,
        n_classes: int,
        hidden_dim: int = 128,
        n_hidden: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if latent_dim < 1 or n_classes < 2 or hidden_dim < 1 or n_hidden < 0:
            raise ValueError("invalid outcome predictor dimensions")
        self.config = dict(
            latent_dim=latent_dim,
            n_classes=n_classes,
            hidden_dim=hidden_dim,
            n_hidden=n_hidden,
            dropout=dropout,
        )
        layers: list[nn.Module] = []
        width = latent_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(width, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            width = hidden_dim
        layers.append(nn.Linear(width, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    @torch.no_grad()
    def probabilities(self, z: torch.Tensor) -> torch.Tensor:
        """Class probabilities; works for any leading batch shape."""
        return self.forward(z).softmax(-1)


@torch.no_grad()
def _mean_ce(model: OutcomePredictor, z: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    return float(F.cross_entropy(model(z), y))


def train_outcome_predictor(
    model: OutcomePredictor,
    z_fit: torch.Tensor,
    y_fit: torch.Tensor,
    z_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    lr: float = 1e-3,
    batch_size: int = 64,
    max_epochs: int = 300,
    patience: int = 20,
    seed: int = 42,
) -> dict[str, Any]:
    """Minimize cross-entropy; early-stop on validation CE and restore the best weights."""
    if len(z_fit) != len(y_fit) or len(z_val) != len(y_val) or not len(z_val):
        raise ValueError("fit/validation latents and labels must align and be nonempty")
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_epoch, best_state = _mean_ce(model, z_val, y_val), 0, copy.deepcopy(
        model.state_dict()
    )
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(z_fit), generator=generator)
        total = 0.0
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            loss = F.cross_entropy(model(z_fit[batch]), y_fit[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        validation = _mean_ce(model, z_val, y_val)
        history.append(dict(epoch=epoch, train_ce=total / len(order), val_ce=validation))
        if validation < best:
            best, best_epoch, best_state = validation, epoch, copy.deepcopy(
                model.state_dict()
            )
        elif epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return dict(best_epoch=best_epoch, best_validation_ce=best, history=history)


def save_outcome_predictor(
    model: OutcomePredictor,
    path: str | Path,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Atomically write constructor config, weights and caller provenance."""
    save_torch(
        path,
        dict(
            format=OUTCOME_PREDICTOR_FORMAT,
            config=model.config,
            metadata=dict(metadata or {}),
            state_dict=model.state_dict(),
        ),
    )


def load_outcome_predictor(path: str | Path) -> OutcomePredictor:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != OUTCOME_PREDICTOR_FORMAT:
        raise ValueError(f"not an outcome predictor checkpoint: {path}")
    model = OutcomePredictor(**payload["config"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.metadata = payload["metadata"]
    return model.eval()


def confusion_matrix(
    pred: torch.Tensor, true: torch.Tensor, n_classes: int
) -> torch.Tensor:
    """Counts with rows = true class, columns = predicted class."""
    if pred.shape != true.shape:
        raise ValueError("predictions and labels must have the same shape")
    flat = true.long() * n_classes + pred.long()
    return torch.bincount(flat, minlength=n_classes * n_classes).reshape(
        n_classes, n_classes
    )


def grouped_confusion(
    pred: torch.Tensor,
    true: torch.Tensor,
    groups: torch.Tensor,
    n_classes: int,
    *,
    group_name: str = "group",
    n_groups: int | None = None,
) -> dict[str, list[list[int]]]:
    """Overall matrix (``"all"``) plus one per group level (``"<group_name>=<k>"``)."""
    n_groups = int(groups.max()) + 1 if n_groups is None else n_groups
    result = {"all": confusion_matrix(pred, true, n_classes).tolist()}
    for k in range(n_groups):
        mask = groups == k
        result[f"{group_name}={k}"] = confusion_matrix(
            pred[mask], true[mask], n_classes
        ).tolist()
    return result
