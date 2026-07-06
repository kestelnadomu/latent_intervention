"""
Semantic decoder: maps latent text representations to a tabular representation.

The target schema defaults to the SCM simulation variables produced by
src/R (all discrete, small cardinality), each predicted by a categorical head:

    SemanticDecoder: z (latent_dim) -> {column: logits (n_categories)}

Train with train_semantic_decoder on (latents, tabular targets) pairs.
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ColumnSpec:
    """One tabular column: name and number of discrete categories (values 0..n-1)."""

    name: str
    n_categories: int


# Variables of the default SCM in src/R/utils/sim_scm.R with their cardinalities.
SCM_COLUMNS: list[ColumnSpec] = [
    ColumnSpec("R", 4),
    ColumnSpec("G", 2),
    ColumnSpec("A", 3),
    ColumnSpec("E", 4),
    ColumnSpec("S", 3),
    ColumnSpec("W", 3),
    ColumnSpec("V", 2),
    ColumnSpec("C", 2),
    ColumnSpec("Q", 3),
]


class SemanticDecoder(nn.Module):
    """MLP trunk with one categorical classification head per tabular column."""

    def __init__(
        self,
        latent_dim: int,
        columns: list[ColumnSpec] = SCM_COLUMNS,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.columns = columns
        layers: list[nn.Module] = []
        in_dim = latent_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleDict(
            {col.name: nn.Linear(in_dim, col.n_categories) for col in columns}
        )

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return per-column logits: {name: (batch, n_categories)}."""
        h = self.trunk(z)
        return {name: head(h) for name, head in self.heads.items()}

    @torch.no_grad()
    def predict(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return per-column predicted category indices: {name: (batch,)}."""
        return {name: logits.argmax(dim=-1) for name, logits in self(z).items()}

    def loss(
        self,
        logits: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Mean cross-entropy over all columns."""
        losses = [F.cross_entropy(logits[col.name], targets[col.name]) for col in self.columns]
        return torch.stack(losses).mean()


def targets_from_dataframe(
    df: pd.DataFrame, columns: list[ColumnSpec] = SCM_COLUMNS
) -> dict[str, torch.Tensor]:
    """Convert tabular data (e.g. loaded from data/sim/scm/) into per-column target tensors."""
    return {col.name: torch.as_tensor(df[col.name].to_numpy(), dtype=torch.long) for col in columns}


def train_semantic_decoder(
    decoder: SemanticDecoder,
    latents: torch.Tensor,
    targets: dict[str, torch.Tensor],
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str | torch.device | None = None,
    verbose: bool = True,
) -> list[float]:
    """
    Train the decoder on (latent, tabular) pairs; returns per-epoch mean losses.

    `latents` has shape (n, latent_dim); `targets` maps column name to a (n,)
    tensor of category indices (see targets_from_dataframe).
    """
    device = torch.device(device) if device is not None else torch.device("cpu")
    decoder.to(device).train()
    latents = latents.to(device)
    targets = {k: v.to(device) for k, v in targets.items()}

    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    n = latents.shape[0]
    history: list[float] = []

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        epoch_losses = []
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            batch_targets = {k: v[idx] for k, v in targets.items()}
            loss = decoder.loss(decoder(latents[idx]), batch_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        history.append(sum(epoch_losses) / len(epoch_losses))
        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"epoch {epoch + 1}/{epochs}  loss {history[-1]:.4f}")

    decoder.eval()
    return history


def accuracy(decoder: SemanticDecoder, latents: torch.Tensor, targets: dict[str, torch.Tensor]) -> dict[str, float]:
    """Per-column prediction accuracy of the decoder on the given data."""
    preds = decoder.predict(latents)
    return {
        col.name: (preds[col.name].cpu() == targets[col.name].cpu()).float().mean().item()
        for col in decoder.columns
    }


if __name__ == "__main__":
    # Smoke test on random data: shapes and a short training run.
    torch.manual_seed(0)
    latent_dim, n = 128, 256
    z = torch.randn(n, latent_dim)
    targets: dict[str, Any] = {
        col.name: torch.randint(col.n_categories, (n,)) for col in SCM_COLUMNS
    }
    decoder = SemanticDecoder(latent_dim)
    history = train_semantic_decoder(decoder, z, targets, epochs=5, verbose=False)
    print(f"loss {history[0]:.4f} -> {history[-1]:.4f}")
    print("accuracy:", accuracy(decoder, z, targets))
