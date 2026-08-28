"""
Semantic kernel g: maps a frozen latent z to a distribution over the structured state S.

    g(s | z): Z -> Delta(S)

Two parameterisations (see docs/architecture/semantic_decoder.md):

- SemanticDecoderA -- Plan A: shared MLP trunk + one independent categorical head per
  column, p(s | z) = prod_i p(s_i | z). Asserts the columns conditionally independent
  given z (technically false); the smallest thing that runs end-to-end.
- SemanticDecoderB -- Plan B: same trunk, each head additionally conditioned on an
  embedding of the already-decoded prefix, p(s | z) = prod_i p(s_i | s_<i, z). Exact
  joint, no independence assumption.

Both are trained with train_semantic_decoder on (latents, tabular targets) pairs under
the cross-entropy objective L_g = E[CE(g(. | z), s)].
"""

from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from src.schema import ColumnSpec


def _mlp_trunk(latent_dim: int, hidden_dim: int, n_hidden: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_dim = latent_dim
    for _ in range(n_hidden):
        layers += [nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        in_dim = hidden_dim
    return nn.Sequential(*layers)


class SemanticDecoder(nn.Module):
    """Parent class of Plan A and Plan B: MLP trunk + one independent categorical head per column."""
    def __init__(
            self,
            latent_dim: int,
            columns: list[ColumnSpec],
            hidden_dim: int = 256,
            n_hidden: int = 2,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.columns = list(columns)
            self._config = {
                "latent_dim": latent_dim,
                "columns": [(c.name, c.n_categories) for c in self.columns],
                "hidden_dim": hidden_dim,
                "n_hidden": n_hidden,
                "dropout": dropout,
            }
            self.trunk = _mlp_trunk(latent_dim, hidden_dim, n_hidden, dropout)
            d = hidden_dim if n_hidden > 0 else latent_dim
            self.heads = nn.ModuleDict(
                {c.name: nn.Linear(d, c.n_categories) for c in self.columns}
            )
    
    
    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return per-column logits: {name: (batch, n_categories)}."""
        h = self.trunk(z)
        return {name: head(h) for name, head in self.heads.items()}

    def log_joint(self, z: torch.Tensor) -> torch.Tensor:
        """Dense (batch, |S|) log-probabilities via outer sum of per-column log-softmaxes.

        Column-major (kron) order over ``self.columns``: index
        ``(((s_0 * k_1) + s_1) * k_2 + ...)``.
        """
        logps = [F.log_softmax(logit, dim=-1) for logit in self.forward(z).values()]
        acc = logps[0]
        for lp in logps[1:]:
            acc = (acc[..., None] + lp[:, None, :]).reshape(acc.shape[0], -1)
        return acc

    def nll(self, z: torch.Tensor, targets: dict[str, torch.Tensor]) -> torch.Tensor:
        """Mean cross-entropy over all columns (the L_g objective)."""
        logits = self.forward(z)
        losses = [F.cross_entropy(logits[c.name], targets[c.name]) for c in self.columns]
        return torch.stack(losses).mean()

    @torch.no_grad()
    def predict(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return per-column predicted category indices: {name: (batch,)}."""
        return {name: logits.argmax(dim=-1) for name, logits in self.forward(z).items()}

    def save(self, path: str | Path) -> None:
        _save(self, path)

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device | None = None) -> "SemanticDecoder":
        return _load(cls, path, device)


class SemanticAutoRegDecoder(SemanticDecoder):
    """Plan B: same trunk, heads conditioned on an embedding of the decoded prefix."""

    def __init__(
            self,
            latent_dim: int,
            columns: list[ColumnSpec],
            hidden_dim: int = 256,
            n_hidden: int = 2,
            dropout: float = 0.1,
            embed_dim: int = 16,
        ) -> None:
            super().__init__(
                latent_dim,
                columns,
                hidden_dim,
                n_hidden,
                dropout
            )
            self._config["embed_dim"] = embed_dim
            d = hidden_dim if n_hidden > 0 else latent_dim
            self.embed = nn.ModuleDict(
                {c.name: nn.Embedding(c.n_categories, embed_dim) for c in self.columns}
            )
            self.heads = nn.ModuleDict(
                {c.name: nn.Linear(d + i * embed_dim, c.n_categories) for i, c in enumerate(self.columns)}
            )

    def _prefix_logits(self, h: torch.Tensor, prefix: list[torch.Tensor], name: str) -> torch.Tensor:
        return self.heads[name](torch.cat([h, *prefix], dim=-1))

    def log_prob(self, z: torch.Tensor, s: dict[str, torch.Tensor]) -> torch.Tensor:
        """Sum of per-column log p(s_i | s_<i, z) for observed rows s: (batch,)."""
        h = self.trunk(z)
        prefix: list[torch.Tensor] = []
        total = torch.zeros(z.shape[0], device=z.device)
        for c in self.columns:
            logits = self._prefix_logits(h, prefix, c.name)
            lp = F.log_softmax(logits, dim=-1).gather(-1, s[c.name][:, None]).squeeze(-1)
            total = total + lp
            prefix.append(self.embed[c.name](s[c.name]))
        return total

    def nll(self, z: torch.Tensor, targets: dict[str, torch.Tensor]) -> torch.Tensor:
        """Mean negative log-likelihood -E[log p(s | z)] (the L_g objective)."""
        return -self.log_prob(z, targets).mean()

    @torch.no_grad()
    def predict(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Greedy sequential argmax decode: {name: (batch,)}."""
        h = self.trunk(z)
        prefix: list[torch.Tensor] = []
        out: dict[str, torch.Tensor] = {}
        for c in self.columns:
            pred = self._prefix_logits(h, prefix, c.name).argmax(dim=-1)
            out[c.name] = pred
            prefix.append(self.embed[c.name](pred))
        return out

    def log_joint(self, z: torch.Tensor) -> torch.Tensor:
        """Dense (batch, |S|) log-probabilities by enumerating the autoregressive product.

        Column-major order over ``self.columns``, matching SemanticDecoderA.log_joint.
        """
        h = self.trunk(z)
        batch = z.shape[0]
        # acc: (batch, n_states_so_far); prefix_embed: (batch, n_states_so_far, sum embed dims)
        acc = torch.zeros(batch, 1, device=z.device)
        prefix_embed = torch.zeros(batch, 1, 0, device=z.device)
        for c in self.columns:
            n_states = acc.shape[1]
            h_exp = h[:, None, :].expand(batch, n_states, -1).reshape(batch * n_states, -1)
            pe = prefix_embed.reshape(batch * n_states, -1)
            logits = self.heads[c.name](torch.cat([h_exp, pe], dim=-1))
            lp = F.log_softmax(logits, dim=-1).reshape(batch, n_states, c.n_categories)
            acc = (acc[..., None] + lp).reshape(batch, n_states * c.n_categories)
            idx = torch.arange(c.n_categories, device=z.device)
            emb = self.embed[c.name](idx)  # (k, embed_dim)
            prefix_embed = torch.cat(
                [
                    prefix_embed[:, :, None, :].expand(batch, n_states, c.n_categories, -1),
                    emb[None, None, :, :].expand(batch, n_states, c.n_categories, -1),
                ],
                dim=-1,
            ).reshape(batch, n_states * c.n_categories, -1)
        return acc


def _save(model: nn.Module, path: str | Path) -> None:
    """Persist constructor config + weights + class name in one file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"class": type(model).__name__, "config": model._config, "state_dict": model.state_dict()},
        path,
    )


def _load(cls: type, path: str | Path, device: str | torch.device | None) -> Any:
    payload = torch.load(Path(path), map_location=device or "cpu", weights_only=True)
    saved = payload.get("class", cls.__name__)
    if saved != cls.__name__:
        raise ValueError(f"checkpoint holds a {saved}, not a {cls.__name__}")
    config = dict(payload["config"])
    config["columns"] = [ColumnSpec(name, card) for name, card in config["columns"]]
    model = cls(**config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def targets_from_dataframe(
    df: pd.DataFrame, columns: list[ColumnSpec]
) -> dict[str, torch.Tensor]:
    """Convert tabular data (e.g. loaded from data/sim/) into per-column target tensors."""
    return {c.name: torch.as_tensor(df[c.name].to_numpy(), dtype=torch.long) for c in columns}


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
    tensor of category indices (see targets_from_dataframe). Works for either
    parameterisation via `decoder.nll`.
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
            loss = decoder.nll(latents[idx], batch_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        history.append(sum(epoch_losses) / len(epoch_losses))
        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"epoch {epoch + 1}/{epochs}  loss {history[-1]:.4f}")

    decoder.eval()
    return history


def accuracy(
    decoder: SemanticDecoder,
    latents: torch.Tensor,
    targets: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Per-column prediction accuracy of the decoder on the given data."""
    preds = decoder.predict(latents)
    return {
        c.name: (preds[c.name].cpu() == targets[c.name].cpu()).float().mean().item()
        for c in decoder.columns
    }


if __name__ == "__main__":
    # Smoke test on random data: shapes and a short training run for both parameterisations.
    torch.manual_seed(0)
    latent_dim, n = 128, 256
    columns = [ColumnSpec(f"c{i}", int(k)) for i, k in enumerate(torch.randint(2, 5, (6,)))]
    z = torch.randn(n, latent_dim)
    targets: dict[str, Any] = {
        c.name: torch.randint(c.n_categories, (n,)) for c in columns
    }
    n_states = 1
    for c in columns:
        n_states *= c.n_categories

    for name, ctor in (("A", SemanticDecoder), ("B", SemanticAutoRegDecoder)):
        decoder = ctor(latent_dim, columns)
        history = train_semantic_decoder(decoder, z, targets, epochs=5, verbose=False)
        joint = decoder.log_joint(z[:4])
        assert joint.shape == (4, n_states), joint.shape
        assert torch.allclose(joint.exp().sum(-1), torch.ones(4), atol=1e-4)
        print(f"[{name}] loss {history[0]:.4f} -> {history[-1]:.4f}")
        print(f"[{name}] accuracy:", accuracy(decoder, z, targets))
