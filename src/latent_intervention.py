"""
Latent intervention: learn to transform a latent text representation
analogously to a symbolic counterfactual operation (the manipulator h_Z).

A single-layer transformer reads the latent vector together with one token per
column of the do() specification and outputs a residual update:

    z' = z + delta(z, do(nodes=values))

Training (train_latent_intervention) keeps a pre-trained SemanticDecoder g
frozen and enforces the paper's consistency constraint h_S(g(z)) = g(h_Z(z)):
the counterfactual targets S' = h_S(S) come from the SCM simulation
(exp/sim/scm.py), and the loss is

    CE(g(z'), S') over all S columns
    + sparsity_weight  * ||z' - z||_1     (alpha, L1)
    + proximity_weight * ||z' - z||_2^2   (beta, L2)

so interventions move descendants where S' says so, keep non-descendants, and
stay close and sparse in latent space.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from src.semantic_decoder import SCM_COLUMNS, ColumnSpec, SemanticDecoder


class LatentIntervention(nn.Module):
    """Single-layer transformer mapping (latent, do() spec) to an intervened latent."""

    def __init__(
        self,
        latent_dim: int,
        columns: list[ColumnSpec] = SCM_COLUMNS,
        d_model: int = 128,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.columns = columns
        self._config = {
            "latent_dim": latent_dim,
            "columns": [(col.name, col.n_categories) for col in columns],
            "d_model": d_model,
            "nhead": nhead,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
        }
        max_card = max(col.n_categories for col in columns)

        self.z_proj = nn.Linear(latent_dim, d_model)
        self.col_embed = nn.Embedding(len(columns), d_model)
        # Value indices 0..max_card-1 are do() values; max_card means "not intervened".
        self.null_value = max_card
        self.val_embed = nn.Embedding(max_card + 1, d_model)

        self.layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.out = nn.Linear(d_model, latent_dim)
        # Zero-init the output so training starts from the identity mapping z' = z.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z: torch.Tensor, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply the intervention.

        Args:
            z: (batch, latent_dim) latent representations.
            values: (batch, n_columns) long tensor of do() category indices
                (entries where mask is False are ignored).
            mask: (batch, n_columns) bool tensor marking intervened columns.

        Returns:
            (batch, latent_dim) intervened latent representations.
        """
        batch = z.shape[0]
        col_idx = torch.arange(len(self.columns), device=z.device).expand(batch, -1)
        val_idx = torch.where(mask, values, torch.full_like(values, self.null_value))

        tokens = torch.cat(
            [
                self.z_proj(z).unsqueeze(1),
                self.col_embed(col_idx) + self.val_embed(val_idx),
            ],
            dim=1,
        )
        h = self.layer(tokens)
        return z + self.out(h[:, 0])

    def save(self, path: str | Path) -> None:
        """Persist constructor config + weights in one file (see LatentIntervention.load)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": self._config, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device | None = None) -> "LatentIntervention":
        """Restore a manipulator saved with save(); returns it in eval mode."""
        payload = torch.load(Path(path), map_location=device or "cpu", weights_only=True)
        config = dict(payload["config"])
        config["columns"] = [ColumnSpec(name, card) for name, card in config["columns"]]
        model = cls(**config)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model


def make_objective(
    intervention: dict[str, int],
    columns: list[ColumnSpec] = SCM_COLUMNS,
    batch_size: int = 1,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build (values, mask) tensors for a do() spec, repeated batch_size times.

    Example: make_objective({"G": 1}) encodes do(G=1).
    """
    names = [col.name for col in columns]
    unknown = set(intervention) - set(names)
    if unknown:
        raise ValueError(f"Unknown columns: {sorted(unknown)}. Available: {names}")
    for col in columns:
        if col.name in intervention and not 0 <= intervention[col.name] < col.n_categories:
            raise ValueError(
                f"do({col.name}={intervention[col.name]}) outside 0..{col.n_categories - 1}"
            )
    values = torch.tensor([intervention.get(n, 0) for n in names], dtype=torch.long, device=device)
    mask = torch.tensor([n in intervention for n in names], dtype=torch.bool, device=device)
    return values.expand(batch_size, -1).clone(), mask.expand(batch_size, -1).clone()


def intervention_loss(
    model: LatentIntervention,
    decoder: SemanticDecoder,
    z: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    s_prime: dict[str, torch.Tensor],
    proximity_weight: float = 1.0,
    sparsity_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Consistency loss against the counterfactual targets S' plus latent-space penalties.

    The consistency term is the decoder's mean cross-entropy over *all* S columns:
    descendants of the intervened node must move to their S' values while
    non-descendants must keep their factual ones.
    """
    z_prime = model(z, values, mask)
    consistency_loss = decoder.loss(decoder(z_prime), s_prime)
    sparsity_loss = F.l1_loss(z_prime, z)
    proximity_loss = F.mse_loss(z_prime, z)

    total = consistency_loss + sparsity_weight * sparsity_loss + proximity_weight * proximity_loss
    return total, {
        "consistency": consistency_loss.item(),
        "sparsity": sparsity_loss.item(),
        "proximity": proximity_loss.item(),
        "total": total.item(),
    }


def train_latent_intervention(
    model: LatentIntervention,
    decoder: SemanticDecoder,
    latents: torch.Tensor,
    intervention: dict[str, int],
    s_prime_targets: dict[str, torch.Tensor],
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-4,
    proximity_weight: float = 1.0,
    sparsity_weight: float = 1.0,
    seed: int | None = None,
    device: str | torch.device | None = None,
    verbose: bool = True,
) -> list[dict[str, float]]:
    """
    Train the manipulator on counterfactual pairs; the decoder stays frozen.

    `latents` are the factual latent representations (n, latent_dim);
    `s_prime_targets` maps each S column to its (n,) counterfactual values
    (see targets_from_dataframe on the counterfactual sim CSV); `intervention`
    is the do() spec those counterfactuals were generated under, e.g. {"G": 1}.
    Returns per-epoch mean loss components.
    """
    device = torch.device(device) if device is not None else torch.device("cpu")
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    model.to(device).train()
    decoder.to(device).eval()
    decoder.requires_grad_(False)
    latents = latents.to(device)
    s_prime_targets = {k: v.to(device) for k, v in s_prime_targets.items()}
    values, mask = make_objective(intervention, model.columns, batch_size=1, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = latents.shape[0]
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        perm = torch.randperm(n, generator=generator, device=device)
        epoch_logs: list[dict[str, float]] = []
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            batch_s_prime = {k: v[idx] for k, v in s_prime_targets.items()}
            loss, logs = intervention_loss(
                model,
                decoder,
                latents[idx],
                values.expand(len(idx), -1),
                mask.expand(len(idx), -1),
                batch_s_prime,
                proximity_weight=proximity_weight,
                sparsity_weight=sparsity_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_logs.append(logs)
        history.append(
            {k: sum(log[k] for log in epoch_logs) / len(epoch_logs) for k in epoch_logs[0]}
        )
        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            log = history[-1]
            print(
                f"epoch {epoch + 1}/{epochs}  total {log['total']:.4f}  "
                f"consistency {log['consistency']:.4f}  "
                f"sparsity {log['sparsity']:.4f}  proximity {log['proximity']:.4f}"
            )

    model.eval()
    return history


if __name__ == "__main__":
    # Smoke test on random data: identity at init, then a short training run
    # against a freshly (randomly) initialized frozen decoder with synthetic
    # counterfactual targets.
    torch.manual_seed(0)
    latent_dim, n = 128, 256
    z = torch.randn(n, latent_dim)

    decoder = SemanticDecoder(latent_dim)
    model = LatentIntervention(latent_dim)

    values, mask = make_objective({"G": 1}, batch_size=n)
    with torch.no_grad():
        assert torch.allclose(model(z, values, mask), z), "expected identity at init"

    s_prime = {col.name: torch.randint(col.n_categories, (n,)) for col in SCM_COLUMNS}
    s_prime["G"] = torch.ones(n, dtype=torch.long)
    history = train_latent_intervention(
        model, decoder, z, {"G": 1}, s_prime, epochs=5, seed=0, verbose=False
    )
    print(f"total loss {history[0]['total']:.4f} -> {history[-1]['total']:.4f}")

    with torch.no_grad():
        z_prime = model(z, values, mask)
    preds = decoder.predict(z_prime)
    consistency = {
        col.name: (preds[col.name] == s_prime[col.name]).float().mean().item()
        for col in SCM_COLUMNS
    }
    print("consistency accuracy:", {k: round(v, 2) for k, v in consistency.items()})
    print(f"mean latent shift: {(z_prime - z).norm(dim=1).mean():.3f}")
