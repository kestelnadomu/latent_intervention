"""
Latent intervention: learn to slightly transform a latent text representation
so that it satisfies a given objective.

The objective is a set of target values for (a subset of) the tabular columns
predicted by the SemanticDecoder. A single-layer transformer reads the latent
vector together with one token per column objective and outputs a residual
update:

    z' = z + delta(z, objective)

Training (train_latent_intervention) samples random objectives, keeps a
pre-trained SemanticDecoder frozen, and minimizes
    cross-entropy(decoder(z')[targeted columns], targets)
    + proximity_weight * ||z' - z||^2
so interventions achieve the objective while staying close to the original
representation.
"""

import random

import torch
import torch.nn.functional as F
from torch import nn

from src.semantic_decoder import SCM_COLUMNS, ColumnSpec, SemanticDecoder


class LatentIntervention(nn.Module):
    """Single-layer transformer mapping (latent, objective) to an intervened latent."""

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
        max_card = max(col.n_categories for col in columns)

        self.z_proj = nn.Linear(latent_dim, d_model)
        self.col_embed = nn.Embedding(len(columns), d_model)
        # Value indices 0..max_card-1 are target values; max_card means "not targeted".
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
            values: (batch, n_columns) long tensor of target category indices
                (entries where mask is False are ignored).
            mask: (batch, n_columns) bool tensor marking targeted columns.

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


def make_objective(
    targets: dict[str, int],
    columns: list[ColumnSpec] = SCM_COLUMNS,
    batch_size: int = 1,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build (values, mask) tensors for a named objective, repeated batch_size times.

    Example: make_objective({"E": 3, "Q": 2}) targets E=3 and Q=2.
    """
    names = [col.name for col in columns]
    unknown = set(targets) - set(names)
    if unknown:
        raise ValueError(f"Unknown columns: {sorted(unknown)}. Available: {names}")
    for col in columns:
        if col.name in targets and not 0 <= targets[col.name] < col.n_categories:
            raise ValueError(
                f"Target {col.name}={targets[col.name]} outside 0..{col.n_categories - 1}"
            )
    values = torch.tensor([targets.get(n, 0) for n in names], dtype=torch.long, device=device)
    mask = torch.tensor([n in targets for n in names], dtype=torch.bool, device=device)
    return values.expand(batch_size, -1).clone(), mask.expand(batch_size, -1).clone()


def sample_objectives(
    batch_size: int,
    columns: list[ColumnSpec] = SCM_COLUMNS,
    p_target: float = 0.3,
    generator: torch.Generator | None = None,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random training objectives: each column targeted with prob p_target (at least one)."""
    n_cols = len(columns)
    mask = torch.rand(batch_size, n_cols, generator=generator, device=device) < p_target
    # Guarantee at least one targeted column per row.
    none_targeted = ~mask.any(dim=1)
    if none_targeted.any():
        forced = torch.randint(n_cols, (int(none_targeted.sum()),), generator=generator, device=device)
        mask[none_targeted, forced] = True
    values = torch.stack(
        [
            torch.randint(col.n_categories, (batch_size,), generator=generator, device=device)
            for col in columns
        ],
        dim=1,
    )
    return values, mask


def intervention_loss(
    model: LatentIntervention,
    decoder: SemanticDecoder,
    z: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    proximity_weight: float = 1.0,
    sparsity_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Objective cross-entropy on targeted columns + proximity penalty to the original latent."""
    z_prime = model(z, values, mask)
    logits = decoder(z_prime)

    objective_terms = []
    for i, col in enumerate(model.columns):
        targeted = mask[:, i]
        if targeted.any():
            objective_terms.append(
                F.cross_entropy(logits[col.name][targeted], values[targeted, i])
            )
    objective_loss = torch.stack(objective_terms).mean()
    proximity_loss = F.mse_loss(z_prime, z)
    sparsity_loss = F.l1_loss(z_prime, z)

    total = objective_loss + proximity_weight * proximity_loss + sparsity_weight * sparsity_loss
    return total, {
        "objective": objective_loss.item(),
        "proximity": proximity_loss.item(),
        "sparsity": sparsity_loss.item(),
        "total": total.item(),
    }


def train_latent_intervention(
    model: LatentIntervention,
    decoder: SemanticDecoder,
    latents: torch.Tensor,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-4,
    p_target: float = 0.3,
    proximity_weight: float = 1.0,
    seed: int | None = None,
    device: str | torch.device | None = None,
    verbose: bool = True,
) -> list[dict[str, float]]:
    """
    Train the intervention model on randomly sampled objectives; the decoder stays frozen.

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

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = latents.shape[0]
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        perm = torch.randperm(n, generator=generator, device=device)
        epoch_logs: list[dict[str, float]] = []
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            values, mask = sample_objectives(
                len(idx), model.columns, p_target, generator=generator, device=device
            )
            loss, logs = intervention_loss(
                model, decoder, latents[idx], values, mask, proximity_weight
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
                f"objective {log['objective']:.4f}  proximity {log['proximity']:.4f}"
            )

    model.eval()
    return history


if __name__ == "__main__":
    # Smoke test on random data: identity at init, then a short training run
    # against a freshly (randomly) initialized frozen decoder.
    torch.manual_seed(0)
    random.seed(0)
    latent_dim, n = 128, 256
    z = torch.randn(n, latent_dim)

    decoder = SemanticDecoder(latent_dim)
    model = LatentIntervention(latent_dim)

    values, mask = make_objective({"E": 3, "Q": 2}, batch_size=n)
    with torch.no_grad():
        assert torch.allclose(model(z, values, mask), z), "expected identity at init"

    history = train_latent_intervention(model, decoder, z, epochs=5, seed=0, verbose=False)
    print(f"total loss {history[0]['total']:.4f} -> {history[-1]['total']:.4f}")
    with torch.no_grad():
        z_prime = model(z, values, mask)
    preds = decoder.predict(z_prime)
    print(
        "objective satisfaction: "
        f"E {(preds['E'] == 3).float().mean():.2f}, Q {(preds['Q'] == 2).float().mean():.2f}, "
        f"mean shift {(z_prime - z).norm(dim=1).mean():.3f}"
    )
