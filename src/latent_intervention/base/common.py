"""Shared building blocks and training loop of the base h_Z family.

`_ManipulatorBase` (config + persistence) and `_train` (the generic loop) are used by
every plan; each plan module supplies only its model body and its per-batch loss.
"""

import math
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from src.schema import ColumnSpec
from src.semantic_decoder.model import SemanticDecoder, SemanticDecoderModel
from src.symbolic_intervention import SymbolicKernel

__all__ = ["make_objective", "consistency_target"]



class _DoSpecTokens(nn.Module):
    """Embed a do() spec (or a full structured state) as one token per column."""

    def __init__(self, columns: list[ColumnSpec], d_model: int) -> None:
        super().__init__()
        self.max_card = max(col.n_categories for col in columns)
        self.null_value = self.max_card  # index for "not intervened"
        self.col_embed = nn.Embedding(len(columns), d_model)
        self.val_embed = nn.Embedding(self.max_card + 1, d_model)
        self.register_buffer("col_idx", torch.arange(len(columns)), persistent=False)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(batch, n_cols) values + mask -> (batch, n_cols, d_model)."""
        val_idx = torch.where(mask, values, torch.full_like(values, self.null_value))
        cols = self.col_idx.expand(values.shape[0], -1)
        return self.col_embed(cols) + self.val_embed(val_idx)


class _DeltaNet(nn.Module):
    """[vec token, (noise token,) condition tokens] -> single-layer transformer -> zero-init residual.

    With `noise_dim > 0` an extra token carries an outsourced noise vector, kept separate
    from the latent so the conditioning information reaches the network intact.
    """

    def __init__(
        self,
        latent_dim: int,
        columns: list[ColumnSpec],
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        noise_dim: int = 0,
    ) -> None:
        super().__init__()
        self.vec_proj = nn.Linear(latent_dim, d_model)
        self.noise_proj = nn.Linear(noise_dim, d_model) if noise_dim > 0 else None
        self.cond = _DoSpecTokens(columns, d_model)
        self.layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.out = nn.Linear(d_model, latent_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)  # identity mapping at init

    def forward(
        self,
        vec: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (noise is None) != (self.noise_proj is None):
            raise ValueError("pass `noise` exactly when the net was built with noise_dim > 0")
        tokens = [self.vec_proj(vec).unsqueeze(1)]
        if noise is not None:
            tokens.append(self.noise_proj(noise).unsqueeze(1))
        tokens.append(self.cond(values, mask))
        return self.out(self.layer(torch.cat(tokens, dim=1))[:, 0])


def _mc_mixture(
    zs: torch.Tensor, decoder: SemanticDecoder
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Monte-Carlo composition over latent samples zs (M, batch, latent_dim).

    Returns log (g . h_Z) as the log-mean-exp of g's dense joints, (batch, |S|), and the
    per-sample joints themselves, (M, batch, |S|).
    """
    log_joints = torch.stack([decoder.log_joint(z_m) for z_m in zs])
    return torch.logsumexp(log_joints, dim=0) - math.log(zs.shape[0]), log_joints


def _forward_kl(pred_log: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """D_KL(target || pred), mean over the batch. `pred_log` are log-probabilities."""
    return F.kl_div(pred_log, target, reduction="batchmean")


def _penalties(
    shifts: torch.Tensor, l1_weight: float, l2_weight: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Latent-shift penalties: (sparsity, proximity, l1_weight*sparsity + l2_weight*proximity)."""
    sparsity = shifts.abs().mean()
    proximity = shifts.pow(2).mean()
    return sparsity, proximity, l1_weight * sparsity + l2_weight * proximity


# --- config + persistence, shared by every plan --------------------------------------


class _ManipulatorBase(nn.Module):
    """Holds `columns`, the reconstruction `_config`, and one-file save/load."""

    def __init__(
        self,
        latent_dim: int,
        columns: list[ColumnSpec],
        *,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        **extra: object,
    ) -> None:
        super().__init__()
        self.columns = list(columns)
        self._config = {
            "latent_dim": latent_dim,
            "columns": [(c.name, c.n_categories) for c in self.columns],
            "d_model": d_model,
            "nhead": nhead,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            **extra,
        }

    def save(self, path: str | Path) -> None:
        """Persist constructor config + weights in one file (see load)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"class": type(self).__name__, "config": self._config, "state_dict": self.state_dict()},
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device | None = None):
        """Restore a manipulator saved with save(); returns it in eval mode."""
        target_device = torch.device(device or "cpu")
        payload = torch.load(Path(path), map_location=target_device, weights_only=True)
        if payload.get("class", cls.__name__) != cls.__name__:
            raise ValueError(f"checkpoint holds a {payload['class']}, not a {cls.__name__}")
        config = dict(payload["config"])
        config["columns"] = [ColumnSpec(name, card) for name, card in config["columns"]]
        model = cls(**config).to(target_device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model


def make_objective(
    intervention: dict[str, int],
    columns: list[ColumnSpec],
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


@torch.no_grad()
def consistency_target(
    decoder: SemanticDecoderModel,
    h_s: SymbolicKernel,
    latents: torch.Tensor,
    intervention: dict[str, int],
    chunk: int = 512,
) -> torch.Tensor:
    """
    Dense (n, |S|) target (h_S . g)(. | z, delta): push g's joint through M_delta.

    Frozen decoder + closed-form h_S, so this is precomputed once before training.
    """
    m_t = h_s.transition_matrix(intervention).t().coalesce()
    out = []
    for start in range(0, latents.shape[0], chunk):
        g_probs = decoder.log_joint(latents[start : start + chunk]).exp()
        out.append(torch.sparse.mm(m_t, g_probs.t()).t())
    return torch.cat(out, dim=0)


# --- shared training loop -----------------------------------------------------------


# batch_loss(phase, ctx, generator, z, values, mask, idx) -> (loss, log-dict)
_BatchLoss = Callable[
    [str, dict, torch.Generator, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, dict[str, float]],
]


def _train(
    model: _ManipulatorBase,
    decoder: SemanticDecoderModel,
    latents: torch.Tensor,
    intervention: dict[str, int],
    *,
    phases: list[tuple[str, int]],
    batch_size: int,
    lr: float,
    seed: int | None,
    device: str | torch.device | None,
    verbose: bool,
    setup: Callable[[torch.device, torch.Generator, torch.Tensor], dict],
    batch_loss: _BatchLoss,
    epoch_callback: Callable | None = None,
    grad_clip: float | None = None,
) -> list[dict[str, float]]:
    """
    Drive minibatch training over `phases` (name, n_epochs); the decoder stays frozen.

    `setup` precomputes per-run context (targets, aligned tensors) once the device is
    known; `batch_loss` computes the loss for one minibatch given that context.
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
    values, mask = make_objective(intervention, model.columns, batch_size=1, device=device)
    ctx = setup(device, generator, latents) or {}

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = latents.shape[0]
    multi = len(phases) > 1
    history: list[dict[str, float]] = []

    for phase, n_epochs in phases:
        for epoch in range(n_epochs):
            perm = torch.randperm(n, generator=generator, device=device)
            logs: list[dict[str, float]] = []
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                v, m = values.expand(len(idx), -1), mask.expand(len(idx), -1)
                loss, log = batch_loss(phase, ctx, generator, latents[idx], v, m, idx)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite latent-intervention loss")
                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(
                        model.parameters(), grad_clip, error_if_nonfinite=True
                    )
                optimizer.step()
                logs.append(log)
            row = {k: sum(x[k] for x in logs) / len(logs) for k in logs[0]}
            history.append(row)
            if verbose and (epoch + 1) % max(1, n_epochs // (5 if multi else 10)) == 0:
                tag = f"[{phase}] " if multi else ""
                body = "  ".join(f"{k} {val:.4f}" for k, val in row.items() if k != "phase")
                print(f"{tag}epoch {epoch + 1}/{n_epochs}  {body}")
            # Optional benchmark control; the ordinary pipeline keeps its fixed-epoch API.
            if epoch_callback is not None and epoch_callback(
                model, optimizer, phase, row
            ):
                model.eval()
                return history

    model.eval()
    return history
