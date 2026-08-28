"""
Latent editor h_Z(z' | z, delta): Z -> Delta(Z).

Freeze f, g, h_S; train h_Z so the two paths from z to a distribution over counterfactual
states agree (h_S . g == g . h_Z). See docs/architecture/latent_intervention.md.

Three parameterisations:

- LatentIntervention  -- Plan 0 (baseline): deterministic residual transformer
  z' = z + T_theta(z, delta). Cannot represent the multimodality of (h_S . g) in the
  ambiguous strata; kept as a baseline, trained by per-column CE against S'.
- LatentInterventionPreAdditive -- Plan A (engression): z' = z + Delta_theta(z + eps, delta),
  eps ~ N(0, sigma^2 I). h_Z is the pushforward of eps; a nonlinear Delta_theta folds a
  unimodal eps onto separated modes. Composition g . h_Z needs Monte Carlo.
- LatentInterventionDist -- Plan B (discrete mixture):
  h_Z(. | z, delta) = sum_s' w_theta(s' | z, delta) * dirac_{z + Delta_phi(z, s')}.
  w_theta is an autoregressive head over S (the SCM internalised, no h_S at inference);
  Delta_phi is a deterministic realiser. Composition is an exact finite sum over the
  retained top-k s'. Trained pretrain-then-joint.

The consistency objective (Plan A/B) is
    L = E_z[ D_KL( (h_S . g)(. | z, delta) || (g . h_Z)(. | z, delta) ) ]
        + alpha ||z' - z||_1 + beta ||z' - z||_2^2
with forward (mass-covering) KL over the dense |S| = prod(cardinalities) vector.

Common scaffolding (config + persistence, the training loop, the latent-space penalties)
lives in `_ManipulatorBase` and `_train`; each plan supplies only its model body and its
per-batch loss.
"""

import math
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from src.schema import ColumnSpec, flat_state_index, unflatten_state_index
from src.semantic_decoder import SemanticDecoder
from src.symbolic_intervention import SymbolicKernel

__all__ = [
    "flat_state_index",
    "unflatten_state_index",
    "LatentIntervention",
    "LatentInterventionPreAdditive",
    "LatentInterventionDist",
    "make_objective",
    "consistency_target",
    "train_latent_intervention",
    "train_latent_intervention_preadditive",
    "train_latent_intervention_dist",
]


# --- shared building blocks -------------------------------------------------------------


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
    """[vec token, condition tokens] -> single-layer transformer -> zero-init residual."""

    def __init__(
        self,
        latent_dim: int,
        columns: list[ColumnSpec],
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.vec_proj = nn.Linear(latent_dim, d_model)
        self.cond = _DoSpecTokens(columns, d_model)
        self.layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.out = nn.Linear(d_model, latent_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)  # identity mapping at init

    def forward(self, vec: torch.Tensor, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tokens = torch.cat([self.vec_proj(vec).unsqueeze(1), self.cond(values, mask)], dim=1)
        return self.out(self.layer(tokens)[:, 0])


def _autoreg_log_joint(
    context: torch.Tensor,
    columns: list[ColumnSpec],
    prefix_embed: nn.ModuleDict,
    heads: nn.ModuleDict,
) -> torch.Tensor:
    """Dense (batch, |S|) log-probabilities of an autoregressive product over `columns`.

    Each head sees `context` plus embeddings of the already-decoded prefix; states are
    enumerated in column-major order (matches SemanticDecoder.log_joint).
    """
    batch = context.shape[0]
    acc = context.new_zeros(batch, 1)
    prefix = context.new_zeros(batch, 1, 0)
    for col in columns:
        n_states = acc.shape[1]
        ctx = context[:, None, :].expand(batch, n_states, -1).reshape(batch * n_states, -1)
        logits = heads[col.name](torch.cat([ctx, prefix.reshape(batch * n_states, -1)], dim=-1))
        lp = F.log_softmax(logits, dim=-1).reshape(batch, n_states, col.n_categories)
        acc = (acc[..., None] + lp).reshape(batch, n_states * col.n_categories)
        emb = prefix_embed[col.name](torch.arange(col.n_categories, device=context.device))
        prefix = torch.cat(
            [
                prefix[:, :, None, :].expand(batch, n_states, col.n_categories, -1),
                emb[None, None].expand(batch, n_states, col.n_categories, -1),
            ],
            dim=-1,
        ).reshape(batch, n_states * col.n_categories, -1)
    return acc


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
        payload = torch.load(Path(path), map_location=device or "cpu", weights_only=True)
        if payload.get("class", cls.__name__) != cls.__name__:
            raise ValueError(f"checkpoint holds a {payload['class']}, not a {cls.__name__}")
        config = dict(payload["config"])
        config["columns"] = [ColumnSpec(name, card) for name, card in config["columns"]]
        model = cls(**config)
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
    decoder: SemanticDecoder,
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
    decoder: SemanticDecoder,
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
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                logs.append(log)
            row = {k: sum(x[k] for x in logs) / len(logs) for k in logs[0]}
            history.append(row)
            if verbose and (epoch + 1) % max(1, n_epochs // (5 if multi else 10)) == 0:
                tag = f"[{phase}] " if multi else ""
                body = "  ".join(f"{k} {val:.4f}" for k, val in row.items() if k != "phase")
                print(f"{tag}epoch {epoch + 1}/{n_epochs}  {body}")

    model.eval()
    return history


# --- Plan 0: deterministic baseline ----------------------------------------------------


class LatentIntervention(_ManipulatorBase):
    """Single-layer transformer mapping (latent, do() spec) to an intervened latent."""

    def __init__(
        self,
        latent_dim: int,
        columns: list[ColumnSpec],
        d_model: int = 128,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            latent_dim,
            columns,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.delta = _DeltaNet(latent_dim, self.columns, d_model, nhead, dim_feedforward, dropout)

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
        return z + self.delta(z, values, mask)


def train_latent_intervention(
    model: LatentIntervention,
    decoder: SemanticDecoder,
    latents: torch.Tensor,
    intervention: dict[str, int],
    s_prime: dict[str, torch.Tensor] | None = None,
    *,
    h_s: SymbolicKernel | None = None,  # unused; accepted for a uniform call site
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
    Train the baseline manipulator on counterfactual pairs; the decoder stays frozen.

    `latents` are the factual latent representations (n, latent_dim); `s_prime` maps each
    S column to its (n,) counterfactual values (see targets_from_dataframe on the
    counterfactual sim CSV); `intervention` is the do() spec those counterfactuals were
    generated under, e.g. {"G": 1}. The consistency term is the decoder's mean
    cross-entropy over *all* S columns: descendants of the intervened node must move to
    their S' values while non-descendants keep their factual ones.
    """
    if s_prime is None:
        raise ValueError("baseline training needs the counterfactual targets `s_prime`")

    def setup(device: torch.device, _gen: torch.Generator, _z: torch.Tensor) -> dict:
        return {"s": {k: v.to(device) for k, v in s_prime.items()}}

    def batch_loss(_phase, ctx, _gen, z, v, m, idx):
        z_prime = model(z, v, m)
        consistency = decoder.nll(z_prime, {k: val[idx] for k, val in ctx["s"].items()})
        sparsity, proximity, penalty = _penalties(z_prime - z, sparsity_weight, proximity_weight)
        total = consistency + penalty
        return total, {
            "total": total.item(),
            "consistency": consistency.item(),
            "sparsity": sparsity.item(),
            "proximity": proximity.item(),
        }

    return _train(
        model,
        decoder,
        latents,
        intervention,
        phases=[("train", epochs)],
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        device=device,
        verbose=verbose,
        setup=setup,
        batch_loss=batch_loss,
    )


# --- Plan A: pre-additive noise (engression) -----------------------------------------


class LatentInterventionPreAdditive(_ManipulatorBase):
    """z' = z + Delta_theta(z + eps, delta), eps ~ N(0, noise_std^2 I)."""

    def __init__(
        self,
        latent_dim: int,
        columns: list[ColumnSpec],
        noise_std: float = 1.0,
        d_model: int = 128,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            latent_dim,
            columns,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            noise_std=noise_std,
        )
        self.noise_std = noise_std
        self.delta = _DeltaNet(latent_dim, self.columns, d_model, nhead, dim_feedforward, dropout)

    def forward(
        self,
        z: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """One sample z' ~ h_Z(. | z, delta)."""
        eps = self.noise_std * torch.randn(z.shape, device=z.device, generator=generator)
        return z + self.delta(z + eps, values, mask)

    def sample(
        self,
        z: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        n_samples: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """(n_samples, batch, latent_dim) draws from h_Z."""
        return torch.stack(
            [self.forward(z, values, mask, generator) for _ in range(n_samples)]
        )

    def composed_log_joint(
        self,
        z: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        decoder: SemanticDecoder,
        n_samples: int,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Monte-Carlo estimate of log (g . h_Z)(. | z, delta), shape (batch, |S|),
        plus the sampled latent shifts (n_samples, batch, latent_dim) for penalties.
        """
        zs = self.sample(z, values, mask, n_samples, generator)
        log_joints = torch.stack([decoder.log_joint(z_m) for z_m in zs])  # (M, batch, |S|)
        composed = torch.logsumexp(log_joints, dim=0) - math.log(n_samples)
        return composed, zs - z


def train_latent_intervention_preadditive(
    model: LatentInterventionPreAdditive,
    decoder: SemanticDecoder,
    latents: torch.Tensor,
    intervention: dict[str, int],
    s_prime: dict[str, torch.Tensor] | None = None,  # unused; uniform call site
    *,
    h_s: SymbolicKernel | None = None,
    n_samples: int = 8,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-4,
    proximity_weight: float = 1.0,
    sparsity_weight: float = 1.0,
    seed: int | None = None,
    device: str | torch.device | None = None,
    verbose: bool = True,
) -> list[dict[str, float]]:
    """Train Plan A on the forward-KL consistency objective (decoder + h_S frozen)."""
    if h_s is None:
        raise ValueError("pre-additive training needs the symbolic kernel `h_s`")

    def setup(device: torch.device, _gen: torch.Generator, z: torch.Tensor) -> dict:
        return {"target": consistency_target(decoder, h_s, z, intervention).to(device)}

    def batch_loss(_phase, ctx, generator, z, v, m, idx):
        composed, shifts = model.composed_log_joint(z, v, m, decoder, n_samples, generator)
        kl = _forward_kl(composed, ctx["target"][idx])
        sparsity, proximity, penalty = _penalties(shifts, sparsity_weight, proximity_weight)
        total = kl + penalty
        return total, {
            "total": total.item(),
            "kl": kl.item(),
            "sparsity": sparsity.item(),
            "proximity": proximity.item(),
        }

    return _train(
        model,
        decoder,
        latents,
        intervention,
        phases=[("train", epochs)],
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        device=device,
        verbose=verbose,
        setup=setup,
        batch_loss=batch_loss,
    )


# --- Plan B: discrete mixture over counterfactual states -----------------------------


class _AutoregWeights(nn.Module):
    """w_theta(s' | z, delta): autoregressive over S, conditioned on z and the do() spec."""

    def __init__(
        self,
        latent_dim: int,
        columns: list[ColumnSpec],
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.columns = list(columns)
        self.z_proj = nn.Linear(latent_dim, d_model)
        self.cond = _DoSpecTokens(self.columns, d_model)
        self.layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.prefix_embed = nn.ModuleDict(
            {c.name: nn.Embedding(c.n_categories, embed_dim) for c in self.columns}
        )
        self.heads = nn.ModuleDict(
            {
                c.name: nn.Linear(d_model + i * embed_dim, c.n_categories)
                for i, c in enumerate(self.columns)
            }
        )

    def context(self, z: torch.Tensor, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tokens = torch.cat([self.z_proj(z).unsqueeze(1), self.cond(values, mask)], dim=1)
        return self.layer(tokens)[:, 0]

    def log_joint(self, z: torch.Tensor, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Dense (batch, |S|) log-weights."""
        return _autoreg_log_joint(
            self.context(z, values, mask), self.columns, self.prefix_embed, self.heads
        )

    def top_k(
        self, z: torch.Tensor, values: torch.Tensor, mask: torch.Tensor, k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(batch, k) retained state indices and their (renormalised) log-weights."""
        log_joint = self.log_joint(z, values, mask)
        logw, idx = log_joint.topk(min(k, log_joint.shape[-1]), dim=-1)
        return idx, logw - torch.logsumexp(logw, dim=-1, keepdim=True)


class LatentInterventionDist(_ManipulatorBase):
    """h_Z(. | z, delta) = sum_s' w_theta(s' | z, delta) dirac_{z + Delta_phi(z, s')}."""

    def __init__(
        self,
        latent_dim: int,
        columns: list[ColumnSpec],
        d_model: int = 128,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        embed_dim: int = 16,
        top_k: int = 16,
    ) -> None:
        super().__init__(
            latent_dim,
            columns,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            embed_dim=embed_dim,
            top_k=top_k,
        )
        self.top_k = top_k
        self.w_theta = _AutoregWeights(
            latent_dim, self.columns, d_model, nhead, dim_feedforward, dropout, embed_dim
        )
        # Realiser Delta_phi conditions on a full state s' (every column masked in).
        self.realiser = _DeltaNet(
            latent_dim, self.columns, d_model, nhead, dim_feedforward, dropout
        )

    def realise(self, z: torch.Tensor, s_prime: torch.Tensor) -> torch.Tensor:
        """z' = z + Delta_phi(z, s'); `s_prime` is (batch, n_cols) category indices."""
        mask = torch.ones_like(s_prime, dtype=torch.bool)
        return z + self.realiser(z, s_prime, mask)

    def forward(self, z: torch.Tensor, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Inference: realise the single most likely counterfactual state (argmax w_theta)."""
        idx, _ = self.w_theta.top_k(z, values, mask, 1)
        return self.realise(z, unflatten_state_index(idx[:, 0], self.columns))

    def composed_log_joint(
        self,
        z: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        decoder: SemanticDecoder,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Exact log (g . h_Z)(. | z, delta) over the retained top-k s', shape (batch, |S|),
        plus the per-component (k, batch, latent_dim) latent shifts for penalties.
        """
        idx, logw = self.w_theta.top_k(z, values, mask, self.top_k)  # (batch, k)
        components, shifts = [], []
        for j in range(idx.shape[1]):
            z_j = self.realise(z, unflatten_state_index(idx[:, j], self.columns))
            components.append(logw[:, j : j + 1] + decoder.log_joint(z_j))
            shifts.append(z_j - z)
        composed = torch.logsumexp(torch.stack(components), dim=0)
        return composed, torch.stack(shifts)


def train_latent_intervention_dist(
    model: LatentInterventionDist,
    decoder: SemanticDecoder,
    latents: torch.Tensor,
    intervention: dict[str, int],
    s_prime: dict[str, torch.Tensor] | None = None,
    *,
    h_s: SymbolicKernel | None = None,
    pretrain_epochs: int = 30,
    joint_epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-4,
    proximity_weight: float = 1.0,
    sparsity_weight: float = 1.0,
    realiser_l1: float = 1.0,
    realiser_l2: float = 1.0,
    seed: int | None = None,
    device: str | torch.device | None = None,
    verbose: bool = True,
) -> list[dict[str, float]]:
    """
    Plan B, pretrain-then-joint (decoder + h_S frozen):

    - **Pretrain** splits the bilevel problem into two supervised ones:
      w_theta distils the dense target (h_S . g)(. | z, delta); the realiser Delta_phi
      minimises -log g(s' | z + Delta_phi(z, s')) + l1||Delta||_1 + l2||Delta||_2^2 on the
      true counterfactual states `s_prime`.
    - **Joint** fine-tunes both on the exact forward-KL consistency objective.

    Returns per-epoch mean loss components; the "phase" key is 0 for pretrain, 1 for joint.
    """
    if h_s is None or s_prime is None:
        raise ValueError("dist training needs both `h_s` and `s_prime`")

    def setup(device: torch.device, _gen: torch.Generator, z: torch.Tensor) -> dict:
        s_cols = {k: v.to(device) for k, v in s_prime.items()}
        return {
            "s_cols": s_cols,
            "s_idx": torch.stack([s_cols[c.name] for c in model.columns], dim=1),  # (n, n_cols)
            "target": consistency_target(decoder, h_s, z, intervention).to(device),
        }

    def batch_loss(phase, ctx, _gen, z, v, m, idx):
        if phase == "pretrain":
            w_kl = _forward_kl(model.w_theta.log_joint(z, v, m), ctx["target"][idx])
            z_r = model.realise(z, ctx["s_idx"][idx])
            realise_ce = decoder.nll(z_r, {k: val[idx] for k, val in ctx["s_cols"].items()})
            _, _, realise_reg = _penalties(z_r - z, realiser_l1, realiser_l2)
            total = w_kl + realise_ce + realise_reg
            return total, {
                "phase": 0.0,
                "total": total.item(),
                "w_kl": w_kl.item(),
                "realise_ce": realise_ce.item(),
            }
        composed, shifts = model.composed_log_joint(z, v, m, decoder)
        kl = _forward_kl(composed, ctx["target"][idx])
        sparsity, proximity, penalty = _penalties(shifts, sparsity_weight, proximity_weight)
        total = kl + penalty
        return total, {
            "phase": 1.0,
            "total": total.item(),
            "kl": kl.item(),
            "sparsity": sparsity.item(),
        }

    return _train(
        model,
        decoder,
        latents,
        intervention,
        phases=[("pretrain", pretrain_epochs), ("joint", joint_epochs)],
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        device=device,
        verbose=verbose,
        setup=setup,
        batch_loss=batch_loss,
    )


if __name__ == "__main__":
    # Smoke tests on a random schema: identity at init, then a short training run for
    # each plan against a randomly initialised frozen decoder and a stub h_S.
    torch.manual_seed(0)
    latent_dim, n = 128, 192
    columns = [ColumnSpec(f"c{i}", int(k)) for i, k in enumerate(torch.randint(2, 4, (5,)))]
    total_states = 1
    for col in columns:
        total_states *= col.n_categories

    class _StubKernel:
        """Identity h_S: every factual state is its own counterfactual (rows sum to 1)."""

        columns = columns

        def state_index(self, state: dict[str, int]) -> int:
            idx = 0
            for c in self.columns:
                idx = idx * c.n_categories + int(state[c.name])
            return idx

        def transition_matrix(self, delta: dict[str, int]) -> torch.Tensor:
            eye = torch.arange(total_states)
            return torch.sparse_coo_tensor(
                torch.stack([eye, eye]), torch.ones(total_states), (total_states, total_states)
            ).coalesce()

        def compose(self, g_probs: torch.Tensor, delta: dict[str, int]) -> torch.Tensor:
            return g_probs

    z = torch.randn(n, latent_dim)
    decoder = SemanticDecoder(latent_dim, columns)
    h_s = _StubKernel()
    intervention = {columns[1].name: 1}
    s_prime = {col.name: torch.randint(col.n_categories, (n,)) for col in columns}
    s_prime[columns[1].name] = torch.ones(n, dtype=torch.long)
    values, mask = make_objective(intervention, columns, batch_size=n)

    # Plan 0 -- baseline
    base = LatentIntervention(latent_dim, columns)
    with torch.no_grad():
        assert torch.allclose(base(z, values, mask), z), "baseline: identity at init"
    h0 = train_latent_intervention(
        base, decoder, z, intervention, s_prime, epochs=5, seed=0, verbose=False
    )
    print(f"[plan 0] total {h0[0]['total']:.4f} -> {h0[-1]['total']:.4f}")

    # Plan A -- engression
    gen = torch.Generator().manual_seed(0)
    a = LatentInterventionPreAdditive(latent_dim, columns, noise_std=0.5)
    with torch.no_grad():
        assert torch.allclose(a(z, values, mask, gen), z), "plan A: identity at init"
    ha = train_latent_intervention_preadditive(
        a, decoder, z, intervention, h_s=h_s, n_samples=4, epochs=3, seed=0, verbose=False
    )
    print(f"[plan A] total {ha[0]['total']:.4f} -> {ha[-1]['total']:.4f}  kl {ha[-1]['kl']:.4f}")

    # Plan B -- discrete mixture
    b = LatentInterventionDist(latent_dim, columns, top_k=8)
    with torch.no_grad():
        z_b = b(z, values, mask)
    assert torch.allclose(z_b, z), "plan B: realiser is identity at init"
    hb = train_latent_intervention_dist(
        b, decoder, z, intervention, s_prime, h_s=h_s,
        pretrain_epochs=3, joint_epochs=2, seed=0, verbose=False,
    )
    pretrain = [r for r in hb if r["phase"] == 0.0][-1]
    joint = [r for r in hb if r["phase"] == 1.0][-1]
    print(f"[plan B] pretrain w_kl {pretrain['w_kl']:.4f} realise_ce {pretrain['realise_ce']:.4f}"
          f"  joint kl {joint['kl']:.4f}")

    with torch.no_grad():
        z_prime = b(z, values, mask)
    preds = decoder.predict(z_prime)
    consistency = {c.name: (preds[c.name] == s_prime[c.name]).float().mean().item() for c in columns}
    print("[plan B] consistency accuracy:", {k: round(v, 2) for k, v in consistency.items()})
    print(f"[plan B] mean latent shift: {(z_prime - z).norm(dim=1).mean():.3f}")
