"""
Latent editor h_Z(z' | z, delta): Z -> Delta(Z).

Freeze f, g, h_S; train h_Z so the two paths from z to a distribution over counterfactual
states agree (h_S . g == g . h_Z). See docs/architecture/latent_intervention.md.

Three parameterisations:

- LatentIntervention  -- Plan 0 (baseline): deterministic residual transformer
  z' = z + T_theta(z, delta). Cannot represent the multimodality of (h_S . g) in the
  ambiguous strata; kept as a baseline, trained by per-column CE against S'.
- LatentInterventionA -- Plan A (engression): z' = z + Delta_theta(z + eps, delta),
  eps ~ N(0, sigma^2 I). h_Z is the pushforward of eps; a nonlinear Delta_theta folds a
  unimodal eps onto separated modes. Composition g . h_Z needs Monte Carlo.
- LatentInterventionB -- Plan B (discrete mixture, preferred):
  h_Z(. | z, delta) = sum_s' w_theta(s' | z, delta) * dirac_{z + Delta_phi(z, s')}.
  w_theta is an autoregressive head over S (the SCM internalised, no h_S at inference);
  Delta_phi is a deterministic realiser. Composition is an exact finite sum over the
  retained top-k s'. Trained pretrain-then-joint.

The consistency objective (Plan A/B) is
    L = E_z[ D_KL( (h_S . g)(. | z, delta) || (g . h_Z)(. | z, delta) ) ]
        + alpha ||z' - z||_1 + beta ||z' - z||_2^2
with forward (mass-covering) KL over the dense |S| = 3456 vector.
"""

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from src.semantic_decoder import SCM_COLUMNS, ColumnSpec, SemanticDecoder
from src.symbolic_intervention import SymbolicIntervention

# --- flat state <-> per-column indices --------------------------------------------------
# Column-major over `columns`, matching SemanticDecoder.log_joint and
# SymbolicIntervention.state_index: idx = ((s_0 * k_1) + s_1) * k_2 + ...


def flat_state_index(values: torch.Tensor, columns: list[ColumnSpec] = SCM_COLUMNS) -> torch.Tensor:
    """(..., n_cols) category indices -> (...,) flat state index."""
    idx = torch.zeros(values.shape[:-1], dtype=torch.long, device=values.device)
    for i, col in enumerate(columns):
        idx = idx * col.n_categories + values[..., i]
    return idx


def unflatten_state_index(flat: torch.Tensor, columns: list[ColumnSpec] = SCM_COLUMNS) -> torch.Tensor:
    """(...,) flat state index -> (..., n_cols) category indices."""
    rem = flat.clone()
    cols: list[torch.Tensor] = []
    for col in reversed(columns):
        cols.append(rem % col.n_categories)
        rem = torch.div(rem, col.n_categories, rounding_mode="floor")
    return torch.stack(list(reversed(cols)), dim=-1)


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


def _save_module(model: nn.Module, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"class": type(model).__name__, "config": model._config, "state_dict": model.state_dict()},
        path,
    )


def _load_module(cls: type, path: str | Path, device: str | torch.device | None):
    payload = torch.load(Path(path), map_location=device or "cpu", weights_only=True)
    if payload.get("class", cls.__name__) != cls.__name__:
        raise ValueError(f"checkpoint holds a {payload['class']}, not a {cls.__name__}")
    config = dict(payload["config"])
    config["columns"] = [ColumnSpec(name, card) for name, card in config["columns"]]
    model = cls(**config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


# --- Plan 0: deterministic baseline ----------------------------------------------------


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
        _save_module(self, path)

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device | None = None) -> "LatentIntervention":
        """Restore a manipulator saved with save(); returns it in eval mode."""
        return _load_module(cls, path, device)


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
    consistency_loss = decoder.nll(z_prime, s_prime)
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
    Train the baseline manipulator on counterfactual pairs; the decoder stays frozen.

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


# --- consistency target shared by Plan A and Plan B -----------------------------------


@torch.no_grad()
def consistency_target(
    decoder: SemanticDecoder,
    h_s: SymbolicIntervention,
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


def _forward_kl(pred_log: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """D_KL(target || pred), mean over the batch. `pred_log` are log-probabilities."""
    return F.kl_div(pred_log, target, reduction="batchmean")


# --- Plan A: pre-additive noise (engression) -----------------------------------------


class LatentInterventionA(nn.Module):
    """z' = z + Delta_theta(z + eps, delta), eps ~ N(0, noise_std^2 I)."""

    def __init__(
        self,
        latent_dim: int,
        columns: list[ColumnSpec] = SCM_COLUMNS,
        noise_std: float = 1.0,
        d_model: int = 128,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.columns = list(columns)
        self.noise_std = noise_std
        self._config = {
            "latent_dim": latent_dim,
            "columns": [(c.name, c.n_categories) for c in self.columns],
            "noise_std": noise_std,
            "d_model": d_model,
            "nhead": nhead,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
        }
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

    def save(self, path: str | Path) -> None:
        _save_module(self, path)

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device | None = None) -> "LatentInterventionA":
        return _load_module(cls, path, device)


def train_latent_intervention_a(
    model: LatentInterventionA,
    decoder: SemanticDecoder,
    h_s: SymbolicIntervention,
    latents: torch.Tensor,
    intervention: dict[str, int],
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
    device = torch.device(device) if device is not None else torch.device("cpu")
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    model.to(device).train()
    decoder.to(device).eval()
    decoder.requires_grad_(False)
    latents = latents.to(device)
    target = consistency_target(decoder, h_s, latents, intervention).to(device)
    values, mask = make_objective(intervention, model.columns, batch_size=1, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = latents.shape[0]
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        perm = torch.randperm(n, generator=generator, device=device)
        logs: list[dict[str, float]] = []
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            composed, shifts = model.composed_log_joint(
                latents[idx],
                values.expand(len(idx), -1),
                mask.expand(len(idx), -1),
                decoder,
                n_samples,
                generator,
            )
            kl = _forward_kl(composed, target[idx])
            sparsity = shifts.abs().mean()
            proximity = shifts.pow(2).mean()
            loss = kl + sparsity_weight * sparsity + proximity_weight * proximity

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logs.append(
                {
                    "total": loss.item(),
                    "kl": kl.item(),
                    "sparsity": sparsity.item(),
                    "proximity": proximity.item(),
                }
            )
        history.append({k: sum(x[k] for x in logs) / len(logs) for k in logs[0]})
        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            log = history[-1]
            print(f"epoch {epoch + 1}/{epochs}  total {log['total']:.4f}  kl {log['kl']:.4f}")

    model.eval()
    return history


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


class LatentInterventionB(nn.Module):
    """h_Z(. | z, delta) = sum_s' w_theta(s' | z, delta) dirac_{z + Delta_phi(z, s')}."""

    def __init__(
        self,
        latent_dim: int,
        columns: list[ColumnSpec] = SCM_COLUMNS,
        d_model: int = 128,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        embed_dim: int = 16,
        top_k: int = 16,
    ) -> None:
        super().__init__()
        self.columns = list(columns)
        self.top_k = top_k
        self._config = {
            "latent_dim": latent_dim,
            "columns": [(c.name, c.n_categories) for c in self.columns],
            "d_model": d_model,
            "nhead": nhead,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "embed_dim": embed_dim,
            "top_k": top_k,
        }
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

    def save(self, path: str | Path) -> None:
        _save_module(self, path)

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device | None = None) -> "LatentInterventionB":
        return _load_module(cls, path, device)


def train_latent_intervention_b(
    model: LatentInterventionB,
    decoder: SemanticDecoder,
    h_s: SymbolicIntervention,
    latents: torch.Tensor,
    intervention: dict[str, int],
    s_prime_targets: dict[str, torch.Tensor],
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
      true counterfactual states `s_prime_targets`.
    - **Joint** fine-tunes both on the exact forward-KL consistency objective.

    Returns per-epoch mean loss components; the "phase" key is 0 for pretrain, 1 for joint.
    """
    device = torch.device(device) if device is not None else torch.device("cpu")
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    model.to(device).train()
    decoder.to(device).eval()
    decoder.requires_grad_(False)
    latents = latents.to(device)
    s_cols = {k: v.to(device) for k, v in s_prime_targets.items()}
    s_prime_idx = torch.stack([s_cols[c.name] for c in model.columns], dim=1)  # (n, n_cols)
    target = consistency_target(decoder, h_s, latents, intervention).to(device)
    values, mask = make_objective(intervention, model.columns, batch_size=1, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = latents.shape[0]
    history: list[dict[str, float]] = []

    def run_epoch(phase: int) -> dict[str, float]:
        perm = torch.randperm(n, generator=generator, device=device)
        logs: list[dict[str, float]] = []
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            z = latents[idx]
            v, m = values.expand(len(idx), -1), mask.expand(len(idx), -1)
            if phase == 0:
                w_kl = _forward_kl(model.w_theta.log_joint(z, v, m), target[idx])
                z_r = model.realise(z, s_prime_idx[idx])
                realise_ce = decoder.nll(z_r, {k: v_[idx] for k, v_ in s_cols.items()})
                delta = z_r - z
                realise_reg = realiser_l1 * delta.abs().mean() + realiser_l2 * delta.pow(2).mean()
                loss = w_kl + realise_ce + realise_reg
                logs.append(
                    {
                        "phase": 0.0,
                        "total": loss.item(),
                        "w_kl": w_kl.item(),
                        "realise_ce": realise_ce.item(),
                    }
                )
            else:
                composed, shifts = model.composed_log_joint(z, v, m, decoder)
                kl = _forward_kl(composed, target[idx])
                sparsity = shifts.abs().mean()
                proximity = shifts.pow(2).mean()
                loss = kl + sparsity_weight * sparsity + proximity_weight * proximity
                logs.append(
                    {"phase": 1.0, "total": loss.item(), "kl": kl.item(), "sparsity": sparsity.item()}
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return {k: sum(x[k] for x in logs) / len(logs) for k in logs[0]}

    for phase, n_epochs in ((0, pretrain_epochs), (1, joint_epochs)):
        for epoch in range(n_epochs):
            row = run_epoch(phase)
            history.append(row)
            if verbose and (epoch + 1) % max(1, n_epochs // 5) == 0:
                tag = "pretrain" if phase == 0 else "joint"
                print(f"[{tag}] epoch {epoch + 1}/{n_epochs}  " + "  ".join(
                    f"{k} {v:.4f}" for k, v in row.items() if k != "phase"
                ))

    model.eval()
    return history


if __name__ == "__main__":
    # Smoke tests on random data: identity at init, then a short training run for each plan
    # against a freshly (randomly) initialised frozen decoder and the closed-form h_S.
    torch.manual_seed(0)
    latent_dim, n = 128, 192
    z = torch.randn(n, latent_dim)
    decoder = SemanticDecoder(latent_dim)
    h_s = SymbolicIntervention.from_scm()
    intervention = {"G": 1}
    s_prime = {col.name: torch.randint(col.n_categories, (n,)) for col in SCM_COLUMNS}
    s_prime["G"] = torch.ones(n, dtype=torch.long)
    values, mask = make_objective(intervention, batch_size=n)

    # Plan 0 -- baseline
    base = LatentIntervention(latent_dim)
    with torch.no_grad():
        assert torch.allclose(base(z, values, mask), z), "baseline: identity at init"
    h0 = train_latent_intervention(base, decoder, z, intervention, s_prime, epochs=5, seed=0, verbose=False)
    print(f"[plan 0] total {h0[0]['total']:.4f} -> {h0[-1]['total']:.4f}")

    # Plan A -- engression
    gen = torch.Generator().manual_seed(0)
    a = LatentInterventionA(latent_dim, noise_std=0.5)
    with torch.no_grad():
        assert torch.allclose(a(z, values, mask, gen), z), "plan A: identity at init"
    ha = train_latent_intervention_a(
        a, decoder, h_s, z, intervention, n_samples=4, epochs=3, seed=0, verbose=False
    )
    print(f"[plan A] total {ha[0]['total']:.4f} -> {ha[-1]['total']:.4f}  kl {ha[-1]['kl']:.4f}")

    # Plan B -- discrete mixture
    b = LatentInterventionB(latent_dim, top_k=8)
    with torch.no_grad():
        z_b = b(z, values, mask)
    assert torch.allclose(z_b, z), "plan B: realiser is identity at init"
    hb = train_latent_intervention_b(
        b, decoder, h_s, z, intervention, s_prime,
        pretrain_epochs=3, joint_epochs=2, seed=0, verbose=False,
    )
    pretrain = [r for r in hb if r["phase"] == 0.0][-1]
    joint = [r for r in hb if r["phase"] == 1.0][-1]
    print(f"[plan B] pretrain w_kl {pretrain['w_kl']:.4f} realise_ce {pretrain['realise_ce']:.4f}"
          f"  joint kl {joint['kl']:.4f}")

    with torch.no_grad():
        z_prime = b(z, values, mask)
    preds = decoder.predict(z_prime)
    consistency = {c.name: (preds[c.name] == s_prime[c.name]).float().mean().item() for c in SCM_COLUMNS}
    print("[plan B] consistency accuracy:", {k: round(v, 2) for k, v in consistency.items()})
    print(f"[plan B] mean latent shift: {(z_prime - z).norm(dim=1).mean():.3f}")
