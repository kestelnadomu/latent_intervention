"""``state_flow``: a conditional density model ``Z = F(S, U)`` fitted only to factual ``(S, Z)``.

Counterfactual transport abducts ``U`` under the factual state and renders the same noise
under a target state. It exposes explicit state-aware operations so oracle-state and
inferred-state evaluations cannot be confused.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
from torch import nn

from src.schema import ColumnSpec
from src.semantic_decoder.model import SemanticDecoderModel

from src.latent_intervention.flow.common import (
    StateLike,
    _ConditionalRealNVP,
    _FlowModelBase,
    _finish_epoch,
    _make_generator,
    _model_device,
    _model_dtype,
    _positive_int,
    _state_tensor,
    _validate_latents,
    _validate_training_inputs,
    _weighted_log_update,
)

__all__ = [
    "StateConditionalFlow",
    "train_state_conditional_flow",
]


class _StateContext(nn.Module):
    def __init__(
        self,
        columns: Sequence[ColumnSpec],
        state_embed_dim: int,
        condition_dim: int,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(column.n_categories, state_embed_dim) for column in columns]
        )
        self.project = nn.Sequential(
            nn.Linear(len(columns) * state_embed_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        embedded = [embedding(states[:, index]) for index, embedding in enumerate(self.embeddings)]
        return self.project(torch.cat(embedded, dim=-1))


class StateConditionalFlow(_FlowModelBase):
    """Conditional factual density ``Z = F(S, U)`` and shared-noise transport."""

    variant = "state_flow"

    def __init__(
        self,
        latent_dim: int,
        columns: Sequence[ColumnSpec],
        *,
        n_blocks: int = 8,
        hidden_dim: int = 128,
        condition_dim: int = 64,
        state_embed_dim: int = 16,
        permutation_seed: int = 0,
        scale_floor: float = 1e-6,
    ) -> None:
        super().__init__(
            latent_dim,
            columns,
            n_blocks=n_blocks,
            hidden_dim=hidden_dim,
            condition_dim=condition_dim,
            state_embed_dim=state_embed_dim,
            permutation_seed=permutation_seed,
            scale_floor=scale_floor,
        )
        self.state_context = _StateContext(self.columns, state_embed_dim, condition_dim)
        self.flow = _ConditionalRealNVP(
            latent_dim, condition_dim, hidden_dim, n_blocks, permutation_seed
        )

    def _condition(self, states: StateLike, batch_size: int) -> torch.Tensor:
        state = _state_tensor(
            states, self.columns, batch_size=batch_size, device=_model_device(self)
        )
        return self.state_context(state)

    def abduct(self, z: torch.Tensor, states: StateLike) -> torch.Tensor:
        """Return factual exogenous coordinates ``U = F^{-1}(Z; S)``."""
        _validate_latents(z, self.latent_dim)
        device = _model_device(self)
        z = z.to(device=device, dtype=self.latent_mean.dtype)
        standardized = (z - self.latent_mean) / self.latent_scale
        return self.flow.inverse_transform(
            standardized, self._condition(states, z.shape[0])
        )[0]

    def render(self, u: torch.Tensor, states: StateLike) -> torch.Tensor:
        """Render latent values from exogenous coordinates and a semantic state."""
        _validate_latents(u, self.latent_dim)
        device = _model_device(self)
        u = u.to(device=device, dtype=self.latent_mean.dtype)
        standardized = self.flow.forward_transform(
            u, self._condition(states, u.shape[0])
        )[0]
        return self.latent_mean + self.latent_scale * standardized

    def transport(
        self,
        z: torch.Tensor,
        source_states: StateLike,
        target_states: StateLike,
    ) -> torch.Tensor:
        """Abduct under ``source_states`` and render the same noise under targets."""
        return self.render(self.abduct(z, source_states), target_states)

    def forward(
        self,
        z: torch.Tensor,
        source_states: StateLike,
        target_states: StateLike,
    ) -> torch.Tensor:
        return self.transport(z, source_states, target_states)

    def log_prob(self, z: torch.Tensor, states: StateLike) -> torch.Tensor:
        """Conditional factual log density ``log p(Z | S)`` for every row."""
        _validate_latents(z, self.latent_dim)
        device = _model_device(self)
        z = z.to(device=device, dtype=self.latent_mean.dtype)
        standardized = (z - self.latent_mean) / self.latent_scale
        base, inverse_log_det = self.flow.inverse_transform(
            standardized, self._condition(states, z.shape[0])
        )
        base_log_prob = -0.5 * (
            base.pow(2) + math.log(2.0 * math.pi)
        ).sum(dim=-1)
        return base_log_prob + inverse_log_det - self.latent_scale.log().sum()

    def sample(
        self,
        states: StateLike,
        n_samples: int = 1,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw ``(n_samples, batch, latent_dim)`` values from ``p(Z | S)``."""
        _positive_int("n_samples", n_samples)
        state = _state_tensor(states, self.columns, device=_model_device(self))
        batch_size = state.shape[0]
        noise = torch.randn(
            n_samples,
            batch_size,
            self.latent_dim,
            device=_model_device(self),
            dtype=self.latent_mean.dtype,
            generator=generator,
        )
        repeated_state = state.unsqueeze(0).expand(n_samples, -1, -1).reshape(
            n_samples * batch_size, -1
        )
        return self.render(
            noise.reshape(n_samples * batch_size, self.latent_dim), repeated_state
        ).reshape(n_samples, batch_size, self.latent_dim)


def train_state_conditional_flow(
    model: StateConditionalFlow,
    latents: torch.Tensor,
    states: StateLike,
    *,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    grad_clip: float = 5.0,
    seed: int | None = None,
    device: str | torch.device | None = None,
    verbose: bool = True,
    epoch_callback: Callable | None = None,
) -> list[dict[str, float]]:
    """Fit ``p(Z | S)`` using factual conditional maximum likelihood only."""
    if not isinstance(model, StateConditionalFlow):
        raise TypeError("model must be a StateConditionalFlow")
    _validate_training_inputs(
        model,
        latents,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
    )
    state = _state_tensor(states, model.columns, batch_size=latents.shape[0])
    model.set_latent_statistics(latents)
    run_device = torch.device(device or "cpu")
    model.to(run_device).train()
    z = latents.to(device=run_device, dtype=model.latent_mean.dtype)
    state = state.to(run_device)
    shuffle = _make_generator(run_device, seed, offset=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        totals: dict[str, float] = {}
        order = torch.randperm(z.shape[0], generator=shuffle, device=run_device)
        for start in range(0, z.shape[0], batch_size):
            index = order[start : start + batch_size]
            nll = -model.log_prob(z[index], state[index]).mean()
            optimizer.zero_grad(set_to_none=True)
            nll.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip, error_if_nonfinite=True
            )
            optimizer.step()
            _weighted_log_update(
                totals,
                {"total": nll.item(), "nll": nll.item(), "grad_norm": float(grad_norm)},
                len(index),
            )
        history.append(_finish_epoch(totals, len(z), epoch, epochs, verbose))
        if epoch_callback is not None and epoch_callback(
            model, optimizer, "train", history[-1]
        ):
            break

    model.eval()
    return history


@torch.no_grad()
def _source_states_for_inference(
    model: StateConditionalFlow,
    z: torch.Tensor,
    source_states: StateLike | None,
    decoder: SemanticDecoderModel | None,
) -> torch.Tensor:
    if source_states is not None:
        return _state_tensor(
            source_states,
            model.columns,
            batch_size=z.shape[0],
            device=_model_device(model),
        )
    if decoder is None:
        raise ValueError("state_flow inference requires `source_states` or a semantic decoder")
    was_training = decoder.training
    decoder.to(_model_device(model)).eval()
    try:
        predictions = decoder.predict(
            z.to(device=_model_device(model), dtype=_model_dtype(decoder))
        )
    finally:
        decoder.train(was_training)
    return _state_tensor(
        predictions,
        model.columns,
        batch_size=z.shape[0],
        device=_model_device(model),
    )
