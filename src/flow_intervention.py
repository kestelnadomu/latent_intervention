"""Normalizing-flow latent interventions.

This module adds three flow-based variants without changing the existing transformer
implementations in :mod:`src.latent_intervention`:

``state_flow``
    A conditional density model ``Z = F(S, U)`` fitted only to factual ``(S, Z)``.
    Counterfactual transport abducts ``U`` under the factual state and renders the
    same noise under a target state.
``distilled_flow``
    A deployable conditional flow ``q(Z' | Z, delta)`` fitted to samples from a
    frozen ``state_flow`` teacher with an energy-distance objective.
``direct_semantic_flow``
    The same deployable architecture fitted directly against the frozen semantic
    decoder and symbolic transition kernel.

The two direct variants need only ``Z`` and ``delta`` after training.  The state flow
also deliberately exposes explicit state-aware operations so oracle-state and
inferred-state evaluations cannot be confused.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any, ClassVar, Self, TypeAlias

import torch
import torch.nn.functional as F
from torch import nn

from src.schema import ColumnSpec, flat_state_index, unflatten_state_index
from src.semantic_decoder import SemanticDecoderModel
from src.symbolic_intervention import SymbolicKernel


FLOW_CHECKPOINT_FORMAT_VERSION = 1
FLOW_VARIANTS = ("state_flow", "distilled_flow", "direct_semantic_flow")

StateLike: TypeAlias = torch.Tensor | Mapping[str, torch.Tensor]

__all__ = [
    "FLOW_CHECKPOINT_FORMAT_VERSION",
    "FLOW_VARIANTS",
    "StateConditionalFlow",
    "DistilledFlowIntervention",
    "DirectSemanticFlowIntervention",
    "make_latent_intervention",
    "train_state_conditional_flow",
    "train_distilled_flow",
    "train_direct_semantic_flow",
    "train_latent_intervention_model",
    "load_latent_intervention",
    "counterfactual",
    "sample_counterfactual",
    "sample_symbolic_states",
    "multivariate_energy_distance",
]


# --- validation and small utilities -------------------------------------------------


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _unit_interval(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _validate_columns(columns: Sequence[ColumnSpec]) -> list[ColumnSpec]:
    result = list(columns)
    if not result:
        raise ValueError("columns must not be empty")
    names: set[str] = set()
    for column in result:
        if not isinstance(column, ColumnSpec):
            raise TypeError("columns must contain ColumnSpec values")
        if not column.name or column.name in names:
            raise ValueError(f"invalid or duplicate column name: {column.name!r}")
        _positive_int(f"cardinality for {column.name!r}", column.n_categories)
        names.add(column.name)
    return result


def _schema_signature(columns: Sequence[ColumnSpec]) -> str:
    raw = json.dumps(
        [(column.name, column.n_categories) for column in columns],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return next(model.buffers()).device


def _model_dtype(model: nn.Module) -> torch.dtype:
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return next(model.buffers()).dtype


def _validate_latents(
    z: torch.Tensor,
    latent_dim: int,
    *,
    nonempty: bool = False,
) -> None:
    if not isinstance(z, torch.Tensor):
        raise TypeError("latents must be a torch.Tensor")
    if z.ndim != 2 or z.shape[1] != latent_dim:
        raise ValueError(f"latents must have shape (n, {latent_dim}), got {tuple(z.shape)}")
    if nonempty and z.shape[0] == 0:
        raise ValueError("latents must not be empty")
    if not z.is_floating_point():
        raise ValueError("latents must have a floating-point dtype")
    if not torch.isfinite(z).all():
        raise ValueError("latents must contain only finite values")


def _state_tensor(
    states: StateLike,
    columns: Sequence[ColumnSpec],
    *,
    batch_size: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    if isinstance(states, Mapping):
        expected = {column.name for column in columns}
        if set(states) != expected:
            raise ValueError(
                "state columns do not match schema; "
                f"missing={sorted(expected - set(states))}, "
                f"extra={sorted(set(states) - expected)}"
            )
        tensors = [states[column.name] for column in columns]
        if any(not isinstance(value, torch.Tensor) for value in tensors):
            raise TypeError("state mappings must contain torch.Tensor values")
        if any(value.ndim != 1 for value in tensors):
            raise ValueError("each mapped state column must be one-dimensional")
        if len({value.shape[0] for value in tensors}) != 1:
            raise ValueError("mapped state columns must have equal lengths")
        result = torch.stack(tensors, dim=-1)
    elif isinstance(states, torch.Tensor):
        result = states
    else:
        raise TypeError("states must be a tensor or a mapping from schema names to tensors")

    if result.ndim != 2 or result.shape[1] != len(columns):
        raise ValueError(
            f"states must have shape (n, {len(columns)}), got {tuple(result.shape)}"
        )
    if batch_size is not None and result.shape[0] != batch_size:
        raise ValueError(f"states must contain {batch_size} rows, got {result.shape[0]}")
    if result.dtype == torch.bool or result.is_floating_point() or result.is_complex():
        raise ValueError("states must have an integer dtype")
    result = result.to(device=device, dtype=torch.long)
    for index, column in enumerate(columns):
        values = result[:, index]
        if values.numel() and (
            int(values.min()) < 0 or int(values.max()) >= column.n_categories
        ):
            raise ValueError(
                f"state column {column.name!r} must be in "
                f"[0, {column.n_categories - 1}]"
            )
    return result


def _validate_objective(
    values: torch.Tensor,
    mask: torch.Tensor,
    columns: Sequence[ColumnSpec],
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected = (batch_size, len(columns))
    if not isinstance(values, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise TypeError("intervention values and mask must be torch.Tensor values")
    if tuple(values.shape) != expected or tuple(mask.shape) != expected:
        raise ValueError(
            f"intervention values and mask must both have shape {expected}"
        )
    if values.dtype == torch.bool or values.is_floating_point() or values.is_complex():
        raise ValueError("intervention values must have an integer dtype")
    if mask.dtype != torch.bool:
        raise ValueError("intervention mask must have dtype torch.bool")
    values = values.to(device=device, dtype=torch.long)
    mask = mask.to(device=device)
    for index, column in enumerate(columns):
        selected = values[:, index][mask[:, index]]
        if selected.numel() and (
            int(selected.min()) < 0 or int(selected.max()) >= column.n_categories
        ):
            raise ValueError(
                f"intervention column {column.name!r} must be in "
                f"[0, {column.n_categories - 1}]"
            )
    return values, mask


def _objective_tensors(
    intervention: Mapping[str, int],
    columns: Sequence[ColumnSpec],
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    from src.latent_intervention import make_objective

    normalized = dict(_canonical_intervention(intervention))
    return make_objective(normalized, list(columns), batch_size, device)


def _make_generator(
    device: torch.device,
    seed: int | None,
    *,
    offset: int = 0,
) -> torch.Generator:
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(int(seed) + offset)
    return generator


def _warn_small_sample(n: int, latent_dim: int) -> None:
    if n <= latent_dim:
        warnings.warn(
            f"only {n} training units for a {latent_dim}-dimensional flow; "
            "this is suitable for smoke testing, not reliable density estimation",
            RuntimeWarning,
            stacklevel=3,
        )


def _canonical_intervention(intervention: Mapping[str, int]) -> tuple[tuple[str, int], ...]:
    normalized: list[tuple[str, int]] = []
    for name, value in intervention.items():
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"intervention value for {name!r} must be an integer")
        normalized.append((str(name), int(value)))
    return tuple(sorted(normalized))


def _intervention_support(intervention: Mapping[str, int]) -> list[dict[str, int]]:
    configured = dict(_canonical_intervention(intervention))
    return [configured, {}] if configured else [{}]


# --- conditional RealNVP ------------------------------------------------------------


class _AffineCoupling(nn.Module):
    """One conditional affine coupling with a fixed preceding permutation."""

    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        hidden_dim: int,
        *,
        parity: int,
        permutation: torch.Tensor,
    ) -> None:
        super().__init__()
        identity = torch.arange(latent_dim)[torch.arange(latent_dim) % 2 == parity]
        transformed = torch.arange(latent_dim)[torch.arange(latent_dim) % 2 != parity]
        if transformed.numel() == 0:  # supports latent_dim == 1
            identity = torch.empty(0, dtype=torch.long)
            transformed = torch.arange(latent_dim)
        self.register_buffer("identity_index", identity)
        self.register_buffer("transform_index", transformed)
        self.register_buffer("permutation", permutation)
        self.register_buffer("inverse_permutation", torch.argsort(permutation))

        self.conditioner = nn.Sequential(
            nn.Linear(identity.numel() + condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * transformed.numel()),
        )
        final = self.conditioner[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def _coupling_parameters(
        self, permuted: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        observed = permuted.index_select(-1, self.identity_index)
        raw_scale, shift = self.conditioner(torch.cat((observed, condition), dim=-1)).chunk(
            2, dim=-1
        )
        # Bounded scale keeps both likelihood and inversion numerically stable.
        log_scale = 2.0 * torch.tanh(raw_scale / 2.0)
        return log_scale, shift

    def forward_transform(
        self, x: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        permuted = x.index_select(-1, self.permutation)
        log_scale, shift = self._coupling_parameters(permuted, condition)
        transformed = permuted.index_select(-1, self.transform_index)
        transformed = transformed * torch.exp(log_scale) + shift
        output = permuted.clone()
        output[..., self.transform_index] = transformed
        return output, log_scale.sum(dim=-1)

    def inverse_transform(
        self, y: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_scale, shift = self._coupling_parameters(y, condition)
        transformed = y.index_select(-1, self.transform_index)
        transformed = (transformed - shift) * torch.exp(-log_scale)
        permuted = y.clone()
        permuted[..., self.transform_index] = transformed
        return permuted.index_select(-1, self.inverse_permutation), -log_scale.sum(dim=-1)


class _ConditionalRealNVP(nn.Module):
    """Eight-block by default conditional affine RealNVP core."""

    def __init__(
        self,
        latent_dim: int,
        condition_dim: int = 64,
        hidden_dim: int = 128,
        n_blocks: int = 8,
        permutation_seed: int = 0,
    ) -> None:
        super().__init__()
        self.latent_dim = _positive_int("latent_dim", latent_dim)
        self.condition_dim = _positive_int("condition_dim", condition_dim)
        self.hidden_dim = _positive_int("hidden_dim", hidden_dim)
        self.n_blocks = _positive_int("n_blocks", n_blocks)
        if self.latent_dim > 1 and self.n_blocks < 2:
            raise ValueError("n_blocks must be at least 2 when latent_dim is greater than 1")
        if isinstance(permutation_seed, bool) or not isinstance(permutation_seed, int):
            raise ValueError("permutation_seed must be an integer")

        generator = torch.Generator(device="cpu").manual_seed(permutation_seed)
        coordinates = torch.arange(self.latent_dim)
        labels = coordinates.clone()
        covered = torch.zeros(self.latent_dim, dtype=torch.bool)
        first = torch.randperm(self.latent_dim, generator=generator)
        permutations = [first]
        labels = labels[first]
        first_transformed = coordinates[coordinates % 2 != 0]
        if first_transformed.numel() == 0:
            first_transformed = coordinates
        covered[labels[first_transformed]] = True

        if self.latent_dim == 1:
            permutations.extend(coordinates.clone() for _ in range(1, self.n_blocks))
        elif self.n_blocks >= 2:
            # Put every coordinate untouched by block 0 on block 1's transformed
            # side. Random order within both halves retains seeded mixing while
            # guaranteeing full affine coverage even for a two-block flow.
            second_transformed = coordinates[coordinates % 2 == 0]
            second_identity = coordinates[coordinates % 2 == 1]
            untouched_positions = coordinates[~covered[labels]]
            covered_positions = coordinates[covered[labels]]
            untouched_positions = untouched_positions[
                torch.randperm(len(untouched_positions), generator=generator)
            ]
            covered_positions = covered_positions[
                torch.randperm(len(covered_positions), generator=generator)
            ]
            second = torch.empty(self.latent_dim, dtype=torch.long)
            second[second_transformed] = untouched_positions
            second[second_identity] = covered_positions
            permutations.append(second)
            labels = labels[second]
            covered[labels[second_transformed]] = True

        for _ in range(len(permutations), self.n_blocks):
            permutation = torch.randperm(self.latent_dim, generator=generator)
            permutations.append(permutation)
        if not covered.all():
            raise AssertionError("coupling construction left a latent coordinate untouched")
        self.blocks = nn.ModuleList(
            [
                _AffineCoupling(
                    self.latent_dim,
                    self.condition_dim,
                    self.hidden_dim,
                    parity=index % 2,
                    permutation=permutations[index],
                )
                for index in range(self.n_blocks)
            ]
        )

    def forward_transform(
        self, base: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result = base
        log_det = base.new_zeros(base.shape[0])
        for block in self.blocks:
            result, update = block.forward_transform(result, condition)
            log_det = log_det + update
        return result, log_det

    def inverse_transform(
        self, value: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result = value
        log_det = value.new_zeros(value.shape[0])
        for block in reversed(self.blocks):
            result, update = block.inverse_transform(result, condition)
            log_det = log_det + update
        return result, log_det


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


class _DirectContext(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        columns: Sequence[ColumnSpec],
        state_embed_dim: int,
        condition_dim: int,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(column.n_categories + 1, state_embed_dim) for column in columns]
        )
        self.null_values = [column.n_categories for column in columns]
        self.z_project = nn.Linear(latent_dim, condition_dim)
        self.do_project = nn.Linear(len(columns) * state_embed_dim, condition_dim)
        self.combine = nn.Sequential(
            nn.Linear(2 * condition_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )

    def forward(
        self, standardized_z: torch.Tensor, values: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        embedded = []
        for index, embedding in enumerate(self.embeddings):
            category = torch.where(
                mask[:, index],
                values[:, index],
                torch.full_like(values[:, index], self.null_values[index]),
            )
            embedded.append(embedding(category))
        do_context = self.do_project(torch.cat(embedded, dim=-1))
        return self.combine(torch.cat((self.z_project(standardized_z), do_context), dim=-1))


# --- checkpointed public models -----------------------------------------------------


class _FlowModelBase(nn.Module):
    variant: ClassVar[str]

    def __init__(
        self,
        latent_dim: int,
        columns: Sequence[ColumnSpec],
        *,
        n_blocks: int,
        hidden_dim: int,
        condition_dim: int,
        state_embed_dim: int,
        permutation_seed: int,
        scale_floor: float,
        **extra_config: Any,
    ) -> None:
        super().__init__()
        self.latent_dim = _positive_int("latent_dim", latent_dim)
        self.columns = _validate_columns(columns)
        self.n_blocks = _positive_int("n_blocks", n_blocks)
        self.hidden_dim = _positive_int("hidden_dim", hidden_dim)
        self.condition_dim = _positive_int("condition_dim", condition_dim)
        self.state_embed_dim = _positive_int("state_embed_dim", state_embed_dim)
        self.scale_floor = _positive_float("scale_floor", scale_floor)
        if isinstance(permutation_seed, bool) or not isinstance(permutation_seed, int):
            raise ValueError("permutation_seed must be an integer")
        self.permutation_seed = permutation_seed
        self.register_buffer("latent_mean", torch.zeros(self.latent_dim))
        self.register_buffer("latent_scale", torch.ones(self.latent_dim))
        self.supported_interventions: list[dict[str, int]] = []
        self.checkpoint_metadata: dict[str, Any] = {}
        self._config = {
            "latent_dim": self.latent_dim,
            "columns": [(column.name, column.n_categories) for column in self.columns],
            "n_blocks": self.n_blocks,
            "hidden_dim": self.hidden_dim,
            "condition_dim": self.condition_dim,
            "state_embed_dim": self.state_embed_dim,
            "permutation_seed": self.permutation_seed,
            "scale_floor": self.scale_floor,
            **extra_config,
        }

    @torch.no_grad()
    def set_latent_statistics(self, latents: torch.Tensor) -> None:
        _validate_latents(latents, self.latent_dim, nonempty=True)
        mean = latents.detach().mean(dim=0)
        scale = latents.detach().std(dim=0, unbiased=False).clamp_min(self.scale_floor)
        self.latent_mean.copy_(mean.to(self.latent_mean))
        self.latent_scale.copy_(scale.to(self.latent_scale))

    def save(
        self,
        path: str | Path,
        *,
        metadata: Mapping[str, Any] | None = None,
        supported_interventions: Sequence[Mapping[str, int]] | None = None,
    ) -> None:
        """Save a versioned, self-reconstructing flow checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        saved_metadata = dict(metadata or self.checkpoint_metadata)
        saved_metadata.setdefault("schema_signature", _schema_signature(self.columns))
        support = [
            dict(_canonical_intervention(intervention))
            for intervention in (
                supported_interventions
                if supported_interventions is not None
                else self.supported_interventions
            )
        ]
        self.checkpoint_metadata = dict(saved_metadata)
        self.supported_interventions = support
        torch.save(
            {
                "format_version": FLOW_CHECKPOINT_FORMAT_VERSION,
                "variant": self.variant,
                "class": type(self).__name__,
                "config": self._config,
                "state_dict": self.state_dict(),
                "supported_interventions": support,
                "metadata": saved_metadata,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device | None = None,
        expected_metadata: Mapping[str, Any] | None = None,
    ) -> Self:
        model = load_latent_intervention(
            path,
            device=device,
            expected_variant=cls.variant,
            expected_metadata=expected_metadata,
        )
        if not isinstance(model, cls):
            raise TypeError(f"checkpoint did not reconstruct {cls.__name__}")
        return model


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


class _DirectFlowIntervention(_FlowModelBase):
    """Conditional flow over the standardized residual ``(Z' - Z) / scale``."""

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
        base_std: float = 0.05,
    ) -> None:
        self.base_std = _positive_float("base_std", base_std)
        super().__init__(
            latent_dim,
            columns,
            n_blocks=n_blocks,
            hidden_dim=hidden_dim,
            condition_dim=condition_dim,
            state_embed_dim=state_embed_dim,
            permutation_seed=permutation_seed,
            scale_floor=scale_floor,
            base_std=self.base_std,
        )
        self.context = _DirectContext(
            latent_dim, self.columns, state_embed_dim, condition_dim
        )
        self.flow = _ConditionalRealNVP(
            latent_dim, condition_dim, hidden_dim, n_blocks, permutation_seed
        )

    def _condition(
        self, z: torch.Tensor, values: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = _model_device(self)
        z = z.to(device=device, dtype=self.latent_mean.dtype)
        values, mask = _validate_objective(
            values, mask, self.columns, z.shape[0], device
        )
        standardized_z = (z - self.latent_mean) / self.latent_scale
        return self.context(standardized_z, values, mask), values, mask

    def _sample_with_log_det(
        self,
        z: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        n_samples: int,
        *,
        generator: torch.Generator | None = None,
        noise: torch.Tensor | None = None,
        enforce_identity: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_latents(z, self.latent_dim)
        _positive_int("n_samples", n_samples)
        device = _model_device(self)
        z = z.to(device=device, dtype=self.latent_mean.dtype)
        condition, values, mask = self._condition(z, values, mask)
        if noise is None:
            noise = self.base_std * torch.randn(
                n_samples,
                z.shape[0],
                self.latent_dim,
                device=device,
                dtype=z.dtype,
                generator=generator,
            )
        else:
            if noise.ndim == 2:
                noise = noise.unsqueeze(0)
            expected = (n_samples, z.shape[0], self.latent_dim)
            if tuple(noise.shape) != expected:
                raise ValueError(f"noise must have shape {expected}, got {tuple(noise.shape)}")
            if not noise.is_floating_point() or not torch.isfinite(noise).all():
                raise ValueError("noise must be finite and floating point")
            noise = noise.to(device=device, dtype=z.dtype)

        repeated_condition = condition.unsqueeze(0).expand(n_samples, -1, -1).reshape(
            n_samples * z.shape[0], -1
        )
        residual, log_det = self.flow.forward_transform(
            noise.reshape(n_samples * z.shape[0], self.latent_dim), repeated_condition
        )
        residual = residual.reshape(n_samples, z.shape[0], self.latent_dim)
        samples = z.unsqueeze(0) + self.latent_scale * residual
        if enforce_identity:
            no_op = ~mask.any(dim=-1)
            samples = torch.where(no_op[None, :, None], z[None], samples)
        return samples, log_det.reshape(n_samples, z.shape[0])

    def sample(
        self,
        z: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        n_samples: int = 1,
        generator: torch.Generator | None = None,
        *,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Draw counterfactual latents, hard-routing empty interventions to ``Z``."""
        return self._sample_with_log_det(
            z,
            values,
            mask,
            n_samples,
            generator=generator,
            noise=noise,
            enforce_identity=True,
        )[0]

    def forward(
        self,
        z: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Return one draw with shape ``(batch, latent_dim)``."""
        return self.sample(z, values, mask, 1, generator)[0]

    def log_prob(
        self,
        z_prime: torch.Tensor,
        z: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``log q(Z' | Z, delta)`` for non-empty interventions."""
        _validate_latents(z, self.latent_dim)
        _validate_latents(z_prime, self.latent_dim)
        if z_prime.shape[0] != z.shape[0]:
            raise ValueError("z and z_prime must have equal batch sizes")
        device = _model_device(self)
        z = z.to(device=device, dtype=self.latent_mean.dtype)
        z_prime = z_prime.to(device=device, dtype=self.latent_mean.dtype)
        condition, _, mask = self._condition(z, values, mask)
        if (~mask.any(dim=-1)).any():
            raise ValueError("the exact no-op route is a point mass and has no flow density")
        residual = (z_prime - z) / self.latent_scale
        base, inverse_log_det = self.flow.inverse_transform(residual, condition)
        scaled = base / self.base_std
        base_log_prob = -0.5 * (
            scaled.pow(2) + math.log(2.0 * math.pi) + 2.0 * math.log(self.base_std)
        ).sum(dim=-1)
        return base_log_prob + inverse_log_det - self.latent_scale.log().sum()


class DistilledFlowIntervention(_DirectFlowIntervention):
    """Standalone direct flow trained from a state-flow teacher."""

    variant = "distilled_flow"


class DirectSemanticFlowIntervention(_DirectFlowIntervention):
    """Standalone direct flow trained only through frozen ``g`` and ``h_S``."""

    variant = "direct_semantic_flow"


# --- symbolic sampling and training objectives --------------------------------------


@torch.no_grad()
def sample_symbolic_states(
    h_s: SymbolicKernel,
    source_states: StateLike,
    columns: Sequence[ColumnSpec],
    intervention: Mapping[str, int],
    n_samples: int = 1,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw target states from ``h_S(. | S, delta)``.

    Returns a tensor with shape ``(n_samples, batch, n_columns)`` in schema order.
    """
    _positive_int("n_samples", n_samples)
    columns = _validate_columns(columns)
    if list(h_s.columns) != columns:
        raise ValueError("symbolic-kernel schema does not match the flow schema")
    source = _state_tensor(source_states, columns)
    device = source.device
    normalized = dict(_canonical_intervention(intervention))
    _objective_tensors(normalized, columns, source.shape[0], device)
    if not normalized:
        return source.unsqueeze(0).expand(n_samples, -1, -1).clone()
    matrix = h_s.transition_matrix(normalized).to(device)
    state_count = math.prod(column.n_categories for column in columns)
    if matrix.ndim != 2 or tuple(matrix.shape) != (state_count, state_count):
        raise ValueError(
            "symbolic transition matrix must have shape "
            f"({state_count}, {state_count})"
        )
    probabilities = matrix.to_dense() if matrix.is_sparse else matrix
    source_index = flat_state_index(source, columns)
    rows = probabilities.index_select(0, source_index).to(dtype=torch.float32)
    if not torch.isfinite(rows).all() or (rows < 0).any():
        raise ValueError("symbolic transition probabilities must be finite and non-negative")
    row_sum = rows.sum(dim=-1, keepdim=True)
    if (row_sum <= 0).any():
        raise ValueError("symbolic transition matrix contains an empty row")
    rows = rows / row_sum
    sampled = torch.multinomial(
        rows, n_samples, replacement=True, generator=generator
    ).transpose(0, 1)
    return unflatten_state_index(sampled, columns)


def multivariate_energy_distance(
    student_samples: torch.Tensor,
    teacher_samples: torch.Tensor,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Biased empirical multivariate energy distance, averaged over observations.

    Samples must have shapes ``(m, batch, dim)`` and ``(k, batch, dim)``.  The biased
    empirical form is non-negative and reaches zero when the two empirical measures
    agree.  Dividing by per-coordinate factual scales and ``sqrt(dim)`` makes its
    magnitude comparable across encoders and latent widths.
    """
    if not isinstance(student_samples, torch.Tensor) or not isinstance(
        teacher_samples, torch.Tensor
    ):
        raise TypeError("energy-distance samples must be torch.Tensor values")
    if student_samples.ndim != 3 or teacher_samples.ndim != 3:
        raise ValueError("energy-distance samples must have shape (samples, batch, dim)")
    if student_samples.shape[1:] != teacher_samples.shape[1:]:
        raise ValueError("student and teacher samples must share batch and latent dimensions")
    if student_samples.shape[0] == 0 or teacher_samples.shape[0] == 0:
        raise ValueError("energy-distance sample sets must not be empty")
    if not torch.isfinite(student_samples).all() or not torch.isfinite(teacher_samples).all():
        raise ValueError("energy-distance samples must be finite")
    if scale is not None:
        if scale.ndim != 1 or scale.shape[0] != student_samples.shape[-1]:
            raise ValueError("scale must have shape (latent_dim,)")
        if not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("scale must be finite and strictly positive")
        student_samples = student_samples / scale
        teacher_samples = teacher_samples / scale
    student = student_samples.transpose(0, 1)
    teacher = teacher_samples.transpose(0, 1)
    normalizer = math.sqrt(student_samples.shape[-1])
    cross = torch.cdist(student, teacher).mean(dim=(1, 2)) / normalizer
    within_student = torch.cdist(student, student).mean(dim=(1, 2)) / normalizer
    within_teacher = torch.cdist(teacher, teacher).mean(dim=(1, 2)) / normalizer
    return (2.0 * cross - within_student - within_teacher).mean().clamp_min(0.0)


def _validate_training_inputs(
    model: _FlowModelBase,
    latents: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
) -> None:
    _validate_latents(latents, model.latent_dim, nonempty=True)
    _positive_int("epochs", epochs)
    _positive_int("batch_size", batch_size)
    _positive_float("lr", lr)
    if not isinstance(weight_decay, (int, float)) or not math.isfinite(float(weight_decay)):
        raise ValueError("weight_decay must be a finite non-negative number")
    if float(weight_decay) < 0:
        raise ValueError("weight_decay must be a finite non-negative number")
    _positive_float("grad_clip", grad_clip)
    _warn_small_sample(latents.shape[0], model.latent_dim)


def _weighted_log_update(
    totals: dict[str, float], log: Mapping[str, float], batch_size: int
) -> None:
    for name, value in log.items():
        totals[name] = totals.get(name, 0.0) + float(value) * batch_size


def _finish_epoch(
    totals: Mapping[str, float], n: int, epoch: int, epochs: int, verbose: bool
) -> dict[str, float]:
    row = {name: value / n for name, value in totals.items()}
    if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
        print(
            f"epoch {epoch + 1}/{epochs}  "
            + "  ".join(f"{name} {value:.4f}" for name, value in row.items())
        )
    return row


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
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            _weighted_log_update(
                totals,
                {"total": nll.item(), "nll": nll.item(), "grad_norm": float(grad_norm)},
                len(index),
            )
        history.append(_finish_epoch(totals, len(z), epoch, epochs, verbose))

    model.eval()
    return history


def train_distilled_flow(
    model: DistilledFlowIntervention,
    teacher: StateConditionalFlow,
    h_s: SymbolicKernel,
    latents: torch.Tensor,
    states: StateLike,
    intervention: Mapping[str, int],
    *,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    n_samples: int = 8,
    identity_fraction: float = 0.2,
    grad_clip: float = 5.0,
    seed: int | None = None,
    device: str | torch.device | None = None,
    verbose: bool = True,
) -> list[dict[str, float]]:
    """Distil a state-flow teacher with scale-normalized energy distance."""
    if not isinstance(model, DistilledFlowIntervention):
        raise TypeError("model must be a DistilledFlowIntervention")
    if not isinstance(teacher, StateConditionalFlow):
        raise TypeError("teacher must be a StateConditionalFlow")
    if teacher.latent_dim != model.latent_dim or teacher.columns != model.columns:
        raise ValueError("teacher and student flow schemas must match")
    _validate_training_inputs(
        model,
        latents,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
    )
    _positive_int("n_samples", n_samples)
    identity_fraction = _unit_interval("identity_fraction", identity_fraction)
    source = _state_tensor(states, model.columns, batch_size=latents.shape[0])
    model.set_latent_statistics(latents)
    run_device = torch.device(device or "cpu")
    model.to(run_device).train()
    teacher.to(run_device).eval().requires_grad_(False)
    z = latents.to(device=run_device, dtype=model.latent_mean.dtype)
    source = source.to(run_device)
    values, mask = _objective_tensors(
        intervention, model.columns, 1, run_device
    )
    shuffle = _make_generator(run_device, seed, offset=1)
    symbolic_rng = _make_generator(run_device, seed, offset=2)
    flow_rng = _make_generator(run_device, seed, offset=3)
    identity_rng = _make_generator(run_device, seed, offset=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        totals: dict[str, float] = {}
        order = torch.randperm(z.shape[0], generator=shuffle, device=run_device)
        for start in range(0, z.shape[0], batch_size):
            index = order[start : start + batch_size]
            batch_z = z[index]
            batch_source = source[index]
            batch_values = values.expand(len(index), -1).clone()
            batch_mask = mask.expand(len(index), -1).clone()
            identity = (
                torch.rand(len(index), generator=identity_rng, device=run_device)
                < identity_fraction
            )
            batch_mask[identity] = False

            with torch.no_grad():
                target_states = sample_symbolic_states(
                    h_s,
                    batch_source,
                    model.columns,
                    intervention,
                    n_samples,
                    generator=symbolic_rng,
                )
                target_states[:, identity] = batch_source[identity]
                u = teacher.abduct(batch_z, batch_source)
                repeated_u = u.unsqueeze(0).expand(n_samples, -1, -1).reshape(
                    n_samples * len(index), model.latent_dim
                )
                teacher_samples = teacher.render(
                    repeated_u,
                    target_states.reshape(n_samples * len(index), -1),
                ).reshape(n_samples, len(index), model.latent_dim)

            student_samples, _ = model._sample_with_log_det(
                batch_z,
                batch_values,
                batch_mask,
                n_samples,
                generator=flow_rng,
                enforce_identity=False,
            )
            energy = multivariate_energy_distance(
                student_samples, teacher_samples, model.latent_scale
            )
            optimizer.zero_grad(set_to_none=True)
            energy.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            spread = (
                (student_samples / model.latent_scale)
                .std(dim=0, unbiased=False)
                .norm(dim=-1)
                .mean()
            )
            _weighted_log_update(
                totals,
                {
                    "total": energy.item(),
                    "energy": energy.item(),
                    "spread": spread.item(),
                    "identity_fraction": identity.float().mean().item(),
                    "grad_norm": float(grad_norm),
                },
                len(index),
            )
        history.append(_finish_epoch(totals, len(z), epoch, epochs, verbose))

    model.supported_interventions = _intervention_support(intervention)
    model.eval()
    return history


@torch.no_grad()
def _semantic_target(
    decoder: SemanticDecoderModel,
    h_s: SymbolicKernel,
    latents: torch.Tensor,
    intervention: Mapping[str, int],
    *,
    chunk_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = h_s.transition_matrix(dict(intervention)).to(latents.device)
    transposed = matrix.t().coalesce() if matrix.is_sparse else matrix.t()
    factual_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    for start in range(0, latents.shape[0], chunk_size):
        factual = decoder.log_joint(latents[start : start + chunk_size]).exp()
        target = (
            torch.sparse.mm(transposed, factual.t()).t()
            if transposed.is_sparse
            else factual @ matrix
        )
        factual_parts.append(factual)
        target_parts.append(target)
    return torch.cat(target_parts), torch.cat(factual_parts)


def train_direct_semantic_flow(
    model: DirectSemanticFlowIntervention,
    decoder: SemanticDecoderModel,
    h_s: SymbolicKernel,
    latents: torch.Tensor,
    intervention: Mapping[str, int],
    *,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    n_samples: int = 8,
    identity_fraction: float = 0.2,
    entropy_weight: float = 0.1,
    proximity_weight: float = 0.1,
    support_weight: float = 0.05,
    identity_weight: float = 1.0,
    support_bank_size: int = 512,
    grad_clip: float = 5.0,
    seed: int | None = None,
    device: str | torch.device | None = None,
    verbose: bool = True,
) -> list[dict[str, float]]:
    """Fit the direct flow by semantic consistency, confidence, and support."""
    if not isinstance(model, DirectSemanticFlowIntervention):
        raise TypeError("model must be a DirectSemanticFlowIntervention")
    if list(decoder.columns) != model.columns or list(h_s.columns) != model.columns:
        raise ValueError("decoder, symbolic kernel, and flow schemas must match")
    _validate_training_inputs(
        model,
        latents,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
    )
    if n_samples < 2 or n_samples % 2:
        raise ValueError("n_samples must be an even integer >= 2 for antithetic sampling")
    identity_fraction = _unit_interval("identity_fraction", identity_fraction)
    for name, weight in (
        ("entropy_weight", entropy_weight),
        ("proximity_weight", proximity_weight),
        ("support_weight", support_weight),
        ("identity_weight", identity_weight),
    ):
        if not isinstance(weight, (int, float)) or not math.isfinite(float(weight)) or weight < 0:
            raise ValueError(f"{name} must be a finite non-negative number")
    _positive_int("support_bank_size", support_bank_size)

    model.set_latent_statistics(latents)
    run_device = torch.device(device or "cpu")
    model.to(run_device).train()
    decoder.to(run_device).eval().requires_grad_(False)
    z = latents.to(device=run_device, dtype=model.latent_mean.dtype)
    with torch.no_grad():
        target, factual = _semantic_target(decoder, h_s, z, intervention)
    values, mask = _objective_tensors(intervention, model.columns, 1, run_device)
    shuffle = _make_generator(run_device, seed, offset=1)
    flow_rng = _make_generator(run_device, seed, offset=2)
    identity_rng = _make_generator(run_device, seed, offset=3)
    bank_rng = _make_generator(run_device, seed, offset=4)
    bank_count = min(support_bank_size, len(z))
    bank_index = torch.randperm(len(z), generator=bank_rng, device=run_device)[:bank_count]
    support_bank = ((z[bank_index] - model.latent_mean) / model.latent_scale).detach()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        totals: dict[str, float] = {}
        order = torch.randperm(z.shape[0], generator=shuffle, device=run_device)
        for start in range(0, z.shape[0], batch_size):
            index = order[start : start + batch_size]
            batch_z = z[index]
            batch_values = values.expand(len(index), -1).clone()
            batch_mask = mask.expand(len(index), -1).clone()
            identity = (
                torch.rand(len(index), generator=identity_rng, device=run_device)
                < identity_fraction
            )
            batch_mask[identity] = False
            batch_target = target[index].clone()
            batch_target[identity] = factual[index][identity]

            half_noise = model.base_std * torch.randn(
                n_samples // 2,
                len(index),
                model.latent_dim,
                device=run_device,
                dtype=z.dtype,
                generator=flow_rng,
            )
            noise = torch.cat((half_noise, -half_noise), dim=0)
            samples, log_det = model._sample_with_log_det(
                batch_z,
                batch_values,
                batch_mask,
                n_samples,
                noise=noise,
                enforce_identity=False,
            )
            log_joints = torch.stack([decoder.log_joint(sample) for sample in samples])
            composed = torch.logsumexp(log_joints, dim=0) - math.log(n_samples)
            semantic_kl = F.kl_div(composed, batch_target, reduction="batchmean")
            entropy = -(log_joints.exp() * log_joints).sum(dim=-1).mean()
            scaled_shift = (samples - batch_z.unsqueeze(0)) / model.latent_scale
            proximity = F.smooth_l1_loss(scaled_shift, torch.zeros_like(scaled_shift))
            standardized_samples = (
                (samples - model.latent_mean) / model.latent_scale
            ).reshape(n_samples * len(index), model.latent_dim)
            support = (
                torch.cdist(standardized_samples, support_bank).min(dim=-1).values
                / math.sqrt(model.latent_dim)
            ).mean()
            identity_loss = (
                F.smooth_l1_loss(
                    scaled_shift[:, identity], torch.zeros_like(scaled_shift[:, identity])
                )
                if identity.any()
                else samples.sum() * 0.0
            )
            total = (
                semantic_kl
                + entropy_weight * entropy
                + proximity_weight * proximity
                + support_weight * support
                + identity_weight * identity_loss
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            spread = scaled_shift.std(dim=0, unbiased=False).norm(dim=-1).mean()
            _weighted_log_update(
                totals,
                {
                    "total": total.item(),
                    "semantic_kl": semantic_kl.item(),
                    "entropy": entropy.item(),
                    "proximity": proximity.item(),
                    "support": support.item(),
                    "identity": identity_loss.item(),
                    "spread": spread.item(),
                    "log_det": log_det.mean().item(),
                    "grad_norm": float(grad_norm),
                },
                len(index),
            )
        history.append(_finish_epoch(totals, len(z), epoch, epochs, verbose))

    model.supported_interventions = _intervention_support(intervention)
    model.eval()
    return history


# --- central construction, persistence, training, and inference ---------------------


def make_latent_intervention(
    variant: str,
    *,
    latent_dim: int,
    columns: Sequence[ColumnSpec],
    seed: int | None = None,
    # Existing transformer defaults.
    d_model: int = 128,
    nhead: int = 4,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    noise_std: float = 1.0,
    noise_dim: int = 4,
    embed_dim: int = 16,
    top_k: int = 16,
    n_particles: int = 16,
    uniform_weights: bool = False,
    # Flow defaults.
    n_blocks: int = 8,
    hidden_dim: int = 128,
    condition_dim: int = 64,
    state_embed_dim: int = 16,
    permutation_seed: int = 0,
    scale_floor: float = 1e-6,
    base_std: float = 0.05,
) -> nn.Module:
    """Construct any existing or flow-based ``h_Z`` variant at one dispatch point."""
    columns = _validate_columns(columns)

    def construct() -> nn.Module:
        if variant == StateConditionalFlow.variant:
            return StateConditionalFlow(
                latent_dim,
                columns,
                n_blocks=n_blocks,
                hidden_dim=hidden_dim,
                condition_dim=condition_dim,
                state_embed_dim=state_embed_dim,
                permutation_seed=permutation_seed,
                scale_floor=scale_floor,
            )
        if variant == DistilledFlowIntervention.variant:
            return DistilledFlowIntervention(
                latent_dim,
                columns,
                n_blocks=n_blocks,
                hidden_dim=hidden_dim,
                condition_dim=condition_dim,
                state_embed_dim=state_embed_dim,
                permutation_seed=permutation_seed,
                scale_floor=scale_floor,
                base_std=base_std,
            )
        if variant == DirectSemanticFlowIntervention.variant:
            return DirectSemanticFlowIntervention(
                latent_dim,
                columns,
                n_blocks=n_blocks,
                hidden_dim=hidden_dim,
                condition_dim=condition_dim,
                state_embed_dim=state_embed_dim,
                permutation_seed=permutation_seed,
                scale_floor=scale_floor,
                base_std=base_std,
            )

        from src.latent_intervention import (
            LatentIntervention,
            LatentInterventionDist,
            LatentInterventionNoiseToken,
            LatentInterventionParticles,
            LatentInterventionPreAdditive,
        )

        common = {
            "latent_dim": latent_dim,
            "columns": list(columns),
            "d_model": d_model,
            "nhead": nhead,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
        }
        if variant == "baseline":
            return LatentIntervention(**common)
        if variant == "pre_additive":
            return LatentInterventionPreAdditive(**common, noise_std=noise_std)
        if variant == "noise_token":
            return LatentInterventionNoiseToken(**common, noise_dim=noise_dim)
        if variant == "dist":
            return LatentInterventionDist(**common, embed_dim=embed_dim, top_k=top_k)
        if variant == "particles":
            return LatentInterventionParticles(
                **common,
                n_particles=n_particles,
                uniform_weights=uniform_weights,
            )
        raise ValueError(
            f"unknown latent intervention variant {variant!r}; expected one of "
            "'baseline', 'pre_additive', 'noise_token', 'dist', 'particles', "
            "'state_flow', 'distilled_flow', or 'direct_semantic_flow'"
        )

    if seed is None:
        return construct()
    # Construction is isolated so the configured seed controls initialization without
    # perturbing unrelated application-level random streams.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return construct()


def _checkpoint_error(message: str) -> ValueError:
    return ValueError(f"{message}; re-encode/retrain the dependent artifacts")


def _metadata_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and all(
            key in actual and _metadata_matches(actual[key], value)
            for key, value in expected.items()
        )
    return actual == expected


def load_latent_intervention(
    path: str | Path,
    *,
    device: str | torch.device | None = None,
    expected_variant: str | None = None,
    expected_columns: Sequence[ColumnSpec] | None = None,
    expected_metadata: Mapping[str, Any] | None = None,
) -> nn.Module:
    """Load a flow checkpoint, or delegate legacy checkpoints to their public loader."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise _checkpoint_error("latent-intervention checkpoint is malformed")

    if payload.get("format_version") != FLOW_CHECKPOINT_FORMAT_VERSION:
        # Existing transformer checkpoints predate the versioned flow format.
        legacy_name = payload.get("class")
        from src.latent_intervention import (
            LatentIntervention,
            LatentInterventionDist,
            LatentInterventionNoiseToken,
            LatentInterventionParticles,
            LatentInterventionPreAdditive,
        )

        legacy_classes = {
            cls.__name__: cls
            for cls in (
                LatentIntervention,
                LatentInterventionPreAdditive,
                LatentInterventionNoiseToken,
                LatentInterventionDist,
                LatentInterventionParticles,
            )
        }
        legacy_variants = {
            "LatentIntervention": "baseline",
            "LatentInterventionPreAdditive": "pre_additive",
            "LatentInterventionNoiseToken": "noise_token",
            "LatentInterventionDist": "dist",
            "LatentInterventionParticles": "particles",
        }
        if legacy_name not in legacy_classes:
            raise _checkpoint_error("unsupported latent-intervention checkpoint format")
        if expected_variant is not None and legacy_variants[legacy_name] != expected_variant:
            raise _checkpoint_error(
                f"latent-intervention variant mismatch: expected {expected_variant!r}, "
                f"got {legacy_variants[legacy_name]!r}"
            )
        if expected_metadata:
            raise _checkpoint_error("legacy checkpoint has no embedded provenance metadata")
        model = legacy_classes[legacy_name].load(path, device=device)
        if expected_columns is not None and list(model.columns) != list(expected_columns):
            raise _checkpoint_error("latent-intervention schema mismatch")
        return model.to(torch.device(device or "cpu")).eval()

    variant = payload.get("variant")
    if variant not in FLOW_VARIANTS:
        raise _checkpoint_error(f"unknown flow variant {variant!r}")
    if expected_variant is not None and variant != expected_variant:
        raise _checkpoint_error(
            f"latent-intervention variant mismatch: expected {expected_variant!r}, got {variant!r}"
        )
    config = payload.get("config")
    if not isinstance(config, dict):
        raise _checkpoint_error("flow checkpoint has no valid constructor config")
    raw_columns = config.get("columns")
    if not isinstance(raw_columns, (list, tuple)):
        raise _checkpoint_error("flow checkpoint has no valid schema")
    try:
        columns = _validate_columns(
            [ColumnSpec(str(name), int(cardinality)) for name, cardinality in raw_columns]
        )
    except (TypeError, ValueError) as error:
        raise _checkpoint_error("flow checkpoint has a malformed schema") from error
    if expected_columns is not None and columns != list(expected_columns):
        raise _checkpoint_error("latent-intervention schema mismatch")

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise _checkpoint_error("flow checkpoint has no provenance metadata")
    if metadata.get("schema_signature") != _schema_signature(columns):
        raise _checkpoint_error("flow checkpoint schema signature mismatch")
    if expected_metadata is not None and not _metadata_matches(metadata, expected_metadata):
        raise _checkpoint_error("flow checkpoint provenance mismatch")

    constructor = dict(config)
    constructor["columns"] = columns
    model = make_latent_intervention(variant, **constructor)
    if not isinstance(model, _FlowModelBase):
        raise _checkpoint_error("checkpoint did not reconstruct a flow model")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise _checkpoint_error("flow checkpoint has no state dictionary")
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise _checkpoint_error("flow checkpoint weights are incompatible") from error
    support = payload.get("supported_interventions", [])
    if not isinstance(support, list) or any(not isinstance(item, dict) for item in support):
        raise _checkpoint_error("flow checkpoint intervention support is malformed")
    model.supported_interventions = [
        {str(name): int(value) for name, value in item.items()} for item in support
    ]
    model.checkpoint_metadata = dict(metadata)
    model.to(torch.device(device or "cpu")).eval()
    return model


def train_latent_intervention_model(
    model: nn.Module,
    *,
    latents: torch.Tensor,
    intervention: Mapping[str, int],
    decoder: SemanticDecoderModel | None = None,
    h_s: SymbolicKernel | None = None,
    states: StateLike | None = None,
    s_prime: Mapping[str, torch.Tensor] | None = None,
    teacher: StateConditionalFlow | None = None,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    n_samples: int = 8,
    identity_fraction: float = 0.2,
    entropy_weight: float = 0.1,
    proximity_weight: float = 0.1,
    support_weight: float = 0.05,
    identity_weight: float = 1.0,
    support_bank_size: int = 512,
    grad_clip: float = 5.0,
    sparsity_weight: float = 1.0,
    seed: int | None = None,
    device: str | torch.device | None = None,
    verbose: bool = True,
    **variant_kwargs: Any,
) -> list[dict[str, float]]:
    """Train any ``h_Z`` variant through one type-safe dispatch point."""
    columns = getattr(model, "columns", None)
    if columns is not None:
        _objective_tensors(intervention, columns, 1, _model_device(model))
    shared_flow = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "seed": seed,
        "device": device,
        "verbose": verbose,
    }
    if isinstance(model, StateConditionalFlow):
        if states is None:
            raise ValueError("state_flow training requires factual `states`")
        history = train_state_conditional_flow(model, latents, states, **shared_flow)
        model.supported_interventions = _intervention_support(intervention)
        return history
    if isinstance(model, DistilledFlowIntervention):
        if states is None or teacher is None or h_s is None:
            raise ValueError("distilled_flow training requires `states`, `teacher`, and `h_s`")
        return train_distilled_flow(
            model,
            teacher,
            h_s,
            latents,
            states,
            intervention,
            n_samples=n_samples,
            identity_fraction=identity_fraction,
            **shared_flow,
        )
    if isinstance(model, DirectSemanticFlowIntervention):
        if decoder is None or h_s is None:
            raise ValueError("direct_semantic_flow training requires `decoder` and `h_s`")
        return train_direct_semantic_flow(
            model,
            decoder,
            h_s,
            latents,
            intervention,
            n_samples=n_samples,
            identity_fraction=identity_fraction,
            entropy_weight=entropy_weight,
            proximity_weight=proximity_weight,
            support_weight=support_weight,
            identity_weight=identity_weight,
            support_bank_size=support_bank_size,
            **shared_flow,
        )

    if decoder is None:
        raise ValueError("existing transformer variants require `decoder`")
    from src.latent_intervention import (
        LatentIntervention,
        LatentInterventionDist,
        LatentInterventionNoiseToken,
        LatentInterventionParticles,
        LatentInterventionPreAdditive,
        train_latent_intervention,
        train_latent_intervention_dist,
        train_latent_intervention_noise_token,
        train_latent_intervention_particles,
        train_latent_intervention_preadditive,
    )

    common = {
        "model": model,
        "decoder": decoder,
        "latents": latents,
        "intervention": dict(intervention),
        "s_prime": dict(s_prime) if s_prime is not None else None,
        "h_s": h_s,
        "batch_size": batch_size,
        "lr": lr,
        "proximity_weight": proximity_weight,
        "sparsity_weight": sparsity_weight,
        "seed": seed,
        "device": device,
        "verbose": verbose,
    }
    if isinstance(model, LatentIntervention):
        return train_latent_intervention(epochs=epochs, **common)
    if isinstance(model, LatentInterventionPreAdditive):
        return train_latent_intervention_preadditive(
            epochs=epochs, n_samples=n_samples, **common
        )
    if isinstance(model, LatentInterventionNoiseToken):
        return train_latent_intervention_noise_token(
            epochs=epochs,
            n_samples=n_samples,
            entropy_weight=entropy_weight,
            **common,
        )
    if isinstance(model, LatentInterventionDist):
        return train_latent_intervention_dist(
            pretrain_epochs=int(variant_kwargs.pop("pretrain_epochs", 30)),
            joint_epochs=int(variant_kwargs.pop("joint_epochs", 20)),
            realiser_l1=float(variant_kwargs.pop("realiser_l1", 1.0)),
            realiser_l2=float(variant_kwargs.pop("realiser_l2", 1.0)),
            **common,
        )
    if isinstance(model, LatentInterventionParticles):
        return train_latent_intervention_particles(
            epochs=epochs,
            entropy_weight=entropy_weight,
            **common,
        )
    raise TypeError(f"unsupported latent-intervention model type: {type(model).__name__}")


def _warn_if_unsupported(model: nn.Module, intervention: Mapping[str, int]) -> None:
    support = getattr(model, "supported_interventions", None)
    if not support or not intervention:
        return
    canonical = _canonical_intervention(intervention)
    if canonical not in {_canonical_intervention(item) for item in support}:
        warnings.warn(
            f"intervention {dict(intervention)!r} was not present in this model's training "
            "support; the result is an unsupported extrapolation",
            RuntimeWarning,
            stacklevel=3,
        )


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


@torch.no_grad()
def sample_counterfactual(
    model: nn.Module,
    z: torch.Tensor,
    intervention: Mapping[str, int] | None = None,
    *,
    n_samples: int = 1,
    source_states: StateLike | None = None,
    target_states: StateLike | None = None,
    decoder: SemanticDecoderModel | None = None,
    h_s: SymbolicKernel | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample from any ``h_Z`` using a uniform ``(samples, batch, dim)`` interface."""
    _positive_int("n_samples", n_samples)
    if not isinstance(z, torch.Tensor) or z.ndim != 2:
        raise ValueError("z must have shape (batch, latent_dim)")
    intervention = dict(intervention or {})
    _warn_if_unsupported(model, intervention)

    if isinstance(model, StateConditionalFlow):
        _validate_latents(z, model.latent_dim)
        device = _model_device(model)
        z = z.to(device=device, dtype=model.latent_mean.dtype)
        _objective_tensors(intervention, model.columns, 1, device)
        source = _source_states_for_inference(model, z, source_states, decoder)
        if target_states is not None:
            target = _state_tensor(
                target_states,
                model.columns,
                batch_size=z.shape[0],
                device=device,
            ).unsqueeze(0).expand(n_samples, -1, -1)
        elif not intervention:
            target = source.unsqueeze(0).expand(n_samples, -1, -1)
        else:
            if h_s is None:
                raise ValueError(
                    "state_flow inference needs explicit `target_states` or `h_s`"
                )
            target = sample_symbolic_states(
                h_s,
                source,
                model.columns,
                intervention,
                n_samples,
                generator=generator,
            )
        u = model.abduct(z, source)
        repeated_u = u.unsqueeze(0).expand(n_samples, -1, -1).reshape(
            n_samples * z.shape[0], model.latent_dim
        )
        return model.render(
            repeated_u, target.reshape(n_samples * z.shape[0], -1)
        ).reshape(n_samples, z.shape[0], model.latent_dim)

    columns = getattr(model, "columns", None)
    latent_dim = getattr(model, "latent_dim", None)
    if latent_dim is None:
        config = getattr(model, "_config", None)
        if isinstance(config, Mapping):
            latent_dim = config.get("latent_dim")
    if columns is None or latent_dim is None:
        raise TypeError(f"unsupported latent-intervention model type: {type(model).__name__}")
    _validate_latents(z, int(latent_dim))
    device = _model_device(model)
    z = z.to(device=device, dtype=_model_dtype(model))
    values, mask = _objective_tensors(intervention, columns, len(z), device)
    if isinstance(model, _DirectFlowIntervention):
        return model.sample(z, values, mask, n_samples, generator)
    sample_method = getattr(model, "sample", None)
    if callable(sample_method):
        return sample_method(z, values, mask, n_samples, generator)
    return torch.stack([model(z, values, mask) for _ in range(n_samples)])


@torch.no_grad()
def counterfactual(
    model: nn.Module,
    z: torch.Tensor,
    intervention: Mapping[str, int] | None = None,
    *,
    source_states: StateLike | None = None,
    target_states: StateLike | None = None,
    decoder: SemanticDecoderModel | None = None,
    h_s: SymbolicKernel | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return one counterfactual draw with shape ``(batch, latent_dim)``."""
    return sample_counterfactual(
        model,
        z,
        intervention,
        n_samples=1,
        source_states=source_states,
        target_states=target_states,
        decoder=decoder,
        h_s=h_s,
        generator=generator,
    )[0]
