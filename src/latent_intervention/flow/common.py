"""Shared normalizing-flow machinery used by all three flow variants.

Validation helpers, the conditional RealNVP, the checkpointed model base class, symbolic
state sampling, the energy distance and the per-epoch training utilities.
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
from torch import nn

from src.schema import ColumnSpec, flat_state_index, unflatten_state_index
from src.symbolic_intervention import SymbolicKernel

__all__ = [
    "FLOW_CHECKPOINT_FORMAT_VERSION",
    "FLOW_VARIANTS",
    "StateLike",
    "sample_symbolic_states",
    "multivariate_energy_distance",
]


FLOW_CHECKPOINT_FORMAT_VERSION = 1


FLOW_VARIANTS = ("state_flow", "distilled_flow", "direct_semantic_flow")


StateLike: TypeAlias = torch.Tensor | Mapping[str, torch.Tensor]


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
    from src.latent_intervention.base import make_objective

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
        from src.latent_intervention.dispatch import load_latent_intervention

        model = load_latent_intervention(
            path,
            device=device,
            expected_variant=cls.variant,
            expected_metadata=expected_metadata,
        )
        if not isinstance(model, cls):
            raise TypeError(f"checkpoint did not reconstruct {cls.__name__}")
        return model


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
