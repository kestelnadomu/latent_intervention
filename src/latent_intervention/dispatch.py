"""Central construction, persistence, training and inference for the h_Z variants.

Covers the five base (transformer) plans and the three flow plans behind one interface;
the oracle-regression reference is dispatched separately by its callers.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.schema import ColumnSpec
from src.semantic_decoder.model import SemanticDecoderModel
from src.symbolic_intervention import SymbolicKernel

from src.latent_intervention.flow.common import (
    FLOW_CHECKPOINT_FORMAT_VERSION,
    FLOW_VARIANTS,
    StateLike,
    _FlowModelBase,
    _canonical_intervention,
    _intervention_support,
    _model_device,
    _model_dtype,
    _objective_tensors,
    _positive_int,
    _schema_signature,
    _state_tensor,
    _validate_columns,
    _validate_latents,
    sample_symbolic_states,
)
from src.latent_intervention.flow.direct import (
    _DirectFlowIntervention,
)
from src.latent_intervention.flow.direct_semantic import (
    DirectSemanticFlowIntervention,
    train_direct_semantic_flow,
)
from src.latent_intervention.flow.distilled import (
    DistilledFlowIntervention,
    train_distilled_flow,
)
from src.latent_intervention.flow.state_conditional import (
    StateConditionalFlow,
    _source_states_for_inference,
    train_state_conditional_flow,
)

__all__ = [
    "make_latent_intervention",
    "load_latent_intervention",
    "train_latent_intervention_model",
    "sample_counterfactual",
    "counterfactual",
]


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

        from src.latent_intervention.base import (
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
        from src.latent_intervention.base import (
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
    epoch_callback: Callable | None = None,
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
        "epoch_callback": epoch_callback,
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
    from src.latent_intervention.base import (
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
        "epoch_callback": epoch_callback,
        # Legacy fixed-epoch calls retain their original unclipped optimizer.
        "grad_clip": grad_clip if epoch_callback is not None else None,
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
