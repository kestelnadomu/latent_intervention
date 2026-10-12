"""Optimization, validation selection and early stopping for semantic decoders.

The numerical procedure is unchanged; model definitions and evaluation metrics
remain in semantic_decoder.py, which re-exports train_semantic_decoder.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from time import perf_counter
from typing import Any, TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from src.semantic_decoder import SemanticDecoderModel


def _reset_trainable_modules(module: nn.Module) -> None:
    reset_parameters = getattr(module, "reset_parameters", None)
    if callable(reset_parameters):
        reset_parameters()


def train_semantic_decoder(
    decoder: SemanticDecoderModel,
    latents: torch.Tensor,
    targets: Mapping[str, torch.Tensor],
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str | torch.device | None = None,
    verbose: bool = True,
    seed: int = 0,
    *,
    validation_latents: torch.Tensor | None = None,
    validation_targets: Mapping[str, torch.Tensor] | None = None,
    patience: int = 30,
    min_delta: float = 1e-4,
    scheduler_patience: int = 10,
    scheduler_factor: float = 0.5,
    min_lr: float = 1e-6,
    weight_decay: float = 0.0,
    epoch_callback: (
        Callable[[SemanticDecoderModel, dict[str, Any]], None] | None
    ) = None,
) -> list[float]:
    """Train from scratch; restore minimum-validation-NLL weights when supplied.

    The returned losses retain the historical optimization scale. Per-epoch
    records and ``decoder.training_summary`` use full joint NLL for both variants.
    Patience counts improvements of at least ``min_delta``; checkpoint selection
    always tracks the literal lowest finite validation loss.
    """
    # Local imports preserve the historical semantic_decoder training API without
    # an import-time cycle between the model definitions and their training loop.
    from src.semantic_decoder import (
        _SemanticDecoderBase,
        _positive_int,
        _nonnegative_int,
        _model_device,
        _model_dtype,
        joint_nll,
    )

    if not isinstance(decoder, _SemanticDecoderBase):
        raise TypeError("decoder must be a semantic decoder")
    _positive_int("epochs", epochs)
    _positive_int("batch_size", batch_size)
    _nonnegative_int("seed", seed)
    if isinstance(lr, bool) or not isinstance(lr, (int, float)):
        raise ValueError("lr must be a positive finite number")
    learning_rate = float(lr)
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("lr must be a positive finite number")
    decoder._validate_latents(latents, nonempty=True)
    decoder._validate_targets(targets, latents.shape[0])
    if (validation_latents is None) != (validation_targets is None):
        raise ValueError("validation latents and targets must be supplied together")
    _positive_int("patience", patience)
    _positive_int("scheduler_patience", scheduler_patience)
    for name, value in (("min_delta", min_delta), ("weight_decay", weight_decay)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be nonnegative and finite")
    if not 0 < scheduler_factor < 1:
        raise ValueError("scheduler_factor must lie between 0 and 1")
    if not math.isfinite(min_lr) or not 0 < min_lr <= learning_rate:
        raise ValueError("min_lr must be positive and no larger than lr")
    if validation_latents is not None:
        decoder._validate_latents(validation_latents, nonempty=True)
        decoder._validate_targets(validation_targets, len(validation_latents))

    training_device = (
        torch.device(device) if device is not None else _model_device(decoder)
    )
    decoder.to(training_device)
    latents = latents.to(device=training_device, dtype=_model_dtype(decoder))
    targets = {
        name: values.to(device=training_device, dtype=torch.long)
        for name, values in targets.items()
    }
    n_observations = latents.shape[0]
    cuda_devices: list[int] = []
    if training_device.type == "cuda":
        cuda_devices = [
            (
                training_device.index
                if training_device.index is not None
                else torch.cuda.current_device()
            )
        ]

    history: list[float] = []
    records: list[dict[str, Any]] = []
    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    patience_reference = float("inf")
    stale_epochs = 0
    optimizer_steps = 0
    started = perf_counter()
    joint_multiplier = len(decoder.columns) if decoder.variant == "independent" else 1
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        decoder.apply(_reset_trainable_modules)
        optimizer = torch.optim.Adam(
            decoder.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            threshold=min_delta,
            threshold_mode="abs",
            min_lr=min_lr,
        )
        decoder.train()

        for epoch in range(epochs):
            epoch_started = perf_counter()
            epoch_lr = float(optimizer.param_groups[0]["lr"])
            permutation = torch.randperm(n_observations, device=training_device)
            weighted_loss = 0.0
            for start in range(0, n_observations, batch_size):
                index = permutation[start : start + batch_size]
                batch_targets = {
                    name: values[index] for name, values in targets.items()
                }
                loss = decoder.nll(latents[index], batch_targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite training loss at epoch {epoch + 1}"
                    )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer_steps += 1
                weighted_loss += loss.item() * index.numel()
            history.append(weighted_loss / n_observations)
            validation_loss = None
            improved = False
            if validation_latents is not None:
                validation_loss = joint_nll(
                    decoder, validation_latents, validation_targets
                )
                if not math.isfinite(validation_loss):
                    raise FloatingPointError(
                        f"non-finite validation loss at epoch {epoch + 1}"
                    )
                if validation_loss < best_loss:
                    best_loss = validation_loss
                    best_epoch = epoch + 1
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in decoder.state_dict().items()
                    }
                    improved = True
                if validation_loss < patience_reference - min_delta:
                    patience_reference = validation_loss
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                scheduler.step(validation_loss)
            record = {
                "epoch": epoch + 1,
                "train_loss": history[-1],
                "train_joint_nll": history[-1] * joint_multiplier,
                "validation_joint_nll": validation_loss,
                "learning_rate": epoch_lr,
                "next_learning_rate": float(optimizer.param_groups[0]["lr"]),
                "best_epoch": best_epoch,
                "improved": improved,
                "stale_epochs": stale_epochs,
                "optimizer_steps": optimizer_steps,
                "seconds": perf_counter() - epoch_started,
            }
            records.append(record)
            if epoch_callback is not None:
                epoch_callback(decoder, record)
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(
                    f"epoch {epoch + 1}/{epochs}  joint NLL {record['train_joint_nll']:.4f}"
                    f"  validation {validation_loss}  best epoch {best_epoch}",
                    flush=True,
                )
            if validation_latents is not None and stale_epochs >= patience:
                break

    if best_state is not None:
        decoder.load_state_dict(best_state)
    if not all(torch.isfinite(parameter).all() for parameter in decoder.parameters()):
        raise FloatingPointError("non-finite trained decoder parameters")
    decoder.eval()
    decoder.training_summary = {
        "max_epochs": epochs,
        "epochs_run": len(history),
        "best_epoch": best_epoch if best_state is not None else len(history),
        "best_validation_joint_nll": best_loss if best_state is not None else None,
        "stopped_early": len(history) < epochs,
        "stop_reason": "early_stopping" if len(history) < epochs else "max_epochs",
        "optimizer_steps": optimizer_steps,
        "seconds": perf_counter() - started,
        "history": records,
    }
    return history
