"""Fixed-intervention residual MLP supervised by privileged training-pair Z'.

This is a conditional-mean reference, not an oracle available at inference or a
guaranteed upper bound. Neither g nor h_S participates in its training objective.
"""

from __future__ import annotations

import math
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from torch import nn

from src.artifact_io import save_torch

ORACLE_VARIANT = "oracle_regression"


def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


class OracleRegression(nn.Module):
    """Z' = Z + MLP(Z), for exactly the intervention recorded in the checkpoint."""

    def __init__(
        self, latent_dim, intervention, hidden_dim=512, n_hidden=2, dropout=0.1
    ):
        super().__init__()
        for name, value in (
            ("latent_dim", latent_dim),
            ("hidden_dim", hidden_dim),
            ("n_hidden", n_hidden),
        ):
            _positive_int(name, value)
        if not isinstance(intervention, dict) or any(
            not isinstance(key, str)
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for key, value in intervention.items()
        ):
            raise ValueError("intervention must map column names to category indices")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        self.latent_dim = latent_dim
        self.intervention = dict(intervention)
        self.config = dict(
            latent_dim=latent_dim,
            intervention=dict(intervention),
            hidden_dim=hidden_dim,
            n_hidden=n_hidden,
            dropout=dropout,
        )
        layers = []
        width = latent_dim
        for _ in range(n_hidden):
            layers.extend(
                [nn.Linear(width, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            )
            width = hidden_dim
        layers.append(nn.Linear(width, latent_dim))
        self.residual = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.residual:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        # Start at the identity reference. Do not normalize predictions: ordinary
        # squared-error regression estimates a mean, which need not have unit norm.
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def validate_latents(self, z, *, nonempty=False):
        if (
            not isinstance(z, torch.Tensor)
            or z.ndim != 2
            or z.shape[1] != self.latent_dim
            or not z.is_floating_point()
            or (nonempty and len(z) == 0)
            or not torch.isfinite(z).all()
        ):
            raise ValueError(
                f"expected finite floating latents of shape (n, {self.latent_dim})"
            )

    def forward(self, z):
        return z + self.residual(z)

    @torch.no_grad()
    def predict(self, z, *, intervention=None):
        if intervention is not None and intervention != self.intervention:
            raise ValueError("oracle_regression supports only its trained intervention")
        self.validate_latents(z)
        parameter = next(self.parameters())
        self.eval()
        return self(z.to(device=parameter.device, dtype=parameter.dtype))

    def save(self, path, *, metadata):
        save_torch(
            path,
            {
                "format_version": 1,
                "variant": ORACLE_VARIANT,
                "config": self.config,
                "state_dict": {
                    name: value.detach().cpu()
                    for name, value in self.state_dict().items()
                },
                "metadata": metadata,
            },
        )

    @classmethod
    def load(
        cls, path, *, device="cpu", expected_metadata=None, expected_intervention=None
    ):
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if (
            not isinstance(payload, dict)
            or payload.get("format_version") != 1
            or payload.get("variant") != ORACLE_VARIANT
            or not isinstance(payload.get("metadata"), dict)
        ):
            raise ValueError("invalid oracle_regression checkpoint")
        if expected_metadata is not None and payload["metadata"] != expected_metadata:
            raise ValueError("oracle_regression checkpoint provenance mismatch")
        model = cls(**payload["config"])
        if (
            expected_intervention is not None
            and model.intervention != expected_intervention
        ):
            raise ValueError("oracle_regression checkpoint intervention mismatch")
        model.load_state_dict(payload["state_dict"])
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("oracle_regression checkpoint has non-finite weights")
        model.checkpoint_metadata = payload["metadata"]
        return model.to(device).eval()


@torch.no_grad()
def regression_metrics(prediction, target):
    if prediction.shape != target.shape or prediction.ndim != 2 or len(target) == 0:
        raise ValueError("predictions and targets must have matching, nonempty shapes")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("predictions and targets must be finite")
    difference = prediction - target
    return {
        "mse": difference.square().mean().item(),
        "squared_l2_mean": difference.square().sum(dim=1).mean().item(),
        "l2_mean": difference.norm(dim=1).mean().item(),
        "cosine_mean": F.cosine_similarity(prediction, target, dim=1).mean().item(),
    }


@torch.no_grad()
def _mse(model, z, target, batch_size):
    model.eval()
    total = 0.0
    for start in range(0, len(z), batch_size):
        stop = start + batch_size
        total += F.mse_loss(
            model(z[start:stop]), target[start:stop], reduction="sum"
        ).item()
    return total / target.numel()


def train_oracle_regression(
    model,
    latents,
    targets,
    *,
    validation_latents,
    validation_targets,
    epochs=500,
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-5,
    seed=42,
    patience=30,
    min_delta=1e-7,
    scheduler_patience=10,
    scheduler_factor=0.5,
    min_lr=1e-6,
    grad_clip=5.0,
    device="cpu",
    epoch_callback=None,
):
    """Fit from scratch using only supplied fit/validation pairs; restore best MSE."""
    for name, value in (
        ("epochs", epochs),
        ("batch_size", batch_size),
        ("patience", patience),
        ("scheduler_patience", scheduler_patience),
    ):
        _positive_int(name, value)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for name, value in (("lr", lr), ("min_lr", min_lr), ("grad_clip", grad_clip)):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    for name, value in (("min_delta", min_delta), ("weight_decay", weight_decay)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be nonnegative and finite")
    if not 0 < scheduler_factor < 1 or min_lr > lr:
        raise ValueError("invalid learning-rate scheduler settings")
    for z, target in ((latents, targets), (validation_latents, validation_targets)):
        model.validate_latents(z, nonempty=True)
        model.validate_latents(target, nonempty=True)
        if z.shape != target.shape:
            raise ValueError("training/validation pairs must align row-for-row")
    device = torch.device(device)
    model.to(device=device, dtype=torch.float32)
    latents, targets, validation_latents, validation_targets = (
        value.detach().to(device=device, dtype=torch.float32)
        for value in (latents, targets, validation_latents, validation_targets)
    )
    cuda_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    best_mse = reference = float("inf")
    best_state, best_epoch, stale = None, 0, 0
    history = []
    started = perf_counter()
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        model.reset_parameters()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=scheduler_patience,
            factor=scheduler_factor,
            threshold=min_delta,
            threshold_mode="abs",
            min_lr=min_lr,
        )
        for epoch in range(1, epochs + 1):
            epoch_lr = optimizer.param_groups[0]["lr"]
            model.train()
            order = torch.randperm(len(latents), device=device)
            total = 0.0
            for start in range(0, len(latents), batch_size):
                index = order[start : start + batch_size]
                loss = F.mse_loss(model(latents[index]), targets[index])
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "non-finite oracle regression training loss"
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip, error_if_nonfinite=True
                )
                optimizer.step()
                total += loss.item() * len(index)
            validation_mse = _mse(
                model, validation_latents, validation_targets, batch_size
            )
            if not math.isfinite(validation_mse):
                raise FloatingPointError("non-finite oracle regression validation loss")
            if validation_mse < best_mse:
                best_mse, best_epoch = validation_mse, epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            if validation_mse < reference - min_delta:
                reference, stale = validation_mse, 0
            else:
                stale += 1
            scheduler.step(validation_mse)
            record = dict(
                epoch=epoch,
                train_mse=total / len(latents),
                validation_mse=validation_mse,
                learning_rate=epoch_lr,
                best_epoch=best_epoch,
                stale_epochs=stale,
            )
            history.append(record)
            if epoch_callback is not None:
                epoch_callback(model, record)
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return dict(
        history=history,
        best_epoch=best_epoch,
        best_validation_mse=best_mse,
        epochs_run=len(history),
        max_epochs=epochs,
        stopped_early=len(history) < epochs,
        seconds=perf_counter() - started,
        loss="mean_squared_error_per_coordinate",
        seed=seed,
    )
