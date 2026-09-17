"""Semantic decoders from frozen text latents to structured-state distributions."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar, Self

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from src.schema import ColumnSpec

SEMANTIC_DECODER_FORMAT_VERSION = 1


def _schema_signature(columns: Sequence[ColumnSpec]) -> str:
    encoded = json.dumps(
        [(column.name, column.n_categories) for column in columns],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_metadata(
    metadata: Mapping[str, Any] | None,
    columns: Sequence[ColumnSpec],
    *,
    add_schema_signature: bool,
) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise ValueError(
            "semantic decoder metadata must include latent and encoder provenance"
        )
    result = dict(metadata)
    if (
        not isinstance(result.get("latent_artifact_sha256"), str)
        or not result["latent_artifact_sha256"]
    ):
        raise ValueError("semantic decoder metadata has no latent artifact hash")
    if not isinstance(result.get("encoder"), Mapping) or not result["encoder"]:
        raise ValueError("semantic decoder metadata has no encoder identity")

    expected_signature = _schema_signature(columns)
    signature = result.get("schema_signature")
    if signature is None and add_schema_signature:
        result["schema_signature"] = expected_signature
    elif signature != expected_signature:
        raise ValueError("semantic decoder metadata has no matching schema signature")
    return result


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _validate_columns(columns: Sequence[ColumnSpec]) -> list[ColumnSpec]:
    result = list(columns)
    if not result:
        raise ValueError("columns must not be empty")

    names: set[str] = set()
    for column in result:
        if not isinstance(column, ColumnSpec):
            raise TypeError("columns must contain ColumnSpec values")
        if not isinstance(column.name, str) or not column.name or "." in column.name:
            raise ValueError("column names must be non-empty and must not contain '.'")
        if column.name in names:
            raise ValueError(f"duplicate column name: {column.name!r}")
        _positive_int(f"cardinality for {column.name!r}", column.n_categories)
        names.add(column.name)
    return result


def _mlp_trunk(
    latent_dim: int,
    hidden_dim: int,
    n_hidden: int,
    dropout: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_dim = latent_dim
    for _ in range(n_hidden):
        layers.extend((nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)))
        in_dim = hidden_dim
    return nn.Sequential(*layers)


class _SemanticDecoderBase(nn.Module):
    """Shared validation, marginalisation, and persistence for decoder variants."""

    variant: ClassVar[str]

    def __init__(
        self,
        latent_dim: int,
        columns: Sequence[ColumnSpec],
        hidden_dim: int,
        n_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.latent_dim = _positive_int("latent_dim", latent_dim)
        self.hidden_dim = _positive_int("hidden_dim", hidden_dim)
        self.n_hidden = _nonnegative_int("n_hidden", n_hidden)
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
            raise ValueError("dropout must be a number in [0, 1)")
        self.dropout = float(dropout)
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be a number in [0, 1)")

        self.columns = _validate_columns(columns)
        self.trunk = _mlp_trunk(
            self.latent_dim,
            self.hidden_dim,
            self.n_hidden,
            self.dropout,
        )
        self.checkpoint_metadata: dict[str, Any] = {}

    @property
    def _head_input_dim(self) -> int:
        return self.hidden_dim if self.n_hidden else self.latent_dim

    def _constructor_config(self) -> dict[str, Any]:
        return {
            "latent_dim": self.latent_dim,
            "columns": [(column.name, column.n_categories) for column in self.columns],
            "hidden_dim": self.hidden_dim,
            "n_hidden": self.n_hidden,
            "dropout": self.dropout,
        }

    def _validate_latents(self, z: torch.Tensor, *, nonempty: bool = False) -> None:
        if not isinstance(z, torch.Tensor):
            raise TypeError("latents must be a torch.Tensor")
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(
                f"latents must have shape (n, {self.latent_dim}), got {tuple(z.shape)}"
            )
        if nonempty and z.shape[0] == 0:
            raise ValueError("latents must not be empty")
        if not z.is_floating_point():
            raise ValueError("latents must have a floating-point dtype")
        if not torch.isfinite(z).all():
            raise ValueError("latents must contain only finite values")

    def _validate_targets(
        self,
        targets: Mapping[str, torch.Tensor],
        batch_size: int,
        *,
        device: torch.device | None = None,
    ) -> None:
        if not isinstance(targets, Mapping):
            raise TypeError("targets must be a mapping from column names to tensors")
        expected = {column.name for column in self.columns}
        actual = set(targets)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                f"target columns do not match schema; missing={missing}, extra={extra}"
            )

        for column in self.columns:
            values = targets[column.name]
            if not isinstance(values, torch.Tensor):
                raise TypeError(f"target {column.name!r} must be a torch.Tensor")
            if values.ndim != 1 or values.shape[0] != batch_size:
                raise ValueError(
                    f"target {column.name!r} must have shape ({batch_size},), "
                    f"got {tuple(values.shape)}"
                )
            try:
                torch.iinfo(values.dtype)
            except TypeError as error:
                raise ValueError(
                    f"target {column.name!r} must have an integer dtype"
                ) from error
            if device is not None and values.device != device:
                raise ValueError(
                    f"target {column.name!r} must be on the same device as latents"
                )
            if values.numel() and (
                int(values.min()) < 0 or int(values.max()) >= column.n_categories
            ):
                raise ValueError(
                    f"target {column.name!r} must be in [0, {column.n_categories - 1}]"
                )

    def marginal_probabilities(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return exact per-column marginals derived from the dense joint."""
        joint = (
            self.log_joint(z)
            .exp()
            .reshape(z.shape[0], *(column.n_categories for column in self.columns))
        )
        marginals: dict[str, torch.Tensor] = {}
        for index, column in enumerate(self.columns):
            reduce_dims = tuple(
                dim + 1 for dim in range(len(self.columns)) if dim != index
            )
            marginals[column.name] = (
                joint.sum(dim=reduce_dims) if reduce_dims else joint
            )
        return marginals

    def save(
        self,
        path: str | Path,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Save a versioned checkpoint with required artifact provenance."""
        saved_metadata = _checkpoint_metadata(
            metadata, self.columns, add_schema_signature=True
        )
        self.checkpoint_metadata = saved_metadata
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": SEMANTIC_DECODER_FORMAT_VERSION,
                "variant": self.variant,
                "config": self._constructor_config(),
                "metadata": saved_metadata,
                "state_dict": self.state_dict(),
            },
            output,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str | torch.device | None = None,
        *,
        expected_columns: Sequence[ColumnSpec] | None = None,
        expected_metadata: Mapping[str, Any] | None = None,
    ) -> Self:
        """Load this decoder variant and enforce optional compatibility expectations."""
        model = load_semantic_decoder(
            path,
            device=device,
            expected_variant=cls.variant,
            expected_columns=expected_columns,
            expected_metadata=expected_metadata,
        )
        if not isinstance(model, cls):
            raise TypeError(f"checkpoint did not reconstruct {cls.__name__}")
        return model

    def nll(
        self,
        z: torch.Tensor,
        targets: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        raise NotImplementedError

    def predict(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def log_joint(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SemanticDecoder(_SemanticDecoderBase):
    """Shared MLP with conditionally independent categorical heads."""

    variant = "independent"

    def __init__(
        self,
        latent_dim: int,
        columns: Sequence[ColumnSpec],
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(latent_dim, columns, hidden_dim, n_hidden, dropout)
        self.heads = nn.ModuleDict(
            {
                column.name: nn.Linear(self._head_input_dim, column.n_categories)
                for column in self.columns
            }
        )

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return one unconditional logit tensor per schema column."""
        self._validate_latents(z)
        hidden = self.trunk(z)
        return {name: head(hidden) for name, head in self.heads.items()}

    def log_joint(self, z: torch.Tensor) -> torch.Tensor:
        """Return dense joint log-probabilities, with the final column varying fastest."""
        log_probabilities = [
            F.log_softmax(logits, dim=-1) for logits in self.forward(z).values()
        ]
        joint = log_probabilities[0]
        for log_probability in log_probabilities[1:]:
            joint = (joint[..., None] + log_probability[:, None, :]).reshape(
                joint.shape[0], -1
            )
        return joint

    def nll(
        self,
        z: torch.Tensor,
        targets: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return the mean per-column cross-entropy (not the full joint NLL)."""
        self._validate_latents(z, nonempty=True)
        self._validate_targets(targets, z.shape[0], device=z.device)
        logits = self.forward(z)
        losses = [
            F.cross_entropy(logits[column.name], targets[column.name].long())
            for column in self.columns
        ]
        return torch.stack(losses).mean()

    @torch.no_grad()
    def predict(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return the categorical argmax for each independent head."""
        return {name: logits.argmax(dim=-1) for name, logits in self.forward(z).items()}


class SemanticAutoRegDecoder(_SemanticDecoderBase):
    """Autoregressive heads conditioned on embeddings of teacher-forced prefixes."""

    variant = "autoregressive"

    def __init__(
        self,
        latent_dim: int,
        columns: Sequence[ColumnSpec],
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: float = 0.1,
        embed_dim: int = 16,
    ) -> None:
        self.embed_dim = _positive_int("embed_dim", embed_dim)
        super().__init__(latent_dim, columns, hidden_dim, n_hidden, dropout)
        self.embeddings = nn.ModuleDict(
            {
                column.name: nn.Embedding(column.n_categories, self.embed_dim)
                for column in self.columns
            }
        )
        self.heads = nn.ModuleDict(
            {
                column.name: nn.Linear(
                    self._head_input_dim + index * self.embed_dim,
                    column.n_categories,
                )
                for index, column in enumerate(self.columns)
            }
        )

    def _constructor_config(self) -> dict[str, Any]:
        config = super()._constructor_config()
        config["embed_dim"] = self.embed_dim
        return config

    def _prefix_logits(
        self,
        hidden: torch.Tensor,
        prefix: list[torch.Tensor],
        name: str,
    ) -> torch.Tensor:
        return self.heads[name](torch.cat([hidden, *prefix], dim=-1))

    def conditional_logits(
        self,
        z: torch.Tensor,
        targets: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Return teacher-forced logits p(s_i | s_<i, z) for every column."""
        self._validate_latents(z)
        self._validate_targets(targets, z.shape[0], device=z.device)
        hidden = self.trunk(z)
        prefix: list[torch.Tensor] = []
        logits: dict[str, torch.Tensor] = {}
        for column in self.columns:
            logits[column.name] = self._prefix_logits(hidden, prefix, column.name)
            prefix.append(self.embeddings[column.name](targets[column.name].long()))
        return logits

    def forward(
        self,
        z: torch.Tensor,
        targets: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return teacher-forced logits; an observed prefix is necessarily required."""
        if targets is None:
            raise TypeError(
                "SemanticAutoRegDecoder.forward requires targets for teacher forcing; "
                "use predict(z) for greedy decoding or log_joint(z) for the full distribution"
            )
        return self.conditional_logits(z, targets)

    def log_prob(
        self,
        z: torch.Tensor,
        targets: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return teacher-forced log p(s | z) for each observation."""
        logits = self.conditional_logits(z, targets)
        total = z.new_zeros(z.shape[0])
        for column in self.columns:
            values = targets[column.name].long()
            total = total + F.log_softmax(logits[column.name], dim=-1).gather(
                -1, values[:, None]
            ).squeeze(-1)
        return total

    def nll(
        self,
        z: torch.Tensor,
        targets: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return the full-joint negative log-likelihood."""
        self._validate_latents(z, nonempty=True)
        return -self.log_prob(z, targets).mean()

    @torch.no_grad()
    def predict(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Greedily decode each column and feed its argmax into later heads."""
        self._validate_latents(z)
        hidden = self.trunk(z)
        prefix: list[torch.Tensor] = []
        predictions: dict[str, torch.Tensor] = {}
        for column in self.columns:
            prediction = self._prefix_logits(hidden, prefix, column.name).argmax(dim=-1)
            predictions[column.name] = prediction
            prefix.append(self.embeddings[column.name](prediction))
        return predictions

    def log_joint(self, z: torch.Tensor) -> torch.Tensor:
        """Enumerate the exact autoregressive joint in schema/mixed-radix order."""
        self._validate_latents(z)
        hidden = self.trunk(z)
        batch_size = z.shape[0]
        joint = hidden.new_zeros(batch_size, 1)
        prefix_embeddings = hidden.new_zeros(batch_size, 1, 0)

        for column in self.columns:
            n_prefixes = joint.shape[1]
            expanded_hidden = (
                hidden[:, None, :]
                .expand(batch_size, n_prefixes, -1)
                .reshape(batch_size * n_prefixes, -1)
            )
            flat_prefix = prefix_embeddings.reshape(batch_size * n_prefixes, -1)
            logits = self.heads[column.name](
                torch.cat((expanded_hidden, flat_prefix), dim=-1)
            )
            conditional = F.log_softmax(logits, dim=-1).reshape(
                batch_size, n_prefixes, column.n_categories
            )
            joint = (joint[..., None] + conditional).reshape(
                batch_size, n_prefixes * column.n_categories
            )

            category = torch.arange(column.n_categories, device=z.device)
            embedding = self.embeddings[column.name](category)
            prefix_embeddings = torch.cat(
                (
                    prefix_embeddings[:, :, None, :].expand(
                        batch_size, n_prefixes, column.n_categories, -1
                    ),
                    embedding[None, None, :, :].expand(
                        batch_size, n_prefixes, column.n_categories, -1
                    ),
                ),
                dim=-1,
            ).reshape(batch_size, n_prefixes * column.n_categories, -1)
        return joint


SemanticDecoderModel = SemanticDecoder | SemanticAutoRegDecoder


def make_semantic_decoder(
    variant: str,
    *,
    latent_dim: int,
    columns: Sequence[ColumnSpec],
    hidden_dim: int = 256,
    n_hidden: int = 2,
    dropout: float = 0.1,
    embed_dim: int = 16,
) -> SemanticDecoderModel:
    """Construct a configured decoder variant with a single explicit dispatch point."""
    common = {
        "latent_dim": latent_dim,
        "columns": columns,
        "hidden_dim": hidden_dim,
        "n_hidden": n_hidden,
        "dropout": dropout,
    }
    if variant == SemanticDecoder.variant:
        return SemanticDecoder(**common)
    if variant == SemanticAutoRegDecoder.variant:
        return SemanticAutoRegDecoder(**common, embed_dim=embed_dim)
    raise ValueError(
        f"unknown semantic decoder variant {variant!r}; "
        "expected 'independent' or 'autoregressive'"
    )


def _checkpoint_error(message: str) -> ValueError:
    return ValueError(f"{message}; re-encode/retrain the dependent artifacts")


def load_semantic_decoder(
    path: str | Path,
    *,
    device: str | torch.device | None = None,
    expected_variant: str | None = None,
    expected_columns: Sequence[ColumnSpec] | None = None,
    expected_metadata: Mapping[str, Any] | None = None,
) -> SemanticDecoderModel:
    """Load a versioned decoder and reject incompatible artifact provenance."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise _checkpoint_error("semantic decoder checkpoint is malformed")
    if payload.get("format_version") != SEMANTIC_DECODER_FORMAT_VERSION:
        raise _checkpoint_error("unsupported semantic decoder checkpoint format")

    variant = payload.get("variant")
    if expected_variant is not None and variant != expected_variant:
        raise _checkpoint_error(
            f"semantic decoder variant mismatch: expected {expected_variant!r}, got {variant!r}"
        )
    config = payload.get("config")
    if not isinstance(config, dict):
        raise _checkpoint_error("semantic decoder checkpoint has no valid config")
    raw_columns = config.get("columns")
    if not isinstance(raw_columns, (list, tuple)):
        raise _checkpoint_error("semantic decoder checkpoint has no valid schema")
    try:
        saved_columns = [
            ColumnSpec(name, cardinality) for name, cardinality in raw_columns
        ]
        saved_columns = _validate_columns(saved_columns)
    except (TypeError, ValueError) as error:
        raise _checkpoint_error(
            "semantic decoder checkpoint has an invalid schema"
        ) from error

    if expected_columns is not None:
        checked_expected_columns = _validate_columns(expected_columns)
        if saved_columns != checked_expected_columns:
            raise _checkpoint_error(
                "semantic decoder schema does not match the active schema"
            )

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise _checkpoint_error("semantic decoder checkpoint has no valid metadata")
    try:
        metadata = _checkpoint_metadata(
            metadata, saved_columns, add_schema_signature=False
        )
    except ValueError as error:
        raise _checkpoint_error(str(error)) from error
    if expected_metadata is not None:
        mismatches = [
            key
            for key, expected_value in expected_metadata.items()
            if key not in metadata or metadata[key] != expected_value
        ]
        if mismatches:
            raise _checkpoint_error(
                f"semantic decoder metadata mismatch for {sorted(mismatches)}"
            )

    constructor_config = dict(config)
    constructor_config["columns"] = saved_columns
    try:
        model = make_semantic_decoder(str(variant), **constructor_config)
        model.load_state_dict(payload["state_dict"])
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise _checkpoint_error(
            "semantic decoder checkpoint is incompatible"
        ) from error
    model.checkpoint_metadata = dict(metadata)
    model.to(torch.device(device) if device is not None else torch.device("cpu"))
    model.eval()
    return model


def targets_from_dataframe(
    frame: pd.DataFrame,
    columns: Sequence[ColumnSpec],
) -> dict[str, torch.Tensor]:
    """Convert and validate structured-state columns as integer target tensors."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    checked_columns = _validate_columns(columns)
    if not frame.columns.is_unique:
        raise ValueError("dataframe column names must be unique")
    missing = [column.name for column in checked_columns if column.name not in frame]
    if missing:
        raise ValueError(f"dataframe is missing target columns {missing}")

    targets: dict[str, torch.Tensor] = {}
    for column in checked_columns:
        numeric = pd.to_numeric(frame[column.name], errors="coerce")
        values = torch.as_tensor(numeric.astype("float64").to_numpy(copy=True))
        if not torch.isfinite(values).all() or not torch.equal(values, values.round()):
            raise ValueError(f"target {column.name!r} must contain finite integers")
        if values.numel() and (
            float(values.min()) < 0 or float(values.max()) >= column.n_categories
        ):
            raise ValueError(
                f"target {column.name!r} must be in [0, {column.n_categories - 1}]"
            )
        targets[column.name] = values.long()
    return targets


def _model_device(decoder: _SemanticDecoderBase) -> torch.device:
    return next(decoder.parameters()).device


def _model_dtype(decoder: _SemanticDecoderBase) -> torch.dtype:
    return next(decoder.parameters()).dtype


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
) -> list[float]:
    """Train from a deterministic reset and return sample-weighted epoch losses."""
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
            training_device.index
            if training_device.index is not None
            else torch.cuda.current_device()
        ]

    history: list[float] = []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        decoder.apply(_reset_trainable_modules)
        optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
        decoder.train()

        for epoch in range(epochs):
            permutation = torch.randperm(n_observations, device=training_device)
            weighted_loss = 0.0
            for start in range(0, n_observations, batch_size):
                index = permutation[start : start + batch_size]
                batch_targets = {
                    name: values[index] for name, values in targets.items()
                }
                loss = decoder.nll(latents[index], batch_targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                weighted_loss += loss.item() * index.numel()
            history.append(weighted_loss / n_observations)
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"epoch {epoch + 1}/{epochs}  loss {history[-1]:.4f}")

    decoder.eval()
    return history


@contextmanager
def _evaluation_mode(decoder: _SemanticDecoderBase) -> Iterator[None]:
    was_training = decoder.training
    decoder.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        decoder.train(was_training)


def _evaluation_data(
    decoder: _SemanticDecoderBase,
    latents: torch.Tensor,
    targets: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    decoder._validate_latents(latents, nonempty=True)
    decoder._validate_targets(targets, latents.shape[0])
    device = _model_device(decoder)
    moved_latents = latents.to(device=device, dtype=_model_dtype(decoder))
    moved_targets = {
        name: values.to(device=device, dtype=torch.long)
        for name, values in targets.items()
    }
    return moved_latents, moved_targets


def accuracy(
    decoder: SemanticDecoderModel,
    latents: torch.Tensor,
    targets: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    """Return per-column greedy accuracy without changing the decoder's mode."""
    moved_latents, moved_targets = _evaluation_data(decoder, latents, targets)
    with _evaluation_mode(decoder):
        predictions = decoder.predict(moved_latents)
    return {
        column.name: (predictions[column.name] == moved_targets[column.name])
        .float()
        .mean()
        .item()
        for column in decoder.columns
    }


def calibration_metrics(
    decoder: SemanticDecoderModel,
    latents: torch.Tensor,
    targets: Mapping[str, torch.Tensor],
    n_bins: int = 10,
) -> dict[str, dict[str, Any]]:
    """Measure marginal accuracy, ECE, and nonempty equal-width reliability bins."""
    _positive_int("n_bins", n_bins)
    moved_latents, moved_targets = _evaluation_data(decoder, latents, targets)
    with _evaluation_mode(decoder):
        marginals = decoder.marginal_probabilities(moved_latents)

    result: dict[str, dict[str, Any]] = {}
    n_observations = moved_latents.shape[0]
    for column in decoder.columns:
        confidence, prediction = marginals[column.name].max(dim=-1)
        correct = prediction.eq(moved_targets[column.name]).float()
        reliability: list[dict[str, float | int]] = []
        ece = 0.0
        for index in range(n_bins):
            lower = index / n_bins
            upper = (index + 1) / n_bins
            in_bin = confidence.ge(lower) & (
                confidence.le(upper) if index == n_bins - 1 else confidence.lt(upper)
            )
            count = int(in_bin.sum().item())
            if count == 0:
                continue
            mean_confidence = float(confidence[in_bin].mean().item())
            bin_accuracy = float(correct[in_bin].mean().item())
            ece += count / n_observations * abs(mean_confidence - bin_accuracy)
            reliability.append(
                {
                    "lower": lower,
                    "upper": upper,
                    "count": count,
                    "mean_confidence": mean_confidence,
                    "accuracy": bin_accuracy,
                }
            )
        result[column.name] = {
            "accuracy": float(correct.mean().item()),
            "ece": ece,
            "reliability": reliability,
        }
    return result
