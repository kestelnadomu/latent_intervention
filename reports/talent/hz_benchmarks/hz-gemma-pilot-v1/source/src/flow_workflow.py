"""Pipeline-facing orchestration for the three normalizing-flow ``h_Z`` models.

The main pipeline delegates flow variants to this module so its established
transformer training and evaluation paths can remain unchanged.  Flow model
definitions and objectives stay in :mod:`src.flow_intervention`; this module owns
only data alignment, artifact provenance, configured training, and reporting.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

from src.flow_intervention import (
    FLOW_VARIANTS,
    load_latent_intervention,
    make_latent_intervention,
    sample_counterfactual,
    train_latent_intervention_model,
)
from src.pair_encoding import LatentArtifact, load_latent_artifact, sha256_file
from src.schema import ColumnSpec, load_intervention, load_schema
from src.semantic_decoder import (
    accuracy,
    calibration_metrics,
    load_semantic_decoder,
    targets_from_dataframe,
)
from src.symbolic_intervention import load_symbolic_kernel


__all__ = [
    "FLOW_VARIANTS",
    "evaluate_flow_manipulator",
    "train_flow_manipulator",
]


# --- data and artifact helpers ----------------------------------------------------


def _flow_variant(config: dict[str, Any]) -> str:
    variant = str(config["latent_intervention"]["variant"])
    if variant not in FLOW_VARIANTS:
        raise ValueError(
            f"not a flow intervention variant: {variant!r}; expected one of "
            f"{', '.join(FLOW_VARIANTS)}"
        )
    return variant


def _aligned_targets(
    csv_path: str | Path,
    ids: list[int],
    columns: list[ColumnSpec],
) -> dict[str, torch.Tensor]:
    """Read structured targets and align them exactly to latent unit IDs."""
    frame = pd.read_csv(csv_path)
    if "id" not in frame:
        raise ValueError(f"target data has no 'id' column: {csv_path}")
    numeric_ids = pd.to_numeric(frame["id"], errors="coerce")
    if numeric_ids.isna().any() or not numeric_ids.eq(numeric_ids.round()).all():
        raise ValueError("target IDs must be integers")
    frame["id"] = numeric_ids.astype(int)
    if frame["id"].duplicated().any():
        raise ValueError("target data contains duplicate IDs")
    if set(frame["id"]) != set(ids):
        raise ValueError("target IDs do not match the latent artifact")
    aligned = frame.set_index("id").loc[ids].reset_index()
    return targets_from_dataframe(aligned, columns)


def _indices_for_ids(all_ids: list[int], selected_ids: list[int]) -> torch.Tensor:
    positions = {unit_id: index for index, unit_id in enumerate(all_ids)}
    missing = set(selected_ids) - set(positions)
    if missing:
        raise ValueError(
            f"split IDs are absent from the latent artifact: {sorted(missing)}"
        )
    return torch.tensor(
        [positions[unit_id] for unit_id in selected_ids], dtype=torch.long
    )


def _official_indices(artifact: LatentArtifact) -> tuple[torch.Tensor, torch.Tensor]:
    """Return official train/test positions in their recorded order."""
    return (
        _indices_for_ids(artifact.ids, artifact.train_ids),
        _indices_for_ids(artifact.ids, artifact.test_ids),
    )


def _subset(
    targets: dict[str, torch.Tensor], idx: torch.Tensor
) -> dict[str, torch.Tensor]:
    return {name: values[idx] for name, values in targets.items()}


def _config_sha256(config_source: dict[str, Any] | str | Path) -> str:
    if isinstance(config_source, dict):
        canonical = json.dumps(
            config_source, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()
    return sha256_file(config_source)


def _variant_path(config: dict[str, Any], key: str, variant: str) -> Path:
    template = str(config["paths"][key])
    try:
        return Path(template.format(variant=variant))
    except (IndexError, KeyError, ValueError) as exc:
        raise ValueError(f"invalid paths.{key} template: {template!r}") from exc


def _flow_teacher_path(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["flow_teacher_model"])


def _flow_model_path(config: dict[str, Any], variant: str) -> Path:
    if variant == "state_flow":
        return _flow_teacher_path(config)
    return _variant_path(config, "flow_manipulator_model", variant)


def _flow_report_path(config: dict[str, Any], variant: str) -> Path:
    return _variant_path(config, "flow_eval_report", variant)


def _schema_signature(columns: list[ColumnSpec]) -> str:
    encoded = json.dumps(
        [(column.name, column.n_categories) for column in columns],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _flow_training_config_sha256(config: dict[str, Any], variant: str) -> str:
    """Hash only architecture and training settings used by this flow."""
    flow = config["latent_intervention"]["flow"]
    signature: dict[str, Any] = {
        key: flow[key]
        for key in (
            "n_blocks",
            "hidden_dim",
            "condition_dim",
            "state_embed_dim",
            "scale_floor",
            "epochs",
            "batch_size",
            "lr",
            "weight_decay",
            "grad_clip",
        )
    }
    if variant in {"distilled_flow", "direct_semantic_flow"}:
        signature.update(
            base_std=flow["base_std"],
            n_samples=flow["n_samples"],
            no_op_fraction=flow["no_op_fraction"],
        )
    if variant == "direct_semantic_flow":
        signature.update(
            support_bank_size=flow["support_bank_size"],
            direct_semantic=flow["direct_semantic"],
        )
    return _config_sha256(signature)


def _flow_checkpoint_metadata(
    config: dict[str, Any],
    artifact: LatentArtifact,
    columns: list[ColumnSpec],
    variant: str,
    *,
    recorded_teacher_sha256: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "format_version": 1,
        "latent_artifact_sha256": artifact.artifact_sha256,
        "encoder": artifact.encoder_info,
        "schema_signature": _schema_signature(columns),
        "sim_config_sha256": _config_sha256(config["sim_config"]),
        "training_config_sha256": _flow_training_config_sha256(config, variant),
        "training_seed": int(config["seed"]),
    }
    if variant == "state_flow":
        metadata.update(
            objective="conditional_factual_nll",
            sim_factual_sha256=sha256_file(config["paths"]["sim_factual"]),
        )
    elif variant == "distilled_flow":
        teacher_sha256 = recorded_teacher_sha256
        if teacher_sha256 is None:
            teacher_sha256 = sha256_file(_flow_teacher_path(config))
        metadata.update(
            flow_teacher_sha256=teacher_sha256,
            sim_factual_sha256=sha256_file(config["paths"]["sim_factual"]),
            objective="scale_normalized_energy_distance",
            sampling_seed=int(config["seed"]),
        )
    elif variant == "direct_semantic_flow":
        metadata.update(
            objective="semantic_forward_kl",
            decoder_checkpoint_sha256=sha256_file(
                config["paths"]["decoder_model"]
            ),
        )
    else:
        raise ValueError(f"not a flow intervention variant: {variant}")
    return metadata


def _decoder_metadata(
    config: dict[str, Any], artifact: LatentArtifact
) -> dict[str, Any]:
    return {
        "latent_artifact_sha256": artifact.artifact_sha256,
        "encoder": artifact.encoder_info,
        "training_inputs": {
            "sim_factual_sha256": sha256_file(config["paths"]["sim_factual"]),
            "sim_config_sha256": _config_sha256(config["sim_config"]),
        },
    }


def _supported_interventions(intervention: dict[str, int]) -> list[dict[str, int]]:
    return [dict(intervention), {}] if intervention else [{}]


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _flow_info_path(model_path: str | Path) -> Path:
    return Path(model_path).with_suffix(".info.json")


def _flow_info(
    config: dict[str, Any],
    artifact: LatentArtifact,
    columns: list[ColumnSpec],
    variant: str,
    *,
    recorded_teacher_sha256: str | None = None,
) -> dict[str, Any]:
    model_path = _flow_model_path(config, variant)
    return {
        "format_version": 2,
        "latent_artifact_sha256": artifact.artifact_sha256,
        "manipulator_checkpoint_sha256": sha256_file(model_path),
        "manipulator_variant": variant,
        "flow_dependencies": _flow_checkpoint_metadata(
            config,
            artifact,
            columns,
            variant,
            recorded_teacher_sha256=recorded_teacher_sha256,
        ),
    }


def _write_flow_info(
    config: dict[str, Any],
    artifact: LatentArtifact,
    columns: list[ColumnSpec],
    variant: str,
) -> None:
    model_path = _flow_model_path(config, variant)
    _write_json(
        _flow_info_path(model_path),
        _flow_info(config, artifact, columns, variant),
    )


def _validate_flow_info(
    config: dict[str, Any],
    artifact: LatentArtifact,
    columns: list[ColumnSpec],
    variant: str,
) -> dict[str, Any]:
    model_path = _flow_model_path(config, variant)
    path = _flow_info_path(model_path)
    if not path.exists():
        raise ValueError(f"flow metadata is missing: {path}; retrain the flow")
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"invalid flow metadata: {path}; retrain the flow") from exc
    recorded_teacher_sha256 = None
    if variant == "distilled_flow":
        dependencies = info.get("flow_dependencies") if isinstance(info, dict) else None
        if not isinstance(dependencies, dict) or not isinstance(
            dependencies.get("flow_teacher_sha256"), str
        ):
            raise ValueError(f"invalid flow metadata: {path}; retrain the flow")
        recorded_teacher_sha256 = dependencies["flow_teacher_sha256"]
        if len(recorded_teacher_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in recorded_teacher_sha256.lower()
        ):
            raise ValueError(f"invalid flow metadata: {path}; retrain the flow")
    expected = _flow_info(
        config,
        artifact,
        columns,
        variant,
        recorded_teacher_sha256=recorded_teacher_sha256,
    )
    if info != expected:
        raise ValueError(
            "flow metadata does not match the current training artifacts; "
            "retrain the flow"
        )
    return expected["flow_dependencies"]


# --- configured construction and training ----------------------------------------


def _flow_model_kwargs(
    config: dict[str, Any],
    latent_dim: int,
    columns: list[ColumnSpec],
    variant: str,
) -> dict[str, Any]:
    flow = config["latent_intervention"]["flow"]
    kwargs: dict[str, Any] = {
        "latent_dim": latent_dim,
        "columns": columns,
        "seed": int(config["seed"]),
        "n_blocks": int(flow["n_blocks"]),
        "hidden_dim": int(flow["hidden_dim"]),
        "condition_dim": int(flow["condition_dim"]),
        "state_embed_dim": int(flow["state_embed_dim"]),
        "permutation_seed": int(config["seed"]),
        "scale_floor": float(flow["scale_floor"]),
    }
    if variant != "state_flow":
        kwargs["base_std"] = float(flow["base_std"])
    return kwargs


def _flow_training_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    flow = config["latent_intervention"]["flow"]
    return {
        "epochs": int(flow["epochs"]),
        "batch_size": int(flow["batch_size"]),
        "lr": float(flow["lr"]),
        "weight_decay": float(flow["weight_decay"]),
        "grad_clip": float(flow["grad_clip"]),
        "seed": int(config["seed"]),
        "device": config["encoder"]["device"],
    }


def _load_or_train_flow_teacher(
    config: dict[str, Any],
    artifact: LatentArtifact,
    columns: list[ColumnSpec],
    train_idx: torch.Tensor,
    factual_states: dict[str, torch.Tensor],
    intervention: dict[str, int],
) -> Any:
    path = _flow_teacher_path(config)
    metadata = _flow_checkpoint_metadata(
        config, artifact, columns, variant="state_flow"
    )
    if path.is_file():
        try:
            teacher = load_latent_intervention(
                path,
                device=config["encoder"]["device"],
                expected_variant="state_flow",
                expected_columns=columns,
                expected_metadata=metadata,
            )
            _write_flow_info(config, artifact, columns, "state_flow")
            return teacher
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            print(f"retraining stale state-flow teacher {path}: {exc}")

    teacher = make_latent_intervention(
        "state_flow",
        **_flow_model_kwargs(
            config,
            artifact.z.shape[1],
            columns,
            "state_flow",
        ),
    )
    train_latent_intervention_model(
        teacher,
        latents=artifact.z[train_idx],
        states=factual_states,
        intervention=intervention,
        **_flow_training_kwargs(config),
    )
    teacher.save(
        path,
        metadata=metadata,
        supported_interventions=_supported_interventions(intervention),
    )
    _write_flow_info(config, artifact, columns, "state_flow")
    print(f"wrote {path} (state-flow teacher)")
    return teacher


def train_flow_manipulator(config: dict[str, Any]) -> None:
    """Train and save the configured flow using only official-training units."""
    variant = _flow_variant(config)
    artifact = load_latent_artifact(config)
    train_idx, _ = _official_indices(artifact)
    columns, _ = load_schema(config.get("sim_config"))
    intervention = load_intervention(config.get("sim_config"))
    flow = config["latent_intervention"]["flow"]

    factual_states = None
    if variant in {"state_flow", "distilled_flow"}:
        all_factual_states = _aligned_targets(
            config["paths"]["sim_factual"], artifact.ids, columns
        )
        factual_states = _subset(all_factual_states, train_idx)

    decoder = None
    if variant == "direct_semantic_flow":
        decoder = load_semantic_decoder(
            config["paths"]["decoder_model"],
            device=config["encoder"]["device"],
            expected_variant=config["semantic_decoder"]["variant"],
            expected_columns=columns,
            expected_metadata=_decoder_metadata(config, artifact),
        )

    h_s = None
    if variant in {"distilled_flow", "direct_semantic_flow"}:
        h_s = load_symbolic_kernel(config.get("sim_config"))
        if list(h_s.columns) != list(columns):
            raise ValueError("symbolic-kernel schema does not match the flow")

    model = make_latent_intervention(
        variant,
        **_flow_model_kwargs(config, artifact.z.shape[1], columns, variant),
    )
    train_kwargs: dict[str, Any] = {
        **_flow_training_kwargs(config),
        "decoder": decoder,
        "h_s": h_s,
        "latents": artifact.z[train_idx],
        "intervention": intervention,
        "states": factual_states,
    }
    if variant == "distilled_flow":
        if factual_states is None or h_s is None:
            raise AssertionError("distilled flow requires factual states and h_S")
        train_kwargs.update(
            teacher=_load_or_train_flow_teacher(
                config,
                artifact,
                columns,
                train_idx,
                factual_states,
                intervention,
            ),
            n_samples=int(flow["n_samples"]),
            identity_fraction=float(flow["no_op_fraction"]),
        )
    elif variant == "direct_semantic_flow":
        direct = flow["direct_semantic"]
        train_kwargs.update(
            n_samples=int(flow["n_samples"]),
            identity_fraction=float(flow["no_op_fraction"]),
            entropy_weight=float(direct["entropy_weight"]),
            proximity_weight=float(direct["proximity_weight"]),
            support_weight=float(direct["support_weight"]),
            identity_weight=float(direct["identity_weight"]),
            support_bank_size=int(flow["support_bank_size"]),
        )

    train_latent_intervention_model(model, **train_kwargs)
    model_path = _flow_model_path(config, variant)
    model.save(
        model_path,
        metadata=_flow_checkpoint_metadata(config, artifact, columns, variant),
        supported_interventions=_supported_interventions(intervention),
    )
    _write_flow_info(config, artifact, columns, variant)
    print(f"wrote {model_path} (intervention {intervention})")


# --- evaluation ------------------------------------------------------------------


def _latent_sample_metrics(
    samples: torch.Tensor,
    factual: torch.Tensor,
    observed_counterfactual: torch.Tensor,
    training_latents: torch.Tensor,
    identity: torch.Tensor,
) -> dict[str, Any]:
    samples = samples.detach().cpu()
    factual = factual.detach().cpu()
    observed_counterfactual = observed_counterfactual.detach().cpu()
    training_latents = training_latents.detach().cpu()
    identity = identity.detach().cpu().bool()
    if samples.ndim != 3 or samples.shape[1:] != factual.shape:
        raise ValueError("counterfactual samples must have shape (samples, test, latent)")
    if observed_counterfactual.shape != factual.shape:
        raise ValueError("held-out Z' does not align with factual test latents")

    scale = training_latents.std(dim=0, unbiased=False).clamp_min(1e-6)
    center = training_latents.mean(dim=0)
    standardized_samples = (samples - center) / scale
    standardized_truth = (observed_counterfactual - center) / scale
    sample_by_unit = standardized_samples.permute(1, 0, 2)
    truth_distance = (
        sample_by_unit - standardized_truth.unsqueeze(1)
    ).norm(dim=-1).mean(dim=1)
    pair_distance = torch.cdist(sample_by_unit, sample_by_unit).mean(dim=(1, 2))
    energy_score = (truth_distance - 0.5 * pair_distance).mean()

    sample_mean = samples.mean(dim=0)
    cosine = F.cosine_similarity(sample_mean, observed_counterfactual, dim=-1)
    heldout_support = torch.cdist(
        standardized_samples.reshape(-1, samples.shape[-1]),
        standardized_truth,
    ).min(dim=1).values
    shifts = samples - factual.unsqueeze(0)

    identity_report: dict[str, float | int | None] = {
        "count": int(identity.sum().item()),
        "l1_mean": None,
        "l2_mean": None,
    }
    if identity.any():
        identity_shift = shifts[:, identity]
        identity_report.update(
            l1_mean=identity_shift.abs().sum(dim=-1).mean().item(),
            l2_mean=identity_shift.norm(dim=-1).mean().item(),
        )

    return {
        "standardized_energy_score": energy_score.item(),
        "sample_mean_recovery": {
            "l2_mean": (sample_mean - observed_counterfactual)
            .norm(dim=-1)
            .mean()
            .item(),
            "cosine_mean": cosine.mean().item(),
        },
        "sample_spread": samples.std(dim=0, unbiased=False)
        .norm(dim=-1)
        .mean()
        .item(),
        "heldout_nearest_neighbor_distance": heldout_support.mean().item(),
        "latent_shift": {
            "l1_mean": shifts.abs().sum(dim=-1).mean().item(),
            "l2_mean": shifts.norm(dim=-1).mean().item(),
            "z_l2_mean": factual.norm(dim=-1).mean().item(),
        },
        "identity_shift": identity_report,
    }


@torch.no_grad()
def _semantic_sample_metrics(
    samples: torch.Tensor,
    factual: torch.Tensor,
    realized_states: dict[str, torch.Tensor],
    decoder: Any,
    h_s: Any,
    intervention: dict[str, int],
    decoder_used_for_training: bool,
    decoder_used_for_generation: bool,
) -> dict[str, Any]:
    try:
        decoder_device = next(decoder.parameters()).device
    except (AttributeError, StopIteration):
        decoder_device = torch.device("cpu")
    samples = samples.to(decoder_device)
    factual = factual.to(decoder_device)
    n_samples, batch_size, latent_dim = samples.shape
    flat = samples.reshape(n_samples * batch_size, latent_dim)
    predicted_joint = decoder.log_joint(flat).exp().reshape(
        n_samples, batch_size, -1
    ).mean(dim=0)
    factual_joint = decoder.log_joint(factual).exp()
    target_joint = h_s.compose(factual_joint.cpu(), intervention).to(predicted_joint)
    predicted_joint = predicted_joint.clamp_min(1e-12)
    positive = target_joint > 0
    semantic_kl = torch.where(
        positive,
        target_joint
        * (target_joint.clamp_min(1e-12).log() - predicted_joint.log()),
        torch.zeros_like(target_joint),
    ).sum(dim=-1).mean()

    accuracy_by_column: dict[str, float] = {}
    marginals = decoder.marginal_probabilities(flat)
    for column in decoder.columns:
        mixture = marginals[column.name].reshape(
            n_samples, batch_size, column.n_categories
        ).mean(dim=0)
        prediction = mixture.argmax(dim=-1).cpu()
        truth = realized_states[column.name].cpu()
        accuracy_by_column[column.name] = prediction.eq(truth).float().mean().item()
    return {
        "semantic_kl": semantic_kl.item(),
        "realized_state_accuracy": accuracy_by_column,
        "realized_state_accuracy_mean": sum(accuracy_by_column.values())
        / len(accuracy_by_column),
        "decoder_used_for_training": decoder_used_for_training,
        "decoder_used_for_generation": decoder_used_for_generation,
        "independent_validation": not (
            decoder_used_for_training or decoder_used_for_generation
        ),
    }


def _evaluate_samples(
    samples: torch.Tensor,
    *,
    factual: torch.Tensor,
    observed_counterfactual: torch.Tensor,
    training_latents: torch.Tensor,
    identity: torch.Tensor,
    realized_states: dict[str, torch.Tensor],
    decoder: Any,
    h_s: Any,
    intervention: dict[str, int],
    decoder_used_for_training: bool,
    decoder_used_for_generation: bool = False,
) -> dict[str, Any]:
    report = _latent_sample_metrics(
        samples,
        factual,
        observed_counterfactual,
        training_latents,
        identity,
    )
    report["semantic_evaluation"] = _semantic_sample_metrics(
        samples,
        factual,
        realized_states,
        decoder,
        h_s,
        intervention,
        decoder_used_for_training,
        decoder_used_for_generation,
    )
    return report


def evaluate_flow_manipulator(config: dict[str, Any]) -> None:
    """Evaluate the configured flow on official-test units and write its report."""
    variant = _flow_variant(config)
    artifact = load_latent_artifact(config)
    train_idx, test_idx = _official_indices(artifact)
    columns, _ = load_schema(config.get("sim_config"))
    intervention = load_intervention(config.get("sim_config"))
    device = config["encoder"]["device"]

    decoder = load_semantic_decoder(
        config["paths"]["decoder_model"],
        device=device,
        expected_variant=config["semantic_decoder"]["variant"],
        expected_columns=columns,
        expected_metadata=_decoder_metadata(config, artifact),
    )
    flow_metadata = _validate_flow_info(config, artifact, columns, variant)
    model = load_latent_intervention(
        _flow_model_path(config, variant),
        device=device,
        expected_variant=variant,
        expected_columns=columns,
        expected_metadata=flow_metadata,
    )
    h_s = load_symbolic_kernel(config.get("sim_config"))
    if list(h_s.columns) != list(decoder.columns):
        raise ValueError("symbolic-kernel schema does not match the semantic decoder")

    s_factual = _aligned_targets(
        config["paths"]["sim_factual"], artifact.ids, decoder.columns
    )
    s_prime = _aligned_targets(
        config["paths"]["sim_counterfactual"], artifact.ids, decoder.columns
    )
    factual_test = _subset(s_factual, test_idx)
    counterfactual_test = _subset(s_prime, test_idx)
    z_test = artifact.z[test_idx].to(device)
    n_samples = int(
        config["latent_intervention"].get("flow", {}).get("eval_samples", 1)
    )
    generator = torch.Generator(device=torch.device(device)).manual_seed(
        int(config["seed"])
    )
    with torch.no_grad():
        if variant == "state_flow":
            samples = sample_counterfactual(
                model,
                z_test,
                intervention,
                n_samples=n_samples,
                source_states=factual_test,
                h_s=h_s,
                generator=generator,
            )
        else:
            samples = sample_counterfactual(
                model,
                z_test,
                intervention,
                n_samples=n_samples,
                generator=generator,
            )

    decoder_accuracy = accuracy(decoder, z_test, factual_test)
    decoder_calibration = calibration_metrics(
        decoder,
        z_test,
        factual_test,
        n_bins=int(config["semantic_decoder"]["calibration_bins"]),
    )
    evaluation = _evaluate_samples(
        samples,
        factual=z_test,
        observed_counterfactual=artifact.z_prime,
        training_latents=artifact.z[train_idx],
        identity=artifact.is_identity,
        realized_states=counterfactual_test,
        decoder=decoder,
        h_s=h_s,
        intervention=intervention,
        decoder_used_for_training=(variant == "direct_semantic_flow"),
    )
    report = {
        "manipulator_variant": variant,
        "intervention": intervention,
        "n_test": len(test_idx),
        "n_samples": n_samples,
        "decoder_factual_accuracy": decoder_accuracy,
        "decoder_factual_calibration": decoder_calibration,
        "counterfactual": evaluation,
    }

    if variant == "state_flow":
        regimes: dict[str, Any] = {"observed_source_sampled_target": evaluation}
        with torch.no_grad():
            oracle_generator = torch.Generator(
                device=torch.device(device)
            ).manual_seed(int(config["seed"]) + 1)
            oracle_samples = sample_counterfactual(
                model,
                z_test,
                intervention,
                n_samples=n_samples,
                source_states=factual_test,
                target_states=counterfactual_test,
                generator=oracle_generator,
            )
            inferred_generator = torch.Generator(
                device=torch.device(device)
            ).manual_seed(int(config["seed"]) + 2)
            inferred_samples = sample_counterfactual(
                model,
                z_test,
                intervention,
                n_samples=n_samples,
                decoder=decoder,
                h_s=h_s,
                generator=inferred_generator,
            )
        regimes["observed_source_realized_target_oracle"] = _evaluate_samples(
            oracle_samples,
            factual=z_test,
            observed_counterfactual=artifact.z_prime,
            training_latents=artifact.z[train_idx],
            identity=artifact.is_identity,
            realized_states=counterfactual_test,
            decoder=decoder,
            h_s=h_s,
            intervention=intervention,
            decoder_used_for_training=False,
        )
        regimes["inferred_source_sampled_target"] = _evaluate_samples(
            inferred_samples,
            factual=z_test,
            observed_counterfactual=artifact.z_prime,
            training_latents=artifact.z[train_idx],
            identity=artifact.is_identity,
            realized_states=counterfactual_test,
            decoder=decoder,
            h_s=h_s,
            intervention=intervention,
            decoder_used_for_training=False,
            decoder_used_for_generation=True,
        )
        report["state_flow_regimes"] = regimes

    report_path = _flow_report_path(config, variant)
    _write_json(report_path, report)
    print(json.dumps(report, indent=2))
    print(f"wrote {report_path}")
