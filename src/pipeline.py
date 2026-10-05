"""
Training and evaluation pipeline: text latents -> semantic decoder -> manipulator.

Stages (hyperparameters from src/config.yaml; data + schema from exp/sim/):
    encode             encode generated input texts into latents (data/latents/)
    train-decoder      train the semantic decoder g: Z -> S on factual pairs
    train-manipulator  train the configured latent manipulator h_Z
    evaluate           manipulator faithfulness on the official test split

Run from the repository root:
    uv run python -m src.pipeline encode --encoder-variant langvae
    uv run python -m src.pipeline train-decoder --encoder-variant langvae --decoder-variant independent
    uv run python -m src.pipeline train-manipulator --encoder-variant langvae --decoder-variant independent
    uv run python -m src.pipeline evaluate --encoder-variant langvae --decoder-variant independent
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.artifact_io import write_json
from src.config import CONFIG_PATH, load_config
from src.semantic_decoder.reporting import DecoderTrainingRun
from src.encoder import add_encoder_arguments
from src.latent_intervention.flow.workflow import (
    FLOW_VARIANTS,
    evaluate_flow_manipulator,
    train_flow_manipulator,
)
from src.latent_intervention.base.workflow import (
    BASE_VARIANTS,
    evaluate_base_manipulator,
    train_base_manipulator,
)
from src.pair_encoding import LatentArtifact, load_latent_artifact, sha256_file
from src.latent_intervention.oracle.regression import ORACLE_VARIANT
from src.schema import ColumnSpec, load_schema
from src.semantic_decoder.model import (
    calibration_metrics,
    joint_nll,
    make_semantic_decoder,
    targets_from_dataframe,
    train_semantic_decoder,
)

def _load_latents(config: dict[str, Any]) -> tuple[torch.Tensor, list[int]]:
    """Compatibility wrapper returning factual latents and their unit IDs."""
    artifact = load_latent_artifact(config)
    return artifact.z, artifact.ids


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
    """Map ordered unit IDs to positions in the latent artifact."""
    positions = {unit_id: i for i, unit_id in enumerate(all_ids)}
    missing = set(selected_ids) - set(positions)
    if missing:
        raise ValueError(
            f"split IDs are absent from the latent artifact: {sorted(missing)}"
        )
    return torch.tensor(
        [positions[unit_id] for unit_id in selected_ids], dtype=torch.long
    )


def _official_indices(artifact: LatentArtifact) -> tuple[torch.Tensor, torch.Tensor]:
    """Return official train/test positions, preserving recorded ID order."""
    return (
        _indices_for_ids(artifact.ids, artifact.train_ids),
        _indices_for_ids(artifact.ids, artifact.test_ids),
    )


def _fit_calibration_split(
    train_idx: torch.Tensor,
    fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split only official training positions into fit and calibration subsets."""
    if not 0 < fraction < 1:
        raise ValueError("semantic_decoder.calibration_split must lie between 0 and 1")
    if len(train_idx) < 2:
        raise ValueError("at least two official training units are required")
    generator = torch.Generator().manual_seed(seed)
    shuffled = train_idx[torch.randperm(len(train_idx), generator=generator)]
    n_calibration = min(max(round(len(train_idx) * fraction), 1), len(train_idx) - 1)
    return shuffled[n_calibration:], shuffled[:n_calibration]


def _subset(
    targets: dict[str, torch.Tensor], idx: torch.Tensor
) -> dict[str, torch.Tensor]:
    return {name: values[idx] for name, values in targets.items()}


def _config_sha256(config_source: dict[str, Any] | str | Path) -> str:
    """Hash either an in-memory simulation config or its exact source file."""
    if isinstance(config_source, dict):
        canonical = json.dumps(
            config_source, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()
    return sha256_file(config_source)


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


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def stage_encode(config: dict[str, Any]) -> None:
    """Encode all factual texts and held-out counterfactual pairs once."""
    from src.pair_encoding import encode_pairs

    encode_pairs(config)


def stage_train_decoder(config: dict[str, Any]) -> dict[str, Any]:
    """Select g by validation NLL within official train; never inspect test metrics."""
    started_at = datetime.now(timezone.utc).isoformat()
    artifact = load_latent_artifact(config)
    columns, _ = load_schema(config.get("sim_config"))
    targets = _aligned_targets(config["paths"]["sim_factual"], artifact.ids, columns)
    cfg = config["semantic_decoder"]
    official_train_idx, official_test_idx = _official_indices(artifact)
    fit_idx, calibration_idx = _fit_calibration_split(
        official_train_idx,
        float(cfg["calibration_split"]),
        int(cfg.get("split_seed", config["seed"])),
    )

    torch.manual_seed(int(config["seed"]))
    decoder = make_semantic_decoder(
        cfg["variant"],
        latent_dim=artifact.z.shape[1],
        columns=columns,
        hidden_dim=int(cfg["hidden_dim"]),
        n_hidden=int(cfg["n_hidden"]),
        dropout=float(cfg["dropout"]),
        embed_dim=int(cfg["autoregressive"]["embed_dim"]),
    )
    fit_targets = _subset(targets, fit_idx)
    validation_targets = _subset(targets, calibration_idx)
    run = DecoderTrainingRun(
        config,
        artifact,
        fit_idx,
        calibration_idx,
        _decoder_metadata(config, artifact),
        started_at,
    )
    history = train_semantic_decoder(
        decoder,
        artifact.z[fit_idx],
        fit_targets,
        epochs=int(cfg["epochs"]),
        batch_size=int(cfg["batch_size"]),
        lr=float(cfg["lr"]),
        seed=int(config["seed"]),
        device=config["encoder"]["device"],
        validation_latents=artifact.z[calibration_idx],
        validation_targets=validation_targets,
        **run.trainer_options(),
    )
    metrics = calibration_metrics(
        decoder,
        artifact.z[calibration_idx],
        validation_targets,
        n_bins=int(cfg["calibration_bins"]),
    )
    validation_loss = joint_nll(
        decoder, artifact.z[calibration_idx], validation_targets
    )
    return run.finish(decoder, history, metrics, validation_loss)


def stage_train_manipulator(config: dict[str, Any]) -> None:
    """Train the configured manipulator on official training units."""
    variant = config["latent_intervention"]["variant"]
    if variant == ORACLE_VARIANT:
        from src.latent_intervention.oracle.workflow import train_oracle_manipulator

        train_oracle_manipulator(config)
    elif variant in FLOW_VARIANTS:
        train_flow_manipulator(config)
    else:
        train_base_manipulator(config)


def stage_evaluate(config: dict[str, Any]) -> None:
    """Evaluate decoder and manipulator behavior on official test units only."""
    variant = config["latent_intervention"]["variant"]
    if variant == ORACLE_VARIANT:
        from src.latent_intervention.oracle.workflow import evaluate_oracle_manipulator

        evaluate_oracle_manipulator(config)
    elif variant in FLOW_VARIANTS:
        evaluate_flow_manipulator(config)
    else:
        evaluate_base_manipulator(config)


STAGES = {
    "encode": stage_encode,
    "train-decoder": stage_train_decoder,
    "train-manipulator": stage_train_manipulator,
    "evaluate": stage_evaluate,
}


def main() -> None:
    """CLI entry point for the training/eval pipeline."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("stage", choices=STAGES, help="pipeline stage to run")
    parser.add_argument(
        "--config", default=None, help="path to an alternative config YAML"
    )
    add_encoder_arguments(parser)
    parser.add_argument(
        "--decoder-variant",
        choices=("independent", "autoregressive"),
        default=None,
        help="override semantic_decoder.variant before artifact paths are resolved",
    )
    parser.add_argument(
        "--manipulator-variant",
        choices=(*BASE_VARIANTS, *FLOW_VARIANTS, ORACLE_VARIANT),
        default=None,
        help="override latent_intervention.variant before artifact paths are resolved",
    )
    args = parser.parse_args()
    config = load_config(
        path=args.config or CONFIG_PATH,
        encoder_variant=args.encoder_variant,
        decoder_variant=args.decoder_variant,
        manipulator_variant=args.manipulator_variant,
        nomic_dim=args.nomic_dim,
        qwen3_dim=args.qwen3_dim,
        embeddinggemma_dim=args.embeddinggemma_dim,
    )
    STAGES[args.stage](config)


if __name__ == "__main__":
    main()
