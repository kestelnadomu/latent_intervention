"""
Training and evaluation pipeline: text latents -> semantic decoder -> manipulator.

Stages (hyperparameters from src/config.yaml; data + schema from exp/sim/):
    encode             encode generated input texts into latents (data/latents/)
    train-decoder      train the semantic decoder g: Z -> S on factual pairs
    train-manipulator  train the latent manipulator h_Z against frozen g and
                       counterfactual targets S' from the SCM simulation
    evaluate           manipulator faithfulness on the official test split

Run from the repository root:
    uv run python -m src.pipeline encode
    uv run python -m src.pipeline train-decoder
    uv run python -m src.pipeline train-manipulator
    uv run python -m src.pipeline evaluate
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.config import load_config
from src.latent_intervention import (
    LatentIntervention,
    LatentInterventionNoiseToken,
    LatentInterventionDist,
    LatentInterventionParticles,
    LatentInterventionPreAdditive,
    make_objective,
    train_latent_intervention,
    train_latent_intervention_dist,
    train_latent_intervention_noise_token,
    train_latent_intervention_particles,
    train_latent_intervention_preadditive,
)
from src.pair_encoding import LatentArtifact, load_latent_artifact, sha256_file
from src.schema import ColumnSpec, load_intervention, load_schema
from src.semantic_decoder import (
    accuracy,
    calibration_metrics,
    load_semantic_decoder,
    make_semantic_decoder,
    targets_from_dataframe,
    train_semantic_decoder,
)
from src.symbolic_intervention import load_symbolic_kernel


INTERVENTION_VARIANTS = {
    "baseline": LatentIntervention,
    "pre_additive": LatentInterventionPreAdditive,
    "noise_token": LatentInterventionNoiseToken,
    "dist": LatentInterventionDist,
    "particles": LatentInterventionParticles,
}


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
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _manipulator_info_path(model_path: str | Path) -> Path:
    return Path(model_path).with_suffix(".info.json")


def _write_manipulator_info(config: dict[str, Any], artifact: LatentArtifact) -> None:
    _write_json(
        _manipulator_info_path(config["paths"]["manipulator_model"]),
        {
            "format_version": 1,
            "latent_artifact_sha256": artifact.artifact_sha256,
            "decoder_checkpoint_sha256": sha256_file(config["paths"]["decoder_model"]),
            "manipulator_checkpoint_sha256": sha256_file(
                config["paths"]["manipulator_model"]
            ),
            "sim_counterfactual_sha256": sha256_file(
                config["paths"]["sim_counterfactual"]
            ),
            "decoder_variant": config["semantic_decoder"]["variant"],
            "manipulator_variant": config["latent_intervention"]["variant"],
        },
    )


def _validate_manipulator_info(
    config: dict[str, Any], artifact: LatentArtifact
) -> None:
    path = _manipulator_info_path(config["paths"]["manipulator_model"])
    if not path.exists():
        raise ValueError(
            f"manipulator metadata is missing: {path}; retrain the manipulator"
        )
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(
            f"invalid manipulator metadata: {path}; retrain the manipulator"
        ) from exc
    expected = {
        "format_version": 1,
        "latent_artifact_sha256": artifact.artifact_sha256,
        "decoder_checkpoint_sha256": sha256_file(config["paths"]["decoder_model"]),
        "manipulator_checkpoint_sha256": sha256_file(
            config["paths"]["manipulator_model"]
        ),
        "sim_counterfactual_sha256": sha256_file(config["paths"]["sim_counterfactual"]),
        "decoder_variant": config["semantic_decoder"]["variant"],
        "manipulator_variant": config["latent_intervention"]["variant"],
    }
    if info != expected:
        raise ValueError(
            "manipulator metadata does not match the current latent/decoder artifacts; "
            "retrain the manipulator"
        )


def stage_encode(config: dict[str, Any]) -> None:
    """Encode all factual texts and held-out counterfactual pairs once."""
    from src.pair_encoding import encode_pairs

    encode_pairs(config)


def stage_train_decoder(config: dict[str, Any]) -> None:
    """Fit the decoder on official-train units and assess a train-only holdout."""
    artifact = load_latent_artifact(config)
    columns, _ = load_schema(config.get("sim_config"))
    targets = _aligned_targets(config["paths"]["sim_factual"], artifact.ids, columns)
    cfg = config["semantic_decoder"]
    official_train_idx, official_test_idx = _official_indices(artifact)
    fit_idx, calibration_idx = _fit_calibration_split(
        official_train_idx,
        float(cfg["calibration_split"]),
        int(config["seed"]),
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
    history = train_semantic_decoder(
        decoder,
        artifact.z[fit_idx],
        _subset(targets, fit_idx),
        epochs=int(cfg["epochs"]),
        batch_size=int(cfg["batch_size"]),
        lr=float(cfg["lr"]),
        seed=int(config["seed"]),
        device=config["encoder"]["device"],
    )
    metrics = calibration_metrics(
        decoder,
        artifact.z[calibration_idx],
        _subset(targets, calibration_idx),
        n_bins=int(cfg["calibration_bins"]),
    )
    decoder.save(
        config["paths"]["decoder_model"],
        metadata=_decoder_metadata(config, artifact),
    )

    report = {
        "decoder_variant": cfg["variant"],
        "seed": int(config["seed"]),
        "latent_artifact_sha256": artifact.artifact_sha256,
        "encoder": artifact.encoder_info,
        "split": {
            "official_train": len(official_train_idx),
            "fit": len(fit_idx),
            "calibration": len(calibration_idx),
            "official_test": len(official_test_idx),
        },
        "training": {
            "epochs": int(cfg["epochs"]),
            "final_loss": history[-1],
        },
        "calibration": metrics,
    }
    _write_json(config["paths"]["decoder_report"], report)
    ece = {name: round(values["ece"], 3) for name, values in metrics.items()}
    print("calibration ECE:", ece)
    print(f"wrote {config['paths']['decoder_model']}")
    print(f"wrote {config['paths']['decoder_report']}")


def stage_train_manipulator(config: dict[str, Any]) -> None:
    """Train the manipulator on official training units against the frozen decoder."""
    artifact = load_latent_artifact(config)
    train_idx, _ = _official_indices(artifact)
    columns, _ = load_schema(config.get("sim_config"))
    intervention = load_intervention(config.get("sim_config"))
    cfg = config["latent_intervention"]

    decoder = load_semantic_decoder(
        config["paths"]["decoder_model"],
        expected_variant=config["semantic_decoder"]["variant"],
        expected_columns=columns,
        expected_metadata=_decoder_metadata(config, artifact),
    )
    s_prime = _aligned_targets(
        config["paths"]["sim_counterfactual"], artifact.ids, decoder.columns
    )
    h_s = load_symbolic_kernel(config.get("sim_config"))
    if list(h_s.columns) != list(decoder.columns):
        raise ValueError("symbolic-kernel schema does not match the semantic decoder")

    model_kwargs = dict(
        latent_dim=artifact.z.shape[1],
        columns=decoder.columns,
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
    )
    train_kwargs = dict(
        decoder=decoder,
        latents=artifact.z[train_idx],
        intervention=intervention,
        h_s=h_s,
        s_prime=_subset(s_prime, train_idx),
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        proximity_weight=cfg["proximity_weight"],
        sparsity_weight=cfg["sparsity_weight"],
        seed=config["seed"],
        device=config["encoder"]["device"],
    )

    if cfg["variant"] == "baseline":
        model = LatentIntervention(**model_kwargs)
        train_latent_intervention(model=model, epochs=cfg["epochs"], **train_kwargs)
    elif cfg["variant"] == "pre_additive":
        pa = cfg["pre_additive"]
        model = LatentInterventionPreAdditive(**model_kwargs, noise_std=pa["noise_std"])
        train_latent_intervention_preadditive(
            model=model,
            epochs=cfg["epochs"],
            n_samples=pa["n_samples"],
            **train_kwargs,
        )
    elif cfg["variant"] == "noise_token":
        nt = cfg["noise_token"]
        model = LatentInterventionNoiseToken(**model_kwargs, noise_dim=nt["noise_dim"])
        train_latent_intervention_noise_token(
            model=model,
            epochs=cfg["epochs"],
            n_samples=nt["n_samples"],
            entropy_weight=nt["entropy_weight"],
            **train_kwargs,
        )
    elif cfg["variant"] == "dist":
        dist = cfg["dist"]
        model = LatentInterventionDist(
            **model_kwargs,
            embed_dim=dist["embed_dim"],
            top_k=dist["top_k"],
        )
        train_latent_intervention_dist(
            model=model,
            pretrain_epochs=dist["pretrain_epochs"],
            joint_epochs=dist["joint_epochs"],
            realiser_l1=dist["realiser_l1"],
            realiser_l2=dist["realiser_l2"],
            **train_kwargs,
        )
    elif cfg["variant"] == "particles":
        pt = cfg["particles"]
        model = LatentInterventionParticles(
            **model_kwargs,
            n_particles=pt["n_particles"],
            uniform_weights=pt["uniform_weights"],
        )
        train_latent_intervention_particles(
            model=model,
            epochs=cfg["epochs"],
            entropy_weight=pt["entropy_weight"],
            **train_kwargs,
        )
    else:
        raise ValueError(f"unknown latent intervention variant: {cfg['variant']}")
    model.save(config["paths"]["manipulator_model"])
    _write_manipulator_info(config, artifact)
    print(f"wrote {config['paths']['manipulator_model']} (intervention {intervention})")


def stage_evaluate(config: dict[str, Any]) -> None:
    """Evaluate decoder and manipulator behavior on official test units only."""
    artifact = load_latent_artifact(config)
    _, test_idx = _official_indices(artifact)
    columns, _ = load_schema(config.get("sim_config"))
    intervention = load_intervention(config.get("sim_config"))

    decoder = load_semantic_decoder(
        config["paths"]["decoder_model"],
        expected_variant=config["semantic_decoder"]["variant"],
        expected_columns=columns,
        expected_metadata=_decoder_metadata(config, artifact),
    )
    _validate_manipulator_info(config, artifact)
    manipulator_type = INTERVENTION_VARIANTS.get(
        config["latent_intervention"]["variant"]
    )
    if manipulator_type is None:
        raise ValueError(
            "unknown latent intervention variant: "
            f"{config['latent_intervention']['variant']}"
        )
    model = manipulator_type.load(config["paths"]["manipulator_model"])
    s_factual = _aligned_targets(
        config["paths"]["sim_factual"], artifact.ids, decoder.columns
    )
    s_prime = _aligned_targets(
        config["paths"]["sim_counterfactual"], artifact.ids, decoder.columns
    )
    z_test = artifact.z[test_idx]
    values, mask = make_objective(
        intervention, decoder.columns, batch_size=len(test_idx)
    )
    with torch.no_grad():
        if isinstance(model, (LatentInterventionPreAdditive, LatentInterventionNoiseToken)):
            # stochastic plans: one seeded draw per text, reproducible across runs
            generator = torch.Generator().manual_seed(config["seed"])
            z_prime = model(z_test, values, mask, generator)
        else:
            z_prime = model(z_test, values, mask)
    preds = decoder.predict(z_prime)

    consistency = {
        column.name: (preds[column.name].cpu() == s_prime[column.name][test_idx].cpu())
        .float()
        .mean()
        .item()
        for column in decoder.columns
    }
    factual_targets = _subset(s_factual, test_idx)
    decoder_accuracy = accuracy(decoder, z_test, factual_targets)
    decoder_calibration = calibration_metrics(
        decoder,
        z_test,
        factual_targets,
        n_bins=int(config["semantic_decoder"]["calibration_bins"]),
    )
    delta = z_prime - z_test
    report = {
        "intervention": intervention,
        "n_test": len(test_idx),
        "decoder_factual_accuracy": decoder_accuracy,
        "decoder_factual_calibration": decoder_calibration,
        "consistency_accuracy": consistency,
        "consistency_accuracy_mean": sum(consistency.values()) / len(consistency),
        "latent_shift": {
            "l1_mean": delta.abs().sum(dim=1).mean().item(),
            "l2_mean": delta.norm(dim=1).mean().item(),
            "z_l2_mean": z_test.norm(dim=1).mean().item(),
        },
    }
    _write_json(config["paths"]["eval_report"], report)
    print(json.dumps(report, indent=2))
    print(f"wrote {config['paths']['eval_report']}")


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
    args = parser.parse_args()
    config = load_config(path=args.config) if args.config else load_config()
    STAGES[args.stage](config)


if __name__ == "__main__":
    main()
