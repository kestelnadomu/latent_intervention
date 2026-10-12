"""Pipeline adapter for the five base h_Z plans (baseline, pre_additive, noise_token, dist,
particles): train on official training units and evaluate on official test units.

``src.pipeline`` dispatches here for every variant that is neither a flow nor the oracle
reference, mirroring ``flow/workflow.py`` and ``oracle/workflow.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.latent_intervention.base import (
    LatentIntervention,
    LatentInterventionDist,
    LatentInterventionNoiseToken,
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
from src.schema import load_intervention, load_schema
from src.semantic_decoder.model import accuracy, calibration_metrics, load_semantic_decoder
from src.symbolic_intervention import load_symbolic_kernel

BASE_VARIANTS = {
    "baseline": LatentIntervention,
    "pre_additive": LatentInterventionPreAdditive,
    "noise_token": LatentInterventionNoiseToken,
    "dist": LatentInterventionDist,
    "particles": LatentInterventionParticles,
}


def _manipulator_info_path(model_path: str | Path) -> Path:
    return Path(model_path).with_suffix(".info.json")


def _write_manipulator_info(config: dict[str, Any], artifact: LatentArtifact) -> None:
    from src.pipeline import _write_json

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


def train_base_manipulator(config: dict[str, Any]) -> None:
    """Train the configured base plan on official training units."""
    from src.pipeline import _aligned_targets, _decoder_metadata, _official_indices, _subset

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

    torch.manual_seed(int(config["seed"]))
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


def evaluate_base_manipulator(config: dict[str, Any]) -> None:
    """Evaluate decoder and base-plan manipulator behavior on official test units only."""
    from src.pipeline import (
        _aligned_targets,
        _decoder_metadata,
        _official_indices,
        _subset,
        _write_json,
    )

    artifact = load_latent_artifact(config)
    _, test_idx = _official_indices(artifact)
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
    _validate_manipulator_info(config, artifact)
    manipulator_type = BASE_VARIANTS.get(
        config["latent_intervention"]["variant"]
    )
    if manipulator_type is None:
        raise ValueError(
            "unknown latent intervention variant: "
            f"{config['latent_intervention']['variant']}"
        )
    model = manipulator_type.load(
        config["paths"]["manipulator_model"], device=device
    )
    s_factual = _aligned_targets(
        config["paths"]["sim_factual"], artifact.ids, decoder.columns
    )
    s_prime = _aligned_targets(
        config["paths"]["sim_counterfactual"], artifact.ids, decoder.columns
    )
    z_test = artifact.z[test_idx].to(device)
    values, mask = make_objective(
        intervention, decoder.columns, batch_size=len(test_idx), device=device
    )
    with torch.no_grad():
        if isinstance(model, (LatentInterventionPreAdditive, LatentInterventionNoiseToken)):
            # stochastic plans: one seeded draw per text, reproducible across runs
            generator = torch.Generator(device=torch.device(device)).manual_seed(
                config["seed"]
            )
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
