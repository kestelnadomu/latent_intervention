"""Training/evaluation adapter for the privileged, fixed-intervention MLP baseline."""

from __future__ import annotations

import fcntl
import json
from pathlib import Path

import torch

from src.artifact_io import sha256_file, write_json
from src.oracle_regression import (
    ORACLE_VARIANT,
    OracleRegression,
    regression_metrics,
    train_oracle_regression,
)
from src.oracle_targets import load_oracle_targets, require_oracle, source_metadata
from src.pair_encoding import load_latent_artifact
from src.schema import load_intervention, load_schema


def _split(config, artifact):
    # Use the same deterministic within-training split convention as g. Positions
    # here index the train-only oracle artifact, not the canonical all-unit tensor.
    from src.pipeline import _fit_calibration_split

    settings = config["latent_intervention"][ORACLE_VARIANT]
    return _fit_calibration_split(
        torch.arange(len(artifact.train_ids)),
        settings["validation_split"],
        settings["split_seed"],
    )


def _metadata(config, artifact, target_sha256):
    fit, validation = _split(config, artifact)
    columns, _ = load_schema(config["sim_config"])
    return {
        "variant": ORACLE_VARIANT,
        "privileged_supervision": "training_pair_Z_prime",
        "decoder_used_for_training": False,
        "symbolic_kernel_used_for_training": False,
        "sources": source_metadata(config, artifact),
        "oracle_target_sha256": target_sha256,
        "schema": [[column.name, column.n_categories] for column in columns],
        "training_seed": config["seed"],
        "training_settings": config["latent_intervention"][ORACLE_VARIANT],
        "fit_ids": [artifact.train_ids[i] for i in fit.tolist()],
        "validation_ids": [artifact.train_ids[i] for i in validation.tolist()],
    }


def _paths(config):
    model = Path(config["paths"]["manipulator_model"])
    report = Path(config["paths"]["oracle_training_report"])
    info = model.with_suffix(".info.json")
    paths = (model, report, info, Path(config["paths"]["eval_report"]))
    protected = {
        Path(config["paths"][key]).resolve()
        for key in (
            "latents",
            "decoder_model",
            "oracle_targets",
            "texts",
            "texts_counterfactual",
            "pair_index",
            "sim_factual",
            "sim_counterfactual",
        )
    }
    protected.update(
        path.with_suffix(".info.json")
        for path in list(protected)
        if path.suffix == ".pt"
    )
    if len({path.resolve() for path in paths}) != len(paths) or any(
        path.resolve() in protected for path in paths
    ):
        raise ValueError(
            "oracle output paths must be distinct from each other and upstream artifacts"
        )
    return model, report, info


def train_oracle_manipulator(config):
    """Train without g/h_S; test targets are never used for fitting or selection."""
    from src.pipeline import _indices_for_ids

    require_oracle(config)
    artifact = load_latent_artifact(config)
    targets = load_oracle_targets(config, artifact)
    fit, validation = _split(config, artifact)
    indices = _indices_for_ids(artifact.ids, targets.ids)
    latents = artifact.z[indices]
    metadata = _metadata(config, artifact, targets.artifact_sha256)
    model_path, report_path, info_path = _paths(config)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with (model_path.parent / ".oracle_regression.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any(
            path.exists()
            for path in (
                model_path,
                report_path,
                info_path,
                Path(config["paths"]["eval_report"]),
            )
        ):
            raise FileExistsError(
                "oracle outputs already exist; choose a new manipulator tag/run path"
            )
        settings = config["latent_intervention"][ORACLE_VARIANT]
        model = OracleRegression(
            artifact.z.shape[1],
            load_intervention(config["sim_config"]),
            **{key: settings[key] for key in ("hidden_dim", "n_hidden", "dropout")},
        )
        progress_path = report_path.with_name("training.progress.json")

        def progress(_model, record):
            write_json(progress_path, {"status": "running", **record})

        summary = train_oracle_regression(
            model,
            latents[fit],
            targets.z_prime[fit],
            validation_latents=latents[validation],
            validation_targets=targets.z_prime[validation],
            seed=config["seed"],
            device=config["encoder"]["device"],
            epoch_callback=progress,
            **{
                key: settings[key]
                for key in (
                    "epochs",
                    "batch_size",
                    "lr",
                    "weight_decay",
                    "patience",
                    "min_delta",
                    "scheduler_patience",
                    "scheduler_factor",
                    "min_lr",
                    "grad_clip",
                )
            },
        )
        # Revalidate immutable dependencies before publishing a new checkpoint.
        current = load_latent_artifact(config)
        current_targets = load_oracle_targets(config, current)
        if _metadata(config, current, current_targets.artifact_sha256) != metadata:
            raise ValueError(
                "oracle training inputs changed; refusing checkpoint publication"
            )
        model.save(model_path, metadata=metadata)
        checkpoint_sha256 = sha256_file(model_path)
        info = {
            "format_version": 1,
            "checkpoint_sha256": checkpoint_sha256,
            "metadata": metadata,
        }
        write_json(info_path, info)
        report = {
            **info,
            "variant": ORACLE_VARIANT,
            "training": summary,
            "split": {
                "official_train": len(targets.ids),
                "fit": len(fit),
                "validation": len(validation),
                "official_test": len(artifact.test_ids),
            },
            "validation": regression_metrics(
                model.predict(latents[validation]).cpu(), targets.z_prime[validation]
            ),
            "validation_identity_reference": regression_metrics(
                latents[validation], targets.z_prime[validation]
            ),
            "official_test_used_for_training_or_selection": False,
            "interpretation": "Oracle-supervised conditional-mean reference, not a guaranteed upper bound.",
        }
        write_json(report_path, report)
        write_json(
            progress_path,
            {
                "status": "complete",
                "best_epoch": summary["best_epoch"],
                "checkpoint_sha256": checkpoint_sha256,
            },
        )
    print(f"wrote {model_path}; validation-selected epoch {summary['best_epoch']}")
    return report


def evaluate_oracle_manipulator(config):
    """Held-out evaluation; does not need or load privileged training-target files."""
    from src.flow_workflow import _evaluate_samples
    from src.pipeline import (
        _aligned_targets,
        _decoder_metadata,
        _official_indices,
        _subset,
    )
    from src.semantic_decoder import accuracy, load_semantic_decoder
    from src.symbolic_intervention import load_symbolic_kernel

    require_oracle(config)
    artifact = load_latent_artifact(config)
    train, test = _official_indices(artifact)
    model_path, _, info_path = _paths(config)
    info = json.loads(info_path.read_text())
    if info.get("format_version") != 1 or info.get("checkpoint_sha256") != sha256_file(
        model_path
    ):
        raise ValueError("oracle checkpoint checksum/sidecar mismatch")
    recorded = info.get("metadata")
    if not isinstance(recorded, dict) or not isinstance(
        recorded.get("oracle_target_sha256"), str
    ):
        raise ValueError("oracle checkpoint has incomplete provenance")
    metadata = _metadata(config, artifact, recorded["oracle_target_sha256"])
    if metadata != recorded:
        raise ValueError(
            "oracle checkpoint does not match the current latent space/configuration"
        )
    intervention = load_intervention(config["sim_config"])
    model = OracleRegression.load(
        model_path,
        device=config["encoder"]["device"],
        expected_metadata=metadata,
        expected_intervention=intervention,
    )
    factual = artifact.z[test]
    prediction = model.predict(factual, intervention=intervention).cpu()
    columns, _ = load_schema(config["sim_config"])
    decoder = load_semantic_decoder(
        config["paths"]["decoder_model"],
        device=config["encoder"]["device"],
        expected_variant=config["semantic_decoder"]["variant"],
        expected_columns=columns,
        expected_metadata=_decoder_metadata(config, artifact),
    )
    h_s = load_symbolic_kernel(config["sim_config"])
    if list(h_s.columns) != list(columns):
        raise ValueError("symbolic-kernel schema does not match the oracle evaluation")
    realized = _aligned_targets(
        config["paths"]["sim_counterfactual"], artifact.ids, columns
    )
    factual_states = _aligned_targets(
        config["paths"]["sim_factual"], artifact.ids, columns
    )
    report = {
        "manipulator_variant": ORACLE_VARIANT,
        "intervention": intervention,
        "n_test": len(test),
        "n_samples": 1,
        "privileged_supervision": "training_pair_Z_prime",
        "checkpoint_sha256": sha256_file(model_path),
        "latent_artifact_sha256": artifact.artifact_sha256,
        "decoder_checkpoint_sha256": sha256_file(config["paths"]["decoder_model"]),
        "sim_counterfactual_sha256": sha256_file(config["paths"]["sim_counterfactual"]),
        "recovery": regression_metrics(prediction, artifact.z_prime),
        "identity_reference": regression_metrics(factual, artifact.z_prime),
        "decoder_factual_accuracy": accuracy(
            decoder, factual, _subset(factual_states, test)
        ),
        "counterfactual": _evaluate_samples(
            prediction.unsqueeze(0),
            factual=factual,
            observed_counterfactual=artifact.z_prime,
            training_latents=artifact.z[train],
            identity=artifact.is_identity,
            realized_states=_subset(realized, test),
            decoder=decoder,
            h_s=h_s,
            intervention=intervention,
            decoder_used_for_training=False,
        ),
    }
    for name, mask in (
        ("identity", artifact.is_identity),
        ("nonidentity", ~artifact.is_identity),
    ):
        report[f"{name}_recovery"] = {
            "count": int(mask.sum()),
            "metrics": (
                regression_metrics(prediction[mask], artifact.z_prime[mask])
                if mask.any()
                else None
            ),
        }
    write_json(config["paths"]["eval_report"], report)
    print(json.dumps(report, indent=2))
    print(f"wrote {config['paths']['eval_report']}")
    return report
