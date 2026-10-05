"""Training protocol, best-checkpoint publication and reports for one g run.

This module owns bookkeeping only: the numerical trainer and model definitions
are separate. The existing pipeline still owns data alignment and split selection.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from src.artifact_io import write_json, sha256_file


class DecoderTrainingRun:
    """Keep per-run reporting out of the pipeline's data/training control flow."""

    def __init__(
        self, config, artifact, fit_idx, calibration_idx, base_metadata, started_at
    ):
        cfg = config["semantic_decoder"]
        stopping = cfg.get("early_stopping", {})
        scheduler = cfg.get("scheduler", {})
        protocol = {
            "seed": int(config["seed"]),
            "split_seed": int(cfg.get("split_seed", config["seed"])),
            "max_epochs": int(cfg["epochs"]),
            "batch_size": int(cfg["batch_size"]),
            "lr": float(cfg["lr"]),
            "weight_decay": float(cfg.get("weight_decay", 0.0)),
            "patience": int(stopping.get("patience", 30)),
            "min_delta": float(stopping.get("min_delta", 1e-4)),
            "scheduler_patience": int(scheduler.get("patience", 10)),
            "scheduler_factor": float(scheduler.get("factor", 0.5)),
            "min_lr": float(scheduler.get("min_lr", 1e-6)),
            "optimizer": "Adam",
            "selection_metric": "validation_joint_nll",
        }
        split_ids = {
            "fit": [artifact.ids[i] for i in fit_idx.tolist()],
            "validation": [artifact.ids[i] for i in calibration_idx.tolist()],
            "official_test": list(artifact.test_ids),
        }
        split_hash = hashlib.sha256(
            json.dumps(
                split_ids, sort_keys=True, separators=(",", ":"), default=str
            ).encode("utf-8")
        ).hexdigest()
        metadata = {
            **base_metadata,
            "training_protocol": protocol,
            "split_sha256": split_hash,
        }
        progress_path = Path(config["paths"]["decoder_report"]).with_suffix(
            ".progress.json"
        )

        self.config, self.artifact = config, artifact
        self.fit_idx, self.calibration_idx = fit_idx, calibration_idx
        self.started_at, self.protocol = started_at, protocol
        self.split_ids, self.split_hash = split_ids, split_hash
        self.metadata, self.progress_path = metadata, progress_path

    def trainer_options(self) -> dict:
        return {
            **{
                key: self.protocol[key]
                for key in (
                    "patience",
                    "min_delta",
                    "scheduler_patience",
                    "scheduler_factor",
                    "min_lr",
                    "weight_decay",
                )
            },
            "epoch_callback": self.checkpoint_epoch,
            "verbose": bool(self.config["semantic_decoder"].get("verbose", True)),
        }

    def checkpoint_epoch(self, model, record):
        if record["improved"]:
            model.save(
                self.config["paths"]["decoder_model"],
                metadata={**self.metadata, "selection": record},
            )
        write_json(self.progress_path, {"status": "running", **record})

    def finish(self, decoder, history, metrics, validation_loss) -> dict:
        config, artifact = self.config, self.artifact
        cfg = config["semantic_decoder"]
        fit_idx, calibration_idx = self.fit_idx, self.calibration_idx
        started_at, protocol = self.started_at, self.protocol
        split_ids, split_hash = self.split_ids, self.split_hash
        metadata, progress_path = self.metadata, self.progress_path
        summary = getattr(decoder, "training_summary", {})
        decoder.save(
            config["paths"]["decoder_model"],
            metadata={
                **metadata,
                "selection": {
                    key: value for key, value in summary.items() if key != "history"
                },
            },
        )

        report = {
            "decoder_variant": cfg["variant"],
            "seed": int(config["seed"]),
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint_sha256": sha256_file(config["paths"]["decoder_model"]),
            "config": config,
            "latent_artifact_sha256": artifact.artifact_sha256,
            "encoder": artifact.encoder_info,
            "split": {
                "official_train": len(artifact.train_ids),
                "fit": len(fit_idx),
                "calibration": len(calibration_idx),
                "official_test": len(artifact.test_ids),
            },
            "training": {
                **summary,
                "epochs": len(history),
                "protocol": protocol,
                "final_loss": history[-1],
            },
            "split_seed": protocol["split_seed"],
            "split_ids": split_ids,
            "split_sha256": split_hash,
            "validation_joint_nll": validation_loss,
            "validation_note": "Used for checkpoint/hyperparameter selection; not an unbiased test estimate. No probability calibration was fitted.",
            "calibration": metrics,
        }
        write_json(config["paths"]["decoder_report"], report)
        write_json(
            progress_path,
            {
                "status": "complete",
                "epochs_run": len(history),
                "validation_joint_nll": validation_loss,
            },
        )
        ece = {name: round(values["ece"], 3) for name, values in metrics.items()}
        print("calibration ECE:", ece)
        print(f"wrote {config['paths']['decoder_model']}")
        print(f"wrote {config['paths']['decoder_report']}")
        return report
