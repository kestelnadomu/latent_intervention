"""Isolated native adaptation. No production pair encoding or downstream training."""

import copy
import gc
import json
import math
import os
import random
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import torch
import yaml

from . import TAG
from .artifacts import (
    ROOT,
    atomic_json,
    atomic_torch,
    export_checkpoint,
    packages,
    read_checkpoint,
    restore_resume,
    save_resume,
    source_hashes,
    trainable_schema,
    write_inference_config,
)
from .data import file_hash, passage_records, split_cvs
from .evaluation import audit, evaluate
from .model import (
    batch_tensors,
    load_native,
    parameter_hash,
    pooled_features,
    train_update,
)


def device_for(name):
    device = torch.device(name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("only FP32 CPU/CUDA is supported by this baseline")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
        device = torch.device(
            "cuda",
            device.index if device.index is not None else torch.cuda.current_device(),
        )
        torch.cuda.set_device(device)
    return device


def read_settings(config_path, device=None):
    recipe = yaml.safe_load(Path(config_path).read_text())
    if recipe.get("tag") != TAG:
        raise ValueError("this workflow is exclusively the langvae_ft baseline")
    integer_fields = (
        "epochs",
        "minimum_epochs",
        "patience",
        "batch_size",
        "accumulation_steps",
        "eval_batch_size",
        "feature_batch_size",
        "validation_samples",
        "audit_cvs",
        "cpu_threads",
        "log_every",
    )
    for key in integer_fields:
        if (
            isinstance(recipe[key], bool)
            or not isinstance(recipe[key], int)
            or recipe[key] < 1
        ):
            raise ValueError(f"{key} must be a positive integer")
    for key in ("encoder_lr", "decoder_lr", "clip_grad_norm", "max_training_hours"):
        if not math.isfinite(recipe[key]) or recipe[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not 0 <= recipe["warmup_epochs"] < recipe["epochs"]:
        raise ValueError("warm-up must end before the last training epoch")
    if not 0 < recipe["validation_fraction"] < 1:
        raise ValueError("invalid validation_fraction")
    if recipe["minimum_epochs"] > recipe["epochs"]:
        raise ValueError("minimum_epochs exceeds epochs")
    if device is not None:
        recipe["device"] = device
    raw = yaml.safe_load((ROOT / recipe["base_config"]).read_text())
    raw.pop("finetune_vae", None)  # do not inherit V1/V2/V3 hyperparameters
    encoder = copy.deepcopy(raw["encoder"])
    encoder.update(
        variant="langvae",
        local_checkpoint=None,
        tag=None,
        deterministic=True,
        device="cpu",
    )
    for name in (
        "model_revision",
        "langvae_encoder_model_revision",
        "langvae_decoder_model_revision",
    ):
        if not encoder.get(name):
            raise ValueError(f"pin {name} before training")
    if encoder["max_len"] != 512:
        raise ValueError("preserve the baseline's full-CV encoder length (512)")
    return recipe, raw, encoder


def _dataset(base, examples, recipe, device):
    records = {
        key: passage_records(rows, base.decoder.tokenizer)
        for key, rows in examples.items()
    }
    features = {
        key: pooled_features(base, rows, recipe["feature_batch_size"], device)
        for key, rows in records.items()
    }
    return {"records": records, "features": features}


def _audit(adapter, dataset, tokenizer, recipe, device):
    return audit(
        adapter,
        dataset["features"]["validation"],
        dataset["records"]["validation"],
        tokenizer,
        count=recipe["audit_cvs"],
        batch_size=recipe["eval_batch_size"],
        device=device,
        seed=recipe["seed"] + 901,
    )


def _evaluate(adapter, dataset, recipe, device):
    return evaluate(
        adapter,
        dataset["features"]["validation"],
        dataset["records"]["validation"],
        batch_size=recipe["eval_batch_size"],
        device=device,
        seed=recipe["seed"] + 902,
        samples=recipe["validation_samples"],
    )


def fit(config_path, *, stage="train", device_name=None, resume=None):
    """A smoke stage performs two updates only and never exports an encoding config.

    Resume is epoch-boundary only, including optimizer/RNG/data/source checks.
    An interrupted partial epoch is replayed; no exact cross-hardware promise.
    """
    if stage not in {"train", "smoke"}:
        raise ValueError("unknown training stage")
    recipe, raw, encoder = read_settings(config_path, device_name)
    if stage == "smoke":
        recipe.update(
            epochs=1,
            minimum_epochs=1,
            warmup_epochs=0,
            batch_size=2,
            accumulation_steps=1,
            eval_batch_size=2,
            audit_cvs=2,
        )
    device = device_for(recipe["device"])
    torch.set_num_threads(recipe["cpu_threads"])
    random.seed(recipe["seed"])
    torch.manual_seed(recipe["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if resume:
        directory = Path(resume).resolve()
        if not directory.is_relative_to(ROOT / "models/langvae_ft"):
            raise ValueError("resume must refer to an isolated langvae_ft run")
        prior = json.loads((directory / "run.json").read_text())
        state = json.loads((directory / "status.json").read_text())["state"]
        if state.startswith("completed"):
            raise ValueError("completed runs are immutable; start a new run")
        if (
            prior["recipe"] != recipe
            or prior["encoder"] != encoder
            or prior["stage"] != stage
            or prior["source_sha256"] != source_hashes()
            or prior["packages"] != packages()
        ):
            raise ValueError("resume configuration/source/environment mismatch")
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        directory = (
            ROOT / "models/langvae_ft" / f"{stamp}-{stage}-{uuid.uuid4().hex[:8]}"
        )
        directory.mkdir(parents=True, exist_ok=False)
        prior = None
    print(f"RUN_DIR={directory}", flush=True)
    started = time.monotonic()
    try:
        atomic_json(
            directory / "status.json",
            {"state": "preparing", "pid": os.getpid(), "stage": stage},
        )
        paths = {key: str(ROOT / raw["paths"][key]) for key in ("texts", "pair_index")}
        examples, manifest = split_cvs(
            paths, recipe["seed"], recipe["validation_fraction"]
        )
        if stage == "smoke":
            examples = {key: rows[:2] for key, rows in examples.items()}
        manifest["used_cv_ids"] = {
            key: [row["id"] for row in rows] for key, rows in examples.items()
        }
        base, adapter = load_native(encoder, device)
        initial = {
            "bert": parameter_hash(base.encoder.encoder),
            "gpt2": parameter_hash(adapter.language_model),
            "projection": parameter_hash(adapter.projection),
            "adapters": parameter_hash(adapter.adapters),
        }
        schema = trainable_schema(adapter)
        if prior:
            if manifest != prior["data_manifest"] or initial != prior["initial_hashes"]:
                raise ValueError("resume input/base weights changed")
            if file_hash(directory / "dataset.pt") != prior["dataset_sha256"]:
                raise ValueError("resume prepared-data checksum mismatch")
            dataset = torch.load(
                directory / "dataset.pt", map_location="cpu", weights_only=True
            )
            run = prior
        else:
            dataset = _dataset(base, examples, recipe, device)
            atomic_torch(directory / "dataset.pt", dataset)
            run = {
                "schema": 1,
                "tag": TAG,
                "stage": stage,
                "recipe": recipe,
                "encoder": encoder,
                "pipeline_template": raw,
                "source_sha256": source_hashes(),
                "packages": packages(),
                "initial_hashes": initial,
                "trainable_schema": schema,
                "data_manifest": manifest,
                "dataset_sha256": file_hash(directory / "dataset.pt"),
                "passages": {
                    key: len(rows) for key, rows in dataset["records"].items()
                },
                "weighting": "uniform passages (not uniform CVs)",
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None,
            }
            atomic_json(directory / "run.json", run)
            atomic_json(directory / "data_manifest.json", manifest)
            atomic_json(
                directory / "baseline_validation.json",
                _evaluate(adapter, dataset, recipe, device),
            )
            metrics, rows = _audit(
                adapter, dataset, base.decoder.tokenizer, recipe, device
            )
            atomic_json(directory / "baseline_audit.json", metrics)
            atomic_json(directory / "baseline_generations.json", rows)
        optimizer = torch.optim.AdamW(
            [
                {"params": adapter.projection.parameters(), "lr": recipe["encoder_lr"]},
                {"params": adapter.adapters.parameters(), "lr": recipe["decoder_lr"]},
            ],
            weight_decay=0.0,
        )
        progress = {
            "epoch": 0,
            "step": 0,
            "best_loss": None,
            "selected": None,
            "stale": 0,
            "history": [],
            "elapsed": 0.0,
        }
        if prior:
            progress = restore_resume(directory / "resume.pt", adapter, optimizer)
        records, features = dataset["records"]["train"], dataset["features"]["train"]
        batch_size = recipe["batch_size"]
        effective = batch_size * recipe["accumulation_steps"]
        updates_per_epoch = math.ceil(len(records) / effective)
        warmup_steps = recipe["warmup_epochs"] * updates_per_epoch
        timing, stop_reason = [], "epochs"
        batches = []
        already_stopped = (
            progress["epoch"] >= recipe["epochs"]
            or (
                progress["epoch"] >= recipe["minimum_epochs"]
                and progress["stale"] >= recipe["patience"]
            )
            or progress["elapsed"] >= recipe["max_training_hours"] * 3600
        )
        first_epoch = recipe["epochs"] if already_stopped else progress["epoch"]
        if already_stopped:
            stop_reason = "resumed_final_verification"
        for epoch in range(first_epoch, recipe["epochs"]):
            order = torch.randperm(len(records)).tolist()
            losses = []
            for offset in range(0, len(order), effective):
                group = order[offset : offset + effective]
                batches = [
                    batch_tensors(features, records, group[i : i + batch_size], device)
                    for i in range(0, len(group), batch_size)
                ]
                beta = (
                    min(1.0, progress["step"] / warmup_steps) if warmup_steps else 1.0
                )
                update_started = time.monotonic()
                metrics = train_update(
                    adapter, optimizer, batches, beta, recipe["clip_grad_norm"]
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                timing.append(time.monotonic() - update_started)
                losses.append((metrics["loss"], len(group)))
                progress["step"] += 1
                if progress["step"] % recipe["log_every"] == 0 or progress["step"] == 1:
                    status = {
                        "state": "training",
                        "pid": os.getpid(),
                        "epoch": epoch + 1,
                        "step": progress["step"],
                        "passages_seen_this_epoch": offset + len(group),
                        "total_passages_per_epoch": len(records),
                        "mean_update_seconds": sum(timing[-50:]) / len(timing[-50:]),
                        **metrics,
                    }
                    atomic_json(directory / "status.json", status)
                    print(json.dumps(status), flush=True)
                if stage == "smoke" and progress["step"] >= 2:
                    break
            validation = _evaluate(adapter, dataset, recipe, device)
            audit_metrics, generations = _audit(
                adapter, dataset, base.decoder.tokenizer, recipe, device
            )
            improved = (
                progress["best_loss"] is None
                or validation["negative_elbo"] < progress["best_loss"]
            )
            if improved:
                progress["selected"] = export_checkpoint(
                    base, adapter, directory, dataset, run, epoch + 1, validation
                )
                progress["best_loss"] = validation["negative_elbo"]
                progress["stale"] = 0
                atomic_json(directory / "selected_audit.json", audit_metrics)
                atomic_json(directory / "selected_generations.json", generations)
            else:
                progress["stale"] += 1
            progress["history"].append(
                {
                    "epoch": epoch + 1,
                    "updates": progress["step"],
                    "train_loss": sum(x * n for x, n in losses)
                    / sum(n for _, n in losses),
                    "validation": validation,
                    "audit": audit_metrics,
                    "selected": improved,
                }
            )
            atomic_json(directory / "history.json", progress["history"])
            progress["epoch"] = epoch + 1
            elapsed = progress["elapsed"] + time.monotonic() - started
            save_resume(
                directory / "resume.pt",
                adapter,
                optimizer,
                **{**progress, "elapsed": elapsed},
            )
            print(
                json.dumps(
                    {"epoch": epoch + 1, "validation": validation, "selected": improved}
                ),
                flush=True,
            )
            if (
                epoch + 1 >= recipe["minimum_epochs"]
                and progress["stale"] >= recipe["patience"]
            ):
                stop_reason = "validation_early_stopping"
                break
            if elapsed >= recipe["max_training_hours"] * 3600:
                stop_reason = "epoch_boundary_time_budget"
                break
        integrity = {
            "bert_unchanged": parameter_hash(base.encoder.encoder) == initial["bert"],
            "gpt2_unchanged": parameter_hash(adapter.language_model) == initial["gpt2"],
            "parameter_shapes_unchanged": schema == trainable_schema(adapter),
            "projection_updated": parameter_hash(adapter.projection)
            != initial["projection"],
            "decoder_adapters_updated": parameter_hash(adapter.adapters)
            != initial["adapters"],
        }
        if not all(integrity.values()):
            raise RuntimeError(f"native fine-tuning integrity failed: {integrity}")
        selected = directory / progress["selected"]["path"]
        read_checkpoint(selected)
        atomic_json(
            directory / "status.json",
            {"state": "verifying_offline_reload", "selected": str(selected)},
        )
        del base, adapter, optimizer, batches
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        command = [
            sys.executable,
            "-m",
            "src.langvae_ft",
            "verify",
            "--checkpoint",
            str(selected),
            "--device",
            str(device),
        ]
        with (directory / "reload.log").open("w") as stream:
            subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        if stage == "train":
            output = directory / "encoding.yaml"
            if not output.exists():
                write_inference_config(run["pipeline_template"], selected, output)
        result = {
            "state": "completed" if stage == "train" else "completed_diagnostic",
            "stage": stage,
            "selected_checkpoint": progress["selected"],
            "integrity": integrity,
            "offline_reload_passed": True,
            "stop_reason": stop_reason,
            "completed_epochs": progress["epoch"],
            "updates": progress["step"],
            "seconds": progress["elapsed"] + time.monotonic() - started,
            "mean_update_seconds": sum(timing) / len(timing) if timing else None,
            "production_encodings_written": False,
            "downstream_models_changed": False,
            "accepted_for_downstream": False,
            "note": "Review validation evidence; final encoding is an explicit destination-cluster step.",
        }
        atomic_json(directory / "status.json", result)
        return directory, result
    except BaseException as exc:
        atomic_json(
            directory / "status.json",
            {
                "state": "interrupted"
                if isinstance(exc, KeyboardInterrupt)
                else "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "stage": stage,
            },
        )
        raise
