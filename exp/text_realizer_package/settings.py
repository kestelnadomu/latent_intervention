"""Explicit package experiment identity; historical decoder settings stay unchanged."""

from __future__ import annotations

from pathlib import Path

import yaml

from exp.text_realizer.protocol import immutable_json, read_json, snapshot
from exp.text_realizer.settings import (
    ROOT,
    digest as digest,
    packages,
    resolve,
    run_paths as run_paths,
    source_hashes as shared_source_hashes,
)
from src.artifact_io import sha256_file
from src.encoder.protocols import EMBEDDINGGEMMA_REVISION, embeddinggemma_protocol

DEFAULT_CONFIG = ROOT / "configs/text_realizer_talent_motivation_256.yaml"


def read_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    config = yaml.safe_load(Path(path).read_text())
    if (
        config["format_version"] != 1
        or config["experiment"] != "talent_motivation_256"
        or config["seed"] != 42
        or config["model"]["name"] != "google/gemma-3-270m"
        or config["model"]["revision"] != "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
        or config["model"]["latent_dim"] != 256
        or config["model"]["prefix_tokens"] != 16
        or config["model"]["input_scaling"] != "sqrt_latent_dim"
        or config["encoder"]["embeddinggemma_latent_dim"] != 256
        or config["encoder"]["embeddinggemma_model_revision"] != EMBEDDINGGEMMA_REVISION
    ):
        raise ValueError("requires the pinned factual-only 256-D / 270M configuration")
    if config["model"].get("attention_implementation") != "eager":
        raise ValueError(
            "package decoder requires the verified eager attention backend"
        )
    embeddinggemma_protocol(config["encoder"])
    training = config["training"]
    if training["precision"] != "float32" or not training["gradient_checkpointing"]:
        raise ValueError("this experiment requires FP32 and gradient checkpointing")
    for key in ("epochs", "effective_batch", "patience", "checkpoint_every"):
        if type(training[key]) is not int or training[key] < 1:
            raise ValueError(f"invalid training.{key}")
    if not 0 < training["warmup_fraction"] < 1:
        raise ValueError("invalid warmup fraction")
    for key in ("projector_lr", "lora_lr", "clip"):
        if training[key] <= 0:
            raise ValueError(f"invalid training.{key}")
    for key in ("smoke_updates", "benchmark_steps", "overfit_updates"):
        if type(config["preflight"][key]) is not int or config["preflight"][key] < 1:
            raise ValueError(f"invalid preflight.{key}")
    if not 0.15 <= config["preflight"]["memory_headroom"] < 1:
        raise ValueError("at least 15% GPU headroom is required")
    # No output directory may reuse the historical 768-D or package cache trees.
    for key in ("prepared", "models", "reports"):
        destination = resolve(config["paths"][key])
        expected = {
            "prepared": ROOT / "data/latents/talent_motivation/embeddinggemma_256",
            "models": ROOT / "models/talent_motivation/embeddinggemma_256",
            "reports": ROOT / "reports/talent_motivation/embeddinggemma_256",
        }[key]
        if (
            destination.parent.resolve() != expected.resolve()
            or not destination.name.startswith("text-realizer")
        ):
            raise ValueError(f"isolated output directory required: {key}")
    return config


def source_hashes() -> dict:
    result = shared_source_hashes()
    paths = [
        ROOT / "latent_lab/encoder.py",
        ROOT / "scripts/run_text_realizer_package.sh",
    ]
    for folder in ("exp/text_realizer_package", "tests/text_realizer_package"):
        paths.extend((ROOT / folder).glob("*.py"))
    result.update({str(path.relative_to(ROOT)): sha256_file(path) for path in paths})
    return result


def identity(config: dict, prepared: Path) -> dict:
    return {
        "config": config,
        "prepared_manifest_sha256": sha256_file(prepared / "manifest.json"),
        "source_sha256": source_hashes(),
        "packages": packages(),
    }


def require_current(reports: Path, config: dict, prepared: Path) -> dict:
    plan = read_json(reports / "plan.json")
    if plan["identity"] != identity(config, prepared):
        raise ValueError(
            "source/config/environment/prepared data changed; use a new run ID"
        )
    return plan


def create_plan(config: dict, prepared: Path, reports: Path) -> dict:
    plan = {
        "identity": identity(config, prepared),
        "initialization": "fresh projector and LoRA from pinned pretrained Gemma",
        "variant": "lora-compact",
        "seed": config["seed"],
        "selection": "minimum development token NLL",
        "evaluation_scope": "development only; package evaluation remains closed",
    }
    snapshot(reports, plan)
    immutable_json(reports / "config.json", config)
    return plan
