"""Local configuration and immutable experiment identity; no pipeline imports."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/text_realizer_embeddinggemma_768.yaml"
VARIANTS = ("prefix-compact", "prefix-wide", "lora-compact", "lora-wide")


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def read_config(path=DEFAULT_CONFIG) -> dict:
    config = yaml.safe_load(Path(path).read_text())
    if config["format_version"] != 1:
        raise ValueError("unsupported text-realizer configuration")
    model = config["model"]
    if (
        model["name"] != "google/gemma-3-270m"
        or not re.fullmatch(r"[0-9a-f]{40}", model["revision"])
        or model["latent_dim"] != 768
        or model["prefix_tokens"] != 16
        or model["input_scaling"] != "sqrt_latent_dim"
    ):
        raise ValueError(
            "this campaign requires the pinned 768-D / 16-prefix Gemma model"
        )
    if config["training"]["precision"] != "float32":
        raise ValueError("this campaign is explicitly FP32")
    if config["selection"]["seeds"] != [42, 43, 44]:
        raise ValueError("the predeclared seeds differ from this experiment")
    if config["selection"]["deployment_seed"] != 42:
        raise ValueError("deployment seed must remain 42")
    for field in ("epochs", "effective_batch", "patience", "checkpoint_every"):
        value = config["training"][field]
        if type(value) is not int or value < 1:
            raise ValueError(f"training.{field} must be a positive integer")
    for field in ("projector_lr", "lora_lr", "clip"):
        if config["training"][field] <= 0:
            raise ValueError(f"training.{field} must be positive")
    if not 0 < config["training"]["warmup_fraction"] < 1:
        raise ValueError("warmup_fraction must be in (0, 1)")
    if not 0 < config["preflight"]["memory_headroom"] < 1:
        raise ValueError("memory_headroom must be in (0, 1)")
    from src.encoder.protocols import EMBEDDINGGEMMA_REVISION, embeddinggemma_protocol

    if config["encoder"]["embeddinggemma_model_revision"] != EMBEDDINGGEMMA_REVISION:
        raise ValueError("canonical EmbeddingGemma revision changed")
    embeddinggemma_protocol(config["encoder"])
    return config


def require_command(config: dict, command: str) -> None:
    """Archived 1B pilot configurations cannot be used by the active CLI."""
    if "pilot" in config:
        raise ValueError("the 1B pilot is archived under .archive/code")


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def run_paths(config: dict, run_id: str) -> tuple[Path, Path]:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", run_id):
        raise ValueError("unsafe run ID")
    return tuple(
        resolve(config["paths"][key]) / run_id for key in ("models", "reports")
    )


def source_hashes() -> dict:
    from src.artifact_io import sha256_file

    files = []
    for folder in ("src/text_realizer", "exp/text_realizer", "tests/text_realizer"):
        files.extend((ROOT / folder).glob("*.py"))
    files.extend(
        ROOT / name
        for name in (
            "src/artifact_io.py",
            "src/config.py",
            "src/encoder/__init__.py",
            "src/encoder/protocols.py",
            "src/encoder/embeddinggemma.py",
            "src/encoder/progress.py",
            "exp/text_realizer/environment/pyproject.toml",
            "exp/sim/codebook.yaml",
            "exp/text_realizer/environment/uv.lock",
            "scripts/setup_text_realizer.sh",
            "scripts/run_text_realizer.sh",
        )
    )
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(set(files))}


def packages() -> dict:
    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "transformers",
            "sentence-transformers",
            "peft",
            "accelerate",
            "huggingface-hub",
            "numpy",
            "safetensors",
            "rapidfuzz",
            "sacrebleu",
        )
    }
    versions["python"] = platform.python_version()
    return versions
