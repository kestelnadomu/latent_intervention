"""Pinned configuration, paths, and immutable identity for the separate pilot."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import yaml

from src.artifact_io import sha256_file

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/text_realizer_counterfactual_exposure.yaml"
ARMS = ("factual", "oracle")


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    config = yaml.safe_load(Path(path).read_text())
    model, training, selection = (
        config["model"],
        config["training"],
        config["selection"],
    )
    if config["format_version"] != 1 or model != {
        "name": "google/gemma-3-270m",
        "revision": "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1",
        "variant": "lora-compact",
        "attention_implementation": "eager",
        "latent_dim": 768,
        "prefix_tokens": 16,
        "projector_hidden": 768,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_targets": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    }:
        raise ValueError(
            "the oracle pilot requires the predeclared 270M/eager architecture"
        )
    if (
        training
        != {
            "seed": 42,
            "updates": 600,
            "effective_batch": 32,
            "oracle_per_batch": 16,
            "projector_lr": 0.0003,
            "lora_lr": 0.00005,
            "weight_decay": 0.01,
            "warmup_fraction": 0.05,
            "clip": 1.0,
            "checkpoint_every": 25,
            "evaluate_every": 100,
            "precision": "float32",
            "gradient_checkpointing": True,
        }
        or selection
        != {
            "factual_tune_ids": 80,
            "bootstrap_samples": 2000,
            "bootstrap_seed": 20261017,
            "minimum_character_gain": 0.02,
            "minimum_macro_factual_gain": 0.02,
            "maximum_factual_regression": 0.02,
            "maximum_field_regression": 0.02,
            "factual_fields": ["country", "degree", "university_tier", "project_count"],
        }
        or config["generation"] != {"max_new_tokens": 1024, "batch_size": 1}
    ):
        raise ValueError("the oracle pilot protocol differs from the predeclared plan")
    if config["preflight"] != {
        "smoke_updates": 20,
        "benchmark_updates": 10,
        "memory_headroom": 0.15,
        "microbatches": [1, 2],
    }:
        raise ValueError("preflight differs from the predeclared pilot checks")
    return config


def prepared_path(config: dict) -> Path:
    return resolve(config["paths"]["prepared"]) / digest(config["inputs"])


def run_paths(config: dict, run_id: str, arm: str | None = None) -> tuple[Path, Path]:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", run_id):
        raise ValueError("unsafe run ID")
    if arm is not None and arm not in ARMS:
        raise ValueError("unknown comparison arm")
    suffix = Path(run_id) if arm is None else Path(run_id) / arm
    return (
        resolve(config["paths"]["models"]) / suffix,
        resolve(config["paths"]["reports"]) / suffix,
    )


def source_hashes() -> dict[str, str]:
    files = []
    for folder in ("exp/text_realizer_cf", "src/text_realizer"):
        files.extend((ROOT / folder).glob("*.py"))
    for name in (
        "exp/text_realizer/data.py",
        "exp/text_realizer/experiment.py",
        "exp/text_realizer/evaluation.py",
        "exp/text_realizer/facts.py",
        "exp/text_realizer/campaign.py",
        "exp/text_realizer/sanity.py",
        "exp/text_realizer/runtime.py",
        "src/artifact_io.py",
        "src/encoder/embeddinggemma.py",
        "src/encoder/protocols.py",
        "exp/sim/codebook.yaml",
        "exp/text_realizer/environment/pyproject.toml",
        "exp/text_realizer/environment/uv.lock",
        "configs/text_realizer_counterfactual_exposure.yaml",
        "scripts/run_text_realizer_cf.sh",
    ):
        files.append(ROOT / name)
    return {
        str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(set(files))
    }
