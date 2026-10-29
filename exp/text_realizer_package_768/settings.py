"""A matched 768-D comparison without changing either running 256-D experiment."""

from __future__ import annotations

from pathlib import Path

import yaml

from exp.text_realizer.protocol import immutable_json, read_json, snapshot
from exp.text_realizer_package.settings import (
    ROOT,
    DEFAULT_CONFIG as REFERENCE_CONFIG,
    digest as digest,
    packages,
    read_config as read_reference,
    resolve as resolve,
    run_paths as run_paths,
    source_hashes as reference_hashes,
)
from src.artifact_io import sha256_file

CONFIGS = {
    "270m": ROOT / "configs/text_realizer_talent_motivation_768.yaml",
    "1b": ROOT / "configs/text_realizer_talent_motivation_768_gemma1b.yaml",
}
DEFAULT_CONFIG = CONFIGS["270m"]
BACKBONES = {
    "google/gemma-3-270m": ("9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1", ""),
    "google/gemma-3-1b-pt": ("fcf18a2a879aab110ca39f8bffbccd5d49d8eb29", "-1b"),
}


def read_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    config = yaml.safe_load(Path(path).read_text())
    name = config["model"]["name"]
    if name not in BACKBONES:
        raise ValueError("requires a pinned 270M or 1B backbone")
    revision, suffix = BACKBONES[name]
    expected = read_reference()
    expected["experiment"] = "talent_motivation_768"
    expected["encoder"]["embeddinggemma_latent_dim"] = 768
    expected["model"].update(name=name, revision=revision, latent_dim=768)
    for key, path in expected["paths"].items():
        expected["paths"][key] = path.replace(
            "embeddinggemma_256/", "embeddinggemma_768/"
        )
    for key in ("models", "reports"):
        expected["paths"][key] += suffix
    if config != expected:
        raise ValueError(
            "768-D comparison may change only latent width, pinned backbone and output paths; "
            "the verified eager backend and factual-only recipe must match"
        )
    return config


def source_hashes() -> dict:
    result = reference_hashes()
    paths = [
        REFERENCE_CONFIG,
        *CONFIGS.values(),
        ROOT / "scripts/run_text_realizer_package_768.sh",
    ]
    for folder in ("exp/text_realizer_package_768", "tests/text_realizer_package_768"):
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
