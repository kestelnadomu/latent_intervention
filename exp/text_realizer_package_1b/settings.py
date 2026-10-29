"""One explicitly pinned size comparison, without editing a running 270M plan."""

from __future__ import annotations

from pathlib import Path

import yaml

from exp.text_realizer_package.settings import (
    DEFAULT_CONFIG as REFERENCE_CONFIG,
    ROOT,
    create_plan as create_shared_plan,
    read_config as read_reference_config,
    require_current as require_shared_current,
    run_paths as run_paths,
)
from src.artifact_io import copy_atomic, sha256_file

DEFAULT_CONFIG = ROOT / "configs/text_realizer_talent_motivation_256_gemma1b.yaml"
MODEL = "google/gemma-3-1b-pt"
REVISION = "fcf18a2a879aab110ca39f8bffbccd5d49d8eb29"


def extension_hashes() -> dict:
    paths = [
        DEFAULT_CONFIG,
        REFERENCE_CONFIG,
        ROOT / "scripts/run_text_realizer_package_1b.sh",
    ]
    for folder in ("exp/text_realizer_package_1b", "tests/text_realizer_package_1b"):
        paths.extend((ROOT / folder).glob("*.py"))
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(paths)}


def require_extension(config: dict) -> None:
    if config.get("package_1b_source_sha256") != extension_hashes():
        raise ValueError("1B source/configuration changed; use a new run ID")


def read_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    config = yaml.safe_load(Path(path).read_text())
    saved_sources = config.pop("package_1b_source_sha256", None)
    # These are the only intended recipe differences from the running 270M job.
    expected = read_reference_config()
    expected["model"].update(name=MODEL, revision=REVISION)
    expected["paths"].update(
        models="models/talent_motivation/embeddinggemma_256/text-realizer-1b",
        reports="reports/talent_motivation/embeddinggemma_256/text-realizer-1b",
    )
    if config != expected:
        raise ValueError(
            "1B comparison may change only the pinned backbone and output paths"
        )
    current = extension_hashes()
    if saved_sources is not None and saved_sources != current:
        raise ValueError("saved 1B source/configuration changed; use a new run ID")
    # The shared immutable identity includes the complete config, so these
    # extra source hashes also bind checkpoints/resume without modifying its code.
    config["package_1b_source_sha256"] = current
    return config


def create_plan(config: dict, prepared: Path, reports: Path) -> dict:
    require_extension(config)
    plan = create_shared_plan(config, prepared, reports)
    for relative, expected in config["package_1b_source_sha256"].items():
        source, destination = ROOT / relative, reports / "source" / relative
        if sha256_file(source) != expected:
            raise ValueError("1B source changed during snapshot")
        if destination.exists() and sha256_file(destination) != expected:
            raise ValueError("1B source snapshot changed")
        if not destination.exists():
            copy_atomic(source, destination)
    return plan


def require_current(reports: Path, config: dict, prepared: Path) -> dict:
    require_extension(config)
    return require_shared_current(reports, config, prepared)
