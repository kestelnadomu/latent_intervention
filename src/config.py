"""Loader for the module hyperparameter config (src/config.yaml)."""

from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config(section: str | None = None, path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """
    Load src/config.yaml (or `path`); return one section if requested.

    Sections mirror the module names: encoder, semantic_decoder,
    latent_intervention, plus paths and the global seed.
    """
    with open(Path(path), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if section is None:
        return config
    if section not in config:
        raise KeyError(f"No section '{section}' in {path}. Available: {list(config)}")
    return config[section]
