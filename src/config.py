"""Loader for the module hyperparameter config (src/config.yaml)."""

from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"
ENCODER_PLACEHOLDER = "{encoder}"


def encoder_tag(encoder: dict[str, Any]) -> str:
    """
    Name of the latent space an encoder config produces.

    ``encoder.tag`` if set; otherwise the variant, with ``_ft`` appended for a
    LangVAE loaded from ``local_checkpoint`` (a fine-tuned checkpoint).
    """
    if encoder.get("tag"):
        return str(encoder["tag"])
    variant = encoder.get("variant", "langvae")
    if variant == "langvae" and encoder.get("local_checkpoint"):
        return "langvae_ft"
    return variant


def resolve_paths(config: dict[str, Any]) -> dict[str, Any]:
    """Replace ``{encoder}`` in ``paths`` with the encoder tag (in place; returns config)."""
    paths = config.get("paths") or {}
    if any(ENCODER_PLACEHOLDER in str(v) for v in paths.values()):
        tag = encoder_tag(config.get("encoder") or {})
        for key, value in paths.items():
            if isinstance(value, str):
                paths[key] = value.replace(ENCODER_PLACEHOLDER, tag)
    return config


def load_config(section: str | None = None, path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """
    Load src/config.yaml (or `path`); return one section if requested.

    Sections mirror the module names: encoder, semantic_decoder,
    latent_intervention, plus paths and the global seed. ``{encoder}`` in
    ``paths`` is resolved via `encoder_tag`, so encoder-dependent artifacts
    (latents, models, reports) never mix latent spaces.
    """
    with open(Path(path), "r", encoding="utf-8") as f:
        config = resolve_paths(yaml.safe_load(f))
    if section is None:
        return config
    if section not in config:
        raise KeyError(f"No section '{section}' in {path}. Available: {list(config)}")
    return config[section]
