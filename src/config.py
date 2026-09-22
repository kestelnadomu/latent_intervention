"""Loader for the module hyperparameter config (src/config.yaml)."""

from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"
ENCODER_PLACEHOLDER = "{encoder}"
DECODER_PLACEHOLDER = "{decoder}"
MANIPULATOR_PLACEHOLDER = "{manipulator}"


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


def component_tag(component: dict[str, Any], default: str) -> str:
    """Return an explicit artifact tag, or the selected component variant."""
    return str(component.get("tag") or component.get("variant", default))


def resolve_paths(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve component tags in artifact paths in place and return ``config``."""
    paths = config.get("paths") or {}
    replacements = {
        ENCODER_PLACEHOLDER: encoder_tag(config.get("encoder") or {}),
        DECODER_PLACEHOLDER: component_tag(
            config.get("semantic_decoder") or {}, "independent"
        ),
        MANIPULATOR_PLACEHOLDER: component_tag(
            config.get("latent_intervention") or {}, "baseline"
        ),
    }
    for key, value in paths.items():
        if not isinstance(value, str):
            continue
        for placeholder, tag in replacements.items():
            value = value.replace(placeholder, tag)
        paths[key] = value
    return config


def load_config(
    section: str | None = None,
    path: str | Path = CONFIG_PATH,
    *,
    encoder_variant: str | None = None,
    decoder_variant: str | None = None,
    manipulator_variant: str | None = None,
) -> dict[str, Any]:
    """
    Load src/config.yaml (or `path`); return one section if requested.

    Sections mirror the module names: encoder, semantic_decoder,
    latent_intervention, plus paths and the global seed. ``{encoder}`` in
    ``paths`` is resolved via `encoder_tag`, so encoder-dependent artifacts
    (latents, models, reports) never mix latent spaces.
    """
    with open(Path(path), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    overrides = {
        "encoder": encoder_variant,
        "semantic_decoder": decoder_variant,
        "latent_intervention": manipulator_variant,
    }
    for component, variant in overrides.items():
        if variant is not None:
            config[component]["variant"] = variant
    resolve_paths(config)
    if section is None:
        return config
    if section not in config:
        raise KeyError(f"No section '{section}' in {path}. Available: {list(config)}")
    return config[section]
