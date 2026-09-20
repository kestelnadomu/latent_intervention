from __future__ import annotations

from pathlib import Path

import yaml

from src.config import encoder_tag, load_config, resolve_paths


def test_encoder_tag_distinguishes_latent_spaces() -> None:
    assert encoder_tag({"variant": "langvae"}) == "langvae"
    assert encoder_tag({}) == "langvae"
    assert encoder_tag({"variant": "langvae", "local_checkpoint": "models/x"}) == "langvae_ft"
    assert encoder_tag({"variant": "nomic", "local_checkpoint": "models/x"}) == "nomic"
    assert encoder_tag({"variant": "langvae", "tag": "langvae_cv5"}) == "langvae_cv5"


def test_resolve_paths_substitutes_encoder_placeholder() -> None:
    config = {
        "encoder": {"variant": "nomic"},
        "paths": {"latents": "data/latents/{encoder}/z.pt", "texts": "data/t.csv"},
    }
    resolve_paths(config)
    assert config["paths"] == {"latents": "data/latents/nomic/z.pt", "texts": "data/t.csv"}


def test_default_config_scopes_encoder_dependent_paths() -> None:
    raw = yaml.safe_load(Path("src/config.yaml").read_text(encoding="utf-8"))
    paths = load_config("paths")
    tag = encoder_tag(raw["encoder"])
    for key in ("latents", "decoder_model", "manipulator_model", "eval_report"):
        assert "{encoder}" in raw["paths"][key]
        assert f"/{tag}/" in paths[key]
    assert not any("{encoder}" in str(value) for value in paths.values())
