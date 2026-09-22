from __future__ import annotations

from pathlib import Path

import yaml

from src.config import component_tag, encoder_tag, load_config, resolve_paths


def test_encoder_tag_distinguishes_latent_spaces() -> None:
    assert encoder_tag({"variant": "langvae"}) == "langvae"
    assert encoder_tag({}) == "langvae"
    assert (
        encoder_tag({"variant": "langvae", "local_checkpoint": "models/x"})
        == "langvae_ft"
    )
    assert encoder_tag({"variant": "nomic", "local_checkpoint": "models/x"}) == "nomic"
    assert encoder_tag({"variant": "langvae", "tag": "langvae_cv5"}) == "langvae_cv5"


def test_component_tag_prefers_explicit_tag() -> None:
    assert component_tag({"variant": "independent"}, "fallback") == "independent"
    assert (
        component_tag({"variant": "independent", "tag": "trial-2"}, "fallback")
        == "trial-2"
    )
    assert component_tag({}, "fallback") == "fallback"


def test_resolve_paths_substitutes_encoder_placeholder() -> None:
    config = {
        "encoder": {"variant": "nomic"},
        "semantic_decoder": {"variant": "autoregressive"},
        "latent_intervention": {"variant": "dist"},
        "paths": {
            "latents": "data/latents/{encoder}/z.pt",
            "decoder": "models/{encoder}/{decoder}/g.pt",
            "manipulator": "models/{encoder}/{decoder}/{manipulator}/h.pt",
            "texts": "data/t.csv",
        },
    }
    resolve_paths(config)
    assert config["paths"] == {
        "latents": "data/latents/nomic/z.pt",
        "decoder": "models/nomic/autoregressive/g.pt",
        "manipulator": "models/nomic/autoregressive/dist/h.pt",
        "texts": "data/t.csv",
    }


def test_default_config_scopes_encoder_dependent_paths() -> None:
    raw = yaml.safe_load(Path("src/config.yaml").read_text(encoding="utf-8"))
    paths = load_config("paths")
    tag = encoder_tag(raw["encoder"])
    for key in (
        "latents",
        "decoder_model",
        "decoder_report",
        "manipulator_model",
        "eval_report",
    ):
        assert "{encoder}" in raw["paths"][key]
        assert f"/{tag}/" in paths[key]
    assert not any("{encoder}" in str(value) for value in paths.values())
    assert not any("{" in str(value) for value in paths.values())


def test_variant_overrides_create_distinct_artifact_matrix() -> None:
    configs = {
        (encoder, decoder): load_config(
            encoder_variant=encoder, decoder_variant=decoder
        )
        for encoder in ("langvae", "nomic")
        for decoder in ("independent", "autoregressive")
    }

    assert len({config["paths"]["latents"] for config in configs.values()}) == 2
    assert len({config["paths"]["decoder_model"] for config in configs.values()}) == 4
    assert len({config["paths"]["decoder_report"] for config in configs.values()}) == 4
    for (encoder, decoder), config in configs.items():
        assert config["encoder"]["variant"] == encoder
        assert config["semantic_decoder"]["variant"] == decoder
        assert f"/{encoder}/g-{decoder}/" in config["paths"]["decoder_model"]


def test_manipulator_override_has_separate_downstream_paths() -> None:
    paths = {
        variant: load_config(manipulator_variant=variant)["paths"]
        for variant in ("baseline", "pre_additive", "dist")
    }
    assert len({value["manipulator_model"] for value in paths.values()}) == 3
    assert len({value["eval_report"] for value in paths.values()}) == 3
