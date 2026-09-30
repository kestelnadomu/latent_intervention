from __future__ import annotations

from pathlib import Path

import yaml

import pytest

from src.config import (
    NOMIC_DIMENSIONS,
    QWEN3_DIMENSIONS,
    component_tag,
    encoder_tag,
    load_config,
    resolve_paths,
)


def test_encoder_tag_distinguishes_latent_spaces() -> None:
    assert encoder_tag({"variant": "langvae"}) == "langvae"
    assert encoder_tag({}) == "langvae"
    assert (
        encoder_tag({"variant": "langvae", "local_checkpoint": "models/x"})
        == "langvae_ft"
    )
    assert (
        encoder_tag({"variant": "nomic", "local_checkpoint": "models/x"}) == "nomic_128"
    )
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
        "latents": "data/latents/nomic_128/z.pt",
        "decoder": "models/nomic_128/autoregressive/g.pt",
        "manipulator": "models/nomic_128/autoregressive/dist/h.pt",
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
        assert (
            f"/{encoder_tag(config['encoder'])}/g-{decoder}/"
            in config["paths"]["decoder_model"]
        )


@pytest.mark.parametrize("dim", NOMIC_DIMENSIONS)
def test_nomic_dimension_scopes_all_downstream_paths(dim: int) -> None:
    config = load_config(encoder_variant="nomic", nomic_dim=dim)
    for key in (
        "latents",
        "decoder_model",
        "decoder_report",
        "manipulator_model",
        "eval_report",
    ):
        assert f"/nomic_{dim}/" in config["paths"][key]
    assert config["encoder"]["nomic_latent_dim"] == dim


@pytest.mark.parametrize("dim", [0, 127, 1024, True, 128.0, "128"])
def test_invalid_nomic_dimensions_rejected(dim) -> None:
    with pytest.raises(ValueError, match="Nomic dimension"):
        load_config(encoder_variant="nomic", nomic_dim=dim)


def test_nomic_dimension_cannot_silently_change_langvae() -> None:
    with pytest.raises(ValueError, match="requires the nomic"):
        load_config(encoder_variant="langvae", nomic_dim=64)


@pytest.mark.parametrize("dim", QWEN3_DIMENSIONS)
def test_qwen3_paths_only_include_dimension(dim) -> None:
    config = load_config(encoder_variant="qwen3", qwen3_dim=dim)
    for key in (
        "latents",
        "decoder_model",
        "decoder_report",
        "manipulator_model",
        "eval_report",
    ):
        assert f"/qwen3_{dim}/" in config["paths"][key]
        assert "0p6b" not in config["paths"][key]
    assert config["encoder"]["max_len"] == 1024
    assert config["encoder"]["batch_size"] == 8


@pytest.mark.parametrize("dim", [0, 31, 127, 2048, True, 128.0, "128"])
def test_qwen3_invalid_dimensions(dim) -> None:
    with pytest.raises(ValueError, match="Qwen3 dimension"):
        load_config(encoder_variant="qwen3", qwen3_dim=dim)


def test_qwen3_size_and_variant_are_guarded() -> None:
    with pytest.raises(ValueError, match="requires the qwen3"):
        load_config(encoder_variant="nomic", qwen3_dim=64)
    with pytest.raises(ValueError, match="reserved"):
        encoder_tag({"variant": "qwen3", "qwen3_model_name": "Qwen/Qwen3-Embedding-4B"})


def test_manipulator_override_has_separate_downstream_paths() -> None:
    paths = {
        variant: load_config(manipulator_variant=variant)["paths"]
        for variant in ("baseline", "pre_additive", "dist")
    }
    assert len({value["manipulator_model"] for value in paths.values()}) == 3
    assert len({value["eval_report"] for value in paths.values()}) == 3
