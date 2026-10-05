from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import src.encoder.embeddinggemma as module
from src.config import (
    EMBEDDINGGEMMA_DIMENSIONS,
    EMBEDDINGGEMMA_PROMPT,
    embeddinggemma_protocol,
    load_config,
)


@pytest.mark.parametrize("dim", EMBEDDINGGEMMA_DIMENSIONS)
def test_gemma_paths_and_protocol(dim):
    config = load_config(encoder_variant="embeddinggemma", embeddinggemma_dim=dim)
    for key in (
        "latents",
        "decoder_model",
        "decoder_report",
        "manipulator_model",
        "eval_report",
    ):
        assert f"/embeddinggemma_{dim}/" in config["paths"][key]
    assert config["encoder"]["max_len"] == 2048
    protocol = embeddinggemma_protocol(config["encoder"])
    assert protocol["prompt"] == EMBEDDINGGEMMA_PROMPT
    assert protocol["bidirectional_attention"] is True
    assert protocol["normalization"] == f"truncate_{dim}+l2"


@pytest.mark.parametrize("dim", [True, 64, 1024, 128.0, "128", 0])
def test_gemma_invalid_dimensions(dim):
    with pytest.raises(ValueError, match="dimension"):
        load_config(encoder_variant="embeddinggemma", embeddinggemma_dim=dim)


@pytest.mark.parametrize(
    "change",
    [
        {"embeddinggemma_model_name": "google/gemma-3-1b"},
        {"embeddinggemma_model_revision": "main"},
        {"embeddinggemma_prompt": "a different prompt"},
    ],
)
def test_gemma_protocol_cannot_silently_change(change):
    with pytest.raises(ValueError):
        embeddinggemma_protocol(change)


def test_gemma_dimension_requires_correct_variant():
    with pytest.raises(ValueError, match="requires the embeddinggemma"):
        load_config(encoder_variant="langvae", embeddinggemma_dim=128)


def test_gemma_loader_requires_isolated_environment(monkeypatch):
    monkeypatch.setattr(module, "version", lambda _: "4.48.0")
    with pytest.raises(RuntimeError, match="venv-embeddinggemma"):
        module.load_sentence_model("unused", torch.device("cpu"))


@pytest.mark.parametrize("dim", EMBEDDINGGEMMA_DIMENSIONS)
def test_gemma_wrapper_preserves_official_output_and_guards_inputs(monkeypatch, dim):
    calls = []

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tokenizer = lambda texts, **kwargs: {
                "input_ids": [[1, 2, 3] for _ in texts]
            }

        def encode(self, texts, **kwargs):
            assert not self.training
            calls.append((texts, kwargs))
            return F.normalize(
                torch.arange(1, kwargs["truncate_dim"] + 1, dtype=torch.float32).expand(
                    len(texts), -1
                ),
                dim=1,
            )

    monkeypatch.setattr(module, "cached_snapshot", lambda config: "/pinned/gemma")
    monkeypatch.setattr(module, "load_sentence_model", lambda *args: Model())
    monkeypatch.setattr(module, "validate_model", lambda model: None)
    encoder = module.make_embeddinggemma_encoder({"embeddinggemma_latent_dim": dim})
    result = encoder.encode(["CV one", "CV two"], batch_size=2)
    assert result.shape == (2, dim)
    assert torch.allclose(result.norm(dim=1), torch.ones(2))
    assert calls[0][0] == [
        EMBEDDINGGEMMA_PROMPT + "CV one",
        EMBEDDINGGEMMA_PROMPT + "CV two",
    ]
    assert calls[0][1]["prompt"] == ""
    assert calls[0][1]["normalize_embeddings"] is True
    assert encoder.encode([]).shape == (0, dim)
    with pytest.raises(ValueError, match="deterministic"):
        encoder.encode(["CV"], deterministic=False)
    with pytest.raises(ValueError, match="positive"):
        encoder.encode(["CV"], batch_size=0)
    with pytest.raises(ValueError, match="non-empty"):
        encoder.encode([""])
    encoder.max_len = 2
    with pytest.raises(ValueError, match="refusing silent truncation"):
        encoder.encode(["CV"])
    assert len(calls) == 1


def fake_stack():
    class Transformer:
        auto_model = SimpleNamespace(
            config=SimpleNamespace(
                model_type="gemma3_text",
                hidden_size=768,
                use_bidirectional_attention=True,
                use_cache=False,
            )
        )

    class Pooling:
        include_prompt = True

        def get_pooling_mode_str(self):
            return "mean"

    class Dense:
        def __init__(self, start, end):
            self.in_features, self.out_features = start, end
            self.linear = SimpleNamespace(bias=None)
            self.activation_function = torch.nn.Identity()

    class Normalize:
        pass

    class Stack(list):
        def get_sentence_embedding_dimension(self):
            return 768

    return Stack(
        [Transformer(), Pooling(), Dense(768, 3072), Dense(3072, 768), Normalize()]
    )


def test_gemma_rejects_causal_backbone_or_missing_projection():
    model = fake_stack()
    module.validate_model(model)
    model[0].auto_model.config.use_bidirectional_attention = False
    with pytest.raises(ValueError, match="bidirectional"):
        module.validate_model(model)
    with pytest.raises(ValueError, match="complete released"):
        module.validate_model(fake_stack()[:2])
    model = fake_stack()
    model[2].activation_function = torch.nn.ReLU()
    with pytest.raises(ValueError, match="projection heads"):
        module.validate_model(model)
