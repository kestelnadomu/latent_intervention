from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import src.encoder as encoder_module


def test_encode_uses_decoder_tokenizer(monkeypatch) -> None:
    encoder_tokenizer = object()
    decoder_tokenizer = object()
    observed = {}

    class FakeDataSet:
        def __init__(self, texts, tokenizer, max_len) -> None:
            observed["tokenizer"] = tokenizer
            self.data = torch.tensor([[1]])

        def __getitem__(self, index):
            return {"data": self.data[index]}

    def fake_encoder(x):
        return SimpleNamespace(embedding=torch.zeros((x.shape[0], 128)))

    fake_encoder.tokenizer = encoder_tokenizer
    text_encoder = encoder_module.TextEncoder.__new__(encoder_module.TextEncoder)
    text_encoder.device = torch.device("cpu")
    text_encoder.max_len = 512
    text_encoder.model = SimpleNamespace(
        encoder=fake_encoder,
        decoder=SimpleNamespace(tokenizer=decoder_tokenizer),
    )
    monkeypatch.setattr(encoder_module, "TokenizedDataSet", FakeDataSet)

    text_encoder.encode(["example CV"], batch_size=1)

    assert observed["tokenizer"] is decoder_tokenizer


def test_make_encoder_selects_variant_and_keeps_legacy_default(monkeypatch) -> None:
    created = []

    def fake_langvae(**kwargs):
        created.append(("langvae", kwargs))
        return "langvae"

    def fake_nomic(**kwargs):
        created.append(("nomic", kwargs))
        return "nomic"

    monkeypatch.setattr(encoder_module, "TextEncoder", fake_langvae)
    monkeypatch.setattr(encoder_module, "NomicTextEncoder", fake_nomic)
    config = {
        "model_name": "langvae-model",
        "model_revision": "langvae-revision",
        "local_checkpoint": None,
        "nomic_model_name": "nomic-model",
        "nomic_model_revision": "nomic-revision",
        "nomic_code_revision": "code-revision",
        "nomic_task": "classification",
        "device": "cpu",
        "max_len": 512,
    }

    assert encoder_module.make_encoder(config) == "langvae"
    config["variant"] = "nomic"
    assert encoder_module.make_encoder(config) == "nomic"
    assert encoder_module.make_encoder(config, variant="langvae") == "langvae"
    assert created[1][1]["code_revision"] == "code-revision"
    with pytest.raises(ValueError, match="unknown encoder variant"):
        encoder_module.make_encoder(config, variant="unknown")


def test_nomic_encoder_applies_documented_128d_processing(monkeypatch) -> None:
    tokenizer_calls = []
    load_calls = {}

    class FakeTokenizer:
        def __call__(self, texts, **kwargs):
            tokenizer_calls.append((list(texts), kwargs))
            batch = len(texts)
            return {
                "input_ids": torch.tensor([[1, 2, 0]] * batch),
                "attention_mask": torch.tensor([[1, 1, 0]] * batch),
            }

    class FakeModel:
        def eval(self):
            return self

        def to(self, device):
            return self

        def __call__(self, input_ids, attention_mask):
            coordinates = torch.arange(768, dtype=torch.float32)
            hidden = torch.stack(
                (coordinates, coordinates.square() / 768, torch.sin(coordinates))
            )
            return (hidden.unsqueeze(0).expand(input_ids.shape[0], -1, -1),)

    def load_tokenizer(model_name, **kwargs):
        load_calls["tokenizer"] = (model_name, kwargs)
        return FakeTokenizer()

    def load_model(model_name, **kwargs):
        load_calls["model"] = (model_name, kwargs)
        return FakeModel()

    monkeypatch.setattr(encoder_module.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(encoder_module.AutoModel, "from_pretrained", load_model)
    encoder = encoder_module.NomicTextEncoder(
        model_name="nomic-model",
        model_revision="model-revision",
        code_revision="code-revision",
        max_len=512,
    )

    texts = ["first CV", "second CV", "third CV"]
    encoded = encoder.encode(texts, batch_size=2)
    repeated = encoder.encode(texts, batch_size=1)

    assert encoded.shape == (3, 128)
    assert torch.isfinite(encoded).all()
    assert torch.allclose(encoded.norm(dim=1), torch.ones(3))
    assert torch.allclose(encoded, repeated)
    coordinates = torch.arange(768, dtype=torch.float32)
    pooled = (coordinates + coordinates.square() / 768) / 2
    expected = F.normalize(F.layer_norm(pooled, (768,))[:128], dim=0)
    assert torch.allclose(encoded, expected.expand_as(encoded))
    assert [text for call, _ in tokenizer_calls[:2] for text in call] == [
        f"classification: {text}" for text in texts
    ]
    assert all(kwargs["max_length"] == 512 for _, kwargs in tokenizer_calls)
    assert load_calls["tokenizer"][1]["revision"] == "model-revision"
    assert load_calls["model"][1]["code_revision"] == "code-revision"
    with pytest.raises(ValueError, match="only supports deterministic"):
        encoder.encode(texts, deterministic=False)
