from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import src.qwen3_encoder as module


@pytest.mark.parametrize("padding", ["left", "right"])
def test_qwen_last_token_pool_respects_padding(padding):
    hidden = torch.arange(24).reshape(2, 3, 4)
    mask = torch.tensor(
        [[0, 1, 1], [1, 1, 1]] if padding == "left" else [[1, 1, 0], [1, 1, 1]]
    )
    expected = torch.stack([hidden[0, 2 if padding == "left" else 1], hidden[1, 2]])
    assert torch.equal(module.last_token_pool(hidden, mask), expected)
    with pytest.raises(ValueError, match="attention mask"):
        module.last_token_pool(hidden, torch.zeros_like(mask))


@pytest.mark.parametrize("dim", [32, 64, 128, 256, 512, 768, 1024])
def test_qwen_encoder_is_last_token_not_mean_pooling(monkeypatch, dim):
    observed = {}

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            observed["texts"] = texts
            observed["tokenization"] = kwargs
            return {
                "input_ids": torch.tensor([[0, 3, 7]] * len(texts)),
                "attention_mask": torch.tensor([[0, 1, 1]] * len(texts)),
            }

    class Model(torch.nn.Module):
        config = SimpleNamespace(hidden_size=1024, model_type="qwen3")

        def forward(self, input_ids, attention_mask, use_cache):
            assert use_cache is False
            assert self.training is False
            coords = torch.arange(1, 1025, dtype=torch.float32)
            hidden = (
                torch.stack([coords * 0, -coords, coords])
                .unsqueeze(0)
                .expand(len(input_ids), -1, -1)
            )
            return SimpleNamespace(last_hidden_state=hidden)

    def loader(source, **kwargs):
        observed["model_kwargs"] = kwargs
        return Model()

    monkeypatch.setattr(module, "version", lambda _: "4.57.6")
    monkeypatch.setattr(module, "cached_snapshot", lambda _: "/cached/pinned/qwen")
    monkeypatch.setattr(
        module.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer()
    )
    monkeypatch.setattr(module.AutoModel, "from_pretrained", loader)
    encoder = module.make_qwen3_encoder({"qwen3_latent_dim": dim, "max_len": 512})
    result = encoder.encode(["one CV", "second CV"], batch_size=2)
    expected = F.normalize(torch.arange(1, dim + 1, dtype=torch.float32), dim=0)
    assert torch.allclose(result, expected.expand(2, -1))
    assert result.dtype == torch.float32
    assert observed["texts"] == ["one CV", "second CV"]
    assert observed["tokenization"]["truncation"] is False
    assert observed["model_kwargs"]["trust_remote_code"] is False
    assert observed["model_kwargs"]["local_files_only"] is True
    assert observed["model_kwargs"]["dtype"] == torch.float32
    assert encoder.encode([]).shape == (0, dim)
    encoder.max_len = 1
    with pytest.raises(ValueError, match="refusing silent truncation"):
        encoder.encode(["too long"])
    with pytest.raises(ValueError, match="deterministic"):
        encoder.encode(["CV"], deterministic=False)


def test_qwen_environment_error_is_actionable(monkeypatch):
    monkeypatch.setattr(module, "version", lambda _: "4.48.0")
    with pytest.raises(RuntimeError, match="venv-qwen3"):
        module.make_qwen3_encoder({})


def test_qwen_optional_instruction_format():
    assert module.format_text("CV", "") == "CV"
    assert (
        module.format_text("CV", "Represent qualifications")
        == "Instruct: Represent qualifications\nQuery:CV"
    )
