from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

import src.encoder as encoder_module
from src.pair_encoding import encode_pairs
from src.pipeline import _load_latents


class StubEncoder:
    latent_dim = 128
    instances = 0
    calls: list[list[str]] = []

    def __init__(self, **kwargs) -> None:
        type(self).instances += 1

    def encode(self, texts, deterministic, batch_size):
        type(self).calls.append(list(texts))
        return torch.tensor(
            [
                [float(len(text)), float(sum(text.encode("utf-8")))] + [0.0] * 126
                for text in texts
            ]
        )


def _config(tmp_path: Path) -> dict:
    pairs = tmp_path / "pair_index.csv"
    factual = tmp_path / "factual.csv"
    counterfactual = tmp_path / "counterfactual.csv"
    pd.DataFrame(
        {
            "id": [3, 1, 4, 2],
            "split": ["train", "test", "train", "test"],
            "is_identity": [False, False, True, True],
        }
    ).to_csv(pairs, index=False)
    pd.DataFrame(
        {"id": [2, 4, 1, 3], "text": ["x2", "x4", "x1", "x3"]}
    ).to_csv(factual, index=False)
    pd.DataFrame({"id": [2, 1], "text": ["x2", "x1-prime"]}).to_csv(
        counterfactual, index=False
    )
    return {
        "encoder": {
            "model_name": "stub",
            "local_checkpoint": None,
            "max_len": 32,
            "batch_size": 4,
            "device": "cpu",
        },
        "paths": {
            "pair_index": str(pairs),
            "texts": str(factual),
            "texts_counterfactual": str(counterfactual),
            "latents": str(tmp_path / "z_pairs.pt"),
        },
    }


def test_encode_pairs_uses_one_encoder_and_one_canonical_artifact(tmp_path: Path) -> None:
    StubEncoder.instances = 0
    StubEncoder.calls = []
    config = _config(tmp_path)

    payload = encode_pairs(config, encoder_factory=StubEncoder)

    assert StubEncoder.instances == 1
    assert StubEncoder.calls == [["x1", "x2", "x3", "x4", "x1-prime"]]
    assert payload["ids"] == [1, 2, 3, 4]
    assert payload["test_ids"] == [1, 2]
    assert payload["is_identity"].tolist() == [False, True]
    assert torch.equal(payload["z_prime"][1], payload["z"][1])
    assert torch.equal(payload["z_prime"][0, :2], torch.tensor([8.0, 755.0]))
    assert torch.isfinite(payload["z"]).all()
    assert (tmp_path / "z_pairs.info.json").exists()

    z, ids = _load_latents(config)
    assert ids == payload["ids"]
    assert torch.equal(z, payload["z"])


def test_encode_pairs_rejects_incomplete_or_incorrect_test_pairs(tmp_path: Path) -> None:
    config = _config(tmp_path)
    counterfactual = Path(config["paths"]["texts_counterfactual"])
    pd.DataFrame({"id": [1], "text": ["x1-prime"]}).to_csv(counterfactual, index=False)

    with pytest.raises(ValueError, match="equal test IDs"):
        encode_pairs(config, encoder_factory=StubEncoder)

    _config(tmp_path)
    pd.DataFrame({"id": [1, 2], "text": ["x1-prime", "not-x2"]}).to_csv(
        counterfactual, index=False
    )
    with pytest.raises(ValueError, match="identity text differs"):
        encode_pairs(config, encoder_factory=StubEncoder)


def test_encode_pairs_rejects_wrong_latent_width(tmp_path: Path) -> None:
    class WrongWidthEncoder(StubEncoder):
        def encode(self, texts, deterministic, batch_size):
            return torch.zeros((len(texts), 127))

    with pytest.raises(ValueError, match="invalid latent vectors"):
        encode_pairs(_config(tmp_path), encoder_factory=WrongWidthEncoder)


def test_encode_pairs_records_active_nomic_encoder(
    tmp_path: Path, monkeypatch
) -> None:
    config = _config(tmp_path)
    config["encoder"].update(
        {
            "variant": "nomic",
            "nomic_model_name": "nomic-model",
            "nomic_model_revision": "model-revision",
            "nomic_code_revision": "code-revision",
            "nomic_task": "classification",
        }
    )

    monkeypatch.setattr(encoder_module, "make_encoder", lambda config: StubEncoder())
    encode_pairs(config)
    info = json.loads((tmp_path / "z_pairs.info.json").read_text(encoding="utf-8"))

    assert info["encoder_variant"] == "nomic"
    assert info["encoder"] == "nomic-model"
    assert info["model_revision"] == "model-revision"
    assert info["code_revision"] == "code-revision"
    assert info["task_prefix"] == "classification: "
