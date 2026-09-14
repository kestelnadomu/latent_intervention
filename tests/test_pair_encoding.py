from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from src.pair_encoding import encode_pairs
from src.pipeline import _load_latents


class StubEncoder:
    instances = 0
    calls: list[list[str]] = []

    def __init__(self, **kwargs) -> None:
        type(self).instances += 1

    def encode(self, texts, deterministic, batch_size):
        type(self).calls.append(list(texts))
        return torch.tensor(
            [[float(len(text)), float(sum(text.encode("utf-8")))] for text in texts]
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
    assert torch.equal(payload["z_prime"][0], torch.tensor([8.0, 755.0]))
    assert torch.isfinite(payload["z"]).all()
    assert (tmp_path / "z_pairs.info.json").exists()

    z, ids = _load_latents(config)
    assert ids == payload["ids"]
    assert torch.equal(z, payload["z"])


def test_encode_pairs_accepts_all_counterfactual_texts_but_encodes_test_only(
    tmp_path: Path,
) -> None:
    StubEncoder.instances = 0
    StubEncoder.calls = []
    config = _config(tmp_path)
    counterfactual = Path(config["paths"]["texts_counterfactual"])
    pd.DataFrame(
        {
            "id": [4, 2, 3, 1],
            "text": ["x4", "x2", "x3-prime", "x1-prime"],
        }
    ).to_csv(counterfactual, index=False)

    payload = encode_pairs(config, encoder_factory=StubEncoder)

    assert StubEncoder.instances == 1
    assert StubEncoder.calls == [["x1", "x2", "x3", "x4", "x1-prime"]]
    assert payload["test_ids"] == [1, 2]
    assert payload["is_identity"].tolist() == [False, True]
    assert payload["z_prime"].shape == (2, 2)
    assert torch.equal(payload["z_prime"][1], payload["z"][1])


def test_encode_pairs_rejects_partial_or_incorrect_pairs(tmp_path: Path) -> None:
    config = _config(tmp_path)
    counterfactual = Path(config["paths"]["texts_counterfactual"])
    pd.DataFrame(
        {"id": [1, 2, 3], "text": ["x1-prime", "x2", "x3-prime"]}
    ).to_csv(counterfactual, index=False)

    with pytest.raises(ValueError, match="either test IDs or all pair-index IDs"):
        encode_pairs(config, encoder_factory=StubEncoder)

    _config(tmp_path)
    pd.DataFrame({"id": [1, 2], "text": ["x1-prime", "not-x2"]}).to_csv(
        counterfactual, index=False
    )
    with pytest.raises(ValueError, match="identity text differs"):
        encode_pairs(config, encoder_factory=StubEncoder)
