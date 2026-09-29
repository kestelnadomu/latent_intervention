from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

import src.encoder as encoder_module
from src.pair_encoding import encode_pairs, load_latent_artifact, sha256_file


class StubEncoder:
    latent_dim = 128
    instances = 0
    calls: list[list[str]] = []
    configs: list[dict] = []

    def __init__(self, config: dict | None = None) -> None:
        type(self).instances += 1
        type(self).configs.append(config)

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
    pd.DataFrame({"id": [2, 4, 1, 3], "text": ["x2", "x4", "x1", "x3"]}).to_csv(
        factual, index=False
    )
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


def test_encode_pairs_uses_one_encoder_and_one_canonical_artifact(
    tmp_path: Path,
) -> None:
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
    info_path = tmp_path / "z_pairs.info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    assert info["artifact_sha256"] == sha256_file(tmp_path / "z_pairs.pt")

    artifact = load_latent_artifact(config)
    assert artifact.ids == payload["ids"]
    assert artifact.test_ids == [1, 2]
    assert artifact.train_ids == [3, 4]
    assert torch.equal(artifact.z, payload["z"])
    assert torch.equal(artifact.z_prime, payload["z_prime"])
    assert artifact.encoder_info["encoder"] == "stub"


def test_load_latent_artifact_preserves_id_row_alignment(tmp_path: Path) -> None:
    config = _config(tmp_path)
    encode_pairs(config, encoder_factory=StubEncoder)

    artifact = load_latent_artifact(config)

    assert artifact.ids == [1, 2, 3, 4]
    assert artifact.z[:, 0].tolist() == [2.0, 2.0, 2.0, 2.0]
    assert artifact.z[:, 1].tolist() == [169.0, 170.0, 171.0, 172.0]
    assert artifact.test_ids == [1, 2]
    assert artifact.z_prime[:, 1].tolist() == [755.0, 170.0]


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
    assert payload["z_prime"].shape == (2, StubEncoder.latent_dim)
    assert torch.equal(payload["z_prime"][1], payload["z"][1])


def test_encode_pairs_rejects_partial_or_incorrect_pairs(tmp_path: Path) -> None:
    config = _config(tmp_path)
    counterfactual = Path(config["paths"]["texts_counterfactual"])
    pd.DataFrame({"id": [1, 2, 3], "text": ["x1-prime", "x2", "x3-prime"]}).to_csv(
        counterfactual, index=False
    )

    with pytest.raises(ValueError, match="either test IDs or all pair-index IDs"):
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


def test_encode_pairs_records_active_nomic_encoder(tmp_path: Path, monkeypatch) -> None:
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
    assert info["task"] == "classification"
    assert info["task_prefix"] == "classification: "
    assert load_latent_artifact(config).encoder_info["encoder_variant"] == "nomic"


@pytest.mark.parametrize("dimension", [64, 128, 256, 512, 768])
def test_nomic_dimension_metadata_and_loading(tmp_path, dimension) -> None:
    config = _config(tmp_path)
    config["encoder"].update(variant="nomic", nomic_latent_dim=dimension)

    class DimensionEncoder:
        latent_dim = dimension

        def __init__(self, config):
            pass

        def encode(self, texts, **kwargs):
            return torch.ones(len(texts), dimension) / dimension**0.5

    encode_pairs(config, encoder_factory=DimensionEncoder)
    artifact = load_latent_artifact(config)
    assert artifact.z.shape == (4, dimension)
    info = json.loads((tmp_path / "z_pairs.info.json").read_text())
    assert info["latent_dimension"] == dimension
    assert info["normalization"] == f"layer_norm+truncate_{dimension}+l2"
    config["encoder"]["nomic_latent_dim"] = 64 if dimension != 64 else 128
    with pytest.raises(ValueError, match="active encoder config"):
        load_latent_artifact(config)


def test_encoding_refuses_to_overwrite_existing_artifacts(tmp_path) -> None:
    config = _config(tmp_path)
    encode_pairs(config, encoder_factory=StubEncoder)
    before = sha256_file(config["paths"]["latents"])
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        encode_pairs(config, encoder_factory=StubEncoder)
    assert sha256_file(config["paths"]["latents"]) == before


def test_encoding_rejects_inputs_changed_during_inference(tmp_path) -> None:
    config = _config(tmp_path)

    class MutatingEncoder(StubEncoder):
        def encode(self, texts, deterministic, batch_size):
            frame = pd.read_csv(config["paths"]["texts"])
            frame.loc[0, "text"] = "changed during encoding"
            frame.to_csv(config["paths"]["texts"], index=False)
            return super().encode(texts, deterministic, batch_size)

    with pytest.raises(ValueError, match="source inputs changed"):
        encode_pairs(config, encoder_factory=MutatingEncoder)
    assert not Path(config["paths"]["latents"]).exists()


def test_load_latent_artifact_rejects_artifact_hash_mismatch(tmp_path: Path) -> None:
    config = _config(tmp_path)
    encode_pairs(config, encoder_factory=StubEncoder)
    info_path = tmp_path / "z_pairs.info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["artifact_sha256"] = "0" * 64
    info_path.write_text(json.dumps(info), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256.*re-encode"):
        load_latent_artifact(config)


def test_load_latent_artifact_rejects_encoder_config_mismatch(tmp_path: Path) -> None:
    config = _config(tmp_path)
    encode_pairs(config, encoder_factory=StubEncoder)
    config["encoder"]["max_len"] = 64

    with pytest.raises(ValueError, match="active encoder config.*re-encode"):
        load_latent_artifact(config)


def test_load_latent_artifact_hashes_local_checkpoint_contents(tmp_path: Path) -> None:
    config = _config(tmp_path)
    checkpoint = tmp_path / "local-langvae"
    checkpoint.mkdir()
    weights = checkpoint / "weights.bin"
    weights.write_bytes(b"first checkpoint")
    config["encoder"]["local_checkpoint"] = str(checkpoint)
    encode_pairs(config, encoder_factory=StubEncoder)

    info = json.loads((tmp_path / "z_pairs.info.json").read_text(encoding="utf-8"))
    assert info["local_checkpoint"] == str(checkpoint.resolve())
    assert info["local_checkpoint_sha256"]
    load_latent_artifact(config)

    weights.write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="active encoder config.*re-encode"):
        load_latent_artifact(config)


def test_load_latent_artifact_rejects_input_hash_mismatch(tmp_path: Path) -> None:
    config = _config(tmp_path)
    encode_pairs(config, encoder_factory=StubEncoder)
    factual_path = Path(config["paths"]["texts"])
    factual = pd.read_csv(factual_path)
    factual.loc[factual["id"] == 1, "text"] = "changed"
    factual.to_csv(factual_path, index=False)

    with pytest.raises(ValueError, match="current pair/text inputs.*re-encode"):
        load_latent_artifact(config)


def test_load_latent_artifact_rejects_split_mismatch(tmp_path: Path) -> None:
    config = _config(tmp_path)
    encode_pairs(config, encoder_factory=StubEncoder)
    pair_path = Path(config["paths"]["pair_index"])
    pairs = pd.read_csv(pair_path)
    pairs.loc[pairs["id"] == 3, "split"] = "test"
    pairs.to_csv(pair_path, index=False)

    # Keep the source hash current to exercise semantic split validation itself.
    info_path = tmp_path / "z_pairs.info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["input_sha256"]["pair_index"] = sha256_file(pair_path)
    info_path.write_text(json.dumps(info), encoding="utf-8")

    with pytest.raises(ValueError, match="test IDs do not match.*re-encode"):
        load_latent_artifact(config)


def test_load_latent_artifact_rejects_changed_identity_latent(tmp_path: Path) -> None:
    config = _config(tmp_path)
    encode_pairs(config, encoder_factory=StubEncoder)
    artifact_path = tmp_path / "z_pairs.pt"
    payload = torch.load(artifact_path, weights_only=True)
    payload["z_prime"][1, 0] += 1
    torch.save(payload, artifact_path)
    info_path = tmp_path / "z_pairs.info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["artifact_sha256"] = sha256_file(artifact_path)
    info_path.write_text(json.dumps(info), encoding="utf-8")

    with pytest.raises(ValueError, match="identity latent differs.*re-encode"):
        load_latent_artifact(config)


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("ids", [1, 1, 3, 4], "duplicate IDs"),
        ("z", torch.full((4, 128), float("nan")), "invalid z"),
        ("z_prime", torch.zeros(2, 127), "invalid z_prime"),
        ("is_identity", torch.tensor([0, 1]), "invalid is_identity"),
    ],
)
def test_load_latent_artifact_rejects_malformed_payload(
    tmp_path: Path, field: str, bad_value, message: str
) -> None:
    config = _config(tmp_path)
    encode_pairs(config, encoder_factory=StubEncoder)
    artifact_path = tmp_path / "z_pairs.pt"
    payload = torch.load(artifact_path, weights_only=True)
    payload[field] = bad_value
    torch.save(payload, artifact_path)
    info_path = tmp_path / "z_pairs.info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["artifact_sha256"] = sha256_file(artifact_path)
    info_path.write_text(json.dumps(info), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"{message}.*re-encode"):
        load_latent_artifact(config)
