import copy
import json
from pathlib import Path

import pytest
import torch

from exp.text_realizer_package.data import (
    load_bundle,
    make_splits,
    prepare,
    project_factual,
)
from src.artifact_io import sha256_file


def test_factual_only_join_and_exact_package_projection(package_config, monkeypatch):
    original = torch.load

    class Protected(dict):
        def __getitem__(self, key):
            assert key in {"ids", "z"}, f"forbidden canonical key: {key}"
            return super().__getitem__(key)

    def guarded(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        if str(path) == package_config["inputs"]["latents"]:
            return Protected(result)
        return result

    monkeypatch.setattr(torch, "load", guarded)
    root = prepare(package_config)
    bundle = load_bundle(root)
    assert bundle["fit"]["ids"] == [0, 1, 2, 3]
    assert bundle["tune"]["ids"] == [4, 5]
    assert bundle["fit"]["texts"] == [f"4 {i + 3} 6" for i in range(4)]
    raw = original(package_config["inputs"]["latents"], weights_only=True)
    expected = torch.nn.functional.normalize(raw["z"][:4, :256].float(), dim=1)
    torch.testing.assert_close(bundle["fit"]["z"], expected, rtol=0, atol=0)
    assert set(bundle) == {"format_version", "split_sha256", "fit", "tune"}
    assert not (root / "audit.pt").exists()
    assert prepare(package_config) == root


def test_projection_matches_real_package_loader_without_oracle_content(tmp_path):
    from latent_lab.encoder import Encoder

    z = torch.nn.functional.normalize(torch.randn(3, 768), dim=1)
    path = tmp_path / "factual_only.pt"
    torch.save({"ids": [1, 2, 3], "z": z}, path)
    package = Encoder("embeddinggemma", 256)._load("unused", "unused", path)
    torch.testing.assert_close(project_factual(z), package["z"], rtol=0, atol=0)


def test_hash_and_missing_record_rejection(package_config):
    path = Path(package_config["inputs"]["factual"])
    path.write_text(path.read_text().replace("4 3 6", "changed target"))
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare(package_config)


def test_partition_duplicates_and_official_test_mismatch():
    pairs = {1: {"split": "train"}, 2: {"split": "train"}, 3: {"split": "test"}}
    partition = {
        1: {"partition": "pool"},
        2: {"partition": "dev"},
        3: {"partition": "eval"},
    }
    counts = {"fit_count": 1, "tune_count": 1, "evaluation_count": 1}
    assert make_splits([3, 1, 2], pairs, partition, counts)["fit"] == [1]
    with pytest.raises(ValueError, match="unique and identical"):
        make_splits([1, 1, 3], pairs, partition, counts)
    wrong = copy.deepcopy(partition)
    wrong[3]["partition"] = "pool"
    with pytest.raises(ValueError, match="disagrees"):
        make_splits([1, 2, 3], pairs, wrong, counts)


def test_incompatible_encoder_provenance_is_rejected(package_config):
    path = Path(package_config["inputs"]["info"])
    data = json.loads(path.read_text())
    data["embeddinggemma_protocol"]["prompt"] = "different task"
    path.write_text(json.dumps(data))
    package_config["inputs"]["sha256"]["info"] = sha256_file(path)
    with pytest.raises(ValueError, match="provenance"):
        prepare(package_config)


def test_invalid_eval_texts_do_not_enter_preparation(package_config):
    # Evaluation targets can be empty without altering any training decision.
    path = Path(package_config["inputs"]["factual"])
    path.write_text(path.read_text().replace("4 9 6", "").replace("4 10 6", ""))
    observed = sha256_file(path)
    package_config["inputs"]["sha256"]["factual"] = observed
    for key in ("info", "manifest"):
        file = Path(package_config["inputs"][key])
        data = json.loads(file.read_text())
        if key == "info":
            data["input_sha256"]["factual_text"] = observed
        else:
            data["protocol"]["input_sha256"]["texts"] = observed
        file.write_text(json.dumps(data))
        package_config["inputs"]["sha256"][key] = sha256_file(file)
    assert len(load_bundle(prepare(package_config))["fit"]["texts"]) == 4


def test_corrupt_prepared_bundle_rejected(package_config):
    root = prepare(package_config)
    (root / "development.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="bundle hash"):
        load_bundle(root)


def test_nonfinite_and_degenerate_canonical_vectors():
    with pytest.raises(ValueError, match="unit normalized"):
        project_factual(torch.zeros(2, 768))
    z = torch.zeros(2, 768)
    z[:, -1] = 1
    with pytest.raises(ValueError, match="degenerate"):
        project_factual(z)
