import json

import pytest
import torch

from exp.text_realizer_package_768.data import prepare
from exp.text_realizer_package_768.evaluation import read_vectors
from exp.text_realizer_package_768.settings import (
    create_plan,
    require_current,
    run_paths,
)


@pytest.mark.parametrize("backend", [None, "sdpa"])
def test_package_rejects_backend_that_failed_real_gpu_parity(tmp_path, backend):
    from exp.text_realizer_package_768.settings import read_config

    config = read_config()
    assert config["model"]["attention_implementation"] == "eager"
    if backend is None:
        del config["model"]["attention_implementation"]
    else:
        config["model"]["attention_implementation"] = backend
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="verified eager"):
        read_config(path)


def test_inference_preserves_off_manifold_edits(tmp_path):
    path = tmp_path / "vectors.pt"
    z = torch.randn(2, 768) * 3
    encoder = {"embeddinggemma_latent_dim": 768}
    payload = {
        "ids": [3, 7],
        "z": z,
        "encoder": encoder,
        "provenance": {"editor": "fixture"},
    }
    torch.save(payload, path)
    ids, actual, _ = read_vectors(path, encoder)
    assert ids == [3, 7]
    torch.testing.assert_close(actual, z, atol=0, rtol=0)
    payload["z_prime"] = z
    torch.save(payload, path)
    with pytest.raises(ValueError, match="exactly"):
        read_vectors(path, encoder)


def test_changed_plan_refuses_continuation(package_config):
    root = prepare(package_config)
    _, reports = run_paths(package_config, "identity")
    create_plan(package_config, root, reports)
    package_config["training"]["projector_lr"] *= 2
    with pytest.raises(ValueError, match="changed"):
        require_current(reports, package_config, root)


def test_missing_preflight_refuses_training(package_config):
    from exp.text_realizer_package_768.training import train

    root = prepare(package_config)
    _, reports = run_paths(package_config, "no-preflight")
    create_plan(package_config, root, reports)
    with pytest.raises(FileNotFoundError, match="ready.json"):
        train(package_config, "no-preflight", device="cpu")


def test_wrong_dimension_and_encoder_are_rejected(tmp_path):
    path = tmp_path / "wrong.pt"
    encoder = {"embeddinggemma_latent_dim": 768}
    data = {
        "ids": [1],
        "z": torch.randn(1, 256),
        "encoder": encoder,
        "provenance": {"run": "fixture"},
    }
    torch.save(data, path)
    with pytest.raises(ValueError, match="768-D"):
        read_vectors(path, encoder)
    data["z"] = torch.randn(1, 768)
    data["encoder"] = {"embeddinggemma_latent_dim": 256}
    torch.save(data, path)
    with pytest.raises(ValueError, match="768-D"):
        read_vectors(path, encoder)


def test_internal_stage_requires_own_gpu_lease(monkeypatch):
    from exp.text_realizer_package_768.__main__ import main

    monkeypatch.delenv("TEXT_REALIZER_PACKAGE_768_LEASE", raising=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "decoder",
            "_stage",
            "train",
            "--config",
            "configs/text_realizer_talent_motivation_768.yaml",
            "--run-id",
            "not-launched",
            "--device",
            "cuda:0",
        ],
    )
    with pytest.raises(PermissionError, match="workflow GPU lease"):
        main()
