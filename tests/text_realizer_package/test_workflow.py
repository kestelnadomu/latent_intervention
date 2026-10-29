import json
from contextlib import contextmanager

import pytest
import torch

from exp.text_realizer_package import workflow
from exp.text_realizer_package.data import prepare
from exp.text_realizer_package.evaluation import read_vectors
from exp.text_realizer_package.settings import create_plan, require_current, run_paths


@pytest.mark.parametrize("backend", [None, "sdpa"])
def test_package_rejects_backend_that_failed_real_gpu_parity(tmp_path, backend):
    from exp.text_realizer_package.settings import read_config

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
    z = torch.randn(2, 256) * 3
    encoder = {"embeddinggemma_latent_dim": 256}
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


def test_busy_gpu_waits_without_touching_other_jobs(tmp_path, monkeypatch):
    calls = []

    @contextmanager
    def lease(device):
        calls.append(device)
        if len(calls) == 1:
            raise RuntimeError("GPU 2 is already using 9000 MiB")
        yield

    monkeypatch.setattr(workflow, "gpu_lease", lease)
    monkeypatch.setattr(workflow.time, "sleep", lambda seconds: None)
    with workflow.available_gpu("cuda:2", tmp_path, wait=True):
        assert len(calls) == 2
    assert (
        json.loads((tmp_path / "status.json").read_text())["state"] == "waiting_for_gpu"
    )


def test_gpu_errors_do_not_become_endless_queue(tmp_path, monkeypatch):
    @contextmanager
    def lease(device):
        raise RuntimeError("invalid GPU configuration")
        yield

    monkeypatch.setattr(workflow, "gpu_lease", lease)
    with pytest.raises(RuntimeError, match="invalid GPU"):
        with workflow.available_gpu("cuda:0", tmp_path, wait=True):
            pass


def test_missing_preflight_refuses_training(package_config):
    from exp.text_realizer_package.training import train

    root = prepare(package_config)
    _, reports = run_paths(package_config, "no-preflight")
    create_plan(package_config, root, reports)
    with pytest.raises(FileNotFoundError, match="ready.json"):
        train(package_config, "no-preflight", device="cpu")
