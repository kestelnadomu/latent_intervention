import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import src.encoders.langvae_ft_encoding as worker


class FakeEncoder:
    latent_dim = 128

    def __init__(self):
        class Projection(torch.nn.Module):
            def forward(self, values):
                return SimpleNamespace(embedding=values)

        self.model = SimpleNamespace(encoder=Projection())

    def encode(self, texts, *, deterministic, batch_size):
        assert deterministic
        return torch.cat(
            [
                self.model.encoder(
                    torch.tensor(texts[start : start + batch_size])
                    .float()
                    .unsqueeze(1)
                    .expand(-1, 128)
                ).embedding
                for start in range(0, len(texts), batch_size)
            ]
        )


def test_progress_observer_preserves_native_batches_and_removes_hook(monkeypatch):
    native = FakeEncoder()
    monkeypatch.setattr(worker, "make_encoder", lambda config: native)
    events = []
    observed = worker.ProgressEncoder({}, lambda **event: events.append(event))
    texts = list(range(645))
    expected = native.encode(texts, deterministic=True, batch_size=16)
    actual = observed.encode(texts, deterministic=True, batch_size=16)
    assert torch.equal(actual, expected)
    assert [event["completed_texts"] for event in events] == [16, 320, 640, 645]
    assert events[-1]["eta_seconds"] == 0
    assert not native.model.encoder._forward_hooks


def test_invalid_batch_fails_and_removes_hook(monkeypatch):
    native = FakeEncoder()
    monkeypatch.setattr(worker, "make_encoder", lambda config: native)
    observed = worker.ProgressEncoder({}, lambda **event: None)
    with pytest.raises(ValueError, match="posterior means"):
        observed.encode([float("nan")], batch_size=1)
    assert not native.model.encoder._forward_hooks


@pytest.fixture
def encoding_job(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    monkeypatch.setattr(
        worker, "selected_checkpoint", lambda config: tmp_path / "checkpoint"
    )
    monkeypatch.setattr(worker, "verify_checkpoint", lambda *args: {"passed": True})
    monkeypatch.setattr(
        worker.subprocess, "check_output", lambda *args, **kwargs: "test-head\n"
    )
    config_path = tmp_path / "encoding.yaml"
    config_path.write_text("fixture")
    output = tmp_path / "data/latents/talent/langvae_ft/z_pairs.pt"
    stock = output.parent.parent / "langvae/z_pairs.pt"
    stock.parent.mkdir(parents=True)
    stock.write_bytes(b"original baseline")
    config = {"encoder": {}, "paths": {"latents": str(output)}}
    monkeypatch.setattr(worker, "load_config", lambda **kwargs: copy.deepcopy(config))

    def fake_encode(config, encoder_factory):
        path = Path(config["paths"]["latents"])
        path.parent.mkdir(parents=True)
        path.write_bytes(b"new fixture")
        path.with_suffix(".info.json").write_text("{}")

    def fake_load(config):
        path = Path(config["paths"]["latents"])
        assert path.is_file() and path.with_suffix(".info.json").is_file()
        return SimpleNamespace(
            ids=[0, 1, 2],
            test_ids=[2],
            z=torch.zeros(3, 128),
            is_identity=torch.tensor([True]),
            artifact_sha256=worker.sha256_file(path),
        )

    monkeypatch.setattr(worker, "encode_pairs", fake_encode)
    monkeypatch.setattr(worker, "load_latent_artifact", fake_load)
    return config_path, output, stock


def test_run_publishes_both_files_and_refuses_overwrite(encoding_job):
    config_path, output, stock = encoding_job
    worker.run(config_path, "test")
    assert output.read_bytes() == b"new fixture"
    assert output.with_suffix(".info.json").is_file()
    assert stock.read_bytes() == b"original baseline"
    status_path = worker.ROOT / "reports/talent/langvae_ft/encoding/test/status.json"
    status = json.loads(status_path.read_text())
    assert status["state"] == "completed" and status["existing_artifacts_unchanged"]
    with pytest.raises(FileExistsError):
        worker.run(config_path, "second")


def test_failed_validation_keeps_output_unpublished(encoding_job, monkeypatch):
    config_path, output, stock = encoding_job

    def invalid(config):
        raise ValueError("invalid fixture")

    monkeypatch.setattr(worker, "load_latent_artifact", invalid)
    with pytest.raises(ValueError, match="invalid fixture"):
        worker.run(config_path, "failed")
    assert not output.parent.exists()
    assert stock.read_bytes() == b"original baseline"
    status_path = worker.ROOT / "reports/talent/langvae_ft/encoding/failed/status.json"
    assert json.loads(status_path.read_text())["state"] == "failed"
