from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest
import torch
import torch.nn.functional as F
import yaml

from src.config import NOMIC_DIMENSIONS
from src.encoders.nomic_encoding import (
    dimension_configs,
    inspect_queue,
    run_queue,
    verify_artifact,
)
from src.pair_encoding import encode_pairs, sha256_file


class FakeNomic:
    calls = []

    def __init__(self, config):
        self.latent_dim = config["nomic_latent_dim"]

    def encode(self, texts, **kwargs):
        type(self).calls.append((self.latent_dim, list(texts)))
        full = torch.stack(
            [
                torch.randn(
                    768, generator=torch.Generator().manual_seed(sum(text.encode()))
                )
                for text in texts
            ]
        )
        full = F.layer_norm(full, (768,))
        return F.normalize(full[:, : self.latent_dim], dim=1)


def make_configs(tmp_path, variant="nomic"):
    pd.DataFrame(
        {
            "id": [3, 1, 2],
            "split": ["train", "test", "test"],
            "is_identity": [False, False, True],
        }
    ).to_csv(tmp_path / "pairs.csv", index=False)
    pd.DataFrame({"id": [1, 2, 3], "text": ["first", "second", "third"]}).to_csv(
        tmp_path / "texts.csv", index=False
    )
    pd.DataFrame({"id": [1, 2], "text": ["counterfactual", "second"]}).to_csv(
        tmp_path / "counterfactual.csv", index=False
    )
    raw = {
        "encoder": {
            "variant": "nomic",
            "tag": None,
            "max_len": 512,
            "batch_size": 2,
            "device": "cpu",
        },
        "paths": {
            "latents": str(tmp_path / "latents/{encoder}/z_pairs.pt"),
            "texts": str(tmp_path / "texts.csv"),
            "texts_counterfactual": str(tmp_path / "counterfactual.csv"),
            "pair_index": str(tmp_path / "pairs.csv"),
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return dimension_configs(path, variant=variant)


class FakeQwen3:
    calls = []

    def __init__(self, config):
        self.latent_dim = config["qwen3_latent_dim"]

    def encode(self, texts, **kwargs):
        type(self).calls.append((self.latent_dim, list(texts)))
        full = torch.stack(
            [
                torch.randn(
                    1024, generator=torch.Generator().manual_seed(sum(text.encode()))
                )
                for text in texts
            ]
        )
        return F.normalize(full[:, : self.latent_dim], dim=1)


def test_qwen_queue_one_full_pass_and_portable_metadata(tmp_path):
    configs = make_configs(tmp_path, variant="qwen3")
    FakeQwen3.calls = []
    state = run_queue(
        configs, tmp_path / "reports", "qwen-test", encoder_factory=FakeQwen3
    )
    assert state["status"] == "complete"
    assert FakeQwen3.calls == [(1024, ["first", "second", "third", "counterfactual"])]
    assert len(state["completed"]) == 7
    for dim, config in configs.items():
        artifact = verify_artifact(config)
        assert artifact.z.shape == (3, dim)
        assert artifact.encoder_info["encoder"] == "Qwen/Qwen3-Embedding-0.6B"
        assert (
            artifact.encoder_info["qwen3_protocol"]["pooling"]
            == "last_nonpadding_token"
        )
        assert torch.equal(artifact.z_prime[1], artifact.z[1])
    run_queue(configs, tmp_path / "reports", "qwen-test", encoder_factory=FakeQwen3)
    assert len(FakeQwen3.calls) == 1
    changed = configs[128]
    changed["encoder"]["qwen3_instruction"] = "different task"
    with pytest.raises(ValueError, match="active encoder config"):
        verify_artifact(changed)


def test_qwen_normalization_metadata_cannot_be_silently_changed(tmp_path):
    configs = make_configs(tmp_path, variant="qwen3")
    encode_pairs(configs[64], encoder_factory=FakeQwen3)
    path = Path(configs[64]["paths"]["latents"]).with_suffix(".info.json")
    info = json.loads(path.read_text())
    info["normalization"] = "layer_norm+truncate_64+l2"
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="Qwen3 normalization"):
        verify_artifact(configs[64])


class FakeGemma:
    calls = []

    def __init__(self, config):
        self.latent_dim = config["embeddinggemma_latent_dim"]

    def encode(self, texts, **kwargs):
        type(self).calls.append((self.latent_dim, list(texts)))
        full = torch.stack(
            [
                torch.randn(
                    768, generator=torch.Generator().manual_seed(sum(text.encode()))
                )
                for text in texts
            ]
        )
        return F.normalize(full[:, : self.latent_dim], dim=1)


def test_gemma_queue_one_pass_roundtrip_and_resume(tmp_path):
    configs = make_configs(tmp_path, variant="embeddinggemma")
    FakeGemma.calls = []
    state = run_queue(
        configs, tmp_path / "reports", "gemma-test", encoder_factory=FakeGemma
    )
    assert state["status"] == "complete"
    assert FakeGemma.calls == [(768, ["first", "second", "third", "counterfactual"])]
    assert len(state["completed"]) == 4
    hashes = []
    for dim, config in configs.items():
        artifact = verify_artifact(config)
        assert artifact.z.shape == (3, dim)
        assert artifact.z_prime.shape == (2, dim)
        assert torch.equal(artifact.z_prime[1], artifact.z[1])
        assert artifact.encoder_info["encoder"] == "google/embeddinggemma-300m"
        assert artifact.encoder_info["embeddinggemma_protocol"][
            "bidirectional_attention"
        ]
        hashes.append(artifact.artifact_sha256)
    run_queue(configs, tmp_path / "reports", "gemma-test", encoder_factory=FakeGemma)
    assert len(FakeGemma.calls) == 1
    assert hashes == [verify_artifact(c).artifact_sha256 for c in configs.values()]


@pytest.mark.parametrize(
    "key", ["normalization", "task_prefix", "embeddinggemma_protocol"]
)
def test_gemma_artifact_rejects_changed_or_missing_protocol(tmp_path, key):
    config = make_configs(tmp_path, variant="embeddinggemma")[128]
    encode_pairs(config, encoder_factory=FakeGemma)
    path = Path(config["paths"]["latents"]).with_suffix(".info.json")
    info = json.loads(path.read_text())
    del info[key]
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="metadata|normalization/prompt"):
        verify_artifact(config)


@pytest.mark.parametrize("mode", ["--prepare", "--run"])
@pytest.mark.parametrize("preflight_state", ["missing", "failed", "stale", "passed"])
@pytest.mark.parametrize("variant", ["qwen3", "embeddinggemma"])
def test_embedding_cli_preflight_gate(
    tmp_path, monkeypatch, mode, preflight_state, variant
):
    import src.encoders.nomic_encoding as queue

    configs = make_configs(tmp_path, variant=variant)
    signature = {"protocol_test_version": 1}
    report_root = tmp_path / f"reports/talent/{variant}_dimensions/cli-test"
    report_root.mkdir(parents=True)
    if preflight_state != "missing":
        report = {
            "status": "failed" if preflight_state == "failed" else "passed",
            "protocol_signature": (
                {"protocol_test_version": 0}
                if preflight_state == "stale"
                else signature
            ),
        }
        (report_root / "preflight.json").write_text(json.dumps(report))
    calls = []

    def fake_run(*args, **kwargs):
        calls.append("run")
        return {"status": "complete"}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(queue, "dimension_configs", lambda *args, **kwargs: configs)
    monkeypatch.setattr(queue, "protocol", lambda *args: signature)
    monkeypatch.setattr(queue, "run_queue", fake_run)
    monkeypatch.setattr(sys, "argv", ["qwen3_encoding", mode, "--run-id", "cli-test"])
    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        if preflight_state == "passed":
            queue.main(variant=variant)
            assert (report_root / "protocol.json").is_file()
            if mode == "--prepare":
                plan = json.loads((report_root / "plan.json").read_text())
                assert plan["status"] == "prepared"
                assert len(plan["queue"]) == len(configs)
                assert calls == []
            else:
                assert calls == ["run"]
        else:
            with pytest.raises(ValueError, match="preflight"):
                queue.main(variant=variant)
            assert not (report_root / "protocol.json").exists()
            assert calls == []
        assert not any(Path(c["paths"]["latents"]).exists() for c in configs.values())
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_deterministic)


def test_queue_one_pass_preserves_128_and_resumes(tmp_path):
    configs = make_configs(tmp_path)
    encode_pairs(configs[128], encoder_factory=FakeNomic)
    original = sha256_file(configs[128]["paths"]["latents"])
    FakeNomic.calls = []
    rows = inspect_queue(configs)
    assert len(rows) == 5
    assert FakeNomic.calls == []  # prepare never loads/runs the encoder
    result = run_queue(configs, tmp_path / "report", "test", encoder_factory=FakeNomic)
    assert result["status"] == "complete"
    assert FakeNomic.calls == [(768, ["first", "second", "third", "counterfactual"])]
    assert sha256_file(configs[128]["paths"]["latents"]) == original
    for dim in NOMIC_DIMENSIONS:
        artifact = verify_artifact(configs[dim])
        assert artifact.z.shape == (3, dim)
        assert artifact.z_prime.shape == (2, dim)
        assert torch.equal(artifact.z_prime[1], artifact.z[1])
        if dim not in (128, 768):
            info = json.loads(
                Path(configs[dim]["paths"]["latents"])
                .with_suffix(".info.json")
                .read_text()
            )
            assert info["derivation"]["source_artifact_sha256"] == sha256_file(
                configs[768]["paths"]["latents"]
            )
    hashes = [sha256_file(c["paths"]["latents"]) for c in configs.values()]
    run_queue(configs, tmp_path / "report", "test", encoder_factory=FakeNomic)
    assert len(FakeNomic.calls) == 1
    assert hashes == [sha256_file(c["paths"]["latents"]) for c in configs.values()]


def test_queue_can_start_without_existing_128(tmp_path):
    configs = make_configs(tmp_path)
    FakeNomic.calls = []
    run_queue(configs, tmp_path / "report", "test", encoder_factory=FakeNomic)
    assert len(FakeNomic.calls) == 1
    assert verify_artifact(configs[128]).z.shape[1] == 128


def test_queue_rejects_partial_output_before_inference(tmp_path):
    configs = make_configs(tmp_path)
    path = Path(configs[64]["paths"]["latents"])
    path.parent.mkdir(parents=True)
    path.write_bytes(b"incomplete")
    FakeNomic.calls = []
    with pytest.raises(ValueError, match="sidecar is missing"):
        run_queue(configs, tmp_path / "report", "test", encoder_factory=FakeNomic)
    assert FakeNomic.calls == []


def test_queue_retains_interrupted_staging_and_restarts(tmp_path):
    configs = make_configs(tmp_path)
    parent = Path(configs[768]["paths"]["latents"]).parent.parent
    pending = parent / ".nomic_768.test.pending"
    pending.mkdir(parents=True)
    (pending / "z_pairs.pt").write_bytes(b"incomplete")
    run_queue(configs, tmp_path / "report", "test", encoder_factory=FakeNomic)
    retained = list(parent.glob(".nomic_768.test.pending.interrupted-*"))
    assert len(retained) == 1
    assert (retained[0] / "z_pairs.pt").read_bytes() == b"incomplete"


def test_queue_detects_128_disagreement_without_replacing_it(tmp_path):
    configs = make_configs(tmp_path)

    class Different128(FakeNomic):
        def encode(self, texts, **kwargs):
            return -super().encode(texts, **kwargs)

    encode_pairs(configs[128], encoder_factory=Different128)
    before = sha256_file(configs[128]["paths"]["latents"])
    with pytest.raises(ValueError, match="disagrees with 768D"):
        run_queue(configs, tmp_path / "report", "test", encoder_factory=FakeNomic)
    assert sha256_file(configs[128]["paths"]["latents"]) == before
    assert not Path(configs[64]["paths"]["latents"]).exists()
    assert (
        json.loads((tmp_path / "report/status.json").read_text())["status"] == "failed"
    )


def test_queue_invalid_norm_is_rejected(tmp_path):
    configs = make_configs(tmp_path)

    class InvalidNorm(FakeNomic):
        def encode(self, texts, **kwargs):
            return 2 * super().encode(texts, **kwargs)

    encode_pairs(configs[128], encoder_factory=InvalidNorm)
    with pytest.raises(ValueError, match="unit-normalized"):
        inspect_queue(configs)
