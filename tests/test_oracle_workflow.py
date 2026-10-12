from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from src import pipeline
from exp.encoding import oracle_targets as oracle_encoding
from src.latent_intervention.oracle import targets as oracle_targets
from src.latent_intervention.oracle import workflow as oracle_workflow
from src.artifact_io import save_torch, sha256_file, write_json
from src.config import load_config
from src.latent_intervention.oracle.regression import ORACLE_VARIANT
from src.pair_encoding import encode_pairs, load_latent_artifact
from src.schema import load_schema
from src.semantic_decoder.model import make_semantic_decoder


class StubGemma:
    calls = []

    def __init__(self, config):
        self.latent_dim = config["embeddinggemma_latent_dim"]

    def encode(self, texts, **kwargs):
        type(self).calls.append(list(texts))
        vectors = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            full = torch.randn(768, generator=torch.Generator().manual_seed(seed))
            vectors.append(F.normalize(full[: self.latent_dim], dim=0))
        return torch.stack(vectors)


@pytest.fixture
def config(tmp_path):
    config = load_config(
        encoder_variant="embeddinggemma",
        embeddinggemma_dim=768,
        manipulator_variant=ORACLE_VARIANT,
    )
    ids = list(range(1, 9))
    identity = [False, False, True, False, False, True, True, False]
    paths = config["paths"]
    paths.update(
        pair_index=str(tmp_path / "pairs.csv"),
        texts=str(tmp_path / "factual.csv"),
        texts_counterfactual=str(tmp_path / "counterfactual.csv"),
        latents=str(tmp_path / "embeddinggemma_768/z_pairs.pt"),
        oracle_targets=str(tmp_path / "embeddinggemma_768/oracle_train/z_prime.pt"),
        sim_factual=str(tmp_path / "states.csv"),
        sim_counterfactual=str(tmp_path / "states_prime.csv"),
        manipulator_model=str(
            tmp_path / "models/h-oracle_regression/latent_intervention.pt"
        ),
        oracle_training_report=str(
            tmp_path / "reports/h-oracle_regression/training.json"
        ),
        eval_report=str(tmp_path / "reports/h-oracle_regression/eval.json"),
        decoder_model=str(tmp_path / "models/g.pt"),
    )
    pd.DataFrame(
        dict(id=ids, split=["train"] * 6 + ["test"] * 2, is_identity=identity)
    ).to_csv(paths["pair_index"], index=False)
    pd.DataFrame(dict(id=ids, text=[f"cv-{i}" for i in ids])).to_csv(
        paths["texts"], index=False
    )
    pd.DataFrame(
        dict(
            id=ids,
            text=[
                f"cv-{i}" if flag else f"cv-{i}-prime" for i, flag in zip(ids, identity)
            ],
        )
    ).to_csv(paths["texts_counterfactual"], index=False)
    states = pd.DataFrame(
        dict(
            id=ids,
            X=[3 if flag else 0 for flag in identity],
            T=[1] * 8,
            D=[1] * 8,
            U=[1] * 8,
        )
    )
    states.to_csv(paths["sim_factual"], index=False)
    states.assign(X=3).to_csv(paths["sim_counterfactual"], index=False)
    config["latent_intervention"][ORACLE_VARIANT].update(
        epochs=3, hidden_dim=8, n_hidden=1, batch_size=3, dropout=0
    )
    encode_pairs(config, encoder_factory=StubGemma)
    StubGemma.calls = []
    return config


def test_targets_only_encode_nonidentity_training_pairs_and_preserve_canonical(config):
    canonical = sha256_file(config["paths"]["latents"])
    targets = oracle_targets.encode_oracle_targets(config, encoder_factory=StubGemma)
    assert targets.ids == list(range(1, 7))
    assert StubGemma.calls == [
        ["cv-1", "cv-2", "cv-3", "cv-4"],
        ["cv-1-prime", "cv-2-prime", "cv-4-prime", "cv-5-prime"],
    ]
    assert not any(
        "cv-7" in text or "cv-8" in text for call in StubGemma.calls for text in call
    )
    artifact = load_latent_artifact(config)
    assert torch.equal(targets.z_prime[2], artifact.z[2])
    assert torch.equal(targets.z_prime[5], artifact.z[5])
    assert sha256_file(config["paths"]["latents"]) == canonical
    again = oracle_targets.encode_oracle_targets(
        config, encoder_factory=lambda _: pytest.fail("must resume without inference")
    )
    assert again.artifact_sha256 == targets.artifact_sha256


def test_missing_targets_and_other_variants_fail_closed(config):
    with pytest.raises(ValueError, match="targets are missing"):
        oracle_workflow.train_oracle_manipulator(config)
    assert not Path(config["paths"]["manipulator_model"]).exists()
    config["latent_intervention"]["variant"] = "baseline"
    with pytest.raises(ValueError, match="only to oracle_regression"):
        oracle_targets.encode_oracle_targets(config, encoder_factory=StubGemma)


@pytest.mark.parametrize(
    "change", ["test_id", "ordering", "identity", "nan", "checksum", "provenance"]
)
def test_oracle_loader_rejects_leakage_and_tampering(config, change):
    oracle_targets.encode_oracle_targets(config, encoder_factory=StubGemma)
    path = oracle_targets.target_path(config)
    info = json.loads(path.with_suffix(".info.json").read_text())
    payload = torch.load(path, weights_only=True)
    if change == "test_id":
        payload["train_ids"][0] = 8
    elif change == "ordering":
        payload["train_ids"].reverse()
    elif change == "identity":
        payload["is_identity"][0] = True
    elif change == "nan":
        payload["z_prime"][0, 0] = float("nan")
    elif change == "checksum":
        info["artifact_sha256"] = "0" * 64
    elif change == "provenance":
        info["source_latent_sha256"] = "0" * 64
    if change not in {"checksum", "provenance"}:
        save_torch(path, payload)
        info["artifact_sha256"] = sha256_file(path)
    write_json(path.with_suffix(".info.json"), info)
    with pytest.raises(ValueError):
        oracle_targets.load_oracle_targets(config)


def test_encoder_drift_fails_before_publishing(config):
    class WrongEncoder(StubGemma):
        def encode(self, texts, **kwargs):
            return torch.ones(len(texts), 768) / 768**0.5

    with pytest.raises(ValueError, match="probe differs"):
        oracle_targets.encode_oracle_targets(config, encoder_factory=WrongEncoder)
    assert not oracle_targets.target_path(config).parent.exists()


def test_queue_derives_256_once_and_resumes_without_inference(config, tmp_path):
    narrow = copy.deepcopy(config)
    narrow["encoder"]["embeddinggemma_latent_dim"] = 256
    narrow["paths"]["latents"] = str(tmp_path / "embeddinggemma_256/z_pairs.pt")
    narrow["paths"]["oracle_targets"] = str(
        tmp_path / "embeddinggemma_256/oracle_train/z_prime.pt"
    )
    encode_pairs(narrow, encoder_factory=StubGemma)
    StubGemma.calls = []
    configs = {768: config, 256: narrow}
    root = tmp_path / "queue/test"
    plan = oracle_encoding.prepare(configs, root)
    assert plan["training_units"] == 6
    assert plan["nonidentity_texts_to_encode"] == 4
    assert not StubGemma.calls
    assert not oracle_targets.target_path(config).exists()
    assert oracle_encoding.run(configs, root, encoder_factory=StubGemma) == [768, 256]
    assert (
        len(StubGemma.calls) == 2
    )  # native compatibility probes + native training targets
    wide = oracle_targets.load_oracle_targets(config)
    short = oracle_targets.load_oracle_targets(narrow)
    assert torch.allclose(
        short.z_prime, F.normalize(wide.z_prime[:, :256], dim=1), atol=1e-6
    )
    assert short.metadata["derivation"]["source_targets_sha256"] == wide.artifact_sha256
    assert oracle_encoding.run(
        configs, root, encoder_factory=lambda _: pytest.fail("should reuse")
    ) == [768, 256]
    assert json.loads((root / "status.json").read_text())["status"] == "complete"


def test_training_needs_no_g_or_hs_and_never_uses_test_targets(config, monkeypatch):
    oracle_targets.encode_oracle_targets(config, encoder_factory=StubGemma)
    original = load_latent_artifact(config)
    poisoned = replace(
        original, z_prime=torch.full_like(original.z_prime, float("nan"))
    )
    monkeypatch.setattr(oracle_workflow, "load_latent_artifact", lambda _: poisoned)
    # Neither dependency even needs to exist for ordinary paired MSE fitting.
    assert not Path(config["paths"]["decoder_model"]).exists()
    monkeypatch.setattr(
        "src.semantic_decoder.model.load_semantic_decoder",
        lambda *a, **k: pytest.fail("g must not be loaded for oracle training"),
    )
    monkeypatch.setattr(
        "src.symbolic_intervention.load_symbolic_kernel",
        lambda *a, **k: pytest.fail("h_S must not be loaded for oracle training"),
    )
    config["sim_config"] = str(Path(config["sim_config"]))
    report = oracle_workflow.train_oracle_manipulator(config)
    fit = set(report["metadata"]["fit_ids"])
    validation = set(report["metadata"]["validation_ids"])
    assert fit | validation == set(range(1, 7))
    assert not fit & validation
    assert not (fit | validation) & set(original.test_ids)
    assert report["split"] == {
        "official_train": 6,
        "fit": 5,
        "validation": 1,
        "official_test": 2,
    }
    assert report["metadata"]["decoder_used_for_training"] is False
    assert report["metadata"]["symbolic_kernel_used_for_training"] is False
    assert report["official_test_used_for_training_or_selection"] is False
    with pytest.raises(FileExistsError, match="already exist"):
        oracle_workflow.train_oracle_manipulator(config)


@pytest.mark.parametrize("dimension", [256, 768])
def test_heldout_evaluation_uses_existing_pairs_not_training_target_files(
    config, monkeypatch, dimension
):
    if dimension == 256:
        parent = Path(config["paths"]["latents"]).parent.parent
        config["encoder"]["embeddinggemma_latent_dim"] = dimension
        config["paths"]["latents"] = str(parent / "embeddinggemma_256/z_pairs.pt")
        config["paths"]["oracle_targets"] = str(
            parent / "embeddinggemma_256/oracle_train/z_prime.pt"
        )
        encode_pairs(config, encoder_factory=StubGemma)
    oracle_targets.encode_oracle_targets(config, encoder_factory=StubGemma)
    oracle_workflow.train_oracle_manipulator(config)
    artifact = load_latent_artifact(config)
    columns, _ = load_schema(config["sim_config"])
    decoder = make_semantic_decoder(
        "independent",
        latent_dim=dimension,
        columns=columns,
        hidden_dim=8,
        n_hidden=1,
        dropout=0,
    )
    decoder.save(
        config["paths"]["decoder_model"],
        metadata=pipeline._decoder_metadata(config, artifact),
    )
    decoder_hash = sha256_file(config["paths"]["decoder_model"])
    monkeypatch.setattr(
        oracle_workflow,
        "load_oracle_targets",
        lambda *args: pytest.fail("evaluation must not load training targets"),
    )
    directory = oracle_targets.target_path(config).parent
    directory.rename(directory.with_name("oracle_train_unavailable"))
    report = oracle_workflow.evaluate_oracle_manipulator(config)
    assert report["n_test"] == 2
    assert report["identity_recovery"]["count"] == 1
    assert report["nonidentity_recovery"]["count"] == 1
    assert report["recovery"]["mse"] >= 0
    assert (
        report["counterfactual"]["semantic_evaluation"]["decoder_used_for_training"]
        is False
    )
    assert sha256_file(config["paths"]["decoder_model"]) == decoder_hash


def test_pipeline_delegates_oracle_before_legacy_or_flow_work(config, monkeypatch):
    seen = []
    monkeypatch.setattr(
        oracle_workflow,
        "train_oracle_manipulator",
        lambda cfg: seen.append(("train", cfg)),
    )
    monkeypatch.setattr(
        oracle_workflow,
        "evaluate_oracle_manipulator",
        lambda cfg: seen.append(("evaluate", cfg)),
    )
    monkeypatch.setattr(
        pipeline, "load_latent_artifact", lambda _: pytest.fail("legacy path")
    )
    pipeline.stage_train_manipulator(config)
    pipeline.stage_evaluate(config)
    assert seen == [("train", config), ("evaluate", config)]
