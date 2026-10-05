from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src import pipeline
from src.artifact_io import write_json
from src.config import load_config
from src.decoder_benchmark.decoder_experiment import (
    candidate_settings,
    publish_results,
    run_case,
    verify_run,
)
from src.schema import ColumnSpec


def test_candidates_are_distinct_reproducible_and_common() -> None:
    a = load_config(encoder_variant="langvae", decoder_variant="independent")
    b = load_config(encoder_variant="nomic", decoder_variant="autoregressive")
    candidates = candidate_settings(a["semantic_decoder"])
    assert (
        len(candidates)
        == len({json.dumps(row, sort_keys=True) for row in candidates})
        == 20
    )
    assert candidates == candidate_settings(b["semantic_decoder"])
    assert candidates[0] == {
        "lr": 1e-3,
        "weight_decay": 0.0,
        "dropout": 0.1,
        "hidden_dim": 256,
    }


def test_atomic_json_failure_preserves_existing_file(tmp_path) -> None:
    path = tmp_path / "report.json"
    write_json(path, {"original": True})
    with pytest.raises(ValueError):
        write_json(path, {"invalid": float("nan")})
    assert json.loads(path.read_text()) == {"original": True}
    assert list(tmp_path.iterdir()) == [path]


def test_small_experiment_resume_archive_publish_and_fixed_split(
    tmp_path, monkeypatch
) -> None:
    config = load_config()
    cfg = config["semantic_decoder"]
    cfg.update(epochs=3, hidden_dim=8, batch_size=4, verbose=False)
    cfg["search"].update(trials=2, final_seeds=[42, 43], hidden_dim=[8, 12])
    sim = tmp_path / "sim.yaml"
    factual = tmp_path / "factual.csv"
    sim.write_text("test: true\n")
    factual.write_text("id,s\n")
    config["sim_config"] = str(sim)
    config["paths"].update(
        sim_factual=str(factual),
        decoder_model=str(tmp_path / "models" / "semantic_decoder.pt"),
        decoder_report=str(tmp_path / "reports" / "semantic_decoder.json"),
    )
    artifact = SimpleNamespace(
        ids=list(range(20)),
        train_ids=list(range(16)),
        test_ids=list(range(16, 20)),
        z=torch.arange(40, dtype=torch.float32).reshape(20, 2) / 40,
        artifact_sha256="test-latent-hash",
        encoder_info={"encoder_variant": "test"},
    )
    targets = {"s": torch.arange(20) % 2}
    monkeypatch.setattr(pipeline, "load_latent_artifact", lambda _: artifact)
    monkeypatch.setattr(pipeline, "load_schema", lambda _: ([ColumnSpec("s", 2)], None))
    monkeypatch.setattr(pipeline, "_aligned_targets", lambda *args: targets)
    model_path = Path(config["paths"]["decoder_model"])
    report_path = Path(config["paths"]["decoder_report"])
    model_path.parent.mkdir()
    report_path.parent.mkdir()
    model_path.write_bytes(b"previous-model")
    report_path.write_text("previous-report")
    thread_count = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        result = run_case(config, "tiny")
        repeat = run_case(config, "tiny")
    finally:
        torch.set_num_threads(thread_count)
        torch.use_deterministic_algorithms(deterministic)
    assert result["selected_trial"] == repeat["selected_trial"]
    assert result["summary"] == repeat["summary"]
    assert result["deployment_seed"] == 42
    assert model_path.read_bytes() == b"previous-model"
    reports = [
        json.loads(Path(row["report"]).read_text()) for row in result["final_runs"]
    ]
    assert reports[0]["split_ids"] == reports[1]["split_ids"]
    assert set(reports[0]["split_ids"]["validation"]).isdisjoint(artifact.test_ids)
    published = publish_results([result], tmp_path / "experiment")
    assert len(published) == 1
    assert (
        model_path.parent / "archive/tiny/semantic_decoder.pt"
    ).read_bytes() == b"previous-model"
    assert (
        report_path.parent / "archive/tiny/semantic_decoder.json"
    ).read_text() == "previous-report"
    active = json.loads(report_path.read_text())
    verify_run(active["config"])
    corrupt = copy.deepcopy(active["config"])
    corrupt["seed"] = 999
    with pytest.raises(ValueError, match="configuration"):
        verify_run(corrupt)
