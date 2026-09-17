from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from src import pipeline
from src.schema import ColumnSpec


def _artifact():
    return SimpleNamespace(
        ids=[30, 10, 20, 40],
        train_ids=[30, 10],
        test_ids=[20, 40],
        z=torch.tensor([[30.0], [10.0], [20.0], [40.0]]),
        artifact_sha256="latent-hash",
        encoder_info={"variant": "stub", "model": "stub-model"},
    )


def _config(tmp_path: Path) -> dict:
    sim_config = tmp_path / "sim.yaml"
    sim_factual = tmp_path / "factual.csv"
    sim_counterfactual = tmp_path / "counterfactual.csv"
    sim_config.write_text("schema: stub\n", encoding="utf-8")
    sim_factual.write_text("id,s\n", encoding="utf-8")
    sim_counterfactual.write_text("id,s\n", encoding="utf-8")
    return {
        "seed": 7,
        "sim_config": str(sim_config),
        "encoder": {"device": "cpu"},
        "semantic_decoder": {
            "variant": "independent",
            "hidden_dim": 8,
            "n_hidden": 1,
            "dropout": 0.0,
            "epochs": 2,
            "batch_size": 2,
            "lr": 1e-3,
            "calibration_split": 0.5,
            "calibration_bins": 4,
            "autoregressive": {"embed_dim": 2},
        },
        "latent_intervention": {
            "variant": "baseline",
            "d_model": 4,
            "nhead": 1,
            "dim_feedforward": 8,
            "dropout": 0.0,
            "epochs": 1,
            "batch_size": 2,
            "lr": 1e-3,
            "sparsity_weight": 1.0,
            "proximity_weight": 1.0,
        },
        "paths": {
            "sim_factual": str(sim_factual),
            "sim_counterfactual": str(sim_counterfactual),
            "decoder_model": str(tmp_path / "decoder.pt"),
            "decoder_report": str(tmp_path / "decoder.json"),
            "manipulator_model": str(tmp_path / "manipulator.pt"),
            "eval_report": str(tmp_path / "eval.json"),
        },
    }


class _Decoder:
    columns = [ColumnSpec("s", 2)]

    def __init__(self) -> None:
        self.saved_metadata = None

    def save(self, path, metadata=None) -> None:
        self.saved_metadata = metadata
        Path(path).write_bytes(b"decoder")

    def predict(self, z):
        return {"s": torch.zeros(len(z), dtype=torch.long)}


def test_aligned_targets_follow_latent_ids_and_reject_bad_id_sets(
    tmp_path: Path,
) -> None:
    path = tmp_path / "targets.csv"
    pd.DataFrame({"id": [20, 30, 10], "s": [0, 1, 2]}).to_csv(path, index=False)

    targets = pipeline._aligned_targets(path, [10, 20, 30], [ColumnSpec("s", 3)])

    assert targets["s"].tolist() == [2, 0, 1]

    pd.DataFrame({"id": [10, 10, 30], "s": [0, 1, 2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate IDs"):
        pipeline._aligned_targets(path, [10, 20, 30], [ColumnSpec("s", 3)])

    pd.DataFrame({"id": [10, 30], "s": [0, 2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="do not match"):
        pipeline._aligned_targets(path, [10, 20, 30], [ColumnSpec("s", 3)])

    pd.DataFrame({"id": [10, 20, 30, 40], "s": [0, 1, 2, 0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="do not match"):
        pipeline._aligned_targets(path, [10, 20, 30], [ColumnSpec("s", 3)])


def test_decoder_fit_and_calibration_never_use_official_test(
    tmp_path: Path, monkeypatch
) -> None:
    config = _config(tmp_path)
    artifact = _artifact()
    decoder = _Decoder()
    observed: dict[str, list[float]] = {}
    targets = {"s": torch.tensor([0, 1, 0, 1])}

    monkeypatch.setattr(pipeline, "load_latent_artifact", lambda _: artifact)
    monkeypatch.setattr(
        pipeline, "load_schema", lambda _: (decoder.columns, ColumnSpec("y", 2))
    )
    monkeypatch.setattr(pipeline, "_aligned_targets", lambda *args: targets)
    monkeypatch.setattr(
        pipeline, "make_semantic_decoder", lambda *args, **kwargs: decoder
    )

    def train(_decoder, latents, _targets, **kwargs):
        observed["fit"] = latents[:, 0].tolist()
        return [1.0, 0.5]

    def metrics(_decoder, latents, _targets, **kwargs):
        observed["calibration"] = latents[:, 0].tolist()
        return {"s": {"accuracy": 1.0, "ece": 0.0, "reliability": []}}

    monkeypatch.setattr(pipeline, "train_semantic_decoder", train)
    monkeypatch.setattr(pipeline, "calibration_metrics", metrics)

    pipeline.stage_train_decoder(config)

    used = observed["fit"] + observed["calibration"]
    assert sorted(used) == [10.0, 30.0]
    assert not {20.0, 40.0}.intersection(used)
    assert decoder.saved_metadata["latent_artifact_sha256"] == "latent-hash"
    assert set(decoder.saved_metadata["training_inputs"]) == {
        "sim_factual_sha256",
        "sim_config_sha256",
    }
    report = json.loads(Path(config["paths"]["decoder_report"]).read_text())
    assert report["split"] == {
        "official_train": 2,
        "fit": 1,
        "calibration": 1,
        "official_test": 2,
    }


def test_manipulator_trains_on_official_train_and_evaluates_on_test(
    tmp_path: Path, monkeypatch
) -> None:
    config = _config(tmp_path)
    artifact = _artifact()
    decoder = _Decoder()
    Path(config["paths"]["decoder_model"]).write_bytes(b"decoder")
    targets = {"s": torch.zeros(4, dtype=torch.long)}
    seen: dict[str, list[float]] = {}
    decoder_loads: list[dict] = []

    monkeypatch.setattr(pipeline, "load_latent_artifact", lambda _: artifact)
    monkeypatch.setattr(
        pipeline, "load_schema", lambda _: (decoder.columns, ColumnSpec("y", 2))
    )
    monkeypatch.setattr(pipeline, "load_intervention", lambda _: {"s": 1})

    def load_decoder(*args, **kwargs):
        decoder_loads.append(kwargs)
        return decoder

    monkeypatch.setattr(pipeline, "load_semantic_decoder", load_decoder)
    monkeypatch.setattr(pipeline, "_aligned_targets", lambda *args: targets)
    monkeypatch.setattr(
        pipeline,
        "load_symbolic_kernel",
        lambda _: SimpleNamespace(columns=decoder.columns),
    )

    class Manipulator:
        def save(self, path):
            Path(path).write_bytes(b"manipulator")

        def __call__(self, z, values, mask):
            seen["test"] = z[:, 0].tolist()
            return z

    monkeypatch.setattr(pipeline, "LatentIntervention", lambda **kwargs: Manipulator())

    def train_manipulator(*, latents, **kwargs):
        seen["train"] = latents[:, 0].tolist()

    monkeypatch.setattr(pipeline, "train_latent_intervention", train_manipulator)
    pipeline.stage_train_manipulator(config)
    assert seen["train"] == [30.0, 10.0]

    class Loader:
        @classmethod
        def load(cls, path):
            return Manipulator()

    monkeypatch.setitem(pipeline.INTERVENTION_VARIANTS, "baseline", Loader)
    monkeypatch.setattr(
        pipeline,
        "accuracy",
        lambda decoder, z, targets: {"s": 1.0},
    )
    monkeypatch.setattr(
        pipeline,
        "calibration_metrics",
        lambda decoder, z, targets, n_bins: {
            "s": {"accuracy": 1.0, "ece": 0.0, "reliability": []}
        },
    )

    pipeline.stage_evaluate(config)

    assert seen["test"] == [20.0, 40.0]
    assert len(decoder_loads) == 2
    for load_args in decoder_loads:
        assert load_args["expected_columns"] == decoder.columns
        assert load_args["expected_metadata"]["latent_artifact_sha256"] == "latent-hash"
        assert set(load_args["expected_metadata"]["training_inputs"]) == {
            "sim_factual_sha256",
            "sim_config_sha256",
        }
    report = json.loads(Path(config["paths"]["eval_report"]).read_text())
    assert report["n_test"] == 2


def test_manipulator_metadata_rejects_changed_decoder(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    artifact = _artifact()
    Path(config["paths"]["decoder_model"]).write_bytes(b"first decoder")
    Path(config["paths"]["manipulator_model"]).write_bytes(b"manipulator")
    pipeline._write_manipulator_info(config, artifact)

    Path(config["paths"]["decoder_model"]).write_bytes(b"changed decoder")

    try:
        pipeline._validate_manipulator_info(config, artifact)
    except ValueError as error:
        assert "retrain the manipulator" in str(error)
    else:
        raise AssertionError("changed decoder checkpoint was accepted")


def test_manipulator_metadata_rejects_changed_model_or_counterfactuals(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    artifact = _artifact()
    decoder_path = Path(config["paths"]["decoder_model"])
    manipulator_path = Path(config["paths"]["manipulator_model"])
    counterfactual_path = Path(config["paths"]["sim_counterfactual"])
    decoder_path.write_bytes(b"decoder")
    manipulator_path.write_bytes(b"first manipulator")
    counterfactual_path.write_text("id,s\n1,0\n", encoding="utf-8")
    pipeline._write_manipulator_info(config, artifact)

    manipulator_path.write_bytes(b"changed manipulator")
    with pytest.raises(ValueError, match="retrain the manipulator"):
        pipeline._validate_manipulator_info(config, artifact)

    manipulator_path.write_bytes(b"first manipulator")
    counterfactual_path.write_text("id,s\n1,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="retrain the manipulator"):
        pipeline._validate_manipulator_info(config, artifact)
