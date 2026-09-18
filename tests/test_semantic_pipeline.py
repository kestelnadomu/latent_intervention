from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from src import flow_workflow, pipeline
from src.schema import ColumnSpec


def _artifact():
    return SimpleNamespace(
        ids=[30, 10, 20, 40],
        train_ids=[30, 10],
        test_ids=[20, 40],
        z=torch.tensor([[30.0], [10.0], [20.0], [40.0]]),
        z_prime=torch.tensor([[200.0], [400.0]]),
        is_identity=torch.tensor([False, True]),
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
            "pre_additive": {"noise_std": 0.1, "n_samples": 2},
            "noise_token": {
                "noise_dim": 2,
                "n_samples": 2,
                "entropy_weight": 0.1,
            },
            "dist": {
                "embed_dim": 2,
                "top_k": 2,
                "pretrain_epochs": 1,
                "joint_epochs": 1,
                "realiser_l1": 1.0,
                "realiser_l2": 1.0,
            },
            "particles": {
                "n_particles": 2,
                "uniform_weights": False,
                "entropy_weight": 0.1,
            },
            "flow": {
                "n_blocks": 2,
                "hidden_dim": 8,
                "condition_dim": 4,
                "state_embed_dim": 2,
                "scale_floor": 1e-6,
                "base_std": 0.05,
                "epochs": 1,
                "batch_size": 2,
                "lr": 1e-3,
                "weight_decay": 1e-5,
                "grad_clip": 5.0,
                "n_samples": 2,
                "eval_samples": 3,
                "no_op_fraction": 0.2,
                "support_bank_size": 4,
                "direct_semantic": {
                    "entropy_weight": 0.1,
                    "proximity_weight": 0.1,
                    "support_weight": 0.05,
                    "identity_weight": 1.0,
                },
            },
        },
        "paths": {
            "sim_factual": str(sim_factual),
            "sim_counterfactual": str(sim_counterfactual),
            "decoder_model": str(tmp_path / "decoder.pt"),
            "decoder_report": str(tmp_path / "decoder.json"),
            "manipulator_model": str(tmp_path / "manipulator.pt"),
            "flow_manipulator_model": str(
                tmp_path / "latent_intervention_{variant}.pt"
            ),
            "flow_teacher_model": str(tmp_path / "state_flow.pt"),
            "eval_report": str(tmp_path / "eval.json"),
            "flow_eval_report": str(tmp_path / "eval_{variant}.json"),
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

    def log_joint(self, z):
        return torch.log(torch.full((len(z), 2), 0.5, device=z.device))

    def marginal_probabilities(self, z):
        return {"s": torch.full((len(z), 2), 0.5, device=z.device)}


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
    seen: dict[str, object] = {}
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

    def make_manipulator(**kwargs):
        seen["initialization_seed"] = torch.initial_seed()
        return Manipulator()

    monkeypatch.setattr(pipeline, "LatentIntervention", make_manipulator)

    def train_manipulator(*, latents, **kwargs):
        seen["train"] = latents[:, 0].tolist()

    monkeypatch.setattr(pipeline, "train_latent_intervention", train_manipulator)
    pipeline.stage_train_manipulator(config)
    assert seen["train"] == [30.0, 10.0]
    assert seen["initialization_seed"] == config["seed"]

    class Loader:
        @classmethod
        def load(cls, path, device=None):
            seen["evaluation_device"] = str(device)
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
    assert seen["evaluation_device"] == "cpu"
    assert len(decoder_loads) == 2
    assert decoder_loads[-1]["device"] == "cpu"
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


@pytest.mark.parametrize(
    "variant", ["state_flow", "distilled_flow", "direct_semantic_flow"]
)
def test_flow_variants_delegate_before_legacy_pipeline_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    config = _config(tmp_path)
    config["latent_intervention"]["variant"] = variant
    calls: list[tuple[str, dict]] = []

    monkeypatch.setattr(
        pipeline,
        "train_flow_manipulator",
        lambda received: calls.append(("train", received)),
    )
    monkeypatch.setattr(
        pipeline,
        "evaluate_flow_manipulator",
        lambda received: calls.append(("evaluate", received)),
    )
    monkeypatch.setattr(
        pipeline,
        "load_latent_artifact",
        lambda *_: pytest.fail("legacy pipeline work ran for a flow variant"),
    )

    pipeline.stage_train_manipulator(config)
    pipeline.stage_evaluate(config)

    assert calls == [("train", config), ("evaluate", config)]


@pytest.mark.parametrize(
    ("variant", "stochastic"),
    [
        ("pre_additive", True),
        ("noise_token", True),
        ("dist", False),
        ("particles", False),
    ],
)
def test_legacy_evaluation_preserves_single_draw_or_deterministic_forward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    stochastic: bool,
) -> None:
    config = _config(tmp_path)
    config["latent_intervention"]["variant"] = variant
    config["latent_intervention"]["flow"]["eval_samples"] = 17
    artifact = _artifact()
    decoder = _Decoder()
    calls: list[dict[str, object]] = []

    class Model:
        def __call__(self, z, values, mask, generator=None):
            calls.append(
                {
                    "z": z[:, 0].tolist(),
                    "seed": None if generator is None else generator.initial_seed(),
                }
            )
            return z

        def sample(self, *args, **kwargs):
            pytest.fail("legacy evaluation must not call the sampling API")

    model = Model()

    class Loader:
        @classmethod
        def load(cls, path, device=None):
            return model

    monkeypatch.setitem(pipeline.INTERVENTION_VARIANTS, variant, Loader)
    if variant == "pre_additive":
        monkeypatch.setattr(pipeline, "LatentInterventionPreAdditive", Model)
    elif variant == "noise_token":
        monkeypatch.setattr(pipeline, "LatentInterventionNoiseToken", Model)

    monkeypatch.setattr(pipeline, "load_latent_artifact", lambda _: artifact)
    monkeypatch.setattr(
        pipeline, "load_schema", lambda _: (decoder.columns, ColumnSpec("y", 2))
    )
    monkeypatch.setattr(pipeline, "load_intervention", lambda _: {"s": 1})
    monkeypatch.setattr(pipeline, "load_semantic_decoder", lambda *a, **k: decoder)
    monkeypatch.setattr(pipeline, "_validate_manipulator_info", lambda *a: None)
    monkeypatch.setattr(
        pipeline,
        "_aligned_targets",
        lambda *a: {"s": torch.tensor([0, 1, 1, 0])},
    )
    monkeypatch.setattr(pipeline, "accuracy", lambda *a, **k: {"s": 1.0})
    monkeypatch.setattr(
        pipeline,
        "calibration_metrics",
        lambda *a, **k: {"s": {"accuracy": 1.0, "ece": 0.0, "reliability": []}},
    )

    pipeline.stage_evaluate(config)

    assert calls == [
        {"z": [20.0, 40.0], "seed": config["seed"] if stochastic else None}
    ]


def test_distillation_reuses_a_compatible_state_flow_teacher(
    tmp_path: Path, monkeypatch
) -> None:
    config = _config(tmp_path)
    config["latent_intervention"]["variant"] = "distilled_flow"
    artifact = _artifact()
    columns = [ColumnSpec("s", 2)]
    teacher_path = Path(config["paths"]["flow_teacher_model"])
    teacher_path.write_bytes(b"teacher")
    teacher = object()
    observed: dict[str, object] = {}

    def load(path, **kwargs):
        observed["path"] = Path(path)
        observed["load_kwargs"] = kwargs
        return teacher

    monkeypatch.setattr(flow_workflow, "load_latent_intervention", load)
    monkeypatch.setattr(flow_workflow, "_write_flow_info", lambda *a: None)
    monkeypatch.setattr(
        flow_workflow,
        "make_latent_intervention",
        lambda *a, **k: pytest.fail("compatible teacher was unnecessarily rebuilt"),
    )

    result = flow_workflow._load_or_train_flow_teacher(
        config,
        artifact,
        columns,
        torch.tensor([0, 1]),
        {"s": torch.tensor([0, 1])},
        {"s": 1},
    )

    assert result is teacher
    assert observed["path"] == teacher_path
    load_kwargs = observed["load_kwargs"]
    assert load_kwargs["expected_variant"] == "state_flow"
    assert load_kwargs["expected_columns"] == columns
    assert load_kwargs["expected_metadata"]["latent_artifact_sha256"] == "latent-hash"


def test_distillation_retrains_a_stale_state_flow_teacher(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config = _config(tmp_path)
    config["latent_intervention"]["variant"] = "distilled_flow"
    artifact = _artifact()
    columns = [ColumnSpec("s", 2)]
    teacher_path = Path(config["paths"]["flow_teacher_model"])
    teacher_path.write_bytes(b"stale")
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        flow_workflow,
        "load_latent_intervention",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("provenance mismatch")),
    )
    monkeypatch.setattr(flow_workflow, "_write_flow_info", lambda *a: None)

    class Teacher:
        def save(self, path, **kwargs):
            observed["save_kwargs"] = kwargs
            Path(path).write_bytes(b"fresh")

    teacher = Teacher()
    monkeypatch.setattr(
        flow_workflow, "make_latent_intervention", lambda *a, **k: teacher
    )

    def train(trained_model, **kwargs):
        assert trained_model is teacher
        observed["latents"] = kwargs["latents"][:, 0].tolist()
        observed["states"] = kwargs["states"]["s"].tolist()

    monkeypatch.setattr(flow_workflow, "train_latent_intervention_model", train)

    result = flow_workflow._load_or_train_flow_teacher(
        config,
        artifact,
        columns,
        torch.tensor([0, 1]),
        {"s": torch.tensor([0, 1])},
        {"s": 1},
    )

    assert result is teacher
    assert observed["latents"] == [30.0, 10.0]
    assert observed["states"] == [0, 1]
    assert observed["save_kwargs"]["supported_interventions"] == [{"s": 1}, {}]
    assert teacher_path.read_bytes() == b"fresh"
    assert "retraining stale state-flow teacher" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("variant", "expected_evaluations"),
    [
        ("state_flow", 3),
        ("distilled_flow", 1),
        ("direct_semantic_flow", 1),
    ],
)
def test_flow_evaluation_uses_only_official_test_pairs(
    tmp_path: Path,
    monkeypatch,
    variant: str,
    expected_evaluations: int,
) -> None:
    config = _config(tmp_path)
    config["latent_intervention"]["variant"] = variant
    artifact = _artifact()
    columns = [ColumnSpec("s", 2)]
    decoder = _Decoder()
    Path(config["paths"]["decoder_model"]).write_bytes(b"decoder")
    Path(config["paths"]["flow_teacher_model"]).write_bytes(b"teacher")
    sample_inputs: list[list[float]] = []
    evaluations: list[dict[str, object]] = []

    monkeypatch.setattr(flow_workflow, "load_latent_artifact", lambda _: artifact)
    monkeypatch.setattr(
        flow_workflow, "load_schema", lambda _: (columns, ColumnSpec("y", 2))
    )
    monkeypatch.setattr(flow_workflow, "load_intervention", lambda _: {"s": 1})
    monkeypatch.setattr(
        flow_workflow, "load_semantic_decoder", lambda *a, **k: decoder
    )
    monkeypatch.setattr(flow_workflow, "_validate_flow_info", lambda *a: None)
    monkeypatch.setattr(
        flow_workflow, "load_latent_intervention", lambda *a, **k: object()
    )
    monkeypatch.setattr(
        flow_workflow,
        "load_symbolic_kernel",
        lambda _: SimpleNamespace(columns=columns),
    )
    monkeypatch.setattr(
        flow_workflow,
        "_aligned_targets",
        lambda *a: {"s": torch.tensor([0, 1, 1, 0])},
    )
    monkeypatch.setattr(flow_workflow, "accuracy", lambda *a, **k: {"s": 1.0})
    monkeypatch.setattr(
        flow_workflow,
        "calibration_metrics",
        lambda *a, **k: {"s": {"accuracy": 1.0, "ece": 0.0, "reliability": []}},
    )

    def sample(_model, z, *args, n_samples, **kwargs):
        sample_inputs.append(z[:, 0].tolist())
        return z.unsqueeze(0).expand(n_samples, -1, -1)

    def evaluate(samples, **kwargs):
        evaluations.append(
            {
                "factual": kwargs["factual"][:, 0].tolist(),
                "counterfactual": kwargs["observed_counterfactual"][:, 0].tolist(),
                "training": kwargs["training_latents"][:, 0].tolist(),
                "samples": len(samples),
            }
        )
        return {"ok": True}

    monkeypatch.setattr(flow_workflow, "sample_counterfactual", sample)
    monkeypatch.setattr(flow_workflow, "_evaluate_samples", evaluate)

    flow_workflow.evaluate_flow_manipulator(config)

    assert sample_inputs == [[20.0, 40.0]] * expected_evaluations
    assert len(evaluations) == expected_evaluations
    for evaluation in evaluations:
        assert evaluation == {
            "factual": [20.0, 40.0],
            "counterfactual": [200.0, 400.0],
            "training": [30.0, 10.0],
            "samples": 3,
        }


@pytest.mark.parametrize(
    ("variant", "changed_dependency"),
    [
        ("state_flow", "factual"),
        ("distilled_flow", "factual"),
        ("direct_semantic_flow", "decoder"),
    ],
)
def test_flow_sidecar_rejects_variant_specific_provenance_changes(
    tmp_path: Path,
    monkeypatch,
    variant: str,
    changed_dependency: str,
) -> None:
    config = _config(tmp_path)
    config["latent_intervention"]["variant"] = variant
    artifact = _artifact()
    columns = [ColumnSpec("s", 2)]
    model_path = flow_workflow._flow_model_path(config, variant)
    model_path.write_bytes(b"manipulator")
    Path(config["paths"]["decoder_model"]).write_bytes(b"decoder")
    Path(config["paths"]["flow_teacher_model"]).write_bytes(b"teacher")
    flow_workflow._write_flow_info(config, artifact, columns, variant)

    if changed_dependency == "factual":
        Path(config["paths"]["sim_factual"]).write_text(
            "id,s\n1,1\n", encoding="utf-8"
        )
    else:
        Path(config["paths"]["decoder_model"]).write_bytes(b"new decoder")

    with pytest.raises(ValueError, match="retrain the flow"):
        flow_workflow._validate_flow_info(config, artifact, columns, variant)


def test_distilled_flow_validation_does_not_require_teacher_at_inference(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config["latent_intervention"]["variant"] = "distilled_flow"
    artifact = _artifact()
    columns = [ColumnSpec("s", 2)]
    model_path = flow_workflow._flow_model_path(config, "distilled_flow")
    teacher_path = Path(config["paths"]["flow_teacher_model"])
    teacher_path.write_bytes(b"teacher")
    model = flow_workflow.make_latent_intervention(
        "distilled_flow",
        **flow_workflow._flow_model_kwargs(
            config, latent_dim=2, columns=columns, variant="distilled_flow"
        ),
    )
    model.save(
        model_path,
        metadata=flow_workflow._flow_checkpoint_metadata(
            config, artifact, columns, "distilled_flow"
        ),
        supported_interventions=[{"s": 1}, {}],
    )
    flow_workflow._write_flow_info(config, artifact, columns, "distilled_flow")

    teacher_path.unlink()

    dependencies = flow_workflow._validate_flow_info(
        config, artifact, columns, "distilled_flow"
    )
    assert dependencies["flow_teacher_sha256"]
    loaded = flow_workflow.load_latent_intervention(
        model_path,
        expected_variant="distilled_flow",
        expected_columns=columns,
        expected_metadata=dependencies,
    )
    assert loaded.variant == "distilled_flow"

    teacher_path.write_bytes(b"unrelated teacher")
    assert flow_workflow._validate_flow_info(
        config, artifact, columns, "distilled_flow"
    ) == dependencies

    info_path = flow_workflow._flow_info_path(model_path)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["flow_dependencies"]["flow_teacher_sha256"] = "0" * 64
    info_path.write_text(json.dumps(info), encoding="utf-8")
    tampered_dependencies = flow_workflow._validate_flow_info(
        config, artifact, columns, "distilled_flow"
    )
    with pytest.raises(ValueError, match="provenance mismatch"):
        flow_workflow.load_latent_intervention(
            model_path,
            expected_variant="distilled_flow",
            expected_columns=columns,
            expected_metadata=tampered_dependencies,
        )


def test_flow_metadata_tracks_training_seed_and_relevant_config(tmp_path: Path) -> None:
    config = _config(tmp_path)
    artifact = _artifact()
    columns = [ColumnSpec("s", 2)]

    original = flow_workflow._flow_checkpoint_metadata(
        config, artifact, columns, "state_flow"
    )
    config["seed"] += 1
    changed_seed = flow_workflow._flow_checkpoint_metadata(
        config, artifact, columns, "state_flow"
    )
    config["seed"] -= 1
    config["latent_intervention"]["flow"]["lr"] *= 2
    changed_training = flow_workflow._flow_checkpoint_metadata(
        config, artifact, columns, "state_flow"
    )

    assert changed_seed["training_seed"] != original["training_seed"]
    assert changed_training["training_config_sha256"] != original[
        "training_config_sha256"
    ]
