"""1B protocol isolation and fail-closed automatic continuation."""

import copy

import pytest
import yaml

from exp.text_realizer_cf import pilot_1b
from exp.text_realizer_cf.settings import GEMMA1B_CONFIG, read_config
from src.artifact_io import write_json


def test_1b_keeps_270m_protocol_and_uses_separate_outputs():
    original, larger = read_config(), read_config(GEMMA1B_CONFIG)
    assert original["model"]["name"] == "google/gemma-3-270m"
    assert larger["model"]["name"] == "google/gemma-3-1b-pt"
    for key in ("training", "selection", "generation"):
        assert larger[key] == original[key]
    assert larger["preflight"]["microbatches"] == [1]
    assert original["preflight"]["microbatches"] == [1, 2]
    for key in ("prepared", "models", "reports"):
        assert larger["paths"][key] != original["paths"][key]
    for key in original["inputs"]["sha256"]:
        if key != "baseline_checkpoint":
            assert larger["inputs"]["sha256"][key] == original["inputs"]["sha256"][key]


@pytest.mark.parametrize(
    "case", ["backbone", "revision", "namespace", "batch", "updates"]
)
def test_1b_refuses_changed_protocol(tmp_path, case):
    config = read_config(GEMMA1B_CONFIG)
    if case == "backbone":
        config["model"]["name"] = "unapproved/model"
    elif case == "revision":
        config["model"]["revision"] = "0" * 40
    elif case == "namespace":
        config["paths"]["models"] = read_config()["paths"]["models"]
    elif case == "batch":
        config["preflight"]["microbatches"] = [1, 2]
    else:
        config["training"]["updates"] = 601
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        read_config(path)


@pytest.fixture
def controller_case(monkeypatch, tmp_path):
    config = copy.deepcopy(read_config(GEMMA1B_CONFIG))
    config["paths"]["models"] = str(tmp_path / "models")
    config["paths"]["reports"] = str(tmp_path / "reports")
    prepared = tmp_path / "prepared"
    manifest = {
        "counts": {"selection_factual": 80},
        "split_ids": {"evaluation": list(range(320))},
    }
    write_json(prepared / "manifest.json", manifest)
    identity = {"source_sha256": {}, "test": True}
    monkeypatch.setattr(pilot_1b, "read_config", lambda *args: config)
    monkeypatch.setattr(pilot_1b, "prepare", lambda *args: prepared)
    monkeypatch.setattr(pilot_1b, "identity", lambda *args: identity)
    monkeypatch.setattr(pilot_1b, "require_current", lambda *args: None)
    ready = {
        "ready": True,
        "identity": identity,
        "seconds_per_diagnostic_update": 20,
        "inference": {
            "seconds_per_generated_cv": 10,
            "seconds_per_nll_cv": 1,
            "seconds_per_cycle_cv": 0.1,
        },
    }
    return config, prepared, ready, manifest


def test_preflight_only_then_parallel_training_then_evaluation(
    controller_case, monkeypatch, tmp_path
):
    config, prepared, ready, _ = controller_case
    calls = []

    def children(tasks, controller, phase):
        calls.append(phase)
        if phase == "preflight":
            write_json(prepared / "preflight.json", ready)
        elif phase == "training":
            assert [task["gpu"] for task in tasks] == ["0", "1"]
            assert [task["name"] for task in tasks] == ["factual", "oracle"]
            for task in tasks:
                assert "train" in task["command"]
                assert "--config" in task["command"]
                write_json(task["progress"] / "training.json", {"complete": True})
        elif phase == "evaluating":
            assert calls[-2] == "training"
            write_json(
                controller.parent / "result.json",
                {
                    "complete": True,
                    "identity": ready["identity"],
                    "decision": {"worth_replication": False},
                },
            )

    monkeypatch.setattr(pilot_1b, "_run_children", children)
    config_path = tmp_path / "config.yaml"
    result = pilot_1b.run(config_path, "unit", ["0", "1"])
    assert result["training_started"] is False
    assert calls == ["preflight"]
    result = pilot_1b.run(
        config_path, "unit", ["0", "1"], resume=True, train_after_preflight=True
    )
    assert calls == ["preflight", "training", "evaluating"]
    assert result == {"worth_replication": False}


def test_changed_preflight_never_starts_training(
    controller_case, monkeypatch, tmp_path
):
    _, prepared, ready, _ = controller_case
    calls = []

    def child(tasks, root, phase):
        calls.append(phase)
        write_json(
            prepared / "preflight.json", {**ready, "identity": {"changed": True}}
        )

    monkeypatch.setattr(pilot_1b, "_run_children", child)
    with pytest.raises(ValueError, match="exact identity"):
        pilot_1b.run(
            tmp_path / "config.yaml", "unit", ["0", "1"], train_after_preflight=True
        )
    assert calls == ["preflight"]


def test_estimate_includes_selection_and_all_six_evaluation_worlds(controller_case):
    config, _, ready, manifest = controller_case
    result = pilot_1b.runtime_estimate(config, ready, manifest)
    assert result["parallel_training_seconds"] == 600 * 20 + 6 * 80 * 11
    assert result["final_comparison_seconds"] == pytest.approx(3 * 2 * 320 * 11.1)
    assert (
        result["planning_seconds_with_25_percent_margin"]
        > result["total_seconds_before_margin"]
    )


def test_same_gpu_refused_before_any_work(tmp_path):
    with pytest.raises(ValueError, match="distinct"):
        pilot_1b.run(tmp_path / "missing.yaml", "unit", ["0", "0"])
