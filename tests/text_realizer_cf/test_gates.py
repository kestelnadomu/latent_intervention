"""No implicit counterfactual evaluation or GPU-training bypass."""

from pathlib import Path

import pytest

from exp.text_realizer_cf import evaluation as evaluation_module
from exp.text_realizer_cf import training as training_module
from exp.text_realizer_cf.settings import read_config


def test_evaluation_refuses_before_opening_heldout_partition(monkeypatch, tmp_path):
    config = read_config()
    config["paths"]["reports"] = str(tmp_path / "reports")
    config["paths"]["models"] = str(tmp_path / "models")
    called = []
    monkeypatch.setattr(
        evaluation_module,
        "load_partition",
        lambda *args: called.append(args)
        or (_ for _ in ()).throw(AssertionError("leak")),
    )
    with pytest.raises(FileNotFoundError):
        evaluation_module.evaluate_pilot(config, "untrained", device="cpu")
    assert not called


def test_training_requires_current_ready_marker(monkeypatch, tmp_path):
    config = read_config()
    prepared = Path(tmp_path)
    monkeypatch.setattr(training_module, "identity", lambda *args: {"v": 1})
    (prepared / "preflight.json").write_text('{"ready": false, "identity": {"v": 1}}')
    with pytest.raises(ValueError, match="passing GPU preflight"):
        training_module.require_preflight(config, prepared, "cpu")
