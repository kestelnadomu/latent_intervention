import copy
import json

import pytest

from exp.text_realizer_package.data import prepared_path
from exp.text_realizer_package.settings import read_config as read_reference
from exp.text_realizer_package_1b import settings


def test_same_data_recipe_and_different_backbone_outputs():
    reference, candidate = read_reference(), settings.read_config()
    assert prepared_path(reference) == prepared_path(candidate)
    assert candidate["model"]["name"] == "google/gemma-3-1b-pt"
    assert candidate["model"]["attention_implementation"] == "eager"
    assert candidate["paths"]["models"] != reference["paths"]["models"]
    assert candidate["paths"]["reports"] != reference["paths"]["reports"]
    comparable = copy.deepcopy(candidate)
    del comparable["package_1b_source_sha256"]
    comparable["model"].update(
        name=reference["model"]["name"], revision=reference["model"]["revision"]
    )
    for key in ("models", "reports"):
        comparable["paths"][key] = reference["paths"][key]
    assert comparable == reference


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("model", "latent_dim", 768),
        ("model", "attention_implementation", "sdpa"),
        ("training", "epochs", 300),
        ("paths", "models", "models/latent_lab"),
        ("split", "fit_count", 3200),
    ],
)
def test_unintended_comparison_changes_rejected(tmp_path, section, key, value):
    config = settings.read_config()
    config[section][key] = value
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="only the pinned backbone"):
        settings.read_config(path)


def test_saved_source_identity_is_checked_before_continuation(tmp_path, monkeypatch):
    config = settings.read_config()
    path = tmp_path / "saved.json"
    path.write_text(json.dumps(config))
    assert settings.read_config(path) == config
    monkeypatch.setattr(settings, "extension_hashes", lambda: {"changed.py": "changed"})
    with pytest.raises(ValueError, match="source/configuration changed"):
        settings.read_config(path)
    with pytest.raises(ValueError, match="source/configuration changed"):
        settings.require_extension(config)


def test_stage_requires_own_gpu_lease(monkeypatch):
    from exp.text_realizer_package_1b.workflow import stage

    monkeypatch.delenv("TEXT_REALIZER_PACKAGE_1B_LEASE", raising=False)
    with pytest.raises(PermissionError, match="1B workflow GPU lease"):
        stage(
            settings.read_config(),
            "not-launched",
            "train",
            device="cuda:0",
            resume=False,
        )
