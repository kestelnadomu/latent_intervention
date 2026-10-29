import copy
import json

import pytest

from exp.text_realizer_package.settings import read_config as reference_config
from exp.text_realizer_package.data import prepared_path as reference_prepared_path
from exp.text_realizer_package_768 import settings
from exp.text_realizer_package_768.data import prepared_path
from src.text_realizer.model import model_spec


@pytest.mark.parametrize("size,hidden", [("270m", 640), ("1b", 1152)])
def test_matched_training_recipe_and_full_width(size, hidden):
    original = reference_config()
    config = settings.read_config(settings.CONFIGS[size])
    assert config["model"]["latent_dim"] == 768
    assert config["encoder"]["embeddinggemma_latent_dim"] == 768
    assert config["inputs"] == original["inputs"]
    for key in (
        "split",
        "training",
        "generation",
        "selection",
        "evaluation",
        "preflight",
    ):
        assert config[key] == original[key]
    assert config["model"]["attention_implementation"] == "eager"
    spec = model_spec(config["model"], "lora-compact", hidden)
    assert spec["latent_dim"] == 768
    assert spec["projector_hidden"] == 768
    assert spec["prefix_tokens"] == 16
    assert prepared_path(config) != reference_prepared_path(original)


def test_backbones_share_prepared_data_but_never_model_outputs():
    compact = settings.read_config(settings.CONFIGS["270m"])
    larger = settings.read_config(settings.CONFIGS["1b"])
    assert prepared_path(compact) == prepared_path(larger)
    for key in ("models", "reports"):
        assert compact["paths"][key] != larger["paths"][key]
    comparable = copy.deepcopy(larger)
    comparable["model"].update(
        name=compact["model"]["name"], revision=compact["model"]["revision"]
    )
    for key in ("models", "reports"):
        comparable["paths"][key] = compact["paths"][key]
    assert comparable == compact


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("model", "latent_dim", 256),
        ("model", "attention_implementation", "sdpa"),
        ("encoder", "embeddinggemma_latent_dim", 256),
        ("training", "epochs", 300),
        ("paths", "models", "models/latent_lab"),
        ("split", "fit_count", 3200),
    ],
)
def test_unmatched_recipe_refused(tmp_path, section, key, value):
    config = settings.read_config()
    config[section][key] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="may change only"):
        settings.read_config(path)
