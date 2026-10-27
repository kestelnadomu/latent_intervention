"""Synthetic encoding tests: no downloads, real-data writes or model fitting."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from exp.encoding import talent_motivation as queue
from exp.encoding.text_artifact import load_text_artifact
from src.artifact_io import sha256_file
from src.config import CONFIG_PATH, encoder_tag, load_config
from src.pair_encoding import _active_encoder_info, load_latent_artifact


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    config = load_config(path=queue.CONFIG)
    config["expected_counts"] = dict(factual=4, counterfactual=2, floor=1, factual_vb=1,
                                      pool=1, dev=1, eval=2)
    for key in queue.INPUT_KEYS:
        config["paths"][key] = str(tmp_path / Path(config["paths"][key]).name)
    for key in ("latents", "floor_latents", "held_out_latents"):
        config["paths"][key] = str(tmp_path / "latents" / "embeddinggemma_768" /
                                    Path(config["paths"][key]).name)
    frames = {
        "texts": {"id": [1, 2, 3, 4], "text": ["fact1", "fact2", "fact3", "fact4"]},
        "texts_counterfactual": {"id": [3, 4], "text": ["cf3", "cf4"]},
        "texts_floor": {"id": [3], "text": ["floor3"]},
        "texts_held_out": {"id": [4], "text": ["vb4"]},
        "pair_index": {"id": [1, 2, 3, 4], "split": ["train", "train", "test", "test"],
                       "is_identity": [False] * 4},
        "partition": {"id": [1, 2, 3, 4], "partition": ["pool", "dev", "eval", "eval"],
                      "floor": [False, False, True, False]},
        "interventions": {"id": [1, 2, 3, 4], "do_value": [1, 2, 3, 0]},
        "sim_factual": {"id": [1, 2, 3, 4], "X": [0, 1, 2, 3]},
        "sim_counterfactual": {"id": [1, 2, 3, 4], "X": [1, 2, 3, 0]},
    }
    for key, values in frames.items():
        pd.DataFrame(values).to_csv(config["paths"][key], index=False)
    monkeypatch.setattr(queue, "input_hashes", lambda c: {
        key: sha256_file(c["paths"][key]) for key in queue.INPUT_KEYS
    })
    return config, tmp_path / "reports"


class FakeEncoder:
    def __init__(self, fail_on_vb=False, latent_dim=768, batch_size=8, scale=1.0):
        self.calls = []
        self.fail_on_vb = fail_on_vb
        self.latent_dim, self.batch_size, self.scale = latent_dim, batch_size, scale

    def encode(self, texts, deterministic=True, batch_size=8):
        assert deterministic and batch_size == self.batch_size
        self.calls.append(list(texts))
        if self.fail_on_vb and "vb4" in texts:
            raise RuntimeError("interrupted auxiliary encoding")
        rows = []
        for text in texts:
            offset = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) / 2**32
            rows.append(torch.sin(torch.arange(self.latent_dim, dtype=torch.float32) / 17 + offset))
        return self.scale * F.normalize(torch.stack(rows), dim=1)


@pytest.mark.parametrize("encoder", sorted(queue.FULL_WIDTH))
def test_configuration_preserves_original_encoder_recipe(encoder):
    config = queue.load_encoding_config(encoder)
    old = queue.load_encoding_config(encoder, CONFIG_PATH)  # the original encoder comparison
    assert _active_encoder_info(config["encoder"]) == _active_encoder_info(old["encoder"])
    assert f"talent_motivation/{encoder_tag(config['encoder'])}/" in config["paths"]["latents"]
    assert sum(config["expected_counts"][name] for name in
               ("factual", "counterfactual", "floor", "factual_vb")) == 5700


def test_langvae_queue_accepts_unnormalized_posterior_means(corpus):
    config, report = corpus
    config["encoder"] = queue.load_encoding_config("langvae")["encoder"]
    for key in ("latents", "floor_latents", "held_out_latents"):
        path = Path(config["paths"][key])
        config["paths"][key] = str(path.parent.with_name("langvae") / path.name)
    sig = {"run_id": "tiny", "input_sha256": queue.input_hashes(config)}
    encoder = FakeEncoder(latent_dim=128, batch_size=16, scale=3.0)
    assert queue.run_queue(config, report, sig, encoder_factory=lambda _: encoder)["status"] == "complete"
    assert load_latent_artifact(config).z.norm(dim=1).allclose(torch.full((4,), 3.0))


def test_queue_roundtrip_and_completed_resume(corpus):
    config, report = corpus
    sig = {"run_id": "tiny", "input_sha256": queue.input_hashes(config)}
    encoder = FakeEncoder()
    result = queue.run_queue(config, report, sig, encoder_factory=lambda _: encoder)
    assert result["status"] == "complete"
    assert [len(call) for call in encoder.calls] == [6, 1, 1]
    artifact = load_latent_artifact(config)
    assert artifact.z.shape == (4, 768) and artifact.z_prime.shape == (2, 768)
    extra = load_text_artifact(
        config["paths"]["floor_latents"], texts=config["paths"]["texts_floor"],
        expected_ids=[3], kind="floor", encoder_config=config["encoder"],
        provenance=sig["input_sha256"],
    )
    assert not torch.equal(extra["z"][0], artifact.z[2])  # same state is not same text
    queue.run_queue(config, report, sig, encoder_factory=lambda _: pytest.fail("must reuse"))
    changed = deepcopy(sig)
    changed["run_id"] = "different"
    with pytest.raises(ValueError, match="conflicting run signature"):
        queue.run_queue(config, report, changed)


def test_interrupted_queue_reuses_only_verified_artifacts(corpus):
    config, report = corpus
    sig = {"run_id": "tiny", "input_sha256": queue.input_hashes(config)}
    encoder = FakeEncoder(fail_on_vb=True)
    with pytest.raises(RuntimeError, match="interrupted"):
        queue.run_queue(config, report, sig, encoder_factory=lambda _: encoder)
    destination = Path(config["paths"]["latents"]).parent
    assert not destination.exists()
    pending = destination.with_name(".embeddinggemma_768.tiny.pending")
    before = sha256_file(pending / "z_pairs.pt")
    resumed = FakeEncoder()
    queue.run_queue(config, report, sig, encoder_factory=lambda _: resumed)
    assert resumed.calls == [["vb4"]]
    assert sha256_file(destination / "z_pairs.pt") == before
    assert json.loads((destination / "manifest.json").read_text())["complete"]


def test_changed_inputs_and_wrong_subsets_are_rejected(corpus):
    config, report = corpus
    sig = {"run_id": "tiny", "input_sha256": queue.input_hashes(config)}
    frame = pd.read_csv(config["paths"]["texts_floor"])
    frame.loc[0, "text"] = "changed floor"
    frame.to_csv(config["paths"]["texts_floor"], index=False)
    with pytest.raises(ValueError, match="source inputs changed"):
        queue.run_queue(config, report, sig, encoder_factory=lambda _: pytest.fail("no inference"))
    frame.loc[0, "id"] = 1
    frame.to_csv(config["paths"]["texts_floor"], index=False)
    with pytest.raises(ValueError, match="non-evaluation"):
        queue.input_tables(config)


def test_partial_pair_is_retained_and_lock_is_exclusive(tmp_path):
    path = tmp_path / "z_floor.pt"
    path.write_bytes(b"interrupted result")
    queue._retain_partial(path)
    assert not path.exists()
    retained = list(tmp_path.glob(".interrupted-z_floor-*/z_floor.pt"))
    assert len(retained) == 1 and retained[0].read_bytes() == b"interrupted result"
    with queue.output_lock(tmp_path):
        with pytest.raises(RuntimeError, match="another worker"):
            with queue.output_lock(tmp_path):
                pytest.fail("lock should be exclusive")


def test_only_published_encoding_artifacts_are_trackable(tmp_path):
    root = Path(__file__).resolve().parents[1]
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text((root / ".gitignore").read_text())
    latents = "data/latents/talent_motivation"
    reports = "reports/talent_motivation/encoding/test-run"
    tracked = [f"{latents}/{tag}/{name}" for tag in ("embeddinggemma_768", "qwen3_1024", "langvae") for name in (
        "z_pairs.pt", "z_pairs.info.json", "z_floor.pt", "z_floor.info.json",
        "z_factual_vb.pt", "z_factual_vb.info.json", "manifest.json",
    )] + [f"{reports}/{name}.json" for name in ("protocol", "preflight", "status", "data_validation")]
    ignored = [f"{latents}/.embeddinggemma_768.test-run.pending/z_pairs.pt",
               f"{latents}/.encoding.lock", f"{reports}/run.log",
               f"{latents}/embeddinggemma_768/.interrupted-z_floor-test/z_floor.pt"]
    for name in tracked + ignored:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    actual = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=tmp_path, text=True,
    ).splitlines()
    assert set(tracked) <= set(actual)
    assert not set(ignored) & set(actual)
