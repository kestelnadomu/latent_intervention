from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess

import pandas as pd
import pytest
import torch
import yaml

from exp.benchmarks.reader.protocol import (
    CONFIG,
    STATE_SPACES,
    partitions,
    read_config,
    select_factual,
)
from exp.benchmarks.reader import run as workflow
from exp.benchmarks.semantic_decoder.experiment import candidate_settings
from src.config import load_config
from src.schema import ColumnSpec
from src.semantic_decoder.model import load_semantic_decoder


@pytest.fixture
def tiny(tmp_path):
    cfg = read_config()
    cfg["paths"] = {
        "models": str(tmp_path / "models"),
        "reports": str(tmp_path / "reports"),
    }
    cfg["run_id"] = "tiny"
    cfg["training"].update(epochs=3, batch_size=4, hidden_dim=8, dropout=0.0)
    cfg["training"]["search"].update(trials=2, final_seeds=[42, 43], hidden_dim=[8, 12])
    signature = {
        "inputs": {},
        "source": {},
        "candidates": candidate_settings(cfg["training"]),
    }
    ids = list(range(20))
    split = {
        "fit": ids[:8],
        "validation": ids[12:16],
        "evaluation_reader_reserved": ids[8:12],
        "unused_pool": [],
        "official_test": ids[16:],
    }
    frame = pd.DataFrame(
        {
            "id": ids,
            **{
                name: [i % (4 if name == "X" else 3) for i in ids]
                for name in ["X", "T", "D", "U", "M", "P", "L", "H", "A", "Y"]
            },
        }
    )
    factual = tmp_path / "factual.csv"
    frame.to_csv(factual, index=False)
    sim = tmp_path / "sim.yaml"
    sim.write_text("schema: synthetic\n")
    artifact = SimpleNamespace(
        ids=ids,
        test_ids=ids[16:],
        z=torch.randn(20, 4, generator=torch.Generator().manual_seed(123)),
        artifact_sha256="fake",
        encoder_info={"encoder_variant": "synthetic"},
    )
    return cfg, signature, artifact, frame, factual, sim, split


def data_for(tiny, state):
    _, _, artifact, _, factual, sim, split = tiny
    return select_factual(
        artifact,
        factual,
        [ColumnSpec(c, 4 if c == "X" else 3) for c in STATE_SPACES[state]],
        split,
        sim,
    )


def test_defaults_match_existing_search_and_scope():
    cfg = read_config()
    previous = load_config()["semantic_decoder"]
    assert candidate_settings(cfg["training"]) == candidate_settings(previous)
    assert cfg["training"]["epochs"] == 500
    assert cfg["training"]["search"]["final_seeds"] == [42, 43, 44, 45, 46]
    assert cfg["state_spaces"] == STATE_SPACES
    assert cfg["expected_counts"] == {
        "pool": 2800,
        "dev": 200,
        "eval": 1000,
        "L": 1000,
        "P": 1000,
    }


def test_roles_and_zero_rank_are_enforced(tiny):
    _, _, artifact, _, _, _, expected_split = tiny
    frame = pd.DataFrame(
        {
            "id": artifact.ids,
            "partition": ["pool"] * 12 + ["dev"] * 4 + ["eval"] * 4,
            "L_rank": list(range(1, 9)) + [0] * 12,
            "in_P": [False] * 8 + [True] * 4 + [False] * 8,
        }
    )
    sizes = {"pool": 12, "dev": 4, "eval": 4, "L": 8, "P": 4}
    assert (
        partitions(frame, artifact.ids, artifact.test_ids, sizes, 8) == expected_split
    )
    smaller = partitions(frame, artifact.ids, artifact.test_ids, sizes, 3)
    assert smaller["fit"] == [0, 1, 2]
    assert smaller["unused_pool"] == [3, 4, 5, 6, 7]
    bad = frame.copy()
    bad.loc[0, "in_P"] = True
    with pytest.raises(ValueError, match="disjoint"):
        partitions(bad, artifact.ids, artifact.test_ids, sizes, 8)
    bad = frame.copy()
    bad.loc[19, "L_rank"] = 9
    with pytest.raises(ValueError, match="pool"):
        partitions(bad, artifact.ids, artifact.test_ids, sizes, 8)
    bad = frame.copy()
    bad.loc[1, "L_rank"] = 1
    with pytest.raises(ValueError, match="unique"):
        partitions(bad, artifact.ids, artifact.test_ids, sizes, 8)
    with pytest.raises(ValueError, match="official test"):
        partitions(frame, artifact.ids, [15, 16, 17, 18], sizes, 8)


@pytest.mark.parametrize("state", STATE_SPACES)
def test_only_allowed_heads_and_labels_are_exposed(tiny, state):
    _, _, _, frame, factual, _, split = tiny
    original = data_for(tiny, state)
    assert list(original.fit_targets) == STATE_SPACES[state]
    assert list(original.validation_targets) == STATE_SPACES[state]
    assert len(original.fit_z) == 8 and len(original.validation_z) == 4
    assert not hasattr(original, "z_prime") and not hasattr(original, "test_z")
    changed = frame[["id", *STATE_SPACES[state]]].copy()
    excluded = ~changed.id.isin(split["fit"] + split["validation"])
    for name in STATE_SPACES[state]:
        changed.loc[
            excluded, name
        ] = -999  # forbidden labels must not even be validated
    changed.to_csv(factual, index=False)
    second = data_for(tiny, state)
    for a, b in (
        (original.fit_targets, second.fit_targets),
        (original.validation_targets, second.validation_targets),
    ):
        assert all(torch.equal(a[k], b[k]) for k in a)
    assert torch.equal(original.fit_z, second.fit_z)


def test_tiny_training_resume_publication_and_reload(tiny, monkeypatch):
    cfg, signature, *_ = tiny
    torch.set_num_threads(1)
    data = data_for(tiny, "S6")
    result = workflow.run_case(cfg, "S6", data, signature)
    assert len(result["final_runs"]) == 2
    assert result["columns"] == STATE_SPACES["S6"]
    for row in result["final_runs"]:
        model = load_semantic_decoder(row["checkpoint"], expected_columns=data.columns)
        assert set(model.heads) == set(STATE_SPACES["S6"])
    monkeypatch.setattr(
        workflow,
        "train_semantic_decoder",
        lambda *a, **kw: pytest.fail("completed fit retrained"),
    )
    repeat = workflow.run_case(cfg, "S6", data, signature)
    assert result["final_runs"] == repeat["final_runs"]
    assert result["selected_trial"] == repeat["selected_trial"]
    workflow.publish(cfg, [repeat], signature)
    target = Path(cfg["paths"]["models"]) / "S6/semantic_decoder.pt"
    deployed = load_semantic_decoder(target)
    source = load_semantic_decoder(repeat["final_runs"][0]["checkpoint"])
    assert torch.equal(
        deployed.log_joint(data.validation_z), source.log_joint(data.validation_z)
    )
    settings = {**cfg["training"], **signature["candidates"][0], "lr": 0.00023}
    with pytest.raises(ValueError, match="signature"):
        workflow.execute_fit(
            cfg, "S6", data, settings, 42, "search", "trial-000", signature
        )
    target.write_bytes(b"unrelated previous model")
    with pytest.raises(ValueError, match="refusing to replace"):
        workflow.publish(cfg, [repeat], signature)


def test_interrupted_fit_restarts_deterministically(tiny, monkeypatch):
    cfg, signature, *_ = tiny
    data = data_for(tiny, "S2")
    settings = {**cfg["training"], **signature["candidates"][0]}
    train = workflow.train_semantic_decoder

    def interrupted(*args, **kwargs):
        callback = kwargs["epoch_callback"]

        def stop(model, record):
            callback(model, record)
            raise RuntimeError("simulated interruption")

        kwargs["epoch_callback"] = stop
        return train(*args, **kwargs)

    monkeypatch.setattr(workflow, "train_semantic_decoder", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        workflow.execute_fit(
            cfg, "S2", data, settings, 42, "final", "seed-42", signature
        )
    checkpoint, report = workflow.fit_paths(cfg, "S2", "final", "seed-42")
    assert checkpoint.exists() and not report.exists()
    monkeypatch.setattr(workflow, "train_semantic_decoder", train)
    resumed = workflow.execute_fit(
        cfg, "S2", data, settings, 42, "final", "seed-42", signature
    )
    clean = workflow.execute_fit(
        cfg, "S2", data, settings, 42, "final", "clean", signature
    )
    a, b = (
        load_semantic_decoder(resumed["checkpoint"]),
        load_semantic_decoder(clean["checkpoint"]),
    )
    assert all(
        torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict()
    )
    assert resumed["validation_joint_nll"] == clean["validation_joint_nll"]
    history = resumed["training"]["history"]
    assert resumed["validation_joint_nll"] == min(
        row["validation_joint_nll"] for row in history
    )


def test_auxiliary_and_reserved_labels_cannot_change_training(tiny):
    cfg, signature, _, frame, factual, _, split = tiny
    settings = {**cfg["training"], **signature["candidates"][0]}
    a = workflow.execute_fit(
        cfg, "S1", data_for(tiny, "S1"), settings, 42, "final", "before", signature
    )
    frame = frame[["id", *STATE_SPACES["S1"]]].copy()
    frame.loc[
        ~frame.id.isin(split["fit"] + split["validation"]), STATE_SPACES["S1"]
    ] = -99
    frame.to_csv(factual, index=False)
    b = workflow.execute_fit(
        cfg, "S1", data_for(tiny, "S1"), settings, 42, "final", "after", signature
    )
    assert a["validation_joint_nll"] == b["validation_joint_nll"]
    assert a["training"]["best_epoch"] == b["training"]["best_epoch"]
    am, bm = (
        load_semantic_decoder(a["checkpoint"]),
        load_semantic_decoder(b["checkpoint"]),
    )
    assert all(
        torch.equal(am.state_dict()[k], bm.state_dict()[k]) for k in am.state_dict()
    )


def test_lock_and_config_guards(tmp_path):
    lock = tmp_path / ".lock"
    with workflow.exclusive(lock):
        with pytest.raises(RuntimeError, match="another writer"):
            with workflow.exclusive(lock):
                pass
    cfg = yaml.safe_load(Path(CONFIG).read_text())
    cfg["state_spaces"]["S2"] = ["X", "T", "D", "U"]
    changed = tmp_path / "bad.yaml"
    changed.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="S1-S6"):
        read_config(changed)
    with pytest.raises(ValueError, match="run-id"):
        read_config(run_id="../escape")


def test_signature_conflicts_are_rejected(tiny, monkeypatch, tmp_path):
    cfg, _, _, _, factual, *_ = tiny
    data = {state: data_for(tiny, state) for state in STATE_SPACES}
    monkeypatch.setattr(workflow, "load_data", lambda _: (data, {}))
    _, signature = workflow.prepare(cfg)
    assert signature["test_evaluation"] is False
    assert (
        json.loads((workflow.report_root(cfg) / "preflight.json").read_text())["status"]
        == "passed"
    )
    different = deepcopy(cfg)
    different["training"]["epochs"] = 9
    with pytest.raises(ValueError, match="signature changed"):
        workflow.prepare(different)
    with pytest.raises(ValueError, match="inputs changed"):
        workflow.unchanged({"inputs": {str(factual): "wrong"}, "source": {}})


def test_tracking_policy():
    visible = [
        "models/talent_motivation/embeddinggemma_768/g-independent/S1/semantic_decoder.pt",
        "models/talent_motivation/embeddinggemma_768/g-independent/S6/semantic_decoder.info.json",
        "reports/talent_motivation/reader_benchmarks/test/summary.md",
        "reports/talent_motivation/reader_benchmarks/test/S6/final/seed-46/training.json",
    ]
    hidden = [
        "models/talent_motivation/embeddinggemma_768/g-independent/S1/runs/test/search/trial-000/semantic_decoder.pt",
        "reports/talent_motivation/reader_benchmarks/test/run.log",
        "reports/talent_motivation/reader_benchmarks/test/source/src/pipeline.py",
        "reports/talent_motivation/reader_benchmarks/test/S1/train.log",
    ]

    def ignored(path):
        output = subprocess.run(
            ["git", "check-ignore", "-v", "--no-index", path],
            text=True,
            capture_output=True,
        ).stdout.strip()
        return bool(output) and not output.split("\t")[0].split(":", 2)[2].startswith(
            "!"
        )

    assert all(not ignored(path) for path in visible)
    assert all(ignored(path) for path in hidden)
