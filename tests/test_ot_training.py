"""Tiny optimization/resume and factual-only protocol tests; never fit real CV data."""

from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from src.config import load_config
from src.schema import ColumnSpec, unflatten_state_index
from src.latent_intervention.ot import OT_VARIANTS, make_model, load_model
from src.latent_intervention.ot import protocol, training
from src.latent_intervention.ot.protocol import FactualPartition, FittingData


@pytest.fixture(autouse=True)
def deterministic():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(21)
        yield
    torch.set_num_threads(previous)


@pytest.fixture
def tiny():
    columns = (ColumnSpec("X", 4), ColumnSpec("T", 2))
    states = unflatten_state_index(torch.arange(8), columns)
    matrix = torch.zeros(8, 8)
    matrix[torch.arange(8), 6 + states[:, 1]] = 1
    parts = []
    for name, copies in (("fit", 3), ("tune", 2)):
        labels = states.repeat(copies, 1)
        z = torch.randn(len(labels), 4) * 0.4 + labels[:, :1].float() * 0.25
        parts.append(
            FactualPartition(z, labels, tuple(f"{name}-{i}" for i in range(len(z))))
        )
    settings = protocol.read_settings()
    settings.update(max_epochs=3, batch_size=8, patience=5, learning_rate=0.003)
    settings["sinkhorn"].update(epsilon=0.2, tolerance=1e-6)
    return FittingData(*parts, columns, matrix), settings


def fresh(variant, columns):
    options = {} if variant == "ot_gaussian" else dict(hidden_dim=8, state_embed_dim=2)
    if variant == "ot_flow":
        options.update(ode_steps=2, ode_max_steps=8)
    with torch.random.fork_rng():
        torch.manual_seed(42)
        return make_model(variant, latent_dim=4, columns=columns, settings=options)


@pytest.mark.parametrize("variant", OT_VARIANTS)
def test_tiny_fit_restore_and_completed_idempotence(variant, tiny, tmp_path):
    data, settings = tiny
    model = fresh(variant, data.columns)
    metadata = {"inputs": {}, "tiny": True, "settings": settings}
    args = (
        data,
        settings,
        tmp_path / "selected.pt",
        tmp_path / "training.json",
        tmp_path / "resume.pt",
        metadata,
    )
    record = training.fit(model, *args)
    assert record["status"] == "complete" and record["non_oracle"]
    loaded = load_model(tmp_path / "selected.pt", expected_metadata=metadata)
    if variant != "ot_gaussian":
        assert record["best_validation"] <= record["initial_validation"]
        assert record["optimizer_updates"] > 0 and len(record["history"]) == 3
        restored = training.validation_score(
            loaded, training.validation_bank(loaded, data, settings), settings
        )
        assert restored == pytest.approx(record["best_validation"], abs=1e-7)
        current = torch.load(tmp_path / "resume.pt", weights_only=True)
        initial = fresh(variant, data.columns).state_dict()
        assert any(
            not torch.equal(current["current"][k], initial[k])
            for k, _ in model.named_parameters()
        )
    again = training.fit(fresh(variant, data.columns), *args)
    assert record == again
    with pytest.raises(ValueError, match="signature"):
        training.fit(fresh(variant, data.columns), *args[:-1], {"different": True})


@pytest.mark.parametrize("variant", ("ot_flow",))
def test_interrupted_epoch_resume_matches_uninterrupted(variant, tiny, tmp_path):
    data, settings = tiny
    metadata = {"tiny": True, "settings": settings}

    def run(name, callback=None):
        folder = tmp_path / name
        return training.fit(
            fresh(variant, data.columns),
            data,
            settings,
            folder / "selected.pt",
            folder / "training.json",
            folder / "resume.pt",
            metadata,
            epoch_callback=callback,
        )

    complete = run("complete")

    def interrupt(row):
        raise InterruptedError("synthetic interruption after an atomic epoch snapshot")

    with pytest.raises(InterruptedError):
        run("resumed", interrupt)
    resumed = run("resumed")
    assert complete["history"] == resumed["history"]
    assert complete["best_epoch"] == resumed["best_epoch"]
    for name in ("selected.pt", "resume.pt"):
        a = torch.load(tmp_path / "complete" / name, weights_only=True)
        b = torch.load(tmp_path / "resumed" / name, weights_only=True)
        field = "state_dict" if name == "selected.pt" else "current"
        for key in a[field]:
            assert torch.equal(a[field][key], b[field][key])


def test_flow_coupling_has_fixed_state_and_validation_rng_isolated(tiny):
    data, settings = tiny
    model = fresh("ot_flow", data.columns)
    model.fit_normalization(data.fit.z, data.fit.states)
    normalized = (data.fit.z - model.mean) / model.scale
    bank = training.groups(data.fit.states, data.columns, 8)
    rng = torch.Generator().manual_seed(7)
    batch = training.flow_batch(normalized, bank[2], model.states[2], settings, rng)
    assert torch.equal(batch["states"], model.states[2].expand(len(batch["x"]), -1))
    changed = normalized.clone()
    changed[torch.cat([part for i, part in enumerate(bank) if i != 2])] = 1e6
    same = training.flow_batch(
        changed, bank[2], model.states[2], settings, torch.Generator().manual_seed(7)
    )
    assert torch.equal(batch["x"], same["x"]) and torch.equal(
        batch["velocity"], same["velocity"]
    )
    state = torch.get_rng_state().clone()
    before = rng.get_state().clone()
    validation = training.validation_bank(model, data, settings)
    training.validation_score(model, validation, settings)
    assert torch.equal(state, torch.get_rng_state()) and torch.equal(
        before, rng.get_state()
    )


def test_fit_boundary_rejects_overlap_and_arbitrary_payload(tiny, tmp_path):
    data, settings = tiny
    model = fresh("ot_flow", data.columns)
    args = (settings, tmp_path / "model", tmp_path / "report", tmp_path / "resume", {})
    with pytest.raises(TypeError, match="factual"):
        training.fit(model, {"z_prime": torch.randn(5, 4)}, *args)
    with pytest.raises(ValueError, match="overlapping"):
        training.fit(
            model, FittingData(data.fit, data.fit, data.columns, data.matrix), *args
        )


def test_loader_never_exposes_oracle_or_auxiliary_labels(tmp_path, monkeypatch):
    # Production-schema loader with poisoned oracle fields and no preservation labels.
    columns = [
        ColumnSpec("X", 4),
        ColumnSpec("T", 3),
        ColumnSpec("D", 3),
        ColumnSpec("U", 3),
    ]
    z = torch.nn.functional.normalize(torch.randn(10, 4), dim=-1)
    artifact = SimpleNamespace(
        ids=list(range(10)),
        train_ids=list(range(8)),
        test_ids=[8, 9],
        z=z,
        z_prime=torch.full((2, 4), float("nan")),
    )
    ids = dict(fit=[0, 1, 2, 3], tune=[4, 5], audit=[6, 7], official_test=[8, 9])
    frame = pd.DataFrame(
        dict(id=range(10), X=[0] * 10, T=[1] * 10, D=[2] * 10, U=[0] * 10)
    )
    path = tmp_path / "factual.csv"
    frame.to_csv(path, index=False)
    config = dict(
        sim_config="unused",
        paths=dict(sim_factual=str(path), sim_counterfactual="DO_NOT_READ"),
    )
    monkeypatch.setattr(protocol, "load_latent_artifact", lambda _: artifact)
    monkeypatch.setattr(protocol, "load_schema", lambda _: (columns, None))
    monkeypatch.setattr(protocol, "load_intervention", lambda _: {"X": 3})
    parts, _ = protocol.load_factual(config, ids)
    assert set(parts) == {"fit", "tune"}
    assert set(vars(parts["fit"])) == {"z", "states", "ids"}
    artifact.z_prime.fill_(9999)
    frame["P"], frame["L"], frame["H"], frame["A"] = -999, -999, -999, -999
    frame.to_csv(path, index=False)
    again, _ = protocol.load_factual(config, ids)
    for name in parts:
        assert torch.equal(parts[name].z, again[name].z)
        assert torch.equal(parts[name].states, again[name].states)
    with pytest.raises(ValueError, match="official-training"):
        protocol.load_factual(config, ids, parts=("official_test",))


def test_existing_split_and_defaults_unchanged():
    from src.latent_intervention.editor.protocol import split_ids as editor_split
    from src.latent_intervention.base.workflow import BASE_VARIANTS
    from src.latent_intervention.flow.common import FLOW_VARIANTS

    artifact = SimpleNamespace(
        ids=list(range(5000)),
        train_ids=list(range(4000)),
        test_ids=list(range(4000, 5000)),
    )
    settings = protocol.read_settings()
    ids = protocol.split_ids(artifact, settings)
    assert ids == editor_split(artifact, settings)
    assert {k: len(v) for k, v in ids.items()} == dict(
        fit=3200, tune=400, audit=400, official_test=1000
    )
    assert len(BASE_VARIANTS) == 5 and len(FLOW_VARIANTS) == 3
    assert load_config()["latent_intervention"]["variant"] == "baseline"
