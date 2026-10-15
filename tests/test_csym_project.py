"""Contract tests for the single supported C-sym method, independent of its archive."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from src.config import load_config
from src.latent_intervention.csym import ProjectedCSym
from src.latent_intervention.csym import evaluation, protocol, training, workflow
from src.latent_intervention.csym.probes import (
    AttributeProbe,
    pack_probe,
    unpack_probe,
    fit_readers,
)
from src.latent_intervention.dispatch import (
    make_latent_intervention,
    load_latent_intervention,
    sample_counterfactual,
)
from src.schema import ColumnSpec, flat_state_index, unflatten_state_index
from src.semantic_decoder.model import SemanticDecoder


@pytest.fixture(autouse=True)
def deterministic_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(29)
        yield
    torch.set_num_threads(previous)


class Kernel:
    def __init__(self, columns):
        self.columns = columns

    def transition_matrix(self, action):
        size = 1
        for column in self.columns:
            size *= column.n_categories
        states = unflatten_state_index(torch.arange(size), self.columns)
        for i, column in enumerate(self.columns):
            if column.name in action:
                states[:, i] = action[column.name]
        destination = flat_state_index(states, self.columns)
        return torch.sparse_coo_tensor(
            torch.stack([torch.arange(size), destination]),
            torch.ones(size),
            (size, size),
        ).coalesce()


@pytest.fixture
def tiny():
    columns = [ColumnSpec("X", 4), ColumnSpec("T", 3)]
    protected = [ColumnSpec(name, 3) for name in ("P", "L", "H", "A")]
    data = dict(
        columns=columns,
        protected=protected,
        kernel=Kernel(columns),
        decoder=SemanticDecoder(16, columns, hidden_dim=12, n_hidden=1).eval(),
    )
    for name in ("fit", "tune", "audit"):
        z = torch.nn.functional.normalize(torch.randn(12, 16), dim=-1)
        labels = torch.stack(
            [torch.randint(c.n_categories, (12,)) for c in columns + protected], -1
        )
        labels[:4, 0] = 3
        data[name] = dict(z=z, labels=labels)
    guide = AttributeProbe(16, protected).eval().requires_grad_(False)
    settings = protocol.read_settings()
    settings.update(
        max_epochs=3,
        warmup_epochs=2,
        warmup_patience=2,
        patience=10,
        hidden_dim=12,
        state_embed_dim=3,
        batch_size=6,
        support_bank_size=8,
    )
    return data, guide, settings


def editor(tiny):
    data, guide, settings = tiny
    return ProjectedCSym(
        16,
        data["columns"],
        guide=guide,
        hidden_dim=settings["hidden_dim"],
        state_embed_dim=settings["state_embed_dim"],
    )


def test_coupling_identity_and_mandatory_projection(tiny):
    data, guide, _ = tiny
    model = editor(tiny)
    z = data["fit"]["z"][:4]
    posterior = torch.rand(4, 12)
    posterior = (posterior / posterior.sum(-1, keepdim=True)).requires_grad_()
    matrix = data["kernel"].transition_matrix({"X": 3}).to_dense().requires_grad_()
    locations, weights = model.components(z, posterior, matrix)
    src, dst = (matrix > 0).nonzero().unbind(1)
    torch.testing.assert_close(
        weights.exp(), posterior[:, src] * matrix[src, dst][None]
    )
    recovered = torch.zeros(4, 12).scatter_add_(
        1, dst[None].expand(4, -1), weights.exp()
    )
    torch.testing.assert_close(recovered, posterior @ matrix)
    for before, after in zip(
        guide.log_probs(z), guide.log_probs(locations.flatten(0, 1)), strict=True
    ):
        torch.testing.assert_close(
            before.exp()[None].expand(len(locations), -1, -1),
            after.reshape(len(locations), len(z), -1).exp(),
            atol=2e-5,
            rtol=0,
        )
    states = model.states[: len(z)]
    assert torch.equal(model.realize(z, states, states), z)
    same, _ = model.components(z, posterior, torch.eye(12))
    assert torch.equal(same, z[None].expand(12, -1, -1))
    locations.square().mean().backward()
    assert posterior.grad is None and matrix.grad is None and not weights.requires_grad
    assert model.effect.net[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in guide.parameters())


def test_zero_mass_and_invalid_inputs(tiny):
    data, _, _ = tiny
    model = editor(tiny)
    z = data["fit"]["z"][:2]
    posterior = torch.zeros(2, 12)
    posterior[:, 0] = 1
    matrix = data["kernel"].transition_matrix({"X": 3}).to_dense()
    locations, weights = model.components(z, posterior, matrix)
    assert torch.isfinite(locations).all() and torch.isneginf(weights).sum() == 22
    for invalid in (
        posterior * 2,
        posterior - 1,
        posterior * float("nan"),
        posterior[:, :3],
    ):
        with pytest.raises(ValueError, match="coupled"):
            model.components(z, invalid, matrix)
    with pytest.raises(ValueError, match="latents"):
        model.components(z.double(), posterior, matrix)
    with pytest.raises(ValueError, match="support"):
        model.sample_symbolic(z, {"X": 1}, decoder=data["decoder"], h_s=data["kernel"])
    with pytest.raises(ValueError, match="positive"):
        model.sample_symbolic(z, {}, decoder=None, h_s=None, n_samples=0)
    assert torch.equal(
        model.sample_symbolic(z, {}, decoder=None, h_s=None, n_samples=3),
        z[None].expand(3, -1, -1),
    )


def test_portable_checkpoint_and_shared_dispatch(tiny, tmp_path):
    data, guide, _ = tiny
    model = make_latent_intervention(
        "c_sym_project",
        latent_dim=16,
        columns=data["columns"],
        preservation_guide=guide,
        hidden_dim=12,
        state_embed_dim=3,
    )
    path = tmp_path / "editor.pt"
    model.save(path, metadata={"non_oracle": True})
    restored = load_latent_intervention(
        path,
        expected_variant="c_sym_project",
        expected_columns=data["columns"],
        expected_metadata={"non_oracle": True},
    )
    z = data["fit"]["z"]
    samples = [
        sample_counterfactual(
            m,
            z,
            {"X": 3},
            n_samples=32,
            decoder=data["decoder"],
            h_s=data["kernel"],
            generator=torch.Generator().manual_seed(77),
        )
        for m in (model, restored)
    ]
    assert torch.equal(*samples)
    assert restored.guard.rank == model.guard.rank
    with pytest.raises(ValueError, match="observed"):
        sample_counterfactual(model, z, {"X": 3}, source_states=data["fit"]["labels"])
    schema_path = tmp_path / "schema.pt"
    model.save(schema_path, metadata={"schema": [("X", 4), ("T", 3)]})
    ProjectedCSym.load(schema_path, expected_metadata={"schema": [["X", 4], ["T", 3]]})
    with pytest.raises(ValueError, match="variant"):
        load_latent_intervention(path, expected_variant="dist")
    with pytest.raises(ValueError, match="provenance"):
        ProjectedCSym.load(path, expected_metadata={"non_oracle": False})
    payload = torch.load(path, weights_only=True)
    payload["state_dict"]["scale"][0] = float("nan")
    torch.save(payload, path)
    with pytest.raises(ValueError, match="nonfinite"):
        ProjectedCSym.load(path)


def test_training_restore_resume_and_frozen_readers(tiny, tmp_path, monkeypatch):
    data, guide, settings = tiny
    frozen = copy.deepcopy(data["decoder"].state_dict())
    metadata = {"tiny": True, "inputs": {}, "settings": settings}

    def run(name):
        folder = tmp_path / name
        return training.fit(
            data,
            guide,
            settings,
            folder / "editor.pt",
            folder / "training.json",
            metadata,
        )

    complete = run("complete")
    assert complete["epochs"] == 3 and complete["roundtrip_verified"]
    assert not complete["used_official_test"] and not complete["used_oracle_targets"]
    assert all(
        torch.equal(v, data["decoder"].state_dict()[k]) for k, v in frozen.items()
    )
    assert all(p.grad is None for p in data["decoder"].parameters())
    original_write = training.write_json

    def stop_after_snapshot(path, record):
        original_write(path, record)
        if record.get("phase") == "joint_fit" and record["epoch"] == 1:
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(training, "write_json", stop_after_snapshot)
    with pytest.raises(RuntimeError, match="interruption"):
        run("resumed")
    monkeypatch.setattr(training, "write_json", original_write)
    resumed = run("resumed")
    assert resumed["best_epoch"] == complete["best_epoch"]
    a, b = [
        ProjectedCSym.load(tmp_path / name / "editor.pt")
        for name in ("complete", "resumed")
    ]
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    assert run("resumed") == resumed

    def stop_before_report(path, record):
        if Path(path).name == "training.json":
            raise RuntimeError("interrupted final publication")
        original_write(path, record)

    monkeypatch.setattr(training, "write_json", stop_before_report)
    with pytest.raises(RuntimeError, match="publication"):
        run("publication")
    monkeypatch.setattr(training, "write_json", original_write)
    restored_report = run("publication")
    restored = ProjectedCSym.load(tmp_path / "publication" / "editor.pt")
    assert restored_report["best_epoch"] == complete["best_epoch"]
    assert all(
        torch.equal(v, restored.state_dict()[k]) for k, v in a.state_dict().items()
    )


def test_factual_loader_rejects_test_before_loading(monkeypatch):
    monkeypatch.setattr(
        protocol,
        "load_latent_artifact",
        lambda _: pytest.fail("must not open artifacts"),
    )
    with pytest.raises(ValueError, match="factual"):
        protocol.load_data({}, {}, parts=("official_test",))


def test_factual_loader_never_exposes_oracle_or_audit_targets(monkeypatch):
    class FactualArtifact(SimpleNamespace):
        @property
        def z_prime(self):
            pytest.fail("oracle embeddings must not enter fitting or selection")

    columns = [ColumnSpec("X", 4), *[ColumnSpec(c, 3) for c in ("T", "D", "U")]]
    artifact = FactualArtifact(
        ids=list(range(50)),
        train_ids=list(range(40)),
        test_ids=list(range(40, 50)),
        z=torch.nn.functional.normalize(torch.randn(50, 16), dim=-1),
    )
    ids = protocol.split_ids(artifact, protocol.read_settings())
    frame = pd.DataFrame(dict(id=artifact.ids, **{c: [0] * 50 for c in "XTDUPLHA"}))

    def factual_only(path):
        assert path == "factual.csv"
        return frame

    monkeypatch.setattr(protocol.pd, "read_csv", factual_only)
    monkeypatch.setattr(protocol, "load_latent_artifact", lambda _: artifact)
    monkeypatch.setattr(protocol, "load_schema", lambda _: (columns, None))
    monkeypatch.setattr(protocol, "load_intervention", lambda _: {"X": 3})
    monkeypatch.setattr(protocol, "load_symbolic_kernel", lambda _: Kernel(columns))
    monkeypatch.setattr(protocol, "decoder_for", lambda *_: None)
    result = protocol.load_data(
        {"sim_config": "schema.yaml", "paths": {"sim_factual": "factual.csv"}}, ids
    )
    assert set(result) == {"fit", "tune", "decoder", "kernel", "columns", "protected"}
    for part in ("fit", "tune"):
        assert set(result[part]) == {"z", "labels"}
        assert torch.equal(result[part]["z"], artifact.z[ids[part]])


def test_truth_is_used_only_for_evaluation_metrics(tiny):
    data, _, settings = tiny
    model = editor(tiny)
    z = data["fit"]["z"][:3]
    matrix = data["kernel"].transition_matrix({"X": 3}).to_dense()
    with torch.no_grad():
        locations, weights = model.components(
            z, data["decoder"].log_joint(z).exp(), matrix
        )
        samples = model.sample_symbolic(
            z,
            {"X": 3},
            decoder=data["decoder"],
            h_s=data["kernel"],
            generator=torch.Generator().manual_seed(77),
        )
    realized = {
        c.name: data["fit"]["labels"][:3, i] for i, c in enumerate(data["columns"])
    }
    scores = [
        evaluation.per_unit_metrics(
            samples,
            z,
            truth,
            torch.ones(16),
            data["decoder"],
            data["kernel"],
            {"X": 3},
            realized,
        )
        for truth in (z, z + 3)
    ]
    assert not torch.equal(scores[0]["energy_score"], scores[1]["energy_score"])
    for key in ("semantic_kl", "sample_spread", "mean_shift_l2"):
        assert torch.equal(scores[0][key], scores[1][key])
    diagnostics = [
        evaluation.exact_metrics(
            locations,
            weights,
            z,
            truth,
            torch.ones(16),
            data["decoder"],
            data["kernel"],
            {"X": 3},
        )
        for truth in (z, z + 3)
    ]
    assert torch.equal(diagnostics[0]["semantic_kl"], diagnostics[1]["semantic_kl"])
    # Training/selection consume only the factual parts and named validation terms.
    terms = dict(semantic_kl=0.01, preservation_kl=0.02, identity=0.03, movement=0.04)
    from src.latent_intervention.csym.objective import selection_key

    assert selection_key(terms, 0.15) == selection_key(
        dict(terms, oracle_energy=-1e6, audit_score=1e6), 0.15
    )


def test_reader_publication_resume_and_checksums(tiny, tmp_path, monkeypatch):
    data, guide, settings = tiny
    paths = dict(readers=tmp_path / "readers.pt", probes=tmp_path / "probes.json")
    calls = []

    def fit_readers(*_):
        calls.append(1)
        return {"guide": guide}, {"readers_qualified": True}

    monkeypatch.setattr(workflow, "fit_readers", fit_readers)
    original_write = workflow.write_json

    def stop(*_):
        raise RuntimeError("reader report interrupted")

    monkeypatch.setattr(workflow, "write_json", stop)
    with pytest.raises(RuntimeError, match="interrupted"):
        workflow.prepare_readers(paths, data, settings, {"tiny": True})
    monkeypatch.setattr(workflow, "write_json", original_write)
    assert "guide" in workflow.prepare_readers(paths, data, settings, {"tiny": True})
    assert calls == [1]
    report = json.loads(paths["probes"].read_text())
    report["readers_qualified"] = False
    original_write(paths["probes"], report)
    with pytest.raises(ValueError, match="protocol"):
        workflow.prepare_readers(paths, data, settings, {"tiny": True})


def test_published_weights_are_immutable(tiny, tmp_path, monkeypatch):
    _, _, settings = tiny
    paths = dict(model=tmp_path / "model.pt", migration=tmp_path / "migration.json")
    paths["migration"].write_text("{}")
    monkeypatch.setattr(workflow, "context", lambda _: (settings, None, paths))
    with pytest.raises(FileExistsError, match="immutable"):
        workflow.train_manipulator({})


def test_provenance_rejects_changed_inputs(tmp_path):
    source = tmp_path / "input.dat"
    source.write_bytes(b"original")
    metadata = dict(inputs={str(source): protocol.sha256_file(source)})
    protocol.verify_inputs(metadata)
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="input changed"):
        protocol.verify_inputs(metadata)


def test_split_without_historical_plan():
    artifact = SimpleNamespace(
        ids=list(range(50)), train_ids=list(range(40)), test_ids=list(range(40, 50))
    )
    settings = protocol.read_settings()
    ids = protocol.split_ids(artifact, settings)
    assert {k: len(v) for k, v in ids.items()} == dict(
        fit=32, tune=4, audit=4, official_test=10
    )
    protocol.validate_ids(artifact, ids)
    ids["fit"][0] = 40
    with pytest.raises(ValueError, match="official"):
        protocol.validate_ids(artifact, ids)


def test_reader_roundtrip_and_training(tiny):
    data, _, settings = tiny
    settings = copy.deepcopy(settings)
    settings["probe"].update(
        max_epochs=1, patience=1, auditor_hidden=[8], batch_size=12
    )
    readers, report = fit_readers(data, settings)
    assert set(readers) == {"guide", "auditor", "stress_auditor"}
    assert "readers_qualified" in report
    for probe in readers.values():
        restored = unpack_probe(pack_probe(probe))
        for a, b in zip(
            probe.log_probs(data["tune"]["z"]),
            restored.log_probs(data["tune"]["z"]),
            strict=True,
        ):
            assert torch.equal(a, b)


def test_pipeline_routes_without_training(monkeypatch):
    from src import pipeline

    calls = []
    monkeypatch.setattr(
        workflow, "train_manipulator", lambda cfg: calls.append("train")
    )
    monkeypatch.setattr(
        workflow, "evaluate_manipulator", lambda cfg: calls.append("evaluate")
    )
    cfg = {"latent_intervention": {"variant": "c_sym_project"}}
    pipeline.stage_train_manipulator(cfg)
    pipeline.stage_evaluate(cfg)
    assert calls == ["train", "evaluate"]
    assert load_config()["latent_intervention"]["variant"] == "baseline"


def test_dedicated_cli_default_and_fresh_tag_queue(monkeypatch):
    import sys
    from src.latent_intervention.csym import __main__ as cli

    calls = []
    monkeypatch.setattr(cli, "evaluate_manipulator", lambda cfg: calls.append(cfg))
    monkeypatch.setattr(cli, "train_manipulator", lambda cfg: calls.append(cfg))
    monkeypatch.setattr(sys, "argv", ["csym", "evaluate"])
    cli.main()
    assert "embeddinggemma_768" in calls[0]["paths"]["manipulator_model"]
    calls.clear()
    monkeypatch.setattr(
        sys, "argv", ["csym", "train", "--all-dimensions", "--tag", "fresh-test"]
    )
    cli.main()
    for cfg, dimension in zip(calls, (768, 256), strict=True):
        path = cfg["paths"]["manipulator_model"]
        assert f"embeddinggemma_{dimension}/g-independent/h-fresh-test/" in path
        assert cfg["latent_intervention"]["variant"] == "c_sym_project"


@pytest.mark.parametrize("dimension", [256, 768])
def test_published_models_load_without_experiments(dimension, monkeypatch):
    config = load_config(
        encoder_variant="embeddinggemma",
        embeddinggemma_dim=dimension,
        decoder_variant="independent",
        manipulator_variant="c_sym_project",
    )
    paths = protocol.paths_for(config)
    if not paths["model"].exists():
        pytest.skip("published artifacts are not installed")
    original_open = Path.open

    def deny_archive(path, *args, **kwargs):
        assert (
            "archives" not in path.parts
            and "hz_preservation" not in path.parts
            and "preservation" not in path.parts
        )
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", deny_archive)
    model = workflow.load_published(paths)
    assert model.latent_dim == dimension and model.guard.rank == 8
    migration = json.loads(paths["migration"].read_text())
    assert (
        migration["component_locations_bitwise_equal"]
        and migration["seeded_samples_bitwise_equal"]
    )
    assert migration["maximum_per_unit_metric_difference"] <= 1e-5
    assert (
        protocol.sha256_file(paths["training"]) == migration["original_training_sha256"]
    )
    assert (
        protocol.sha256_file(paths["original_evaluation"])
        == migration["original_evaluation_sha256"]
    )
    original = json.loads(paths["original_evaluation"].read_text())
    evaluated = json.loads(paths["evaluation"].read_text())
    assert original["test_ids"] == evaluated["test_ids"]
    for key, values in original["per_unit"].items():
        torch.testing.assert_close(
            torch.tensor(evaluated["per_unit"][key]),
            torch.tensor(values),
            atol=1e-5,
            rtol=0,
        )
