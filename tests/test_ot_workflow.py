"""Operational isolation, audit/evaluation plumbing and read-only preflight."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from src import pipeline
from src.config import load_config
from src.schema import ColumnSpec, unflatten_state_index
from src.semantic_decoder.model import SemanticDecoder
from src.latent_intervention.dispatch import train_latent_intervention_model
from src.latent_intervention.ot import OT_VARIANTS, make_model
from src.latent_intervention.ot import evaluation, protocol, workflow
from src.latent_intervention.ot.protocol import FactualPartition, FittingData


@pytest.fixture(autouse=True)
def isolated_rng():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(19)
        yield
    torch.set_num_threads(previous)


@pytest.fixture
def synthetic():
    columns = (ColumnSpec("X", 4), ColumnSpec("T", 2))
    states = unflatten_state_index(torch.arange(8), columns)
    matrix = torch.zeros(8, 8)
    matrix[torch.arange(8), 6 + states[:, 1]] = 1
    # A genuinely stochastic transition as well as exact zero-probability edges.
    matrix[0, 6], matrix[0, 7] = 0.3, 0.7
    z = torch.nn.functional.normalize(torch.randn(48, 4), dim=-1)
    parts = {}
    labels = states.repeat(6, 1)
    for name, indices in dict(
        fit=range(24), tune=range(24, 32), audit=range(32, 40)
    ).items():
        parts[name] = FactualPartition(
            z[list(indices)], labels[list(indices)], tuple(indices)
        )
    data = FittingData(parts["fit"], parts["tune"], columns, matrix)
    decoder = (
        SemanticDecoder(4, columns, hidden_dim=8, n_hidden=1)
        .eval()
        .requires_grad_(False)
    )
    kernel = SimpleNamespace(transition_matrix=lambda action: matrix.to_sparse())
    artifact = SimpleNamespace(
        ids=list(range(48)),
        train_ids=list(range(40)),
        test_ids=list(range(40, 48)),
        z=z,
        z_prime=z[40:] + 0.02,
        is_identity=labels[40:, 0] == 3,
    )
    ids = {name: list(part.ids) for name, part in parts.items()}
    ids["official_test"] = artifact.test_ids
    settings = protocol.read_settings()
    settings.update(evaluation_samples=4, evaluation_batch_size=4)
    settings["probe"].update(
        max_epochs=1,
        patience=1,
        batch_size=8,
        auditor_hidden=[4],
        auditor_warm_start=False,
    )
    metadata = dict(ids=ids, inputs={}, settings=settings)
    model = make_model("ot_gaussian", latent_dim=4, columns=columns).fit(
        data.fit.z, data.fit.states
    )
    model.checkpoint_metadata = metadata
    return SimpleNamespace(
        data=data,
        parts=parts,
        decoder=decoder,
        kernel=kernel,
        artifact=artifact,
        settings=settings,
        model=model,
        metadata=metadata,
        labels=labels,
    )


def test_read_only_preflight_and_writer_lock(synthetic, monkeypatch, tmp_path):
    config = load_config(
        encoder_variant="embeddinggemma",
        embeddinggemma_dim=256,
        manipulator_variant="ot_gaussian",
    )
    paths = {
        name: tmp_path / "never_created" / name
        for name in ("model", "protocol", "lock")
    }
    # Preflight still instantiates real 256D capacities; all data/decoder are synthetic.
    artifact = SimpleNamespace(z=torch.empty(48, 256))
    monkeypatch.setattr(
        workflow, "context", lambda _: (synthetic.settings, artifact, paths)
    )
    monkeypatch.setattr(workflow, "split_ids", lambda *_: synthetic.metadata["ids"])
    monkeypatch.setattr(workflow, "load_fitting_data", lambda *_: synthetic.data)
    monkeypatch.setattr(workflow, "decoder_for", lambda *_: synthetic.decoder)
    monkeypatch.setattr(workflow, "input_hashes", lambda *_: {})
    before = set(tmp_path.rglob("*"))
    record = workflow.preflight(config)
    assert not record["fitting_launched"] and not record["used_official_test"]
    assert record["capacity"]["1"]["training_total"] == 0
    assert set(tmp_path.rglob("*")) == before
    with workflow.run_lock(paths):
        with pytest.raises(RuntimeError, match="another writer"):
            with workflow.run_lock(paths):
                pytest.fail("concurrent writer acquired the same lock")
    with workflow.run_lock(paths):
        pass  # release also works after exceptions


def test_pipeline_routes_cli_queue_and_training_boundary(synthetic, monkeypatch):
    from src.latent_intervention.ot import __main__ as cli

    calls = []
    monkeypatch.setattr(
        workflow, "train_manipulator", lambda c: calls.append(("train", c))
    )
    monkeypatch.setattr(
        workflow, "evaluate_manipulator", lambda c: calls.append(("evaluate", c))
    )
    for variant in OT_VARIANTS:
        config = {"latent_intervention": {"variant": variant}}
        pipeline.stage_train_manipulator(config)
        pipeline.stage_evaluate(config)
    assert [stage for stage, _ in calls] == ["train", "evaluate"] * 2
    calls.clear()
    monkeypatch.setattr(
        cli, "preflight", lambda c: calls.append(("preflight", c)) or {"ready": True}
    )
    monkeypatch.setattr(cli, "train_manipulator", lambda c: calls.append(("train", c)))
    monkeypatch.setattr(sys, "argv", ["ot", "train", "--all", "--run-id", "synthetic"])
    cli.main()
    assert [stage for stage, _ in calls] == ["preflight"] * 4 + ["train"] * 4
    roots = [
        protocol.paths_for(c, dict(synthetic.settings, run_id="synthetic"))["model"]
        for stage, c in calls
        if stage == "train"
    ]
    assert len(set(roots)) == 4
    with pytest.raises(ValueError, match="dedicated workflow"):
        train_latent_intervention_model(
            synthetic.model, latents=synthetic.data.fit.z, intervention={"X": 3}
        )


def test_audit_readers_are_separate_and_portable(synthetic, monkeypatch, tmp_path):
    # One-epoch reader fitting on synthetic data only; no OT model fitting here.
    protected = [ColumnSpec(c, 3) for c in ("P", "L", "H", "A")]
    columns = [*synthetic.data.columns, *protected]
    data = {
        name: dict(
            z=part.z,
            labels=torch.cat([part.states, torch.randint(3, (len(part.z), 4))], -1),
        )
        for name, part in synthetic.parts.items()
    }
    paths = dict(readers=tmp_path / "readers.pt", model=tmp_path / "model.pt")
    synthetic.model.save(paths["model"], metadata=synthetic.metadata)
    before = {k: v.clone() for k, v in synthetic.model.state_dict().items()}
    monkeypatch.setattr(evaluation, "audit_data", lambda *_: (data, columns))
    monkeypatch.setattr(evaluation, "decoder_for", lambda *_: synthetic.decoder)
    monkeypatch.setattr(evaluation, "load_symbolic_kernel", lambda *_: synthetic.kernel)
    record = evaluation.audit(
        {"sim_config": "synthetic"}, synthetic.artifact, synthetic.model, paths
    )
    assert not record["used_oracle_targets"] and not record["used_official_test"]
    assert record["audit_ids"] == list(synthetic.parts["audit"].ids)
    assert set(record["reader_quality"]) == {"independent", "stress"}
    assert record["predictions"]["readers"]["independent"]["all"]["units"] == 8
    monkeypatch.setattr(
        evaluation, "fit_probe", lambda *a, **kw: pytest.fail("reader refit")
    )
    monkeypatch.setattr(
        evaluation,
        "fit_independent_auditor",
        lambda *a, **kw: pytest.fail("reader refit"),
    )
    readers, _ = evaluation.readers_for(paths, data, columns, synthetic.metadata)
    assert all(
        not p.requires_grad for reader in readers.values() for p in reader.parameters()
    )
    assert all(
        torch.equal(before[k], v) for k, v in synthetic.model.state_dict().items()
    )
    assert "readers" not in torch.load(paths["model"], weights_only=True)


def test_explicit_synthetic_evaluation_uses_common_metrics(synthetic, monkeypatch):
    from exp.benchmarks.latent_intervention.evaluation import per_unit_metrics

    realized = {
        c.name: synthetic.labels[:, i].clone()
        for i, c in enumerate(synthetic.data.columns)
    }
    realized["X"].fill_(3)
    monkeypatch.setattr(evaluation, "decoder_for", lambda *_: synthetic.decoder)
    monkeypatch.setattr(evaluation, "load_symbolic_kernel", lambda *_: synthetic.kernel)
    monkeypatch.setattr(pipeline, "_aligned_targets", lambda *_: realized)
    config = dict(sim_config="synthetic", paths={"sim_counterfactual": "synthetic"})
    result = evaluation.evaluate(config, synthetic.artifact, synthetic.model)
    assert result["official_test_ids"] == list(range(40, 48))
    rng = torch.Generator().manual_seed(synthetic.settings["evaluation_seed"])
    rows = []
    for start in (40, 44):
        z = synthetic.artifact.z[start : start + 4]
        samples = synthetic.model.sample_symbolic(
            z,
            {"X": 3},
            decoder=synthetic.decoder,
            h_s=synthetic.kernel,
            n_samples=4,
            generator=rng,
        )
        rows.append(
            per_unit_metrics(
                samples,
                z,
                synthetic.artifact.z_prime[start - 40 : start - 36],
                synthetic.data.fit.z.std(0, unbiased=False).clamp_min(1e-6),
                synthetic.decoder,
                synthetic.kernel,
                {"X": 3},
                {k: v[start : start + 4] for k, v in realized.items()},
            )
        )
    for key in ("energy_score", "semantic_kl", "mean_mse"):
        assert result["metrics"]["all"]["metrics"][key] == pytest.approx(
            float(torch.cat([r[key] for r in rows]).mean()), abs=1e-7
        )
    assert result["selection_frozen"] and result["regime"] == "latent_only"


def test_selected_load_checks_provenance(synthetic, monkeypatch, tmp_path):
    from src.artifact_io import sha256_file

    model_path = tmp_path / "selected.pt"
    synthetic.model.save(model_path, metadata=synthetic.metadata)
    info_path = tmp_path / "info.json"
    info_path.write_text(
        json.dumps(
            dict(checkpoint_sha256=sha256_file(model_path), metadata=synthetic.metadata)
        )
    )
    paths = dict(model=model_path, info=info_path)
    monkeypatch.setattr(workflow, "input_hashes", lambda _: {})
    config = {"latent_intervention": {"variant": "ot_gaussian"}}
    loaded = workflow.load_selected(config, synthetic.artifact, paths)
    assert torch.equal(loaded.state_mean, synthetic.model.state_mean)
    changed = json.loads(info_path.read_text())
    changed["checkpoint_sha256"] = "wrong"
    info_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="checksum"):
        workflow.load_selected(config, synthetic.artifact, paths)


def test_git_allowlist_and_launcher_syntax(tmp_path):
    root = Path(__file__).resolve().parents[1]
    subprocess.run(["bash", "-n", str(root / "scripts/run_hz_ot.sh")], check=True)
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text((root / ".gitignore").read_text())
    portable, local = set(), {"reports/talent/ot_logs/run.log"}
    for dim in (256, 768):
        for variant in OT_VARIANTS:
            model = f"models/talent/embeddinggemma_{dim}/g-independent/h-{variant}/runs/example"
            report = f"reports/talent/embeddinggemma_{dim}/g-independent/h-{variant}/runs/example"
            portable.update(
                f"{model}/selected/{name}"
                for name in (
                    "latent_intervention.pt",
                    "latent_intervention.info.json",
                    "audit_readers.pt",
                )
            )
            portable.update(
                f"{report}/{name}.json"
                for name in ("protocol", "training", "audit", "evaluation")
            )
            local.update(
                (
                    f"{model}/resume.pt",
                    f"{model}/.run.lock",
                    f"{report}/run.log",
                    f"{model}/selected/temporary.pt",
                    f"{report}/unknown.json",
                )
            )
    local.update(
        p.replace("embeddinggemma_256", "embeddinggemma_128")
        for p in list(portable)
        if "embeddinggemma_256" in p
    )
    run = subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "core.excludesFile=/dev/null",
            "check-ignore",
            "--no-index",
            "--verbose",
            "--stdin",
        ],
        input="\n".join(sorted(portable | local)) + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    matches = [line.split("\t", 1) for line in run.stdout.splitlines()]
    assert {p for _, p in matches} == portable | local
    ignored = {p for rule, p in matches if not rule.split(":", 2)[2].startswith("!")}
    assert ignored == local
