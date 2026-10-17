"""Clean-editor mathematical, leakage, persistence and operational contracts."""

import copy
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from src.config import load_config
from src.schema import ColumnSpec, unflatten_state_index
from src.semantic_decoder.model import SemanticDecoder
from src.latent_intervention.dispatch import (
    make_latent_intervention,
    load_latent_intervention,
    sample_counterfactual,
)
from src.latent_intervention.editor import EDITOR_VARIANTS, make_model, load_model
from src.latent_intervention.editor import protocol, training, workflow, evaluation
from src.latent_intervention.editor.capacity import capacity_report, potential_count
from src.latent_intervention.editor.common import target_marginal
from src.latent_intervention.editor.models import TargetPotential
from src.latent_intervention.editor.objective import terms_for, total_loss
from src.latent_intervention.editor.protocol import FactualPartition, FittingData

COLS = (ColumnSpec("X", 4), ColumnSpec("T", 2))


@pytest.fixture(autouse=True)
def deterministic():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(27)
        yield
    torch.set_num_threads(threads)


def fresh(variant):
    options = (
        dict(d_model=8, nhead=2, dim_feedforward=16)
        if variant == "semantic_editor"
        else dict(hidden_dim=9, state_embed_dim=2)
    )
    with torch.random.fork_rng():
        torch.manual_seed(42)
        return make_model(variant, latent_dim=4, columns=COLS, settings=options)


@pytest.fixture
def tiny():
    states = unflatten_state_index(torch.arange(8), COLS)
    matrix = torch.zeros(8, 8)
    matrix[torch.arange(8), 6 + states[:, 1]] = 1
    matrix[0, 6], matrix[0, 7] = 0.3, 0.7
    labels = states.repeat(6, 1)
    z = torch.nn.functional.normalize(torch.randn(48, 4), dim=-1)
    fit = FactualPartition(z[:24], labels[:24], tuple(range(24)))
    tune = FactualPartition(z[24:32], labels[24:32], tuple(range(24, 32)))
    data = FittingData(fit, tune, COLS, matrix)
    decoder = (
        SemanticDecoder(4, COLS, hidden_dim=8, n_hidden=1).eval().requires_grad_(False)
    )
    kernel = SimpleNamespace(
        columns=COLS,
        transition_matrix=lambda a: (matrix if a else torch.eye(8)).to_sparse(),
    )
    settings = protocol.read_settings()
    settings.update(max_epochs=3, batch_size=8, learning_rate=0.003)
    artifact = SimpleNamespace(
        ids=list(range(48)),
        train_ids=list(range(40)),
        test_ids=list(range(40, 48)),
        z=z,
        z_prime=z[40:] + 0.02,
        is_identity=labels[40:, 0] == 3,
    )
    ids = dict(
        fit=list(range(24)),
        tune=list(range(24, 32)),
        audit=list(range(32, 40)),
        official_test=list(range(40, 48)),
    )
    return SimpleNamespace(
        data=data,
        decoder=decoder,
        kernel=kernel,
        settings=settings,
        artifact=artifact,
        ids=ids,
        labels=labels,
    )


@pytest.mark.parametrize("variant", EDITOR_VARIANTS)
def test_complete_target_marginal_zeros_identity_and_seeded_samples(
    variant, tiny, tmp_path
):
    model = fresh(variant).eval()
    model.fit_normalization(tiny.data.fit.z)
    if variant == "semantic_editor":
        with torch.no_grad():
            model.delta.out.weight.normal_(0, 0.05)
    z = tiny.data.fit.z[:3]
    posterior = torch.zeros(3, 8)
    posterior[0, 0], posterior[1, 1], posterior[2, 6] = 1, 1, 1
    custom = copy.deepcopy(tiny.decoder)
    custom.log_joint = lambda point: posterior.log()
    q = target_marginal(custom, tiny.kernel, z, {"X": 3}, COLS)
    torch.testing.assert_close(q, posterior @ tiny.data.matrix, atol=0, rtol=0)
    assert not q.requires_grad
    locations, weights = model.components(z, q.requires_grad_())
    torch.testing.assert_close(weights.exp(), q.detach()[:, [6, 7]])
    assert torch.isneginf(weights).sum() == 2 and not weights.requires_grad
    assert locations.shape == (2, 3, 4)
    expected_picks = torch.multinomial(
        weights.exp(), 13, replacement=True, generator=torch.Generator().manual_seed(8)
    )
    actual = sample_counterfactual(
        model,
        z,
        {"X": 3},
        decoder=custom,
        h_s=tiny.kernel,
        n_samples=13,
        generator=torch.Generator().manual_seed(8),
    )
    # Transformer inference fast paths may differ from the grad-enabled kernel by rounding.
    torch.testing.assert_close(
        actual, locations[expected_picks.t(), torch.arange(3)], atol=1e-7, rtol=1e-6
    )
    assert actual.shape == (13, 3, 4)
    assert torch.equal(
        model.sample_symbolic(z, {}, n_samples=3), z[None].expand(3, -1, -1)
    )
    assert model.sample_symbolic(z[:0], {}, n_samples=3).shape == (3, 0, 4)
    with torch.inference_mode():
        result = model.sample_symbolic(z, {"X": 3}, decoder=custom, h_s=tiny.kernel)
        assert torch.isfinite(result).all()
    for action in ({"X": 1}, {"X": 3.0}, {"T": 1}, {"X": 3, "T": 1}):
        with pytest.raises(ValueError, match="support"):
            model.sample_symbolic(z, action)
    with pytest.raises(ValueError, match="observed"):
        sample_counterfactual(model, z, {}, target_states=tiny.data.fit.states[:3])
    with pytest.raises(ValueError, match="probabilities"):
        model.components(z, torch.zeros(3, 8))
    path = tmp_path / "selected.pt"
    model.save(path, metadata={"test": True})
    restored = load_latent_intervention(
        path,
        expected_variant=variant,
        expected_columns=COLS,
        expected_metadata={"test": True},
    )
    torch.testing.assert_close(restored.components(z, q)[0], locations, atol=0, rtol=0)
    torch.testing.assert_close(
        restored.sample_symbolic(
            z,
            {"X": 3},
            decoder=custom,
            h_s=tiny.kernel,
            n_samples=13,
            generator=torch.Generator().manual_seed(8),
        ),
        actual,
        atol=0,
        rtol=0,
    )
    with pytest.raises(ValueError, match="provenance"):
        load_model(path, expected_metadata={"test": False})
    with pytest.raises(ValueError, match="variant"):
        load_model(path, expected_variant="state_flow")
    payload = torch.load(path, weights_only=True)
    payload["state_dict"]["scale"][0] = float("nan")
    torch.save(payload, path)
    with pytest.raises(ValueError, match="normalization"):
        load_model(path)


def test_conditional_convexity_exact_gradient_contraction_and_second_derivatives():
    potential = TargetPotential(3, COLS, 9, state_embed_dim=2).double()
    x, y = (
        torch.randn(12, 3, dtype=torch.float64),
        torch.randn(12, 3, dtype=torch.float64),
    )
    targets = torch.stack([torch.randint(4, (12,)), torch.randint(2, (12,))], -1)
    for fraction in (0.1, 0.5, 0.9):
        left = potential(fraction * x + (1 - fraction) * y, targets)
        right = fraction * potential(x, targets) + (1 - fraction) * potential(
            y, targets
        )
        assert (left <= right + 1e-10).all()
    point = x.clone().requires_grad_(True)
    (expected,) = torch.autograd.grad(potential(point, targets).sum(), point)
    torch.testing.assert_close(potential.gradient(x, targets), expected)
    for mode in (torch.no_grad, torch.inference_mode):
        with mode():
            torch.testing.assert_close(potential.gradient(x, targets), expected)
    moved = potential.gradient(x, targets, create_graph=True)
    moved.square().mean().backward()
    assert (
        potential.positive_hidden.grad is not None
        and torch.isfinite(potential.positive_hidden.grad).all()
    )
    with torch.no_grad():
        potential.positive_output.fill_(-50)
        potential.quadratic.fill_(math.log(math.expm1(0.5 - potential.quadratic_floor)))
        potential.linear.copy_(torch.tensor([0.1, -0.2, 0.3]))
    torch.testing.assert_close(
        potential.gradient(x, targets), 0.5 * x + potential.linear, atol=1e-8, rtol=1e-8
    )


@pytest.mark.parametrize("variant", EDITOR_VARIANTS)
def test_common_loss_matches_manual_mixture_and_freezes_symbolic_components(
    variant, tiny
):
    model = fresh(variant)
    frozen_g, frozen_h = (
        copy.deepcopy(tiny.decoder.state_dict()),
        tiny.data.matrix.clone(),
    )
    bank, targets, indices = training.prepare(
        model, tiny.data, tiny.decoder, tiny.settings
    )
    torch.testing.assert_close(model.mean, tiny.data.fit.z.mean(0), atol=0, rtol=0)
    torch.testing.assert_close(
        model.scale,
        tiny.data.fit.z.std(0, unbiased=False).clamp_min(1e-4),
        atol=0,
        rtol=0,
    )
    assert torch.equal(
        indices, torch.randperm(24, generator=torch.Generator().manual_seed(42))
    )
    z, q = tiny.data.fit.z[:8], targets["fit"][:8]
    identity = tiny.data.fit.states[:8, 0] == 3
    terms = terms_for(model, z, q, identity, tiny.decoder, bank, create_graph=True)
    samples, log_weights = model.components(z, q, create_graph=True)
    weights = log_weights.exp().t()
    decoded = tiny.decoder.log_joint(samples.flatten(0, 1)).exp().reshape(2, 8, 8)
    mixture = (decoded * weights[:, :, None]).sum(0)
    manual_kl = (q * (q.clamp_min(1e-30).log() - mixture.log())).sum(-1).mean()
    movement = (((samples - z) / model.scale).square().mean(-1) * weights).sum(0)
    support = (
        torch.cdist(((samples - model.mean) / model.scale).flatten(0, 1), bank)
        .min(-1)
        .values.reshape(2, 8)
        / 2
    )
    torch.testing.assert_close(terms["semantic_kl"], manual_kl)
    torch.testing.assert_close(terms["movement"], movement.mean())
    torch.testing.assert_close(terms["identity"], movement[identity].mean())
    torch.testing.assert_close(terms["support"], (support * weights).sum(0).mean())
    assert set(terms) == {"semantic_kl", "movement", "identity", "support"}
    total = total_loss(terms, tiny.settings)
    torch.testing.assert_close(
        total,
        manual_kl
        + 0.1 * terms["movement"]
        + 0.05 * terms["support"]
        + terms["identity"],
    )
    total.backward()
    assert all(p.grad is None for p in tiny.decoder.parameters())
    assert all(
        torch.equal(v, tiny.decoder.state_dict()[k]) for k, v in frozen_g.items()
    )
    assert torch.equal(frozen_h, tiny.data.matrix) and tiny.data.matrix.grad is None
    terms = terms_for(model, z, q, torch.zeros(8, dtype=torch.bool), tiny.decoder, bank)
    assert terms["identity"] == 0


@pytest.mark.parametrize("variant", EDITOR_VARIANTS)
@pytest.mark.parametrize("dimension", (256, 768))
@pytest.mark.parametrize("tier", (1, 2))
def test_actual_capacity_budget(variant, dimension, tier):
    columns = [
        ColumnSpec("X", 4),
        ColumnSpec("T", 3),
        ColumnSpec("D", 3),
        ColumnSpec("U", 3),
    ]
    model = make_latent_intervention(
        variant,
        latent_dim=dimension,
        columns=columns,
        seed=42,
        editor_settings={"tier": tier},
    )
    result = capacity_report(model, tier=tier)
    assert result["relative_difference"] <= 0.007 and result["training_only"] == 0
    if variant == "ot_editor":
        assert result["inference_active"] == potential_count(
            dimension, columns, model.hidden_dim, 16
        )
    else:
        assert result["inference_active"] == result["target"]
        assert model.delta.layer.dropout.p == 0


@pytest.mark.parametrize("variant", EDITOR_VARIANTS)
def test_tiny_optimization_restore_idempotence_and_exact_resume(
    variant, tiny, tmp_path
):
    metadata = dict(tiny=True, settings=tiny.settings)

    def run(name, callback=None):
        folder = tmp_path / name
        return training.fit(
            fresh(variant),
            tiny.data,
            tiny.decoder,
            tiny.settings,
            folder / "selected.pt",
            folder / "training.json",
            folder / "resume.pt",
            metadata,
            on_epoch=callback,
        )

    complete = run("complete")
    assert complete["optimizer_updates"] == 9 and complete["epochs_completed"] == 3
    assert (
        complete["selected_validation"]["objective"]
        <= complete["initial_validation"]["objective"]
    )
    saved = torch.load(tmp_path / "complete/resume.pt", weights_only=True)
    assert any(
        not torch.equal(saved["model"][k], p)
        for k, p in fresh(variant).named_parameters()
    )
    assert run("complete") == complete

    def interrupt(epoch):
        raise InterruptedError("after epoch snapshot")

    with pytest.raises(InterruptedError):
        run("resumed", interrupt)
    resumed = run("resumed")
    assert resumed["selected_epoch"] == complete["selected_epoch"]
    assert resumed["selected_validation"] == complete["selected_validation"]
    for a, b in zip(complete["history"], resumed["history"], strict=True):
        assert {k: v for k, v in a.items() if k != "elapsed_seconds"} == {
            k: v for k, v in b.items() if k != "elapsed_seconds"
        }
    for filename, field in (("selected.pt", "state_dict"), ("resume.pt", "model")):
        a = torch.load(tmp_path / "complete" / filename, weights_only=True)
        b = torch.load(tmp_path / "resumed" / filename, weights_only=True)
        assert all(torch.equal(v, b[field][k]) for k, v in a[field].items())
    loaded = load_model(tmp_path / "complete/selected.pt")
    bank, target, _ = training.prepare(loaded, tiny.data, tiny.decoder, tiny.settings)
    validation = training.validate(
        loaded, tiny.data.tune, target["tune"], tiny.decoder, bank, tiny.settings
    )
    assert validation == complete["selected_validation"]
    with pytest.raises(ValueError, match="provenance"):
        training.fit(
            fresh(variant),
            tiny.data,
            tiny.decoder,
            tiny.settings,
            tmp_path / "complete/selected.pt",
            tmp_path / "complete/training.json",
            tmp_path / "complete/resume.pt",
            {"changed": True},
        )


def test_boundary_rejects_unscoped_data_and_overlap(tiny, tmp_path):
    args = (
        tiny.decoder,
        tiny.settings,
        tmp_path / "selected",
        tmp_path / "training",
        tmp_path / "resume",
        {},
    )
    with pytest.raises(TypeError, match="factual"):
        training.fit(
            fresh("semantic_editor"), {"z_prime": tiny.artifact.z_prime}, *args
        )
    with pytest.raises(ValueError, match="overlapping"):
        training.fit(
            fresh("ot_editor"),
            FittingData(tiny.data.fit, tiny.data.fit, COLS, tiny.data.matrix),
            *args,
        )


@pytest.mark.parametrize("variant", EDITOR_VARIANTS)
def test_auxiliary_columns_and_oracle_targets_cannot_change_fit_or_selection(
    variant, tiny, monkeypatch, tmp_path
):
    columns = (
        ColumnSpec("X", 4),
        ColumnSpec("T", 3),
        ColumnSpec("D", 3),
        ColumnSpec("U", 3),
    )
    frame = pd.DataFrame(dict(id=range(48), X=tiny.labels[:, 0], T=0, D=0, U=0))
    path = tmp_path / "factual.csv"
    frame.to_csv(path, index=False)
    config = dict(
        sim_config="synthetic",
        paths=dict(sim_factual=str(path), sim_counterfactual="DO_NOT_READ"),
    )
    matrix = torch.zeros(108, 108)
    matrix[torch.arange(108), 81 + torch.arange(108) % 27] = 1
    monkeypatch.setattr(protocol, "load_latent_artifact", lambda _: tiny.artifact)
    monkeypatch.setattr(protocol, "load_schema", lambda _: (columns, None))
    monkeypatch.setattr(protocol, "load_intervention", lambda _: {"X": 3})
    monkeypatch.setattr(
        protocol,
        "load_symbolic_kernel",
        lambda _: SimpleNamespace(transition_matrix=lambda a: matrix.to_sparse()),
    )
    before = protocol.load_fitting_data(config, tiny.ids)
    assert set(vars(before)) == {"fit", "tune", "columns", "matrix"}
    assert set(vars(before.fit)) == {"z", "states", "ids"}
    frame["P"], frame["L"], frame["H"], frame["A"] = "poison", float("nan"), -9999, 9999
    frame.to_csv(path, index=False)
    tiny.artifact.z_prime.fill_(float("nan"))
    tiny.artifact.z[32:].fill_(
        float("nan")
    )  # reserved-audit and test Z are also unused.
    after = protocol.load_fitting_data(config, tiny.ids)
    for part in ("fit", "tune"):
        assert torch.equal(getattr(before, part).z, getattr(after, part).z)
        assert torch.equal(getattr(before, part).states, getattr(after, part).states)
    decoder = (
        SemanticDecoder(4, columns, hidden_dim=8, n_hidden=1)
        .eval()
        .requires_grad_(False)
    )
    settings = dict(tiny.settings, max_epochs=1)
    records = []
    for name, data in (("before", before), ("after", after)):
        options = (
            dict(d_model=8, nhead=2, dim_feedforward=16)
            if variant == "semantic_editor"
            else dict(hidden_dim=8, state_embed_dim=2)
        )
        with torch.random.fork_rng():
            torch.manual_seed(42)
            model = make_model(variant, latent_dim=4, columns=columns, settings=options)
        folder = tmp_path / name
        records.append(
            training.fit(
                model,
                data,
                decoder,
                settings,
                folder / "selected.pt",
                folder / "training.json",
                folder / "resume.pt",
                {},
            )
        )
    assert records[0]["selected_epoch"] == records[1]["selected_epoch"]
    assert records[0]["selected_validation"] == records[1]["selected_validation"]
    assert records[0]["checkpoint_sha256"] == records[1]["checkpoint_sha256"]


def test_retired_checkpoint_rejection_and_no_archive_runtime_dependencies(tmp_path):
    for payload in (
        dict(format="csym_project_v1"),
        dict(format="latent_ot_v1", variant="ot_editor"),
    ):
        path = tmp_path / "old.pt"
        torch.save(payload, path)
        with pytest.raises(ValueError, match="retired"):
            load_latent_intervention(path)
        with pytest.raises(ValueError, match="incompatible"):
            load_model(path)
    for path in Path("src/latent_intervention/editor").glob("*.py"):
        source = path.read_text()
        assert (
            "audit_probes" not in source
            and "csym." not in source
            and "archives/" not in source
        )


def test_pipeline_routing_twenty_case_queue_and_exclusive_locks(
    tiny, monkeypatch, tmp_path
):
    from src import pipeline
    from src.latent_intervention.editor import __main__ as cli

    calls = []
    monkeypatch.setattr(
        workflow, "train_manipulator", lambda c: calls.append(("train", c))
    )
    monkeypatch.setattr(
        workflow, "evaluate_manipulator", lambda c: calls.append(("evaluate", c))
    )
    monkeypatch.setattr(
        workflow,
        "preflight",
        lambda c: calls.append(("preflight", c)) or {"ready": True},
    )
    for variant in EDITOR_VARIANTS:
        pipeline.stage_train_manipulator({"latent_intervention": {"variant": variant}})
        pipeline.stage_evaluate({"latent_intervention": {"variant": variant}})
    assert [s for s, _ in calls] == ["train", "evaluate"] * 2
    calls.clear()
    monkeypatch.setattr(sys, "argv", ["editor", "train", "--all"])
    cli.main()
    assert [s for s, _ in calls] == ["preflight"] * 20 + ["train"] * 20
    roots = [
        protocol.paths_for(
            c, dict(tiny.settings, **c["latent_intervention"]["editor"]["overrides"])
        )["model"]
        for s, c in calls
        if s == "train"
    ]
    assert len(set(roots)) == 20
    with workflow.run_lock(dict(lock=tmp_path / "run.lock")):
        with pytest.raises(RuntimeError, match="another writer"):
            with workflow.run_lock(dict(lock=tmp_path / "run.lock")):
                pytest.fail("concurrent writer")


@pytest.mark.parametrize("variant", EDITOR_VARIANTS)
def test_explicit_evaluation_common_metrics_without_proxies(
    variant, tiny, monkeypatch, tmp_path
):
    from exp.benchmarks.latent_intervention.evaluation import per_unit_metrics

    model = fresh(variant).eval()
    model.fit_normalization(tiny.data.fit.z)
    model.checkpoint_metadata = dict(ids=tiny.ids, settings=tiny.settings)
    path = tmp_path / "counterfactual.csv"
    pd.DataFrame(dict(id=range(48), X=3, T=tiny.labels[:, 1])).to_csv(path, index=False)
    monkeypatch.setattr(evaluation, "decoder_for", lambda *_: tiny.decoder)
    monkeypatch.setattr(evaluation, "load_symbolic_kernel", lambda *_: tiny.kernel)
    config = dict(sim_config="synthetic", paths=dict(sim_counterfactual=str(path)))
    actual = evaluation.evaluate(config, tiny.artifact, model)
    z = tiny.artifact.z[40:]
    samples = model.sample_symbolic(
        z,
        {"X": 3},
        decoder=tiny.decoder,
        h_s=tiny.kernel,
        n_samples=64,
        generator=torch.Generator().manual_seed(20260930),
    )
    expected = per_unit_metrics(
        samples,
        z,
        tiny.artifact.z_prime,
        tiny.data.fit.z.std(0, unbiased=False).clamp_min(1e-6),
        tiny.decoder,
        tiny.kernel,
        {"X": 3},
        dict(X=torch.full((8,), 3), T=tiny.labels[40:, 1]),
    )
    for name in ("energy_score", "semantic_kl", "realized_joint_accuracy"):
        assert actual["metrics"]["all"]["metrics"][name] == pytest.approx(
            float(expected[name].mean()), abs=1e-7
        )
    assert not actual["used_auxiliary_labels"] and actual["reserved_audit"] == "unused"
    assert actual["metrics"]["identity"]["units"] == 2


def test_tracking_policy_and_launcher(tmp_path):
    root = Path(__file__).resolve().parents[1]
    subprocess.run(["bash", "-n", str(root / "scripts/run_hz_editor.sh")], check=True)
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text((root / ".gitignore").read_text())
    portable, local = (
        set(),
        {
            "reports/talent/editor_logs/run.log",
            "archives/editor-cleanup-example/archive.json",
        },
    )
    for dim in (256, 768):
        for variant in EDITOR_VARIANTS:
            m = f"models/talent/embeddinggemma_{dim}/g-independent/h-{variant}/runs/example"
            r = f"reports/talent/embeddinggemma_{dim}/g-independent/h-{variant}/runs/example"
            portable.update(
                f"{m}/seed-42/selected/{name}"
                for name in ("latent_intervention.pt", "latent_intervention.info.json")
            )
            portable.update(
                f"{r}/seed-{seed}/{name}.json"
                for seed in range(42, 47)
                for name in ("protocol", "training", "evaluation")
            )
            local.update(
                (
                    f"{m}/seed-43/selected/latent_intervention.pt",
                    f"{m}/seed-42/resume.pt",
                    f"{m}/seed-42/.run.lock",
                    f"{m}/selected/latent_intervention.pt",
                    f"{r}/seed-42/probes.json",
                    f"{r}/seed-42/run.log",
                )
            )
    process = subprocess.run(
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
    matches = [line.split("\t", 1) for line in process.stdout.splitlines()]
    assert {p for _, p in matches} == portable | local
    assert {
        p for rule, p in matches if not rule.split(":", 2)[2].startswith("!")
    } == local


def test_report_does_not_train_or_evaluate(tmp_path):
    from src.latent_intervention.editor.reporting import report
    from src.artifact_io import write_json

    config = load_config(
        encoder_variant="embeddinggemma",
        embeddinggemma_dim=256,
        manipulator_variant="semantic_editor",
    )
    config["paths"]["manipulator_model"] = str(
        tmp_path / "model/latent_intervention.pt"
    )
    config["paths"]["eval_report"] = str(tmp_path / "report/eval.json")
    text = report([(config, protocol.read_settings())])
    assert "not run" in text and "semantic_editor" in text
    assert not list(tmp_path.iterdir())
    cases = []
    for seed in (42, 43):
        settings = dict(protocol.read_settings(), seed=seed)
        cases.append((config, settings))
        paths = protocol.paths_for(config, settings)
        write_json(
            paths["training"],
            dict(
                selected_epoch=2,
                selected_validation={"objective": 0.1},
                capacity={"tier": 2},
                checkpoint_sha256="synthetic",
            ),
        )
        write_json(
            paths["evaluation"],
            dict(
                variant="semantic_editor",
                dimension=256,
                seed=seed,
                tier=2,
                checkpoint_sha256="synthetic",
                metrics={
                    part: dict(
                        units=8, metrics={"energy_score": 0.2 + (seed - 42) * 0.2}
                    )
                    for part in ("all", "identity", "nonidentity")
                },
            ),
        )
    text = report(cases)
    assert "2 training seed(s): multi-seed" in text and "256D, tier 2" in text
    assert "energy_score: 0.3 ±" in text


def test_decoder_compatibility_is_not_a_proxy_or_oracle_gate(
    tiny, tmp_path, monkeypatch
):
    path = tmp_path / "g.pt"
    encoder = {"encoder": "synthetic", "latent_dimension": 4}
    tiny.decoder.save(
        path,
        metadata=dict(
            latent_artifact_sha256="historical-paired-payload",
            encoder=encoder,
            training_inputs={"sim_factual_sha256": "historical-full-csv"},
        ),
    )
    config = dict(
        sim_config="synthetic",
        paths=dict(decoder_model=str(path), sim_factual="DO_NOT_READ"),
    )
    artifact = SimpleNamespace(
        z=tiny.data.fit.z,
        encoder_info=encoder,
        artifact_sha256="changed-oracle-payload",
    )
    monkeypatch.setattr(protocol, "load_schema", lambda _: (COLS, None))
    model = protocol.decoder_for(config, artifact)
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    for key, value in tiny.decoder.state_dict().items():
        assert torch.equal(model.state_dict()[key], value)
    artifact.encoder_info = dict(encoder, encoder="incompatible-encoder")
    with pytest.raises(ValueError, match="metadata mismatch"):
        protocol.decoder_for(config, artifact)


@pytest.mark.parametrize("variant", EDITOR_VARIANTS)
def test_selected_loading_and_evaluation_with_archives_unavailable(
    variant, tiny, monkeypatch, tmp_path
):
    import builtins
    import io
    from src.artifact_io import write_json, sha256_file

    def deny_archive(original):
        def guarded(path, *args, **kwargs):
            if isinstance(path, (str, Path)) and "archives" in Path(path).parts:
                raise FileNotFoundError("historical archives unavailable")
            return original(path, *args, **kwargs)

        return guarded

    monkeypatch.setattr(builtins, "open", deny_archive(builtins.open))
    monkeypatch.setattr(io, "open", deny_archive(io.open))
    paths = dict(model=tmp_path / "selected.pt", info=tmp_path / "info.json")
    metadata = dict(ids=tiny.ids, inputs={}, settings=tiny.settings)
    model = fresh(variant).eval()
    model.fit_normalization(tiny.data.fit.z)
    model.save(paths["model"], metadata=metadata)
    write_json(
        paths["info"],
        dict(checkpoint_sha256=sha256_file(paths["model"]), metadata=metadata),
    )
    monkeypatch.setattr(workflow, "input_hashes", lambda *_: {})
    monkeypatch.setattr(workflow, "load_fitting_data", lambda *_: tiny.data)
    config = dict(latent_intervention=dict(variant=variant), sim_config="synthetic")
    restored = workflow.load_selected(config, tiny.artifact, paths)
    path = tmp_path / "cf.csv"
    pd.DataFrame(dict(id=range(48), X=3, T=tiny.labels[:, 1])).to_csv(path, index=False)
    config["paths"] = dict(sim_counterfactual=str(path))
    monkeypatch.setattr(evaluation, "decoder_for", lambda *_: tiny.decoder)
    monkeypatch.setattr(evaluation, "load_symbolic_kernel", lambda *_: tiny.kernel)
    assert (
        evaluation.evaluate(config, tiny.artifact, restored)["metrics"]["all"]["units"]
        == 8
    )
    monkeypatch.setattr(workflow, "input_hashes", lambda *_: {"changed": True})
    with pytest.raises(ValueError, match="inputs differ"):
        workflow.load_selected(config, tiny.artifact, paths)


def test_conflicting_run_signatures_reject_before_fitting(tiny, tmp_path, monkeypatch):
    from src.artifact_io import write_json

    paths = {
        name: tmp_path / name
        for name in ("protocol", "lock", "model", "info", "resume", "training")
    }
    write_json(paths["protocol"], {"different": True})
    monkeypatch.setattr(
        workflow, "context", lambda _: (tiny.settings, tiny.artifact, paths)
    )
    monkeypatch.setattr(workflow, "split_ids", lambda *_: tiny.ids)
    monkeypatch.setattr(workflow, "load_fitting_data", lambda *_: tiny.data)
    monkeypatch.setattr(workflow, "decoder_for", lambda *_: tiny.decoder)
    monkeypatch.setattr(workflow, "metadata_for", lambda *_: {"expected": True})
    monkeypatch.setattr(
        training,
        "fit",
        lambda *a, **kw: pytest.fail("fit ran with a conflicting signature"),
    )
    with pytest.raises(ValueError, match="signature changed"):
        workflow.train_manipulator({})
    assert not paths["model"].exists() and not paths["resume"].exists()
